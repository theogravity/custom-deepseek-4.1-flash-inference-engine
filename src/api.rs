//! Local HTTP adapter for LiteLLM's Anthropic and OpenAI routes.
use crate::{
    chat::{self, Delta, Parser},
    model::{Model, Session, Step},
};
#[cfg(test)]
mod context_tests;
mod scheduler;
use crate::swap::{SwapStats, SwapStore};
use anyhow::{anyhow, ensure, Context, Result};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::{get, post},
    Json, Router,
};
#[cfg(test)]
use scheduler::BatchEngine;
use scheduler::{schedule, Limits, ServingModel};
use serde_json::{json, Value};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokenizers::{
    AddedVocabulary, DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper, Tokenizer,
};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

pub const MODEL_NAME: &str = "DeepSeek V4.1 Flash";
const QUEUE: usize = 8;
const MAX_SEQS: usize = 6;
const WRITE_TIMEOUT: Duration = Duration::from_secs(3);
const BODY_LIMIT: usize = 32 << 20;
const DEFAULT_MAX_TOKENS: u64 = 4096;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct ContextOverflow {
    window: usize,
    prompt: usize,
    output: Option<(&'static str, u64)>,
}
impl std::fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            window,
            prompt,
            output,
        } = self;
        match output {
            Some((field, requested)) => write!(
                f,
                "This model's maximum context length is {window} tokens ({MODEL_NAME}). \
                 The prompt contains {prompt} tokens and {field} requests {requested} output tokens. \
                 At most {} output tokens fit with this prompt.",
                window.saturating_sub(*prompt),
            ),
            None => write!(
                f,
                "prompt is too long: {prompt} tokens > {} maximum for generation. \
                 This model's maximum context length is {window} tokens ({MODEL_NAME}). \
                 Shorten the input to leave room for at least one output token.",
                window.saturating_sub(1),
            ),
        }
    }
}
impl std::error::Error for ContextOverflow {}

#[derive(Clone, Copy, PartialEq)]
enum Wire {
    Anthropic,
    OpenAi,
}
impl Wire {
    fn error(self, code: StatusCode, msg: &str) -> Response {
        let ty = match code {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::INTERNAL_SERVER_ERROR => "api_error",
            _ => "invalid_request_error",
        };
        let body = match self {
            Self::Anthropic => json!({"type":"error","error":{"type":ty,"message":msg}}),
            Self::OpenAi => json!({"error":{"message":msg,"type":ty,"param":null,"code":null}}),
        };
        (code, Json(body)).into_response()
    }

    fn bad_request(self, error: anyhow::Error) -> Response {
        if let (Self::OpenAi, Some(e)) = (self, error.downcast_ref::<ContextOverflow>()) {
            let param = e.output.map_or("messages", |(field, _)| field);
            let body = json!({"error": {"message":e.to_string(), "type":"invalid_request_error",
                                        "param":param, "code":"context_length_exceeded"}});
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
        self.error(StatusCode::BAD_REQUEST, &format!("{error:#}"))
    }
}

struct ServerState {
    key: String,
    tokenizer: Arc<Tokenizer>,
    plain: Tokenizer,
    queue: mpsc::Sender<Job>,
    context: usize,
    encoders: Arc<Semaphore>,
    metrics: Arc<Mutex<Metrics>>,
    swap: SwapStore,
}
struct Prepared {
    ids: Vec<u32>,
    thinking: bool,
    tools: bool,
    max_tokens: usize,
    requested: Option<u64>,
    temperature: f32,
    stops: Vec<String>,
    stream: bool,
    include_usage: bool,
}
fn prepare(req: Value, state: &ServerState, count: bool) -> Result<Prepared> {
    ensure!(
        req["model"].as_str() == Some(MODEL_NAME),
        "unknown or missing model (expected {MODEL_NAME})"
    );
    let (prompt, thinking, tools) = chat::render(&req)?;
    if !count {
        let last = req["messages"]
            .as_array()
            .and_then(|m| m.last())
            .context("a final user or tool message is required")?;
        ensure!(
            matches!(
                last["role"].as_str(),
                Some("user" | "tool" | "developer" | "system")
            ),
            "assistant-prefill continuations are unsupported"
        );
        ensure!(
            req["messages"].as_array().unwrap().len() > 1
                || !matches!(last["role"].as_str(), Some("system" | "developer")),
            "a user message is required"
        );
    }
    let ids = prompt.encode(&state.tokenizer, &state.plain)?;
    if !count && ids.len() >= state.context {
        return Err(ContextOverflow {
            window: state.context,
            prompt: ids.len(),
            output: None,
        }
        .into());
    }
    let field = if !req["max_completion_tokens"].is_null() {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    let requested = match req.get(field) {
        None | Some(Value::Null) => None,
        Some(v) => {
            let value = v
                .as_u64()
                .filter(|i| *i > 0)
                .with_context(|| format!("{field} must be a positive integer"))?;
            if !count && value > state.context as u64 {
                return Err(ContextOverflow {
                    window: state.context,
                    prompt: ids.len(),
                    output: Some((field, value)),
                }
                .into());
            }
            Some(value)
        }
    };
    let max_tokens = requested
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .min(state.context.saturating_sub(ids.len()) as u64) as usize;
    let temperature = match req.get("temperature") {
        None | Some(Value::Null) => 1.,
        Some(v) => v
            .as_f64()
            .filter(|t| (0.0..=2.0).contains(t))
            .context("temperature must be between 0 and 2")? as f32,
    };
    ensure!(
        req["n"].is_null() || req["n"].as_u64() == Some(1),
        "only n=1 is supported"
    );
    let stop = req.get("stop_sequences").or(req.get("stop"));
    let stops = match stop {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .context("stop sequences must be strings")
            })
            .collect::<Result<_>>()?,
        _ => anyhow::bail!("invalid stop sequences"),
    };
    ensure!(
        stops.len() <= 16 && stops.iter().all(|s| !s.is_empty() && s.len() <= 1024),
        "up to 16 nonempty stop sequences of at most 1024 bytes are supported"
    );
    let stream = match req.get("stream") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().context("stream must be a boolean")?,
    };
    Ok(Prepared {
        ids,
        thinking,
        tools,
        max_tokens,
        requested,
        temperature,
        stops,
        stream,
        include_usage: req["stream_options"]["include_usage"] == true,
    })
}

// A step may emit a burst of verified tokens. Stage only the remainder of
// that step if the HTTP body cannot take it; other sessions stay runnable.
struct EventChannel {
    sender: mpsc::Sender<Result<Event, Infallible>>,
    pending: RefCell<VecDeque<Event>>,
    last_progress: Cell<Option<Instant>>,
}
impl EventChannel {
    fn new(sender: mpsc::Sender<Result<Event, Infallible>>) -> Self {
        Self {
            sender,
            pending: RefCell::default(),
            last_progress: Cell::new(None),
        }
    }
    fn send(&self, event: Event) -> Result<()> {
        if !self.pending.borrow().is_empty() {
            self.pending.borrow_mut().push_back(event);
            return Ok(());
        }
        match self.sender.try_send(Ok(event)) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(anyhow!("client disconnected")),
            Err(mpsc::error::TrySendError::Full(Ok(event))) => {
                self.pending.borrow_mut().push_back(event);
                self.last_progress.set(Some(Instant::now()));
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(Err(never))) => match never {},
        }
    }
    fn ready(&self) -> Result<bool> {
        ensure!(!self.sender.is_closed(), "client disconnected");
        let mut pending = self.pending.borrow_mut();
        while !pending.is_empty() {
            match self.sender.try_reserve() {
                Ok(slot) => {
                    slot.send(Ok(pending.pop_front().unwrap()));
                    self.last_progress.set(Some(Instant::now()));
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(anyhow!("client disconnected"))
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    ensure!(
                        self.last_progress
                            .get()
                            .is_some_and(|t| t.elapsed() < WRITE_TIMEOUT),
                        "client stopped reading its response"
                    );
                    return Ok(false);
                }
            }
        }
        self.last_progress.set(None);
        Ok(true)
    }
    // Once inference finishes, drain any terminal frames on the async runtime.
    // A slow socket never holds a finished model session or the compute thread.
    fn finish(self, runtime: &tokio::runtime::Handle) {
        let pending = self.pending.into_inner();
        if !pending.is_empty() && !self.sender.is_closed() {
            runtime.spawn(async move {
                for event in pending {
                    if self
                        .sender
                        .send_timeout(Ok(event), WRITE_TIMEOUT)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
    }
}
enum Reply {
    Json(oneshot::Sender<Result<Value>>),
    Stream(EventChannel),
}
impl Reply {
    fn stream(&self) -> Option<&EventChannel> {
        match self {
            Self::Json(_) => None,
            Self::Stream(s) => Some(s),
        }
    }
    fn closed(&self) -> bool {
        match self {
            Self::Json(s) => s.is_closed(),
            Self::Stream(s) => s.sender.is_closed(),
        }
    }
}
struct Job {
    input: Prepared,
    wire: Wire,
    reply: Reply,
}

async fn post_request(state: Arc<ServerState>, body: Bytes, wire: Wire, count: bool) -> Response {
    let slot = match state.encoders.clone().try_acquire_owned() {
        Ok(slot) => slot,
        Err(_) => return wire.error(StatusCode::TOO_MANY_REQUESTS, "too many pending requests"),
    };
    let task_state = state.clone();
    let input = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let req: Value = serde_json::from_slice(&body).context("invalid request JSON")?;
        prepare(req, &task_state, count)
    })
    .await;
    let input = match input {
        Ok(Ok(i)) => i,
        Ok(Err(e)) => return wire.bad_request(e),
        Err(_) => return wire.error(StatusCode::INTERNAL_SERVER_ERROR, "tokenizer worker failed"),
    };
    if count {
        return Json(json!({"input_tokens":input.ids.len()})).into_response();
    }
    let slot = match state.queue.try_reserve() {
        Ok(s) => s,
        Err(mpsc::error::TrySendError::Full(_)) => {
            return wire.error(StatusCode::TOO_MANY_REQUESTS, "inference queue is full")
        }
        Err(_) => {
            return wire.error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference worker is unavailable",
            )
        }
    };
    let requested = input.requested;
    let effective = input.max_tokens as u64;
    let mut response = if input.stream {
        let (send, recv) = mpsc::channel(32);
        slot.send(Job {
            input,
            wire,
            reply: Reply::Stream(EventChannel::new(send)),
        });
        Sse::new(ReceiverStream::new(recv))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let (send, recv) = oneshot::channel();
        slot.send(Job {
            input,
            wire,
            reply: Reply::Json(send),
        });
        match recv.await {
            Ok(Ok(v)) => Json(v).into_response(),
            Ok(Err(e)) => wire.error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
            Err(_) => wire.error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference worker stopped",
            ),
        }
    };
    if response.status().is_success() && requested.unwrap_or(DEFAULT_MAX_TOKENS) > effective {
        let headers = response.headers_mut();
        headers.insert("x-ds4-effective-max-tokens", effective.into());
        if let Some(n) = requested {
            headers.insert("x-ds4-requested-max-tokens", n.into());
        } else {
            headers.insert("x-ds4-default-max-tokens", DEFAULT_MAX_TOKENS.into());
        }
    }
    response
}
fn allowed(headers: &HeaderMap, key: &str) -> bool {
    let bearer = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, v)| v);
    bearer == Some(key) || headers.get("x-api-key").and_then(|h| h.to_str().ok()) == Some(key)
}
async fn authenticate(State(state): State<Arc<ServerState>>, req: Request, next: Next) -> Response {
    if allowed(req.headers(), &state.key) {
        next.run(req).await
    } else {
        let wire = if req.uri().path().starts_with("/v1/messages") {
            Wire::Anthropic
        } else {
            Wire::OpenAi
        };
        wire.error(StatusCode::UNAUTHORIZED, "a valid API key is required")
    }
}
async fn models() -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":MODEL_NAME,"object":"model","created":0,"owned_by":"local"}]}),
    )
}
#[derive(Default, Clone, serde::Serialize)]
struct Metrics {
    max_seqs: usize,
    max_live_seqs: usize,
    swap_quantum_steps: usize,
    requests_started: u64,
    prefix_reused_tokens: u64,
    logical_active: usize,
    peak_logical_active: usize,
    resident: usize,
    waiting: usize,
    swapped: usize,
    peak_swapped: usize,
    rotations: u64,
    resumes: u64,
    suspend_failures: u64,
    resume_failures: u64,
    swap: SwapStats,
    active: usize,
    prefilling: usize,
    decoding: usize,
    peak_active: usize,
    batch_steps: u64,
    multi_sequence_steps: u64,
    multi_decode_steps: u64,
    peak_batch_size: usize,
    peak_decode_batch: usize,
    processed_tokens: usize,
    peak_processed_tokens: usize,
    six_target_passes: u64,
    multi_target_passes: u64,
    peak_target_batch: usize,
}
async fn metrics(State(s): State<Arc<ServerState>>) -> Json<Metrics> {
    let mut sample = s.metrics.lock().unwrap().clone();
    (sample.multi_target_passes, sample.peak_target_batch) =
        crate::model::batch::target_batch_stats();
    sample.six_target_passes = crate::model::batch::six_target_passes();
    sample.swap = s.swap.stats();
    Json(sample)
}
fn router(state: Arc<ServerState>) -> Router {
    let private = Router::new()
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route(
            "/v1/messages",
            post(|State(s), b| post_request(s, b, Wire::Anthropic, false)),
        )
        .route(
            "/v1/messages/count_tokens",
            post(|State(s), b| post_request(s, b, Wire::Anthropic, true)),
        )
        .route(
            "/v1/chat/completions",
            post(|State(s), b| post_request(s, b, Wire::OpenAi, false)),
        )
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    private
        .route("/health", get(|| async { StatusCode::OK }))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
}
pub fn serve(model: Model, tokenizer: Tokenizer, addr: &str, key: String) -> Result<()> {
    ensure!(!key.trim().is_empty(), "a nonempty API key is required");
    let mut plain = tokenizer.clone();
    plain.with_added_vocabulary(AddedVocabulary::new());
    let limits = Limits::configured()?;
    let swap = SwapStore::configured()?;
    let (queue, jobs) = mpsc::channel(limits.live.max(QUEUE));
    let metrics = Arc::new(Mutex::new(Metrics {
        max_seqs: limits.resident,
        max_live_seqs: limits.live,
        swap_quantum_steps: limits.quantum,
        ..Metrics::default()
    }));
    let state = Arc::new(ServerState {
        key,
        tokenizer: Arc::new(tokenizer),
        plain,
        queue,
        context: model.max_context(),
        encoders: Arc::new(Semaphore::new(limits.live.max(QUEUE))),
        swap: swap.clone(),
        metrics: metrics.clone(),
    });
    let tokenizer = state.tokenizer.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let app = router(state);
    let handle = runtime.handle().clone();
    let mut engine = ServingModel { model, swap };
    let worker = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || schedule(&mut engine, &tokenizer, jobs, limits, &metrics, &handle))?;
    let served = runtime.block_on(async move {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        let mut interrupt =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!("serving {MODEL_NAME} on {}", listener.local_addr()?);
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tokio::select! { _ = terminate.recv() => (), _ = interrupt.recv() => () }
            })
            .await
            .map_err(anyhow::Error::from)
    });
    // Dropping the router closes the job queue. Keep Tokio's runtime alive
    // while outstanding encoders and the worker release their resources.
    let joined = worker
        .join()
        .map_err(|_| anyhow!("inference worker panicked"));
    served.and(joined)
}

#[derive(PartialEq, Clone, Copy)]
enum Kind {
    Text,
    Reason,
}
struct Output {
    wire: Wire,
    id: String,
    created: u64,
    prompt: usize,
    blocks: Vec<Value>,
    active: Option<Kind>,
    n_calls: usize,
    include_usage: bool,
}
impl Output {
    fn new(wire: Wire, input: &Prepared) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let prefix = if wire == Wire::Anthropic {
            "msg"
        } else {
            "chatcmpl"
        };
        let id = format!(
            "{prefix}-{created:x}-{:x}",
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        );
        Self {
            wire,
            id,
            created,
            prompt: input.ids.len(),
            blocks: vec![],
            active: None,
            n_calls: 0,
            include_usage: input.include_usage,
        }
    }
    fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":MODEL_NAME,
            "choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
    }
    fn anthropic(
        &self,
        blocks: &[Value],
        reason: Option<&str>,
        stop: Option<&str>,
        count: usize,
    ) -> Value {
        json!({"id":self.id,"type":"message","role":"assistant","model":MODEL_NAME,"content":blocks,
            "stop_reason":reason,"stop_sequence":stop,"usage":{"input_tokens":self.prompt,"output_tokens":count}})
    }
    fn start(&self, tx: Option<&EventChannel>) -> Result<()> {
        if tx.is_none() {
            return Ok(());
        }
        match self.wire {
            Wire::Anthropic => send(
                tx,
                Some("message_start"),
                json!({"type":"message_start","message":self.anthropic(&[],None,None,0)}),
            ),
            Wire::OpenAi => send(
                tx,
                None,
                self.chunk(json!({"role":"assistant","content":""}), None),
            ),
        }
    }
    fn close_block(&mut self, tx: Option<&EventChannel>) -> Result<()> {
        if self.active.take().is_some() && self.wire == Wire::Anthropic && tx.is_some() {
            let index = self.blocks.len() - 1;
            send(
                tx,
                Some("content_block_stop"),
                json!({"type":"content_block_stop","index":index}),
            )?;
        }
        Ok(())
    }
    fn append(&mut self, delta: Delta, tx: Option<&EventChannel>) -> Result<()> {
        let (kind, text, field, openai) = match delta {
            Delta::Reason(s) => (Kind::Reason, s, "thinking", "reasoning_content"),
            Delta::Text(s) => (Kind::Text, s, "text", "content"),
            Delta::Tool(c) => {
                self.close_block(tx)?;
                let id = format!("call-{}-{}", self.id, self.n_calls);
                let i = self.blocks.len();
                if tx.is_some() {
                    match self.wire {
                    Wire::Anthropic => {
                        send(tx,Some("content_block_start"),json!({"type":"content_block_start","index":i,"content_block":{"type":"tool_use","id":id,"name":c.name,"input":{}}}))?;
                        send(tx,Some("content_block_delta"),json!({"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":c.args.to_string()}}))?;
                        send(tx,Some("content_block_stop"),json!({"type":"content_block_stop","index":i}))?;
                    }
                    Wire::OpenAi => send(tx,None,self.chunk(json!({"tool_calls":[{"index":self.n_calls,"type":"function","id":id,"function":{"name":c.name,"arguments":c.args.to_string()}}]}),None))?
                }
                }
                self.n_calls += 1;
                self.blocks
                    .push(json!({"type":"tool_use","id":id,"name":c.name,"input":c.args}));
                return Ok(());
            }
        };
        if self.active != Some(kind) {
            self.close_block(tx)?;
            let mut block = json!({"type":field,field:""});
            if kind == Kind::Reason {
                block["signature"] = json!("")
            }
            if self.wire == Wire::Anthropic && tx.is_some() {
                send(
                    tx,
                    Some("content_block_start"),
                    json!({"type":"content_block_start","index":self.blocks.len(),"content_block":block}),
                )?;
            }
            self.blocks.push(block);
            self.active = Some(kind);
        }
        let index = self.blocks.len() - 1;
        if tx.is_none() {
            if let Value::String(s) = &mut self.blocks[index][field] {
                s.push_str(&text)
            }
        } else if self.wire == Wire::Anthropic {
            send(
                tx,
                Some("content_block_delta"),
                json!({"type":"content_block_delta","index":index,"delta":{"type":format!("{field}_delta"),field:text}}),
            )?;
        } else {
            send(tx, None, self.chunk(json!({openai:text}), None))?;
        }
        Ok(())
    }
    fn finish(
        &mut self,
        tx: Option<&EventChannel>,
        length: bool,
        stop: Option<&str>,
        count: usize,
    ) -> Result<Value> {
        self.close_block(tx)?;
        let reason = if stop.is_some() {
            "stop_sequence"
        } else if length {
            "max_tokens"
        } else if self.n_calls > 0 {
            "tool_use"
        } else {
            "end_turn"
        };
        if self.wire == Wire::Anthropic {
            if tx.is_some() {
                send(
                    tx,
                    Some("message_delta"),
                    json!({"type":"message_delta","delta":{"stop_reason":reason,"stop_sequence":stop},"usage":{"output_tokens":count}}),
                )?;
                send(tx, Some("message_stop"), json!({"type":"message_stop"}))?;
                return Ok(Value::Null);
            }
            return Ok(self.anthropic(&self.blocks, Some(reason), stop, count));
        }
        let reason = match reason {
            "max_tokens" => "length",
            "tool_use" => "tool_calls",
            _ => "stop",
        };
        let usage = json!({"prompt_tokens":self.prompt,"completion_tokens":count,"total_tokens":self.prompt+count});
        if tx.is_some() {
            send(tx, None, self.chunk(json!({}), Some(reason)))?;
            if self.include_usage {
                let mut last = self.chunk(json!({}), None);
                last["choices"] = json!([]);
                last["usage"] = usage;
                send(tx, None, last)?;
            }
            done(tx)?;
            return Ok(Value::Null);
        }
        let mut answer = String::new();
        let mut reasoning = String::new();
        let mut calls = vec![];
        for b in &self.blocks {
            match b["type"].as_str() {
                Some("text") => answer.push_str(b["text"].as_str().unwrap()),
                Some("thinking") => reasoning.push_str(b["thinking"].as_str().unwrap()),
                Some("tool_use") => calls.push(json!({"id":b["id"],"type":"function","function":{"name":b["name"],"arguments":b["input"].to_string()}})),
                _ => unreachable!()
            }
        }
        let mut msg = json!({"role":"assistant","content":answer});
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        if !calls.is_empty() {
            msg["tool_calls"] = json!(calls);
        }
        Ok(
            json!({"id":self.id,"object":"chat.completion","created":self.created,"model":MODEL_NAME,
            "choices":[{"index":0,"message":msg,"finish_reason":reason}],"usage":usage}),
        )
    }
}
fn send(tx: Option<&EventChannel>, name: Option<&str>, value: Value) -> Result<()> {
    if let Some(tx) = tx {
        let mut event = Event::default().data(value.to_string());
        if let Some(n) = name {
            event = event.event(n);
        }
        tx.send(event)?;
    }
    Ok(())
}
fn done(tx: Option<&EventChannel>) -> Result<()> {
    if let Some(tx) = tx {
        tx.send(Event::default().data("[DONE]"))?
    }
    Ok(())
}
type Decoder<'t> = DecodeStream<
    't,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

struct Active<'t, S> {
    session: S,
    decoder: Decoder<'t>,
    parser: Parser,
    output: Output,
    reply: Reply,
    count: usize,
    ready: bool,
    slice_steps: usize,
    suspend_retry: Option<(Instant, Duration)>,
}
impl<S> Active<'_, S> {
    fn consume(&mut self, step: Step) -> Result<Option<Value>> {
        let stream = self.reply.stream();
        for token in step.tokens {
            self.count += 1;
            if let Some(text) = self.decoder.step(token).map_err(|e| anyhow!("{e}"))? {
                for delta in self.parser.push(&text, false)? {
                    self.output.append(delta, stream)?;
                }
            }
            if self.parser.stop.is_some() {
                break;
            }
        }
        if !step.finished && self.parser.stop.is_none() {
            return Ok(None);
        }
        let stopped = self.parser.stop.is_some();
        if step.eos && !stopped {
            self.count += 1;
        }
        for delta in self.parser.push("", true)? {
            self.output.append(delta, stream)?;
        }
        let length = step.finished && !step.eos && !stopped;
        ensure!(
            !self.parser.incomplete_call() || length,
            "model produced an unterminated DSML call"
        );
        Ok(Some(self.output.finish(
            stream,
            length,
            self.parser.stop.as_deref(),
            self.count,
        )?))
    }
}
fn respond(reply: Reply, wire: Wire, result: Result<Value>, runtime: &tokio::runtime::Handle) {
    match reply {
        Reply::Json(tx) => {
            let _ = tx.send(result);
        }
        Reply::Stream(tx) => {
            if let Err(e) = result {
                if tx.sender.is_closed() {
                    return;
                }
                let detail = json!({"type":"api_error", "message":format!("{e:#}")});
                match wire {
                    Wire::Anthropic => {
                        let _ = send(
                            Some(&tx),
                            Some("error"),
                            json!({"type":"error","error":detail}),
                        );
                    }
                    Wire::OpenAi => {
                        let _ = send(Some(&tx), None, json!({"error":detail}));
                        let _ = done(Some(&tx));
                    }
                }
            }
            tx.finish(runtime);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{models::wordlevel::WordLevel, AddedToken};

    struct TestSession {
        id: u32,
        rounds: usize,
        limit: usize,
        prefill_ticks: usize,
        saved: Option<crate::swap::SwapHandle>,
        reused: usize,
    }
    #[derive(Default)]
    struct TestEngine {
        calls: Vec<Vec<u32>>,
        // Inject requests after the first quantum of an existing request.
        arrivals: Option<(mpsc::Sender<Job>, Vec<Job>)>,
        stalled: Option<mpsc::Receiver<Result<Event, Infallible>>>,
        output_token: u32,
        prefill_ticks: usize,
        reused: usize,
        store: Option<SwapStore>,
        fail_resume: Option<u32>,
        large_state: Option<u32>,
        cancel_on_suspend: Option<(u32, oneshot::Receiver<Result<Value>>)>,
        restored: Vec<u32>,
    }
    impl BatchEngine for TestEngine {
        type Session = TestSession;
        fn start(&mut self, ids: Vec<u32>, limit: usize, _: f32) -> Result<TestSession> {
            Ok(TestSession {
                id: ids[0],
                rounds: 0,
                limit,
                prefill_ticks: self.prefill_ticks,
                saved: None,
                reused: self.reused,
            })
        }
        fn prefill(s: &TestSession) -> bool {
            s.rounds <= s.prefill_ticks
        }
        fn processed(s: &TestSession) -> usize {
            s.rounds + s.reused
        }
        fn suspend(&mut self, s: &mut TestSession) -> Result<()> {
            assert!(s.saved.is_none());
            let store = self
                .store
                .as_ref()
                .context("unexpected suspend in resident-only test")?;
            let mut bytes = s.rounds.to_le_bytes().to_vec();
            if self.large_state == Some(s.id) {
                bytes.resize(1024, 0);
            }
            s.saved = Some(store.put(bytes)?);
            s.rounds = 0;
            if self
                .cancel_on_suspend
                .as_ref()
                .is_some_and(|(id, _)| *id == s.id)
            {
                self.cancel_on_suspend.take();
            }
            Ok(())
        }
        fn resume(&mut self, s: &mut TestSession) -> Result<()> {
            self.restored.push(s.id);
            ensure!(self.fail_resume != Some(s.id), "unreadable test snapshot");
            let image = s.saved.take().unwrap().read()?;
            s.rounds = usize::from_le_bytes(image.as_ref().try_into()?);
            Ok(())
        }
        fn step(&mut self, sessions: &mut [&mut TestSession]) -> Vec<Result<Step>> {
            self.calls.push(sessions.iter().map(|s| s.id).collect());
            if let Some((tx, arrivals)) = self.arrivals.take() {
                for job in arrivals {
                    tx.try_send(job)
                        .unwrap_or_else(|_| panic!("test queue full"));
                }
            }
            if self.calls.len() == 4 {
                self.stalled.take();
            }
            sessions
                .iter_mut()
                .map(|s| {
                    assert!(s.saved.is_none());
                    ensure!(s.id != 99, "bad session");
                    s.rounds += 1;
                    Ok(Step {
                        tokens: if s.rounds > s.prefill_ticks {
                            vec![self.output_token]
                        } else {
                            vec![]
                        },
                        finished: s.rounds == s.limit + s.prefill_ticks,
                        eos: false,
                    })
                })
                .collect()
        }
    }
    fn tokenizer() -> (Tokenizer, u32) {
        let mut tokenizer = Tokenizer::new(WordLevel::default());
        tokenizer.add_tokens(&[AddedToken::from("abc", false)]);
        let token = tokenizer.token_to_id("abc").unwrap();
        (tokenizer, token)
    }
    fn job(id: u32, max_tokens: usize) -> (Job, oneshot::Receiver<Result<Value>>) {
        let (send, recv) = oneshot::channel();
        let input = Prepared {
            ids: vec![id],
            thinking: false,
            tools: false,
            max_tokens,
            requested: Some(max_tokens as u64),
            temperature: 0.,
            stops: vec![],
            stream: false,
            include_usage: false,
        };
        (
            Job {
                input,
                wire: Wire::OpenAi,
                reply: Reply::Json(send),
            },
            recv,
        )
    }
    #[test]
    fn continuous_admission_bounds_slots_and_isolates_a_failing_row() {
        let (tok, token) = tokenizer();
        let (tx, rx) = mpsc::channel(8);
        let (first, first_reply) = job(0, 8);
        tx.try_send(first).unwrap_or_else(|_| panic!());
        let (arrivals, replies): (Vec<_>, Vec<_>) = [1, 2, 99, 4, 5, 6, 7]
            .into_iter()
            .map(|id| job(id, 3))
            .unzip();
        let mut engine = TestEngine {
            calls: vec![],
            arrivals: Some((tx, arrivals)),
            stalled: None,
            output_token: token,
            ..Default::default()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(
            &mut engine,
            &tok,
            rx,
            Limits {
                resident: MAX_SEQS,
                live: 12,
                quantum: 32,
            },
            &metrics,
            runtime.handle(),
        );
        assert_eq!(engine.calls[0], [0]);
        assert_eq!(engine.calls[1], [0, 1, 2, 99, 4, 5]);
        assert!(engine.calls[2].contains(&0) && engine.calls[2].contains(&6));
        assert!(engine.calls.iter().all(|rows| rows.len() <= 6));
        assert_eq!(
            first_reply.blocking_recv().unwrap().unwrap()["usage"]["completion_tokens"],
            8
        );
        for (index, reply) in replies.into_iter().enumerate() {
            let result = reply.blocking_recv().unwrap();
            if index == 2 {
                assert!(result.unwrap_err().to_string().contains("bad session"));
            } else {
                assert_eq!(result.unwrap()["usage"]["completion_tokens"], 3);
            }
        }
        let m = metrics.into_inner().unwrap();
        assert_eq!(m.active, 0);
        assert_eq!((m.peak_active, m.peak_batch_size), (6, 6));
        assert!(m.multi_decode_steps > 0);
    }
    #[test]
    fn slow_stream_pauses_only_its_own_session() {
        let (tok, token) = tokenizer();
        let (tx, rx) = mpsc::channel(8);
        let (mut slow, _) = job(1, 100);
        let (send, stalled) = mpsc::channel(1);
        slow.reply = Reply::Stream(EventChannel::new(send));
        let (fast, answer) = job(2, 6);
        tx.try_send(slow).unwrap_or_else(|_| panic!());
        tx.try_send(fast).unwrap_or_else(|_| panic!());
        drop(tx);
        let mut engine = TestEngine {
            calls: vec![],
            arrivals: None,
            stalled: Some(stalled),
            output_token: token,
            ..Default::default()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(
            &mut engine,
            &tok,
            rx,
            Limits {
                resident: 2,
                live: 12,
                quantum: 32,
            },
            &metrics,
            runtime.handle(),
        );
        assert_eq!(engine.calls[0], [1, 2]);
        assert!(engine.calls[1..].iter().all(|rows| rows == &[2]));
        assert_eq!(
            answer.blocking_recv().unwrap().unwrap()["usage"]["completion_tokens"],
            6
        );
    }
    fn enqueue(
        count: usize,
        limit: usize,
    ) -> (mpsc::Receiver<Job>, Vec<oneshot::Receiver<Result<Value>>>) {
        let (tx, rx) = mpsc::channel(count);
        let mut replies = Vec::new();
        for id in 0..count {
            let (job, reply) = job(id as u32, limit);
            tx.try_send(job)
                .unwrap_or_else(|_| panic!("test queue full"));
            replies.push(reply);
        }
        (rx, replies)
    }

    #[test]
    fn twelve_live_requests_take_fair_prefill_and_decode_quanta_across_both_tiers() {
        let (tok, token) = tokenizer();
        let (jobs, replies) = enqueue(12, 12);
        let dir = tempfile::tempdir().unwrap();
        let store = SwapStore::new(2 * size_of::<usize>(), 1024, Some(dir.path().into())).unwrap();
        let mut engine = TestEngine {
            output_token: token,
            prefill_ticks: 16,
            reused: 1,
            store: Some(store.clone()),
            ..Default::default()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(
            &mut engine,
            &tok,
            jobs,
            Limits {
                resident: 6,
                live: 12,
                quantum: 3,
            },
            &metrics,
            runtime.handle(),
        );
        for (id, reply) in replies.into_iter().enumerate() {
            let message = reply.blocking_recv().unwrap().unwrap();
            assert_eq!(message["usage"]["completion_tokens"], 12);
            // Every admitted job began prefill before an original resident could finish prefill.
            assert!(engine.calls[..16]
                .iter()
                .any(|rows| rows.contains(&(id as u32))));
            assert_eq!(
                engine
                    .calls
                    .iter()
                    .filter(|rows| rows.contains(&(id as u32)))
                    .count(),
                28
            );
        }
        assert!(engine.calls.iter().all(|rows| rows.len() <= 6));
        for pair in engine.calls[..16].windows(2) {
            assert!(pair[1].iter().filter(|id| !pair[0].contains(id)).count() <= 1);
        }
        let m = metrics.into_inner().unwrap();
        assert_eq!(
            (m.peak_logical_active, m.peak_active, m.peak_swapped),
            (12, 6, 6)
        );
        assert_eq!(
            (m.logical_active, m.resident, m.waiting, m.swapped),
            (0, 0, 0, 0)
        );
        assert!(m.rotations > 6 && m.resumes == m.rotations);
        assert_eq!((m.requests_started, m.prefix_reused_tokens), (12, 12));
        let tiers = store.stats();
        assert!(tiers.ram_reads > 0 && tiers.disk_reads > 0);
        assert_eq!((tiers.ram_bytes, tiers.disk_bytes), (0, 0));
    }

    #[test]
    fn unreadable_and_cancelled_swap_handles_release_and_leave_peers_running() {
        let (tok, token) = tokenizer();
        let (jobs, mut replies) = enqueue(5, 12);
        let store = SwapStore::new(128, 0, None).unwrap();
        let mut engine = TestEngine {
            output_token: token,
            store: Some(store.clone()),
            cancel_on_suspend: Some((0, replies.remove(0))),
            fail_resume: Some(1),
            ..Default::default()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(
            &mut engine,
            &tok,
            jobs,
            Limits {
                resident: 2,
                live: 5,
                quantum: 2,
            },
            &metrics,
            runtime.handle(),
        );
        assert!(format!(
            "{:#}",
            replies.remove(0).blocking_recv().unwrap().unwrap_err()
        )
        .contains("unreadable test snapshot"));
        for reply in replies {
            assert_eq!(
                reply.blocking_recv().unwrap().unwrap()["usage"]["completion_tokens"],
                12
            );
        }
        assert!(!engine.restored.contains(&0));
        assert_eq!(engine.restored.iter().filter(|&&id| id == 1).count(), 1);
        let m = metrics.into_inner().unwrap();
        assert_eq!((m.logical_active, m.swapped, m.resume_failures), (0, 0, 1));
        assert_eq!(store.stats().ram_bytes, 0);
    }

    #[test]
    fn full_backing_keeps_residents_runnable_and_bounds_failed_eviction_attempts() {
        let (tok, token) = tokenizer();
        let (jobs, replies) = enqueue(4, 25);
        let store = SwapStore::new(0, 0, None).unwrap();
        let mut engine = TestEngine {
            output_token: token,
            store: Some(store.clone()),
            ..Default::default()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(
            &mut engine,
            &tok,
            jobs,
            Limits {
                resident: 2,
                live: 4,
                quantum: 1,
            },
            &metrics,
            runtime.handle(),
        );
        for reply in replies {
            assert_eq!(
                reply.blocking_recv().unwrap().unwrap()["usage"]["completion_tokens"],
                25
            );
        }
        let m = metrics.into_inner().unwrap();
        assert_eq!(
            (
                m.peak_logical_active,
                m.peak_active,
                m.rotations,
                m.suspend_failures
            ),
            (4, 2, 0, 2)
        );
        assert_eq!(store.stats().write_failures, 2);
    }

    #[test]
    fn releasing_small_tickets_does_not_retry_an_oversized_resident_every_round() {
        let (tok, token) = tokenizer();
        let (jobs, replies) = enqueue(4, 32);
        let store = SwapStore::new(64, 0, None).unwrap();
        let mut engine = TestEngine {
            output_token: token,
            store: Some(store.clone()),
            large_state: Some(0),
            ..Default::default()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(
            &mut engine,
            &tok,
            jobs,
            Limits {
                resident: 2,
                live: 4,
                quantum: 1,
            },
            &metrics,
            runtime.handle(),
        );
        for reply in replies {
            assert_eq!(
                reply.blocking_recv().unwrap().unwrap()["usage"]["completion_tokens"],
                32
            );
        }
        let m = metrics.into_inner().unwrap();
        assert!(m.resumes > 10);
        assert_eq!((m.suspend_failures, m.resume_failures), (1, 0));
        assert_eq!(store.stats().ram_bytes, 0);
    }

    #[test]
    fn backed_up_stream_expires_without_waiting_on_the_compute_thread() {
        let (tx, _rx) = mpsc::channel(1);
        let stream = EventChannel::new(tx);
        stream.send(Event::default().data("first")).unwrap();
        stream.send(Event::default().data("staged")).unwrap();
        assert!(!stream.ready().unwrap());
        stream
            .last_progress
            .set(Some(Instant::now() - WRITE_TIMEOUT));
        assert!(stream
            .ready()
            .unwrap_err()
            .to_string()
            .contains("stopped reading"));
    }
}
