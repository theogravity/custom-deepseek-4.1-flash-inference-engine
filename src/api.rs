//! Local HTTP adapter for LiteLLM's Anthropic and OpenAI routes.
use crate::{
    chat::{self, Delta, Parser},
    model::{Model, Session, Step},
};
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
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

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
}

struct ServerState {
    key: String,
    tokenizer: Arc<Tokenizer>,
    plain: Tokenizer,
    queue: mpsc::Sender<Job>,
    context: usize,
    encoders: Arc<Semaphore>,
    metrics: Arc<Mutex<Metrics>>,
}
struct Prepared {
    ids: Vec<u32>,
    thinking: bool,
    tools: bool,
    max_tokens: usize,
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
    if !count {
        ensure!(
            ids.len() < state.context,
            "prompt contains {} tokens, model context is {}",
            ids.len(),
            state.context
        );
    }
    let limit = req.get("max_completion_tokens").or(req.get("max_tokens"));
    let requested = match limit {
        None | Some(Value::Null) => 4096,
        Some(v) => v
            .as_u64()
            .filter(|i| *i > 0)
            .context("max_tokens must be a positive integer")?,
    };
    let max_tokens = requested.min(state.context.saturating_sub(ids.len()) as u64) as usize;
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
        Ok(Err(e)) => return wire.error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
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
    if input.stream {
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
    }
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
pub fn serve(mut model: Model, tokenizer: Tokenizer, addr: &str, key: String) -> Result<()> {
    ensure!(!key.trim().is_empty(), "a nonempty API key is required");
    let mut plain = tokenizer.clone();
    plain.with_added_vocabulary(AddedVocabulary::new());
    let max_seqs = match std::env::var("DS4_MAX_SEQS") {
        Ok(s) => s
            .parse::<usize>()
            .context("DS4_MAX_SEQS must be an integer")?,
        Err(std::env::VarError::NotPresent) => MAX_SEQS,
        Err(err) => return Err(err.into()),
    };
    ensure!(
        (1..=MAX_SEQS).contains(&max_seqs),
        "DS4_MAX_SEQS must be between 1 and {MAX_SEQS}"
    );
    let (queue, jobs) = mpsc::channel(QUEUE);
    let metrics = Arc::new(Mutex::new(Metrics {
        max_seqs,
        ..Metrics::default()
    }));
    let state = Arc::new(ServerState {
        key,
        tokenizer: Arc::new(tokenizer),
        plain,
        queue,
        context: model.max_context(),
        encoders: Arc::new(Semaphore::new(QUEUE)),
        metrics: metrics.clone(),
    });
    let tokenizer = state.tokenizer.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let app = router(state);
    let handle = runtime.handle().clone();
    let worker = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || schedule(&mut model, &tokenizer, jobs, max_seqs, &metrics, &handle))?;
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
trait BatchEngine {
    type Session;
    fn start(&mut self, ids: Vec<u32>, limit: usize, temperature: f32) -> Result<Self::Session>;
    fn prefill(session: &Self::Session) -> bool;
    fn processed(session: &Self::Session) -> usize;
    fn step(&mut self, sessions: &mut [&mut Self::Session]) -> Vec<Result<Step>>;
}
impl BatchEngine for Model {
    type Session = Session;
    fn start(&mut self, ids: Vec<u32>, limit: usize, temperature: f32) -> Result<Session> {
        self.start_request(ids, limit, temperature)
    }
    fn prefill(session: &Session) -> bool {
        session.is_prefill()
    }
    fn processed(session: &Session) -> usize {
        session.processed_tokens()
    }
    fn step(&mut self, sessions: &mut [&mut Session]) -> Vec<Result<Step>> {
        self.step_batch(sessions)
    }
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
fn admit<'t, E: BatchEngine>(
    engine: &mut E,
    tokenizer: &'t Tokenizer,
    job: Job,
    runtime: &tokio::runtime::Handle,
) -> Option<Active<'t, E::Session>> {
    let Job { input, wire, reply } = job;
    if reply.closed() {
        return None;
    }
    let output = Output::new(wire, &input);
    let session = output
        .start(reply.stream())
        .and_then(|_| engine.start(input.ids, input.max_tokens, input.temperature));
    match session {
        Ok(session) => Some(Active {
            session,
            decoder: tokenizer.decode_stream(false),
            parser: Parser::new(input.thinking, input.tools, input.stops),
            output,
            reply,
            count: 0,
            ready: true,
        }),
        Err(err) => {
            respond(reply, wire, Err(err), runtime);
            None
        }
    }
}
fn record<E: BatchEngine>(
    metrics: &Mutex<Metrics>,
    slots: &[Option<Active<'_, E::Session>>],
    batch: bool,
) {
    let mut sample = metrics.lock().unwrap();
    let mut runnable = 0;
    let mut decode_rows = 0;
    sample.active = 0;
    sample.prefilling = 0;
    sample.processed_tokens = 0;
    for a in slots.iter().flatten() {
        sample.active += 1;
        let prefill = E::prefill(&a.session);
        sample.prefilling += usize::from(prefill);
        sample.processed_tokens += E::processed(&a.session);
        if a.ready {
            runnable += 1;
            decode_rows += usize::from(!prefill);
        }
    }
    sample.decoding = sample.active - sample.prefilling;
    sample.peak_active = sample.peak_active.max(sample.active);
    sample.peak_processed_tokens = sample.peak_processed_tokens.max(sample.processed_tokens);
    if batch && runnable > 0 {
        sample.batch_steps += 1;
        sample.multi_sequence_steps += u64::from(runnable > 1);
        sample.multi_decode_steps += u64::from(decode_rows > 1);
        sample.peak_batch_size = sample.peak_batch_size.max(runnable);
        sample.peak_decode_batch = sample.peak_decode_batch.max(decode_rows);
    }
}
fn schedule<E: BatchEngine>(
    engine: &mut E,
    tokenizer: &Tokenizer,
    mut jobs: mpsc::Receiver<Job>,
    max_seqs: usize,
    metrics: &Mutex<Metrics>,
    runtime: &tokio::runtime::Handle,
) {
    let mut slots: Vec<_> = (0..max_seqs).map(|_| None).collect();
    loop {
        if slots.iter().all(Option::is_none) {
            record::<E>(metrics, &slots, false);
            let Some(job) = jobs.blocking_recv() else {
                break;
            };
            slots[0] = admit(engine, tokenizer, job, runtime);
        }
        // Traverse the fixed number of slots once before each model step so an
        // endless queue of cancelled requests cannot starve admitted sessions.
        for slot in slots.iter_mut().filter(|s| s.is_none()) {
            let Ok(job) = jobs.try_recv() else { break };
            *slot = admit(engine, tokenizer, job, runtime);
        }
        for slot in &mut slots {
            let Some(a) = slot else { continue };
            if a.reply.closed() {
                *slot = None;
            } else if let Some(tx) = a.reply.stream() {
                match tx.ready() {
                    Ok(ready) => a.ready = ready,
                    // The stream already stopped reading or disconnected. Closing
                    // this sender signals EOF without blocking the other rows.
                    Err(_) => *slot = None,
                }
            }
        }
        record::<E>(metrics, &slots, true);
        let mut states: Vec<_> = slots
            .iter_mut()
            .flatten()
            .filter(|a| a.ready)
            .map(|a| &mut a.session)
            .collect();
        if states.is_empty() {
            if slots.iter().any(Option::is_some) {
                std::thread::sleep(Duration::from_millis(2));
            }
            continue;
        }
        let outcomes = engine.step(&mut states);
        let expected = states.len();
        // Include the newly committed target KV before removing any finished row.
        record::<E>(metrics, &slots, false);
        let cardinality_ok = outcomes.len() == expected;
        let mut outcomes = outcomes.into_iter();
        for slot in &mut slots {
            let Some(a) = slot else { continue };
            if !a.ready {
                continue;
            }
            let result = if cardinality_ok {
                outcomes.next().unwrap().and_then(|step| a.consume(step))
            } else {
                Err(anyhow!(
                    "inference returned a different number of results than sessions"
                ))
            };
            let terminal = match result {
                Ok(None) => continue,
                Ok(Some(v)) => Ok(v),
                Err(e) => Err(e),
            };
            let a = slot.take().unwrap();
            respond(a.reply, a.output.wire, terminal, runtime);
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
    }
    struct TestEngine {
        calls: Vec<Vec<u32>>,
        // Inject requests after the first quantum of an existing request.
        arrivals: Option<(mpsc::Sender<Job>, Vec<Job>)>,
        stalled: Option<mpsc::Receiver<Result<Event, Infallible>>>,
        output_token: u32,
    }
    impl BatchEngine for TestEngine {
        type Session = TestSession;
        fn start(&mut self, ids: Vec<u32>, limit: usize, _: f32) -> Result<TestSession> {
            Ok(TestSession {
                id: ids[0],
                rounds: 0,
                limit,
            })
        }
        fn prefill(s: &TestSession) -> bool {
            s.rounds == 0
        }
        fn processed(s: &TestSession) -> usize {
            s.rounds
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
                    ensure!(s.id != 99, "bad session");
                    s.rounds += 1;
                    Ok(Step {
                        tokens: vec![self.output_token],
                        finished: s.rounds == s.limit,
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
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(&mut engine, &tok, rx, MAX_SEQS, &metrics, runtime.handle());
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
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let metrics = Mutex::new(Metrics::default());
        schedule(&mut engine, &tok, rx, 2, &metrics, runtime.handle());
        assert_eq!(engine.calls[0], [1, 2]);
        assert!(engine.calls[1..].iter().all(|rows| rows == &[2]));
        assert_eq!(
            answer.blocking_recv().unwrap().unwrap()["usage"]["completion_tokens"],
            6
        );
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
