//! Local HTTP adapter for LiteLLM's Anthropic and OpenAI routes.
use crate::{
    chat::{self, Delta, Parser},
    model::Model,
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
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokenizers::{AddedVocabulary, Tokenizer};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

pub const MODEL_NAME: &str = "DeepSeek V4.1 Flash";
const QUEUE: usize = 8;
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

struct EventChannel {
    sender: mpsc::Sender<Result<Event, Infallible>>,
    runtime: tokio::runtime::Handle,
}
impl EventChannel {
    fn send(&self, event: Event) -> Result<()> {
        use mpsc::error::TrySendError;
        match self.sender.try_send(Ok(event)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Closed(_)) => Err(anyhow!("client disconnected")),
            Err(TrySendError::Full(frame)) => self
                .runtime
                .block_on(
                    self.sender
                        .send_timeout(frame, std::time::Duration::from_secs(3)),
                )
                .map_err(|_| anyhow!("client stopped reading its response")),
        }
    }
}
enum Reply {
    Json(oneshot::Sender<Result<Value>>),
    Stream(EventChannel),
}
impl Reply {
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
            reply: Reply::Stream(EventChannel {
                sender: send,
                runtime: tokio::runtime::Handle::current(),
            }),
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
fn router(state: Arc<ServerState>) -> Router {
    let private = Router::new()
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
    let (queue, mut jobs) = mpsc::channel(QUEUE);
    let state = Arc::new(ServerState {
        key,
        tokenizer: Arc::new(tokenizer),
        plain,
        queue,
        context: model.max_context(),
        encoders: Arc::new(Semaphore::new(QUEUE)),
    });
    let tokenizer = state.tokenizer.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let app = router(state);
    let worker = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || {
            while let Some(job) = jobs.blocking_recv() {
                if !job.reply.closed() {
                    run(&mut model, &tokenizer, job);
                }
            }
        })?;
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
fn run(model: &mut Model, tokenizer: &Tokenizer, job: Job) {
    let Job { wire, input, reply } = job;
    let stream = match &reply {
        Reply::Json(_) => None,
        Reply::Stream(tx) => Some(tx),
    };
    let mut output = Output::new(wire, &input);
    let result = (|| -> Result<Value> {
        output.start(stream)?;
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = Parser::new(input.thinking, input.tools, input.stops);
        let mut failure = None;
        let mut emitted = 0;
        let count = model.generate_cancellable(
            &input.ids,
            input.max_tokens,
            input.temperature,
            || reply.closed(),
            |token| {
                emitted += 1;
                if reply.closed() {
                    return false;
                }
                let res = (|| -> Result<()> {
                    if let Some(text) = decoder.step(token).map_err(|e| anyhow!("{e}"))? {
                        for delta in parser.push(&text, false)? {
                            output.append(delta, stream)?;
                        }
                    }
                    Ok(())
                })();
                failure = res.err();
                failure.is_none() && parser.stop.is_none()
            },
        )?;
        if let Some(e) = failure {
            return Err(e);
        }
        ensure!(!reply.closed(), "client disconnected");
        for delta in parser.push("", true)? {
            output.append(delta, stream)?;
        }
        let length = count == emitted && count == input.max_tokens;
        ensure!(
            !parser.incomplete_call() || length,
            "model produced an unterminated DSML call"
        );
        output.finish(stream, length, parser.stop.as_deref(), count)
    })();
    match reply {
        Reply::Json(tx) => {
            let _ = tx.send(result);
        }
        Reply::Stream(tx) => {
            if let Err(e) = result {
                // A failed send has already exhausted the slow-reader timeout.
                // Leave a full channel to drain (and the body to end) immediately.
                if tx.sender.capacity() == 0 || tx.sender.is_closed() {
                    return;
                }
                let detail = json!({"type":"api_error","message":format!("{e:#}")});
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stalled_sse_client_releases_inference_thread() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_time()
            .build()
            .unwrap();
        let (sender, _receiver) = mpsc::channel(1);
        let stream = EventChannel {
            sender,
            runtime: runtime.handle().clone(),
        };
        stream.send(Event::default().data("first")).unwrap();
        let start = std::time::Instant::now();
        assert!(stream.send(Event::default().data("stalled")).is_err());
        assert!(start.elapsed() < std::time::Duration::from_secs(6));
    }
}
