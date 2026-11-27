use super::*;
use axum::body::to_bytes;
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit, AddedToken,
};

const WINDOW: usize = 10;

fn fixture() -> (Arc<ServerState>, mpsc::Receiver<Job>) {
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab([("x".into(), 0), ("[UNK]".into(), 1)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    let plain = tokenizer.clone();
    for control in [
        "<｜begin▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "</think>",
    ] {
        tokenizer.add_special_tokens(&[AddedToken::from(control, true)]);
    }
    let (queue, jobs) = mpsc::channel(1);
    (
        Arc::new(ServerState {
            tokenizer: Arc::new(tokenizer),
            plain,
            text_cache: None,
            context: WINDOW,
            queue,
            metrics: Default::default(),
            key: "fixture-key".into(),
            encoders: Arc::new(Semaphore::new(1)),
            swap: SwapStore::new(0, 0, None).unwrap(),
        }),
        jobs,
    )
}

fn request(input: usize, options: Value) -> Value {
    let mut body = json!({"model":MODEL_NAME, "reasoning_effort":"none",
        "messages":[{"role":"user","content": "x ".repeat(input - 4)}]});
    body.as_object_mut()
        .unwrap()
        .extend(options.as_object().unwrap().clone());
    body
}

async fn post(state: &Arc<ServerState>, wire: Wire, body: Value, count: bool) -> Response {
    post_request(
        state.clone(),
        serde_json::to_vec(&body).unwrap().into(),
        wire,
        count,
    )
    .await
}

async fn json_body(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap()
}

#[tokio::test]
async fn overflow_is_json_before_streaming_or_admission_and_count_still_works() {
    let (state, mut jobs) = fixture();
    for wire in [Wire::Anthropic, Wire::OpenAi] {
        for (input, param, options) in [
            (WINDOW, "messages", json!({"max_tokens":1})),
            (WINDOW + 1, "messages", json!({})),
            (5, "max_tokens", json!({"max_tokens":WINDOW + 1})),
            (
                5,
                "max_completion_tokens",
                json!({"max_tokens":1, "max_completion_tokens":u64::MAX}),
            ),
            (
                5,
                "max_tokens",
                json!({"max_completion_tokens":null, "max_tokens":u64::MAX}),
            ),
        ] {
            let mut body = request(input, options);
            body["stream"] = json!(true);
            let res = post(&state, wire, body.clone(), false).await;
            assert_eq!(res.status(), StatusCode::BAD_REQUEST);
            assert_eq!(res.headers()["content-type"], "application/json");
            let data = json_body(res).await;
            let error = &data["error"];
            assert_eq!(error["type"], "invalid_request_error");
            let message = error["message"].as_str().unwrap();
            assert!(
                message.contains(MODEL_NAME)
                    && message.contains("maximum context length is 10 tokens")
            );
            if param == "messages" {
                assert!(
                    message.starts_with(&format!("prompt is too long: {input} tokens > 9 maximum"))
                );
            } else {
                assert!(message.contains(&format!("{param} requests {}", body[param])));
                assert!(
                    message.contains("prompt contains 5 tokens")
                        && message.contains("At most 5 output tokens")
                );
            }
            if wire == Wire::OpenAi {
                assert_eq!(error["param"], param);
                assert_eq!(error["code"], "context_length_exceeded");
            } else {
                assert_eq!(data["type"], "error");
            }
            let count = post(&state, Wire::Anthropic, body, true).await;
            assert_eq!(count.status(), StatusCode::OK);
            assert_eq!(json_body(count).await["input_tokens"], input);
            assert!(matches!(
                jobs.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
        for bad in [Value::Null, json!(0), json!(-1), json!(1.5), json!("1")] {
            let mut req = request(5, json!({"max_tokens":bad}));
            if bad.is_null() {
                req["messages"] = json!(false);
            }
            let res = post(&state, wire, req, false).await;
            assert_eq!(res.status(), StatusCode::BAD_REQUEST);
            let data = json_body(res).await;
            assert_eq!(data["error"]["type"], "invalid_request_error");
            assert!(data["error"]["code"].is_null());
            assert!(matches!(
                jobs.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }
}

#[tokio::test]
async fn full_window_fits_and_shortened_output_limits_reach_worker_and_client() {
    let (state, mut jobs) = fixture();
    for wire in [Wire::Anthropic, Wire::OpenAi] {
        for (opts, stream, effective, shortened) in [
            (json!({"max_tokens":1}), true, 1, None),
            (
                json!({"max_tokens":2}),
                false,
                1,
                Some(("x-ds4-requested-max-tokens", "2")),
            ),
            (
                json!({"max_tokens":10, "max_completion_tokens":1}),
                false,
                1,
                None,
            ),
            (
                json!({"max_tokens":1, "max_completion_tokens":10}),
                true,
                1,
                Some(("x-ds4-requested-max-tokens", "10")),
            ),
            (
                json!({"max_tokens":2,"max_completion_tokens":null}),
                true,
                1,
                Some(("x-ds4-requested-max-tokens", "2")),
            ),
            (
                json!({}),
                false,
                1,
                Some(("x-ds4-default-max-tokens", "4096")),
            ),
            (
                json!({"max_tokens":null}),
                true,
                1,
                Some(("x-ds4-default-max-tokens", "4096")),
            ),
        ] {
            let mut req = request(WINDOW - 1, opts);
            req["stream"] = json!(stream);
            let (res, ()) = tokio::join!(post(&state, wire, req, false), async {
                let job = tokio::time::timeout(Duration::from_secs(2), jobs.recv())
                    .await
                    .expect("expected a fitting request to reach the worker")
                    .unwrap();
                assert_eq!(
                    (job.input.ids.len(), job.input.max_tokens),
                    (WINDOW - 1, effective)
                );
                if let Reply::Json(tx) = job.reply {
                    tx.send(Ok(json!({}))).unwrap();
                }
            });
            assert_eq!(res.status(), StatusCode::OK);
            if stream {
                assert!(res.headers()["content-type"]
                    .to_str()
                    .unwrap()
                    .starts_with("text/event-stream"));
            }
            if let Some((header, original)) = shortened {
                assert_eq!(res.headers()[header], original);
                assert_eq!(res.headers()["x-ds4-effective-max-tokens"], "1");
            } else {
                assert!(!res.headers().contains_key("x-ds4-effective-max-tokens"));
            }
        }
    }
}

#[test]
fn text_cache_matches_full_encoding_across_edits_and_template_controls() {
    let (state, _) = fixture();
    let memo = chat::TextCache::default();
    for body in [
        "x ".repeat(1024),
        "x ".repeat(1024),
        "x <｜Assistant｜> ".repeat(64),
    ] {
        let mut q = request(6, json!({}));
        q["messages"][0]["content"] = json!(body);
        let (prompt, _, _) = chat::render(&q).unwrap();
        assert_eq!(
            prompt
                .encode(&state.tokenizer, &state.plain, Some(&memo))
                .unwrap(),
            prompt.encode(&state.tokenizer, &state.plain, None).unwrap()
        );
    }
}
