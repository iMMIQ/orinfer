use super::*;
use futures_util::StreamExt;

pub(super) struct Http {
    pub(super) url: String,
    pub(super) client: reqwest::Client,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Http {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Http {
    pub(super) async fn new(manager: Arc<Manager>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let router = routes::public_router(manager);
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(key) = std::env::var("ORINFER_API_KEY")
            && !key.is_empty()
        {
            headers.insert("authorization", format!("Bearer {key}").parse().unwrap());
        }
        Self {
            url,
            client: reqwest::Client::builder()
                .no_proxy()
                .default_headers(headers)
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            task,
        }
    }
    pub(super) async fn post(&self, request: &Value) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/chat/completions", self.url))
            .header("x-request-id", "ollama-fixture")
            .json(request)
            .send()
            .await
            .unwrap()
    }
}

#[tokio::test]
async fn unloaded_models_are_listed_without_starting_a_worker() {
    let fixture = Fixture::new();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let models: Value = http
        .client
        .get(format!("{}/v1/models", http.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "fixture:2b");
    assert_eq!(fixture.starts(), 0);
    let bad = http.post(&json!({"model":"missing","messages":[]})).await;
    assert_eq!(bad.status(), 404);
    assert_eq!(
        bad.json::<Value>().await.unwrap()["error"]["code"],
        "not_found"
    );
    assert_eq!(fixture.manager.requests.available_permits(), 128);
    assert_eq!(fixture.manager.body_memory.available_permits(), 512);
    assert_eq!(fixture.starts(), 0);
}
#[tokio::test]
async fn render_forwards_defaults_and_aliases_without_generation_memory_reservation() {
    let fixture = Fixture::new();
    drop(fixture.manager.acquire("fixture:2b").await.unwrap());
    fixture
        .alias("render", |record| {
            record.sequence_bytes = usize::MAX / 4;
            record
                .options
                .default_request_params
                .insert("enable_thinking".into(), json!(true));
        })
        .await;
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let response = http.client.post(format!("{}/v1/chat/completions/render", http.url)).json(&json!({"model":"render","messages":[{"role":"user","content":"hello"}],"chat_template_kwargs":{"thinking_budget":8}})).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["enable_thinking"], true);
    assert_eq!(body["chat_template_kwargs"]["thinking_budget"], 8);
    assert_ne!(body["model"], "render");
    fixture.unload().await;
}

#[tokio::test]
async fn ollama_request_vectors_survive_the_http_gateway_and_alias_rewrite() {
    let fixture = Fixture::new();
    fixture
        .alias("alias", |record| {
            record
                .options
                .default_request_params
                .insert("temperature".into(), json!(0.7));
        })
        .await;
    let original = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    drop(original);
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    for case in crate::test_fixtures::ollama()["requests"]
        .as_array()
        .unwrap()
    {
        let mut request = case["request"].clone();
        request["model"] = json!("alias");
        request["temperature"] = json!(0);
        let response = http.post(&request).await;
        assert_eq!(response.status(), 200);
        let echoed: Value = response.json().await.unwrap();
        request["model"] = json!("fixture:2b");
        assert_eq!(echoed, request, "{}", case["id"]);
    }
    assert_eq!(fixture.starts(), 1);
    fixture.unload().await;
}

#[tokio::test]
async fn streamed_response_owns_lease_and_budgets_until_disconnect() {
    let mut fixture = Fixture::new();
    Arc::get_mut(&mut fixture.manager).unwrap().requests = Arc::new(Semaphore::new(2));
    fixture
        .manager
        .records
        .lock()
        .await
        .get_mut("fixture:2b")
        .unwrap()
        .sequence_bytes = 1024;
    fs::write(fixture.root.join("hold_response"), "").unwrap();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let request = json!({"model":"fixture:2b","messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":16});
    let first = http.post(&request).await;
    assert_eq!(first.status(), 200);
    let mut first = first.bytes_stream();
    assert!(!first.next().await.unwrap().unwrap().is_empty());
    let second = http.post(&request).await;
    assert_eq!(second.status(), 200);
    assert_eq!(fixture.manager.requests.available_permits(), 0);
    assert_eq!(fixture.manager.budgets.lock().unwrap().requests, 2048);
    let third = http.post(&request).await;
    assert_eq!(third.status(), 429);
    assert_eq!(
        third.json::<Value>().await.unwrap()["error"]["code"],
        "queue_full"
    );
    let runner = fixture
        .manager
        .runners
        .lock()
        .await
        .values()
        .next()
        .unwrap()
        .clone();
    runner.keep_alive_s.store(0, Ordering::Release);
    fixture.manager.sweep().await;
    assert_eq!(runner.active.load(Ordering::Acquire), 2);
    assert!(runner.child.lock().await.try_wait().unwrap().is_none());
    drop(first);
    eventually(|| fixture.manager.requests.available_permits() == 1).await;
    assert_eq!(runner.active.load(Ordering::Acquire), 1);
    assert_eq!(fixture.manager.budgets.lock().unwrap().requests, 1024);
    eventually(|| fixture.root.join("disconnected").exists()).await;
    drop(second);
    eventually(|| fixture.manager.requests.available_permits() == 2).await;
    assert_eq!(fixture.manager.body_memory.available_permits(), 512);
    assert_eq!(fixture.manager.budgets.lock().unwrap().requests, 0);
    fixture.manager.sweep().await;
    assert!(fixture.manager.runners.lock().await.is_empty());
    fixture.assert_gpu_released().await;
}

#[tokio::test]
async fn alias_defaults_respect_explicit_alternative_parameter_names() {
    let fixture = Fixture::new();
    fixture
        .alias("alias", |record| {
            record.options.default_request_params = json!({
                "max_completion_tokens":8192, "enable_thinking":true,
                "stream_options":{"include_usage":true}, "top_logprobs":3,
                "temperature":0.7,
            })
            .as_object()
            .unwrap()
            .clone();
        })
        .await;
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let request = json!({"model":"alias","messages":[{"role":"user","content":"Hello"}],
        "max_tokens":16,"reasoning_effort":"none","stream":false,"logprobs":false});
    let echoed: Value = http.post(&request).await.json().await.unwrap();
    for name in [
        "max_completion_tokens",
        "enable_thinking",
        "stream_options",
        "top_logprobs",
    ] {
        assert!(
            echoed.get(name).is_none(),
            "unexpected deployment default: {name}"
        );
    }
    assert_eq!(echoed["temperature"], 0.7);
    assert_eq!(echoed["max_tokens"], 16);
    fixture.unload().await;
}

#[tokio::test]
async fn proxy_preserves_sse_metadata_usage_and_worker_errors() {
    let fixture = Fixture::new();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let request =
        json!({"model":"fixture:2b","messages":[{"role":"user","content":"Hello"}],"stream":true});
    let frame = |delta: Value, finish: Value| json!({"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"fixture:2b","choices":[{"index":0,"delta":delta,"finish_reason":finish}],"usage":null});
    let events = vec![
        frame(json!({"role":"assistant"}), Value::Null),
        frame(json!({"reasoning_content":"need a tool"}), Value::Null),
        frame(
            json!({"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":"}}]}),
            Value::Null,
        ),
        frame(
            json!({"tool_calls":[{"index":0,"function":{"arguments":"\"Seattle\"}"}}]}),
            Value::Null,
        ),
        frame(json!({}), json!("tool_calls")),
        json!({"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"fixture:2b","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":4}}}),
    ];
    fs::write(
        fixture.root.join("response.json"),
        serde_json::to_vec(&json!({"events":events})).unwrap(),
    )
    .unwrap();
    let response = http.post(&request).await;
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.headers()["x-request-id"], "ollama-fixture");
    let text = response.text().await.unwrap();
    let frames: Vec<_> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .collect();
    assert_eq!(frames.last(), Some(&"[DONE]"));
    assert_eq!(frames.len(), events.len() + 1);
    let actual: Vec<Value> = frames[..events.len()]
        .iter()
        .map(|raw| serde_json::from_str(raw).unwrap())
        .collect();
    assert_eq!(actual, events);
    let error = json!({"error":{"message":"worker rejected parameters","type":"invalid_request_error","code":"invalid_request"}});
    fs::write(
        fixture.root.join("response.json"),
        serde_json::to_vec(&json!({"status":400,"body":error})).unwrap(),
    )
    .unwrap();
    let mut request = request;
    request["stream"] = json!(false);
    let response = http.post(&request).await;
    assert_eq!(response.status(), 400);
    assert_eq!(response.json::<Value>().await.unwrap(), error);
    fixture.unload().await;
}
