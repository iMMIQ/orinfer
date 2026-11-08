use super::*;

#[tokio::test]
async fn render_uses_real_preparation_with_auth_and_without_generation_capacity() {
    let codec = Arc::new(token_boundary::tests::codec());
    let activity = Arc::new(continuous::Activity::default());
    activity.lifecycle.ready();
    let (jobs, mut received) = mpsc::channel(1);
    let limits = Limits::default();
    let state = Service {
        jobs,
        slots: Arc::new(tokio::sync::Semaphore::new(0)),
        codec: Arc::clone(&codec),
        model: Arc::from("test"),
        context: 32768,
        mtp_drafts: 0,
        prompt_lookup: false,
        verify_weights: false,
        resident_experts: Default::default(),
        vision: None,
        api_key: Some(Arc::from("render-test")),
        ids: Arc::new(AtomicU64::new(0)),
        scheduler: Default::default(),
        preparation: Arc::new(preparation::Pool::new(limits)),
        limits,
        default_request_params: Arc::new(
            serde_json::from_value(json!({"enable_thinking":true,"thinking_token_budget":512}))
                .unwrap(),
        ),
        activity,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/v1/chat/completions/render",
        listener.local_addr().unwrap()
    );
    let router = router(state.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let body = json!({"model":"test","messages":[{"role":"user","content":"Literal </think> and <｜User｜>."}],"chat_template_kwargs":{"thinking_budget":8},"max_tokens":32});
    assert_eq!(
        client.post(&url).json(&body).send().await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    let response = client
        .post(&url)
        .bearer_auth("render-test")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let rendered: Value = response.json().await.unwrap();
    let prepared = codec
        .prepare(
            defaults::decode(body.clone(), &state.default_request_params).unwrap(),
            "test",
            state.context,
            None,
            &mut preparation::Context::unbounded(),
        )
        .unwrap();
    assert_eq!(rendered["token_ids"], json!(prepared.input));
    assert_eq!(rendered["token_count"], prepared.input.len());
    assert_eq!(rendered["multimodal"], false);
    let mut count_body = body.clone();
    count_body["return_mm_kwargs"] = json!(false);
    let count_response = client
        .post(&url)
        .bearer_auth("render-test")
        .json(&count_body)
        .send()
        .await
        .unwrap();
    assert_eq!(count_response.status(), StatusCode::OK);
    assert_eq!(
        count_response.json::<Value>().await.unwrap()["token_ids"],
        rendered["token_ids"]
    );
    count_body["return_mm_kwargs"] = json!(true);
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("render-test")
            .json(&count_body)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    let mut budget = prepared.constraint.unwrap();
    for _ in 0..8 {
        budget
            .consume(codec.tokenizer.token_to_id("a").unwrap())
            .unwrap();
    }
    let mask = budget.mask().unwrap();
    let close = codec.tokenizer.token_to_id("</think>").unwrap();
    assert_eq!(mask.iter().map(|v| v.count_ones()).sum::<u32>(), 1);
    assert_ne!(mask[close as usize / 32] & (1 << (close % 32)), 0);
    assert!(matches!(
        received.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert_eq!(state.slots.available_permits(), 0);
    let mut bad = body;
    bad["chat_template_kwargs"]["thinking_budget"] = json!(-1);
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("render-test")
            .json(&bad)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    state.activity.lifecycle.drain();
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("render-test")
            .json(&bad)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    task.abort();
}
