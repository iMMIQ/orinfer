use super::*;

fn response(codec: &ChatCodec) -> (Response<'_>, mpsc::Receiver<ModelEvent>) {
    let (mut mailbox, receiver) = super::super::tests::mailbox();
    mailbox.job.prepared.stream = false;
    mailbox.job.prepared.max_tokens = 4096;
    (Response::new(mailbox.job, codec), receiver)
}

fn packet(ids: &[u32]) -> Tokens {
    Tokens {
        tokens: ids.to_vec(),
        scores: ids
            .iter()
            .map(|&token| orinfer_engine::sampling::TokenLogprob {
                token,
                logprob: -0.25,
                top: vec![(token, -0.25)],
            })
            .collect(),
    }
}

#[test]
fn delayed_packets_keep_utf8_tool_arguments_and_probability_records_in_order() {
    let codec = token_boundary::tests::codec();
    let (mut inline, _inline_client) = response(&codec);
    let (mut pipelined, _pipeline_client) = response(&codec);
    for response in [&mut inline, &mut pipelined] {
        response.mailbox.job.prepared.sampling.top_logprobs = Some(1);
        response.parser.tool_format(output::ToolFormat::DeepSeekV41);
        // The schema is also used to convert DSML parameter values to JSON.
        let tools = vec![
            json!({"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}),
        ];
        response.parser = output::Output::new(false, tools, vec![], "pipeline-test");
        response.parser.tool_format(output::ToolFormat::DeepSeekV41);
        response.parser.token_boundaries();
    }
    let text = "中文🙂\n\n<｜DSML｜ calls><｜DSML｜ invoke name=\"read\"><｜DSML｜ parameter name=\"path\" string=\"true\">目录/文件.rs</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    let ids = codec.tokenizer.encode(text, false).unwrap();
    let mut statistics = Statistics::default();
    let chunks: Vec<_> = ids.get_ids().chunks(3).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        pipelined.drain(true, &codec, "test", &mut statistics);
        inline.deliver(packet(chunk), false, &codec, "test", &mut statistics);
        pipelined.deliver(
            packet(chunk),
            i + 1 < chunks.len(),
            &codec,
            "test",
            &mut statistics,
        );
    }
    assert!(pipelined.pending.is_none());
    assert!(pipelined.ending.is_none());
    assert!(inline.ending.is_none());
    assert!(statistics.maintenance_drains > 0);
    for response in [&mut inline, &mut pipelined] {
        for delta in response
            .parser
            .finish(false, &json!("auto"), false)
            .unwrap()
        {
            response.delta_scores(&delta);
        }
    }
    assert_eq!(inline.parser.message(), pipelined.parser.message());
    assert_eq!(inline.count, pipelined.count);
    assert_eq!(inline.content_scores, pipelined.content_scores);
    assert_eq!(
        pipelined.parser.message()["tool_calls"][0]["function"]["arguments"],
        "{\"path\":\"目录/文件.rs\"}"
    );
}

#[test]
fn first_terminal_eos_and_cross_packet_stop_are_processed_inline() {
    let codec = token_boundary::tests::codec();
    let ids = codec.tokenizer.encode("hello STOP world", false).unwrap();
    let (mut output, _client) = response(&codec);
    let mut statistics = Statistics::default();
    output.deliver(
        packet(&ids.get_ids()[..1]),
        true,
        &codec,
        "test",
        &mut statistics,
    );
    assert!(output.pending.is_none());
    output.deliver(
        packet(&ids.get_ids()[1..2]),
        true,
        &codec,
        "test",
        &mut statistics,
    );
    assert!(output.pending.is_some());
    output.drain(false, &codec, "test", &mut statistics);
    // A terminal request (length or grammar completion) sets defer=false.
    output.deliver(
        packet(&ids.get_ids()[2..3]),
        false,
        &codec,
        "test",
        &mut statistics,
    );
    assert!(output.pending.is_none());
    let eos = codec
        .tokenizer
        .token_to_id("<｜end▁of▁sentence｜>")
        .unwrap();
    output.deliver(packet(&[eos]), true, &codec, "test", &mut statistics);
    assert!(matches!(output.ending, Some(Ending::Eos)));
    assert!(output.pending.is_none());

    let (mut output, _client) = response(&codec);
    output.mailbox.job.prepared.stops = vec!["STOP".into()];
    output.parser = output::Output::new(false, vec![], vec!["STOP".into()], "stop");
    for &id in ids.get_ids() {
        output.deliver(packet(&[id]), true, &codec, "test", &mut statistics);
        assert!(output.pending.is_none());
    }
    assert!(matches!(output.ending, Some(Ending::Stop)));
    assert_eq!(output.parser.message()["content"], "hello ");
}

#[test]
fn pending_packet_follows_owner_through_reordering_and_drains_once() {
    let codec = token_boundary::tests::codec();
    let (a, _a_client) = response(&codec);
    let (b, _b_client) = response(&codec);
    let mut outputs = [a, b];
    let mut statistics = Statistics::default();
    for (output, text) in outputs.iter_mut().zip(["A", "B"]) {
        let ids = codec.tokenizer.encode(text, false).unwrap();
        output.deliver(
            packet(ids.get_ids()),
            false,
            &codec,
            "test",
            &mut statistics,
        );
        output.deliver(packet(ids.get_ids()), true, &codec, "test", &mut statistics);
        assert!(output.pending.is_some());
    }
    outputs.swap(0, 1);
    for output in &mut outputs {
        output.drain(true, &codec, "test", &mut statistics);
        // Fallback after an empty/cached forward or failure is idempotent.
        output.drain(false, &codec, "test", &mut statistics);
    }
    assert_eq!(outputs[0].parser.message()["content"], "BB");
    assert_eq!(outputs[1].parser.message()["content"], "AA");
    assert_eq!(statistics.maintenance_drains, 2);
    assert_eq!(statistics.barrier_drains, 0);
}

#[test]
fn cancellation_discards_pending_output_before_owner_reuse() {
    let codec = token_boundary::tests::codec();
    let ids = codec.tokenizer.encode("A", false).unwrap();
    let mut statistics = Statistics::default();
    let (mut output, client) = response(&codec);
    output.deliver(
        packet(ids.get_ids()),
        false,
        &codec,
        "test",
        &mut statistics,
    );
    let count = output.count;
    output.deliver(packet(ids.get_ids()), true, &codec, "test", &mut statistics);
    assert!(output.pending.is_some());
    drop(client);
    output.ending = Some(Ending::Cancelled);
    output.drain(false, &codec, "test", &mut statistics);
    assert!(output.pending.is_none());
    assert_eq!(output.count, count);
    let (replacement, _client) = response(&codec);
    assert_eq!(replacement.count, 0);
    assert!(replacement.pending.is_none());
}
