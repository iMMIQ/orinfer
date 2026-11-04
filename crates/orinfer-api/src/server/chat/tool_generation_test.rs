//! Real-model tool generation retains raw tokens before protocol parsing.
use super::*;
use orinfer_engine::{execution::LoadOptions, model::GenerationInput, scheduler};
use std::{os::fd::AsRawFd, time::Instant};

#[test]
#[ignore = "Requires a real model and its GPU/NPU; configured by ORINFER_TOOL_GENERATION_SPEC"]
fn generated_tools_match_their_token_stream() {
    let spec: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("ORINFER_TOOL_GENERATION_SPEC").unwrap()).unwrap(),
    )
    .unwrap();
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: The descriptor retains the experiment lock through model teardown.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let directory = Path::new(spec["model"].as_str().unwrap());
    let codec = ChatCodec::load(directory).unwrap();
    let mut model = orinfer_engine::model::Model::load_with_options(
        directory,
        LoadOptions {
            mtp_drafts: spec["drafts"].as_u64().map(|n| n as usize),
            prefix_cache_bytes: (spec["prefix_cache_mib"].as_u64().unwrap_or(8192) as usize) << 20,
            kv_offload_bytes: (spec["kv_offload_mib"].as_u64().unwrap_or(16384) as usize) << 20,
            ..Default::default()
        },
    )
    .unwrap();
    if spec["prefix_cache_mib"] == 0 && spec["kv_offload_mib"] == 0 {
        assert!(!model.prefix_cache_enabled());
    }
    let batch = spec["batch"].as_u64().unwrap() as usize;
    assert!((1..=32).contains(&batch));
    let mut report = vec![];
    for repetition in 0..spec["repetitions"].as_u64().unwrap() {
        let row_budget = spec["speculative_row_budgets"][repetition as usize]
            .as_u64()
            .unwrap_or(192) as usize;
        let mut prepared = (0..batch)
            .map(|_| {
                let request: ChatRequest = serde_json::from_value(spec["request"].clone()).unwrap();
                let name = request.model.clone();
                codec
                    .prepare(
                        request,
                        &name,
                        model.max_context(),
                        None,
                        &mut crate::server::preparation::Context::unbounded(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut requests = prepared
            .iter_mut()
            .map(|p| {
                let mut request = model
                    .start_request(
                        GenerationInput {
                            input_tokens: p.input.clone(),
                            images: p.images.clone(),
                            audios: p.audios.clone(),
                            max_new_tokens: p.max_tokens,
                            sampling: p.sampling.clone(),
                            prefix_hints: p.prefix_hints.clone(),
                        },
                        || false,
                    )
                    .unwrap();
                if let Some(constraint) = p.constraint.take() {
                    request.set_constraint(constraint).unwrap();
                }
                request
            })
            .collect::<Vec<_>>();
        let parser = |p: &Prepared| {
            let mut parser = crate::server::output::Output::new(
                p.thinking,
                p.tools.clone(),
                p.stops.clone(),
                "token-stream",
            );
            parser.tool_format(p.tool_format);
            parser.structured_json(p.structured);
            parser.constrained_tools(p.constrained_tools);
            parser
        };
        let mut parsers = prepared.iter().map(parser).collect::<Vec<_>>();
        let mut decoders = (0..batch)
            .map(|_| codec.tokenizer.decode_stream(false))
            .collect::<Vec<_>>();
        let mut tokens = vec![vec![]; batch];
        let mut scores = vec![vec![]; batch];
        let mut packets = vec![vec![]; batch];
        let mut done = vec![false; batch];
        let started = Instant::now();
        while done.iter().any(|v| !v) {
            assert!(started.elapsed().as_secs() < 180);
            let indices = (0..batch).filter(|&i| !done[i]).collect::<Vec<_>>();
            let steps = model
                .advance_requests(
                    &mut requests
                        .iter_mut()
                        .enumerate()
                        .filter_map(|(i, r)| (!done[i]).then_some(r))
                        .collect::<Vec<_>>(),
                    &scheduler::Options {
                        max_speculative_tokens: row_budget,
                        ..Default::default()
                    },
                )
                .unwrap();
            for step in steps {
                let i = indices[step.request];
                packets[i].push(json!({
                    "position":tokens[i].len(),"tokens":step.tokens.len(),
                    "mtp_rounds":requests[i].speculation_statistics().rounds,
                }));
                scores[i].extend(
                    requests[i]
                        .take_logprobs()
                        .into_iter()
                        .map(|p| json!({"token":p.token,"logprob":p.logprob,"top":p.top})),
                );
                for token in step.tokens {
                    tokens[i].push(token);
                    if codec.eos.contains(&token) && !prepared[i].ignore_eos {
                        done[i] = true;
                        break;
                    }
                    let text = decoders[i].step(token).unwrap();
                    parsers[i]
                        .push(text.as_deref().unwrap_or(""), false)
                        .unwrap();
                    if parsers[i].stopped {
                        done[i] = true;
                        break;
                    }
                }
                done[i] |= requests[i].is_finished() || requests[i].constraint_finished();
                if done[i] {
                    model.finish_request(&mut requests[i], true).unwrap();
                }
            }
        }
        let mut all_correct = true;
        let mut all_parsers_match = true;
        for i in 0..batch {
            let p = &prepared[i];
            let exhausted = tokens[i].len() >= p.max_tokens;
            parsers[i]
                .finish(exhausted, &p.tool_choice, p.parallel)
                .unwrap();
            let text = codec.tokenizer.decode(&tokens[i], false).unwrap();
            let mut whole = parser(p);
            whole.push(&text, false).unwrap();
            whole.finish(exhausted, &p.tool_choice, p.parallel).unwrap();
            let message = parsers[i].message();
            let whole_message = whole.message();
            let parser_matches = message == whole_message;
            all_parsers_match &= parser_matches;
            let arguments = message["tool_calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let correct = json!(arguments) == spec["expected_arguments"];
            all_correct &= correct;
            report.push(
                json!({"repetition":repetition,"owner":i,"semantic_correct":correct,
                "speculative_row_budget":row_budget,"prefix":requests[i].prefix_statistics(),
                "speculation":requests[i].speculation_statistics(),
                "input_tokens":p.input,"raw_tokens":tokens[i],"raw_text":text,
                "logprobs":scores[i],"packets":packets[i],"message":message,"whole_message":whole_message,
                "parser_matches":parser_matches,"wall_s":started.elapsed().as_secs_f64()}),
            );
        }
        std::fs::write(
            spec["output"].as_str().unwrap(),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        assert!(
            all_parsers_match,
            "Incremental tool parser changed output; raw tokens retained"
        );
        if !spec["defer_semantic_assertion"].as_bool().unwrap_or(false) {
            assert!(
                all_correct,
                "Tool semantic mismatch; raw tokens and probabilities retained"
            );
        }
    }
    assert!(
        report.iter().all(|r| r["semantic_correct"] == true),
        "Tool semantic mismatch; raw tokens and probabilities retained"
    );
}
