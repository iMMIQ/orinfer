use super::*;
#[test]
fn prompt_planner_reserves_final_windows_without_losing_owner_tokens() {
    let requests = [17usize, 129, 2048, 2049, 4097, 8193]
        .into_iter()
        .enumerate()
        .map(|(owner, n)| Request {
            owner,
            tokens: (0..n as u32).collect(),
            output: true,
        })
        .collect::<Vec<_>>();
    let frames = plan_prompt_frames(&requests, 2048, 128).unwrap();
    for frame in &frames {
        assert!(frame.requests.iter().map(|r| r.tokens.len()).sum::<usize>() <= 2048);
    }
    for r in &requests {
        let parts = frames
            .iter()
            .flat_map(|f| &f.requests)
            .filter(|p| p.owner == r.owner)
            .collect::<Vec<_>>();
        assert_eq!(
            parts
                .iter()
                .flat_map(|r| &r.tokens)
                .copied()
                .collect::<Vec<_>>(),
            r.tokens
        );
        assert!(parts.last().unwrap().output);
        assert!(parts.last().unwrap().tokens.len() >= r.tokens.len().min(128));
        assert!(parts[..parts.len() - 1].iter().all(|p| !p.output));
    }
    assert!(plan_prompt_frames(&requests, 127, 128).is_err());
}
#[test]
fn explicit_teacher_decode_frames_keep_their_phase_and_output_count() {
    let case: Case = serde_json::from_value(serde_json::json!({
        "name": "teacher", "frames": [
            {"requests":[{"owner":17,"tokens":[1,2,3],"output":true}]},
            {"decode":true,"requests":[{"owner":17,"tokens":[7],"output":true}]},
            {"decode":true,"requests":[{"owner":17,"tokens":[9]}]}
        ]
    }))
    .unwrap();
    let mut times = Timings::default();
    for (frame, (outputs, seconds)) in case.frames.iter().zip([(1, 1.), (1, 0.2), (0, 0.3)]) {
        times.record(frame, outputs, seconds);
    }
    assert_eq!(times.input_tokens, 3);
    assert_eq!(times.prefill_s, 1.);
    assert_eq!(times.decode_s, 0.5);
    assert_eq!(times.decode_tokens, 1);
}
#[test]
fn verification_exposes_interior_causal_positions_for_each_owner() {
    let mut frame: Frame = serde_json::from_value(serde_json::json!({
        "verification":true,"requests":[
            {"owner":17,"tokens":[1,2,3,4,5,6],"output":true},
            {"owner":3,"tokens":[7,8],"output":true}
        ]
    }))
    .unwrap();
    let layout = frame_layout(&frame, 2048, 32).unwrap();
    assert_eq!(layout.outputs, [0, 0, 0, 0, 0, 0, 1, 1]);
    assert_eq!(layout.output_positions, [0, 1, 2, 3, 4, 5, 0, 1]);
    assert!(
        layout
            .decoder_roles
            .iter()
            .all(|r| *r == super::super::replay::Role::Continue)
    );
    let mut timings = Timings::default();
    timings.record(&frame, 8, 0.2);
    assert_eq!(timings.decode_tokens, 8);
    assert_eq!(timings.input_tokens, 0);
    frame.continuation = true;
    assert!(frame_layout(&frame, 2048, 32).is_err());
    frame.continuation = false;
    frame.requests[0].output = false;
    assert!(frame_layout(&frame, 2048, 32).is_err());
}

#[test]
fn mixed_verification_journals_the_prefix_and_keeps_prompt_roles() {
    let mut frame: Frame = serde_json::from_value(serde_json::json!({
        "verification":true,"verification_requests":2,"requests":[
            {"owner":17,"tokens":[1,2,3,4,5,6],"output":true},
            {"owner":3,"tokens":[7,8,9,10,11,12],"output":true},
            {"owner":7,"tokens":vec![13;788],"output":false}
        ]
    }))
    .unwrap();
    let layout = frame_layout(&frame, 2048, 32).unwrap();
    assert_eq!(layout.rows.total, 800);
    assert_eq!(frame.verification_count(), 2);
    assert_eq!(layout.outputs, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    assert_eq!(
        layout.decoder_roles[2],
        super::super::replay::Role::Producer
    );
    assert_eq!(layout.decoder(128, true).unwrap().rows.total, 12);
    let mut timings = Timings::default();
    timings.record(&frame, 12, 0.2);
    assert_eq!(timings.mixed_prefill_tokens, 788);
    assert_eq!(timings.mixed_decode_tokens, 12);
    assert_eq!(timings.mixed_s, 0.2);
    assert_eq!(timings.prefill_s + timings.decode_s, 0.);
    assert!((timings.request_s(0.1, 0.3) - 0.6).abs() < 1e-12);
    frame.requests[2].output = true;
    let layout = frame_layout(&frame, 2048, 32).unwrap();
    assert_eq!(layout.outputs.last(), Some(&2));
    assert_eq!(layout.output_positions.last(), Some(&787));
    assert_eq!(layout.decoder_roles[2], super::super::replay::Role::Final);
    assert_eq!(layout.decoder(128, true).unwrap().rows.total, 140);
    for count in [0, 4] {
        frame.verification_requests = Some(count);
        assert!(frame_layout(&frame, 2048, 32).is_err());
    }
    frame.verification_requests = Some(2);
    frame.verification = false;
    assert!(frame_layout(&frame, 2048, 32).is_err());
}

#[test]
fn strided_draft_keeps_owner_endpoints_and_counts_proposals_separately() {
    let mut frame: Frame = serde_json::from_value(serde_json::json!({
        "draft":true,"requests":[
            {"owner":17,"tokens":[271,128799,128799,128799,128799],"output":true},
            {"owner":3,"tokens":[271,128799,128799,128799,128799],"output":true}
        ]
    }))
    .unwrap();
    let layout = frame_layout(&frame, 2048, 32).unwrap();
    assert_eq!(layout.role, super::super::PacketRole::Draft);
    assert_eq!(layout.outputs, [0, 1]);
    assert_eq!(layout.output_positions, [4, 4]);
    let mut timings = Timings::default();
    timings.record(&frame, 10, 0.2);
    assert_eq!(timings.draft_proposals, 10);
    assert_eq!(timings.draft_s, 0.2);
    assert_eq!(timings.decode_tokens, 0);
    assert_eq!(timings.input_tokens, 0);
    frame.decode = true;
    assert!(frame_layout(&frame, 2048, 32).is_err());
    frame.decode = false;
    frame.requests[0].output = false;
    assert!(frame_layout(&frame, 2048, 32).is_err());
}

#[test]
fn teacher_continuation_preserves_verifier_history_and_owner_endpoints() {
    let frame: Frame = serde_json::from_value(serde_json::json!({
        "continuation":true,"requests":[
            {"owner":17,"tokens":[1,2,3,4,5,6],"output":true},
            {"owner":3,"tokens":[7,8],"output":true}
        ]
    }))
    .unwrap();
    let layout = frame_layout(&frame, 2048, 32).unwrap();
    let decoder = layout.decoder(128, true).unwrap();
    assert!(!layout.decode);
    assert_eq!(decoder.rows.total, 8);
    assert_eq!(decoder.output_positions, [5, 1]);
    assert_eq!(decoder.segments[0].slot, 17);
    assert_eq!(decoder.segments[1].slot, 3);
    assert!(
        decoder
            .decoder_roles
            .iter()
            .all(|r| *r == super::super::replay::Role::Continue)
    );
    let mut timings = Timings::default();
    timings.record(&frame, 2, 0.2);
    assert_eq!(timings.decode_tokens, 2);
    assert_eq!(timings.input_tokens, 0);
}
#[test]
fn model_chunk_quantum_fills_packets_without_short_long_request_tails() {
    for batch in [1, 2, 3, 4, 5, 8, 16, 31, 64, 127, 128] {
        let requests = (0..batch)
            .map(|i| Request {
                owner: i * 17 + 9,
                tokens: vec![i as u32; 2048],
                output: false,
            })
            .collect::<Vec<_>>();
        let frames = plan_frames(&requests, 2048, 512).unwrap();
        assert_eq!(frames.len(), batch);
        for frame in &frames {
            assert_eq!(
                frame.requests.iter().map(|r| r.tokens.len()).sum::<usize>(),
                2048
            );
            assert!(frame.requests.iter().all(|r| r.tokens.len() >= 512));
        }
        for r in &requests {
            let parts = frames
                .iter()
                .flat_map(|f| &f.requests)
                .filter(|p| p.owner == r.owner)
                .collect::<Vec<_>>();
            assert_eq!(
                parts
                    .iter()
                    .flat_map(|p| &p.tokens)
                    .copied()
                    .collect::<Vec<_>>(),
                r.tokens
            );
            assert!(parts.last().unwrap().output);
            assert!(parts[..parts.len() - 1].iter().all(|p| !p.output));
        }
    }
}
#[test]
fn unequal_requests_keep_identity_and_exact_tokens_across_chunks() {
    let requests = [
        Request {
            owner: 91,
            tokens: (0..17).collect(),
            output: false,
        },
        Request {
            owner: 8,
            tokens: (30..33).collect(),
            output: false,
        },
        Request {
            owner: 44,
            tokens: (80..111).collect(),
            output: false,
        },
    ];
    let frames = plan_frames(&requests, 11, 1).unwrap();
    for frame in &frames {
        assert!(frame.requests.iter().map(|r| r.tokens.len()).sum::<usize>() <= 11);
        assert!(frame.requests.iter().all(|r| !r.tokens.is_empty()));
    }
    for original in &requests {
        let segments = frames
            .iter()
            .flat_map(|f| &f.requests)
            .filter(|r| r.owner == original.owner)
            .collect::<Vec<_>>();
        let joined = segments
            .iter()
            .flat_map(|r| r.tokens.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(joined, original.tokens);
        assert!(segments.last().unwrap().output);
        assert!(segments[..segments.len() - 1].iter().all(|r| !r.output));
    }
    assert!(plan_frames(&requests, 2, 1).is_err());
}
