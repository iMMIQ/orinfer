use super::*;
#[test]
fn cache_reclamation_survives_request_snapshots_and_is_counted_once() {
    let mut events = CacheReclamation {
        device: 3,
        demotions: 2,
        host: 1,
    };
    let mut first = crate::prefix::Statistics {
        host_stores: 4,
        ..Default::default()
    };
    events.drain_into(&mut first);
    assert_eq!(
        (first.evictions, first.host_stores, first.host_evictions),
        (3, 6, 1)
    );
    let mut next = crate::prefix::Statistics::default();
    events.drain_into(&mut next);
    assert_eq!(
        (next.evictions, next.host_stores, next.host_evictions),
        (0, 0, 0)
    );
    events.device = 1;
    events.host = 2;
    events.drain_into(&mut next);
    assert_eq!(
        (next.evictions, next.host_stores, next.host_evictions),
        (1, 0, 2)
    );
}
fn output() -> Arc<Output> {
    Arc::new(Output {
        owner: 17,
        token: 2,
        value: 3.,
        logits: vec![-1., 0., 3., 1.],
        ..Default::default()
    })
}

#[test]
fn device_sampling_keeps_cached_heads_raw_and_respects_host_fallbacks() {
    let mut request = request(output(), crate::sampling::Options::default());
    request.device_sampling = true;
    assert!(
        request.needs_logits(),
        "The first token and cached head retain raw logits"
    );
    request.prefilling = false;
    assert!(!request.needs_logits());
    let sampled = Output {
        owner: request.slot,
        token: 3,
        value: 1.0,
        logits: vec![],
        ..Default::default()
    };
    assert_eq!(request.accept_output(&sampled).unwrap(), 3);
    assert_eq!(request.history.last(), Some(&3));
    request.sampling.top_p = 0.9;
    assert!(request.needs_logits());
    request.sampling.top_p = 1.0;
    request.sampling.top_logprobs = Some(0);
    assert!(request.needs_logits());
    request.sampling.top_logprobs = None;
    request.device_sampling = false;
    assert!(
        request.needs_logits(),
        "Older execution packages keep the host sampler"
    );
}
#[test]
fn zero_and_positive_temperatures_share_device_selection_and_owner_controls() {
    for temperature in [0., 0.7, 1., 2.] {
        let mut r = request(
            output(),
            crate::sampling::Options {
                temperature,
                ..Default::default()
            },
        );
        r.device_sampling = true;
        r.prefilling = false;
        assert!(r.device_sample());
        assert!(!r.needs_logits());
        let readout = packet::draft_readout(&[&mut r]);
        let draw = &readout.draws[&r.slot];
        assert_eq!(draw.temperature, temperature);
        let control = draw.control(0).unwrap();
        assert_eq!(
            f32::from_bits(control[2] as u32),
            if temperature == 0. {
                0.
            } else {
                (1. / temperature) as f32
            }
        );
    }
    let mut r = request(
        output(),
        crate::sampling::Options {
            temperature: 0.,
            top_k: 20,
            top_p: 0.9,
            ..Default::default()
        },
    );
    r.device_sampling = true;
    r.prefilling = false;
    assert!(
        r.device_sample(),
        "Truncation does not change greedy argmax"
    );
    r.sampling.frequency_penalty = 0.5;
    assert!(!r.device_sample());
    assert!(
        r.needs_logits(),
        "History processors retain their established fallback"
    );
}
#[test]
fn device_prefix_commit_is_shared_by_zero_and_positive_temperatures() {
    for temperature in [0., 1.] {
        let mut r = request(
            output(),
            crate::sampling::Options {
                temperature,
                ..Default::default()
            },
        );
        r.device_sampling = true;
        r.processed_sampling = true;
        r.prefilling = false;
        let values = [1, 3, 4].map(|token| Output {
            owner: r.slot,
            token,
            value: 1.,
            ..Default::default()
        });
        let actual = r
            .verify_device(&[1, 2], &values.iter().collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            actual,
            crate::mtp::greedy_commit(&[1, 2], &[1, 3, 4]).unwrap()
        );
    }
}
#[test]
fn shared_banks_and_cpu_heads_are_charged_once_after_parent_eviction() {
    let mut cache = Cache::new(4096);
    let head = output();
    let head_bytes = head.logits.len() * 4 + std::mem::size_of::<Output>();
    let parent = cache.insert(Entry::new(
        vec![1, 2],
        Media::default(),
        Snapshot {
            id: 2,
            banks: [(1, 128), (2, 256)].into(),
            output: Some(head),
            host_overhead: 0,
        },
        384 + head_bytes,
        0,
        true,
    ));
    cache.insert(Entry::new(
        vec![1, 2, 3],
        Media::default(),
        Snapshot {
            id: 3,
            banks: [(1, 128), (2, 256), (3, 512)].into(),
            output: None,
            host_overhead: 0,
        },
        896,
        0,
        false,
    ));
    let mut statistics = crate::prefix::Statistics::default();
    cache.update_statistics(&mut statistics);
    assert_eq!(statistics.resident_bytes, 896 + head_bytes);
    assert_eq!(statistics.logical_bytes, 1280 + head_bytes);
    assert_eq!(statistics.shared_bytes, 384);
    cache.remove(parent).unwrap();
    cache.update_statistics(&mut statistics);
    assert_eq!(statistics.resident_bytes, 896);
    assert_eq!(statistics.shared_bytes, 0);
    assert_eq!(statistics.entries, 1);
}
pub(super) fn request(output: Arc<Output>, sampling: crate::sampling::Options) -> Request {
    Request {
        owner: 1,
        slot: 91,
        abandoned: Default::default(),
        input: vec![0, 1],
        media: Default::default(),
        pending_images: Default::default(),
        image_features: Default::default(),
        history: vec![0, 1],
        offset: 2,
        generated: 0,
        limit: 2,
        prefilling: true,
        released: false,
        sampling,
        device_sampling: false,
        processed_sampling: false,
        decoder: Default::default(),
        draft_feedback: Default::default(),
        shared_work: false,
        prefix: Default::default(),
        mtp: Default::default(),
        last_output: Instant::now(),
        last_prefill: Instant::now(),
        ready_output: Some(output),
        cache_prompt: true,
        cache_generated: true,
    }
}
#[test]
fn complete_head_matches_only_identical_prompt_and_keeps_prefix_backoff() {
    let mut cache = Cache::new(1024);
    let input = (0..256).collect::<Vec<u32>>();
    let prefix = cache.insert(Entry::new(
        input[..128].to_vec(),
        Media::default(),
        Snapshot {
            id: 1,
            banks: BTreeMap::new(),
            output: None,
            host_overhead: 0,
        },
        0,
        0,
        false,
    ));
    let headless = cache.insert(Entry::new(
        input.clone(),
        Media::default(),
        Snapshot {
            id: 2,
            banks: BTreeMap::new(),
            output: None,
            host_overhead: 0,
        },
        0,
        0,
        false,
    ));
    assert_eq!(matched(&cache, &input, 128), Some(prefix));
    let full = cache.insert(Entry::new(
        input.clone(),
        Media::default(),
        Snapshot {
            id: 3,
            banks: BTreeMap::new(),
            output: Some(output()),
            host_overhead: 0,
        },
        0,
        0,
        true,
    ));
    assert_eq!(matched(&cache, &input, 128), Some(full));
    let mut changed = input.clone();
    changed[255] = 999;
    assert_eq!(matched(&cache, &changed, 128), Some(prefix));
    let mut extended = input.clone();
    extended.push(999);
    assert_eq!(matched(&cache, &extended, 128), Some(prefix));
    extended.extend(1000..1127);
    assert!([Some(full), Some(headless)].contains(&matched(&cache, &extended, 128)));
    cache.remove(full).unwrap();
    assert_eq!(matched(&cache, &input, 128), Some(prefix));
}
#[test]
fn cached_head_reuses_logits_but_applies_current_sampling_and_reports() {
    let shared = output();
    let options = crate::sampling::Options {
        temperature: 0.,
        ..Default::default()
    };
    let mut greedy = request(Arc::clone(&shared), options.clone());
    let steps = cached_outputs(&mut [&mut greedy]).unwrap();
    assert_eq!(steps[0].tokens, vec![2]);
    assert_eq!(greedy.history, vec![0, 1, 2]);
    assert!(!greedy.prefilling && greedy.generated == 1 && greedy.offset == 2);
    assert!(cached_outputs(&mut [&mut greedy]).unwrap().is_empty());
    let mut biased = request(
        Arc::clone(&shared),
        crate::sampling::Options {
            logit_bias: [(0, 100.)].into(),
            top_logprobs: Some(4),
            ..options
        },
    );
    assert_eq!(
        cached_outputs(&mut [&mut biased]).unwrap()[0].tokens,
        vec![0]
    );
    let records = biased.take_logprobs();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].token, 0);
    assert!(records[0].logprob.is_finite());
    assert_eq!(shared.logits, vec![-1., 0., 3., 1.]);
    let mut selected = BTreeSet::new();
    for seed in 0..32 {
        let mut sampled = request(
            Arc::clone(&shared),
            crate::sampling::Options {
                temperature: 2.,
                seed,
                ..Default::default()
            },
        );
        selected.insert(cached_outputs(&mut [&mut sampled]).unwrap()[0].tokens[0]);
    }
    assert!(
        selected.len() > 1,
        "cache must store logits, not a previous sampled token"
    );
}
#[test]
fn constraint_attached_after_cache_restore_controls_the_head() {
    #[derive(Clone)]
    struct Fixed;
    impl Constraint for Fixed {
        fn fork(&self) -> Box<dyn Constraint> {
            Box::new(self.clone())
        }
        fn mask(&mut self) -> Result<Vec<u32>> {
            Ok(vec![1 << 1])
        }
        fn consume(&mut self, token: u32) -> Result<()> {
            assert_eq!(token, 1);
            Ok(())
        }
        fn finished(&self) -> bool {
            false
        }
    }
    let mut r = request(
        output(),
        crate::sampling::Options {
            temperature: 0.,
            ..Default::default()
        },
    );
    r.set_constraint(Box::new(Fixed)).unwrap();
    assert_eq!(cached_outputs(&mut [&mut r]).unwrap()[0].tokens, vec![1]);
    assert!(r.ready_output.is_none());
}

#[test]
fn packet_order_finishes_cached_suffixes_before_cold_prompts_and_keeps_decoders_first() {
    let mut long = request(output(), Default::default());
    long.ready_output = None;
    long.input = vec![7; 40_000];
    long.offset = 0;
    let mut suffix = request(output(), Default::default());
    suffix.ready_output = None;
    suffix.input = vec![8; 128_000];
    suffix.offset = 127_740;
    let mut decoder = request(output(), Default::default());
    decoder.ready_output = None;
    decoder.prefilling = false;
    let mut cached = request(output(), Default::default());
    assert_eq!(
        packet::request_order(&[&mut long, &mut suffix, &mut decoder, &mut cached], 4000.),
        vec![2, 1, 0]
    );
}

#[test]
fn packet_order_promotes_waiting_long_prompt_then_resets_on_progress() {
    let mut long = request(output(), Default::default());
    long.ready_output = None;
    long.input = vec![7; 40_000];
    long.offset = 0;
    long.last_prefill = Instant::now() - std::time::Duration::from_secs(31);
    let mut suffix = request(output(), Default::default());
    suffix.ready_output = None;
    suffix.input = vec![8; 128_000];
    suffix.offset = 127_740;
    assert_eq!(
        packet::request_order(&[&mut long, &mut suffix], 4000.),
        vec![0, 1]
    );
    long.offset += 800;
    long.last_prefill = Instant::now();
    assert_eq!(
        packet::request_order(&[&mut long, &mut suffix], 4000.),
        vec![1, 0]
    );
}

#[test]
fn shared_scheduling_membership_survives_a_single_owner_tail() {
    let mut a = request(output(), Default::default());
    let mut b = request(output(), Default::default());
    a.slot = 7;
    b.slot = 11;
    assert!(packet::mark_shared(
        &[(0, 1), (1, 1)],
        &mut [&mut a, &mut b]
    ));
    assert!(packet::mark_shared(&[(0, 1)], &mut [&mut a]));
    let mut fresh = request(output(), Default::default());
    assert!(!packet::mark_shared(&[(0, 1)], &mut [&mut fresh]));
}

#[derive(Clone)]
struct Steps(usize);
impl Constraint for Steps {
    fn fork(&self) -> Box<dyn Constraint> {
        Box::new(self.clone())
    }
    fn mask(&mut self) -> Result<Vec<u32>> {
        Ok(vec![match self.0 {
            0 => 1 << 2,
            1 => 1 << 3,
            _ => 0,
        }])
    }
    fn consume(&mut self, token: u32) -> Result<()> {
        if token != 2 + self.0 as u32 {
            return Err("Invalid step token".into());
        }
        self.0 += 1;
        Ok(())
    }
    fn finished(&self) -> bool {
        self.0 == 2
    }
}

#[test]
fn processed_masks_follow_proposals_without_mutating_the_live_decoder() {
    let mut r = request(output(), Default::default());
    r.device_sampling = true;
    r.processed_sampling = true;
    r.prefilling = false;
    r.set_constraint(Box::new(Steps(0))).unwrap();
    assert!(!r.needs_logits());
    for (draft, expected) in [
        (vec![2, 3], vec![Some(4), Some(8), None]),
        (vec![1, 3], vec![Some(4), None, None]),
    ] {
        let readout =
            packet::target_readout(&[&mut r], false, &[(0, draft)].into(), 4, &[(0, 3)]).unwrap();
        let rows = readout.parameters.get().unwrap();
        assert_eq!(
            rows[&r.slot]
                .iter()
                .map(|p| p.mask.as_ref().map(|m| m[0]))
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(r.decoder.gpu_mask().unwrap(), Some(vec![4]));
    }
}

#[test]
fn verification_commits_only_accepted_device_reports_and_grammar_states() {
    let make = |token| Output {
        token,
        report: Some(TokenLogprob {
            token,
            logprob: -0.5,
            top: vec![(token, -0.5)],
        }),
        ..Default::default()
    };
    let values = [make(2), make(3), make(0)];
    for (draft, accepted) in [(vec![1, 3], vec![2]), (vec![2, 3], vec![2, 3])] {
        let mut r = request(
            output(),
            crate::sampling::Options {
                top_logprobs: Some(1),
                ..Default::default()
            },
        );
        r.set_constraint(Box::new(Steps(0))).unwrap();
        assert_eq!(
            r.verify_device(&draft, &values.iter().collect::<Vec<_>>())
                .unwrap(),
            accepted
        );
        assert_eq!(
            r.take_logprobs()
                .iter()
                .map(|v| v.token)
                .collect::<Vec<_>>(),
            accepted
        );
        assert_eq!(r.decoder.finished(), accepted.len() == 2);
    }
}
