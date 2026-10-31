use super::*;
fn output() -> Arc<Output> {
    Arc::new(Output {
        owner: 17,
        token: 2,
        value: 3.,
        logits: vec![-1., 0., 3., 1.],
    })
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
fn request(output: Arc<Output>, sampling: crate::sampling::Options) -> Request {
    Request {
        owner: 1,
        slot: 91,
        abandoned: Default::default(),
        input: vec![0, 1],
        history: vec![0, 1],
        offset: 2,
        generated: 0,
        limit: 2,
        prefilling: true,
        released: false,
        sampling,
        decoder: Default::default(),
        prefix: Default::default(),
        mtp: Default::default(),
        last_output: Instant::now(),
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
