use super::*;
use orinfer_model_sdk::architecture::BatchSegment;
#[test]
#[ignore = "Offline preparation-stage probe; needs the real A2 package"]
fn measure_packet_preparation_stages() {
    use super::super::replay::Role;
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: The owned descriptor outlives all native resources.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let directory = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_STATE_MODEL").unwrap());
    let descriptor: serde_json::Value =
        crate::model::read(&directory.join("cache/model.json")).unwrap();
    let plan = directory
        .join("cache/packages")
        .join(descriptor["execution_package"].as_str().unwrap())
        .join("model.json");
    let team = Team::load(&plan, &directory).unwrap();
    team.exchange(Command::Admit(vec![7, 17])).unwrap();
    for (lengths, roles) in [
        (vec![128], vec![Role::Final]),
        (vec![2048], vec![Role::Final]),
        (vec![1, 799], vec![Role::Continue, Role::Producer]),
        (vec![128], vec![Role::Final]),
    ] {
        let mut l = Layout::new(
            lengths
                .iter()
                .zip([7, 17])
                .map(|(&tokens, slot)| BatchSegment { tokens, slot })
                .collect(),
            roles
                .iter()
                .enumerate()
                .filter_map(|(i, r)| (*r != Role::Producer).then_some(i))
                .collect(),
            false,
            team.rows,
            32,
        )
        .unwrap();
        l.decoder_roles = roles;
        team.exchange(Command::Invalidate(l.clone())).unwrap();
        team.exchange(Command::Prepare(l, false)).unwrap();
    }
    drop(team);
    drop(lock);
}
#[test]
#[ignore = "Offline release-mode TP8 timing probe; needs the real A2 package"]
// Reject only an explicitly selected debug probe, not the debug build.
#[allow(clippy::assertions_on_constants)]
fn measure_bound_direct_and_graph_packet_costs() {
    use super::super::replay::Role;
    use std::os::fd::AsRawFd;
    assert!(
        !cfg!(debug_assertions),
        "Measure the optimized current-build test executable"
    );
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: This owned descriptor outlives every native object below.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    {
        let session = Session::new(0, Limits::default()).unwrap();
        for count in [1, 20, 40] {
            let mut created = vec![];
            let mut destroyed = vec![];
            for _ in 0..5 {
                let start = Instant::now();
                let branches = super::super::super::graph::Branches::new(
                    &session,
                    &vec![0; count],
                    Box::new(()),
                )
                .unwrap();
                created.push(start.elapsed().as_secs_f64());
                let start = Instant::now();
                drop(branches);
                destroyed.push(start.elapsed().as_secs_f64());
            }
            eprintln!(
                "A2 branch probe {}",
                serde_json::json!({"forks":count,
                    "create_s":created,"destroy_s":destroyed})
            );
        }
    }
    let directory = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_STATE_MODEL").unwrap());
    let descriptor: serde_json::Value =
        crate::model::read(&directory.join("cache/model.json")).unwrap();
    let plan = directory
        .join("cache/packages")
        .join(descriptor["execution_package"].as_str().unwrap())
        .join("model.json");
    let team = Team::load(&plan, &directory).unwrap();
    team.exchange(Command::Admit(vec![7, 17])).unwrap();
    let layout = |lengths: &[usize], slots: &[usize], roles: &[Role], decode| {
        let mut l = Layout::new(
            lengths
                .iter()
                .zip(slots)
                .map(|(&tokens, &slot)| BatchSegment { tokens, slot })
                .collect(),
            roles
                .iter()
                .enumerate()
                .filter_map(|(i, r)| (*r != Role::Producer).then_some(i))
                .collect(),
            decode,
            if decode { 32 } else { team.rows },
            32,
        )
        .unwrap();
        l.decoder_roles = roles.to_vec();
        l
    };
    let tokens = |l: &Layout| {
        l.segments
            .iter()
            .map(|s| {
                (0..s.tokens)
                    .map(|i| ((i * 73 + s.slot * 19 + 11) % 129280) as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let prefix = layout(&[128], &[7], &[Role::Final], false);
    team.forward(prefix.clone(), tokens(&prefix), true).unwrap();
    team.snapshot_allocations(7, 1, true).unwrap();
    for (name, l) in [
        ("decode1", layout(&[1], &[7], &[Role::Continue], true)),
        ("final128", layout(&[128], &[7], &[Role::Final], false)),
        (
            "mixed800",
            layout(
                &[1, 799],
                &[7, 17],
                &[Role::Continue, Role::Producer],
                false,
            ),
        ),
        ("prefill2048", layout(&[2048], &[7], &[Role::Final], false)),
    ] {
        team.exchange(Command::Invalidate(l.clone())).unwrap();
        team.exchange(Command::Prepare(l.clone(), false)).unwrap();
        let mut timings = vec![];
        let mut promotion_s = 0.;
        for capture in [false, true] {
            if capture {
                let start = Instant::now();
                team.exchange(Command::Prepare(l.clone(), true)).unwrap();
                promotion_s = start.elapsed().as_secs_f64();
            }
            let mut times = vec![];
            for _ in 0..5 {
                for owner in [7, 17] {
                    team.exchange(Command::Restore { owner, snapshot: 1 })
                        .unwrap();
                }
                let start = Instant::now();
                let result = team
                    .exchange(Command::Forward(l.clone(), tokens(&l), false))
                    .unwrap();
                assert!(result.iter().all(|r| matches!(r, Reply::Outputs(_))));
                times.push(start.elapsed().as_secs_f64());
            }
            timings.push(times);
        }
        eprintln!(
            "A2 packet mode probe {}",
            serde_json::json!({"name":name,
                "direct_s":timings[0],"graph_s":timings[1],"promotion_s":promotion_s})
        );
    }
    team.exchange(Command::Evict(1)).unwrap();
    drop(team);
    drop(lock);
}
#[test]
#[ignore = "Requires the native TP8 A2 package and eight NPUs"]
fn graph_launch_identity_reuses_changed_lengths_and_decoder_roles() {
    use super::super::replay::Role;
    use std::os::fd::AsRawFd;
    assert_eq!(std::env::var("HCCL_DETERMINISTIC").unwrap(), "strict");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: The descriptor outlives all native workers and device resources.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let directory = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_STATE_MODEL").unwrap());
    let descriptor: serde_json::Value =
        crate::model::read(&directory.join("cache/model.json")).unwrap();
    let plan = directory
        .join("cache/packages")
        .join(descriptor["execution_package"].as_str().unwrap())
        .join("model.json");
    let team = Team::load(&plan, &directory).unwrap();
    team.exchange(Command::Admit(vec![7, 17])).unwrap();
    let layout = |lengths: &[usize], slots: &[usize], roles: &[Role]| {
        let mut layout = Layout::new(
            lengths
                .iter()
                .zip(slots)
                .map(|(&tokens, &slot)| BatchSegment { slot, tokens })
                .collect(),
            roles
                .iter()
                .enumerate()
                .filter_map(|(i, r)| (*r != Role::Producer).then_some(i))
                .collect(),
            false,
            team.rows,
            32,
        )
        .unwrap();
        layout.decoder_roles = roles.to_vec();
        layout
    };
    let tokens = |l: &Layout| {
        l.segments
            .iter()
            .map(|s| {
                (0..s.tokens)
                    .map(|i| ((i * 73 + s.slot * 19 + 11) % 129280) as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let same = |a: &[Output], b: &[Output]| {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.owner, b.owner);
            assert_eq!(a.logits, b.logits);
        }
    };
    // These layouts have identical local launches on some ranks and
    // different launches on others. Every rank must keep them distinct.
    let partial_a = layout(&[1, 15], &[7, 17], &[Role::Final; 2]);
    let partial_b = layout(&[15, 1], &[7, 17], &[Role::Final; 2]);
    team.exchange(Command::Reset(vec![7, 17])).unwrap();
    team.forward(partial_a.clone(), tokens(&partial_a), true)
        .unwrap();
    assert!(
        team.exchange(Command::Cached(partial_a.clone()))
            .unwrap()
            .iter()
            .all(|r| matches!(r, Reply::Cached(CacheState::Direct(1))))
    );
    assert!(
        team.exchange(Command::Cached(partial_b.clone()))
            .unwrap()
            .iter()
            .all(|r| matches!(r, Reply::Cached(CacheState::Missing))),
        "Partial rank aliases must never share a global cache class"
    );
    team.exchange(Command::Reset(vec![7, 17])).unwrap();
    team.forward(partial_b.clone(), tokens(&partial_b), true)
        .unwrap();
    assert!(
        team.exchange(Command::Cached(partial_a.clone()))
            .unwrap()
            .iter()
            .all(|r| matches!(r, Reply::Cached(CacheState::Direct(_) | CacheState::Graph)))
    );
    team.exchange(Command::Reset(vec![7, 17])).unwrap();
    for expected in 2..=8 {
        team.forward(partial_a.clone(), tokens(&partial_a), false)
            .unwrap();
        assert!(
            team.exchange(Command::Cached(partial_a.clone()))
                .unwrap()
                .iter()
                .all(|r| matches!(r, Reply::Cached(CacheState::Direct(n)) if *n == expected))
        );
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
    }
    let reused = team
        .forward(partial_a.clone(), tokens(&partial_a), true)
        .unwrap()
        .0;
    assert!(
        team.exchange(Command::Cached(partial_a.clone()))
            .unwrap()
            .iter()
            .all(|r| matches!(r, Reply::Cached(CacheState::Graph)))
    );
    team.exchange(Command::Reset(vec![7, 17])).unwrap();
    team.exchange(Command::Invalidate(partial_a.clone()))
        .unwrap();
    let fresh = team
        .forward(partial_a.clone(), tokens(&partial_a), true)
        .unwrap()
        .0;
    same(&reused, &fresh);
    eprintln!(
        "A2 graph partial-rank aliases remain distinct; alternating graph cache hits equal fresh direct execution"
    );
    for (name, a, b, roles) in [
        ("short", [33, 95], [35, 93], [Role::Final; 2]),
        (
            "partitioned producer",
            [769, 1279],
            [785, 1263],
            [Role::Producer; 2],
        ),
    ] {
        let a = layout(&a, &[7, 17], &roles);
        let b = layout(&b, &[7, 17], &roles);
        let tail = layout(&[128, 128], &[7, 17], &[Role::Final; 2]);
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        team.forward(a.clone(), tokens(&a), true).unwrap();
        assert!(
            team.exchange(Command::Cached(b.clone()))
                .unwrap()
                .iter()
                .all(|r| matches!(r, Reply::Cached(CacheState::Direct(_) | CacheState::Graph))),
            "{name} must reuse the captured launches"
        );
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        team.exchange(Command::Prepare(b.clone(), true)).unwrap();
        let mut reused = team.forward(b.clone(), tokens(&b), true).unwrap().0;
        if roles[0] == Role::Producer {
            reused = team.forward(tail.clone(), tokens(&tail), true).unwrap().0;
        }
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        team.exchange(Command::Invalidate(b.clone())).unwrap();
        let mut fresh = team.forward(b.clone(), tokens(&b), true).unwrap().0;
        if roles[0] == Role::Producer {
            fresh = team.forward(tail.clone(), tokens(&tail), true).unwrap().0;
        }
        same(&reused, &fresh);
        eprintln!(
            "A2 graph {name}: changed length reuse equals a fresh direct execution, complete logits"
        );
    }
    team.exchange(Command::Reset(vec![7, 17])).unwrap();
    let prefix = layout(&[128], &[7], &[Role::Final]);
    team.forward(prefix.clone(), tokens(&prefix), true).unwrap();
    team.snapshot_allocations(7, 1, true).unwrap();
    let restore = || {
        for owner in [7, 17] {
            team.exchange(Command::Restore { owner, snapshot: 1 })
                .unwrap();
        }
    };
    let a = layout(&[1, 128], &[7, 17], &[Role::Continue, Role::Final]);
    let b = layout(&[128, 1], &[17, 7], &[Role::Final, Role::Continue]);
    restore();
    team.forward(a.clone(), tokens(&a), true).unwrap();
    assert!(
        team.exchange(Command::Cached(b.clone()))
            .unwrap()
            .iter()
            .all(|r| matches!(r, Reply::Cached(CacheState::Direct(_) | CacheState::Graph)))
    );
    restore();
    team.exchange(Command::Prepare(b.clone(), true)).unwrap();
    let reused = team.forward(b.clone(), tokens(&b), true).unwrap().0;
    restore();
    team.exchange(Command::Invalidate(b.clone())).unwrap();
    let fresh = team.forward(b.clone(), tokens(&b), true).unwrap().0;
    same(&reused, &fresh);
    restore();
    let mut changed = tokens(&b);
    changed[0].fill(999);
    let changed = team.forward(b, changed, true).unwrap().0;
    assert!(
        changed
            .iter()
            .zip(&fresh)
            .any(|(a, b)| a.logits != b.logits)
    );
    team.exchange(Command::Evict(1)).unwrap();
    eprintln!(
        "A2 graph mixed: reversed owner roles reuse equals fresh direct execution; changed input executes"
    );
    drop(team);
    drop(lock);
}
#[test]
#[ignore = "Requires the complete native TP8 A2 service package and eight NPUs"]
fn native_prefix_restore_keeps_both_compression_ratios_and_owner_identity() {
    use std::os::fd::AsRawFd;
    assert!(
        std::env::var("HCCL_DETERMINISTIC").is_ok_and(|value| value.eq_ignore_ascii_case("strict")),
        "Exact cross-request logits diagnostic uses explicit strict communication; normal serving uses the default"
    );
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: the owned descriptor outlives the model and its worker threads.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let directory = std::path::PathBuf::from(
        std::env::var_os("ORINFER_A2_STATE_MODEL")
            .expect("Set ORINFER_A2_STATE_MODEL to the published native service directory"),
    );
    let descriptor: serde_json::Value =
        crate::model::read(&directory.join("cache/model.json")).unwrap();
    let plan = directory
        .join("cache/packages")
        .join(descriptor["execution_package"].as_str().unwrap())
        .join("model.json");
    let team = Team::load(&plan, &directory).unwrap();
    if team
        .model
        .batch
        .as_ref()
        .unwrap()
        .decoder_replay
        .as_ref()
        .is_some_and(|r| r.final_prompt_only)
    {
        verify_prompt_replay(&team);
        return;
    }
    let forward = |slot, tokens: Vec<u32>, full| {
        let layout = Layout::new(
            vec![BatchSegment {
                slot,
                tokens: tokens.len(),
            }],
            vec![0],
            false,
            team.rows,
            128,
        )
        .unwrap();
        team.forward(layout, vec![tokens], full)
            .unwrap()
            .0
            .remove(0)
    };
    for length in [17usize, 129, 2049, 8193] {
        team.exchange(Command::Admit(vec![17, 7])).unwrap();
        let input = (0..length)
            .map(|i| ((i * 73 + 11) % 129280) as u32)
            .collect::<Vec<_>>();
        for chunk in input.chunks(team.rows) {
            forward(17, chunk.to_vec(), false);
        }
        team.exchange(Command::Save {
            owner: 17,
            snapshot: length as u64,
            complete: false,
        })
        .unwrap();
        let expected = forward(17, vec![13, 17, 23], true);
        team.exchange(Command::Release(Some(17))).unwrap();
        team.exchange(Command::Admit(vec![91])).unwrap();
        team.exchange(Command::Restore {
            owner: 91,
            snapshot: length as u64,
        })
        .unwrap();
        let actual = forward(91, vec![13, 17, 23], true);
        assert_eq!(expected.token, actual.token, "prefix length {length}");
        assert_eq!(expected.logits, actual.logits, "prefix length {length}");
        for chunk in input.chunks(team.rows) {
            forward(7, chunk.iter().map(|&v| (v + 1) % 129280).collect(), false);
        }
        let changed = forward(7, vec![13, 17, 23], true);
        assert_ne!(
            actual.logits, changed.logits,
            "changed input must execute with the reused Graph"
        );
        team.exchange(Command::Evict(length as u64)).unwrap();
        team.exchange(Command::Release(None)).unwrap();
        eprintln!(
            "A2 prefix {length}: complete logits equal after owner remap; changed input differs"
        );
    }
    drop(team);
    drop(lock);
}

fn verify_prompt_replay(team: &Team) {
    let contract = team.model.batch.as_ref().unwrap();
    assert!(!contract.state.iter().any(|s| {
        s.buffer
            .strip_prefix("cache.swa.")
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
            .is_some_and(|layer| layer >= 21)
    }));
    let forward = |slot, tokens: Vec<u32>, output, decode| {
        let layout = Layout::new(
            vec![BatchSegment {
                slot,
                tokens: tokens.len(),
            }],
            if output { vec![0] } else { vec![] },
            decode,
            if decode { contract.owners } else { team.rows },
            contract.owners,
        )
        .unwrap();
        team.forward(layout, vec![tokens], output).unwrap().0
    };
    for length in [17usize, 2049, 8193] {
        team.exchange(Command::Admit(vec![17, 7])).unwrap();
        let input = (0..length)
            .map(|i| ((i * 73 + 11) % 129280) as u32)
            .collect::<Vec<_>>();
        let mut previous: Option<(u64, BTreeMap<u64, usize>)> = None;
        for (index, chunk) in input.chunks(team.rows).enumerate() {
            assert!(forward(17, chunk.to_vec(), false, false).is_empty());
            let id = if (index + 1) * team.rows >= length {
                length as u64
            } else {
                (2u64 << 32) + ((length as u64) << 16) + index as u64
            };
            let banks = team.snapshot_allocations(17, id, false).unwrap();
            if let Some((old_id, old_banks)) = previous {
                if contract.state.iter().any(|s| s.append_only) {
                    assert!(
                        banks.keys().any(|id| old_banks.contains_key(id)),
                        "closed KV banks must survive ancestor eviction"
                    );
                }
                team.exchange(Command::Evict(old_id)).unwrap();
            }
            previous = Some((id, banks));
        }
        let tail = (0..128).map(|i| (i * 13 + 17) as u32).collect::<Vec<_>>();
        let mut expected = forward(17, tail.clone(), true, false);
        expected.extend(forward(17, vec![13], true, true));
        team.exchange(Command::Release(Some(17))).unwrap();
        team.exchange(Command::Admit(vec![91])).unwrap();
        team.exchange(Command::Restore {
            owner: 91,
            snapshot: length as u64,
        })
        .unwrap();
        let mut actual = forward(91, tail.clone(), true, false);
        let complete = !contract
            .decoder_replay
            .as_ref()
            .unwrap()
            .decoder_state
            .is_empty();
        let complete_id = (1u64 << 32) + length as u64;
        if complete {
            team.exchange(Command::Save {
                owner: 91,
                snapshot: complete_id,
                complete: true,
            })
            .unwrap();
            team.exchange(Command::Evict(length as u64)).unwrap();
        }
        actual.extend(forward(91, vec![13], true, true));
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(
                a.logits, b.logits,
                "prompt cache {length}, final/decode logits"
            );
        }
        if complete {
            team.exchange(Command::Admit(vec![92])).unwrap();
            team.exchange(Command::Restore {
                owner: 92,
                snapshot: complete_id,
            })
            .unwrap();
            let resumed = forward(92, vec![13], true, true);
            assert_eq!(
                resumed[0].logits, actual[1].logits,
                "complete prompt {length}: consumer SWA must survive restore"
            );
            for token in [17, 23] {
                let expected = forward(91, vec![token], true, true);
                let resumed = forward(92, vec![token], true, true);
                assert_eq!(
                    expected[0].logits, resumed[0].logits,
                    "complete prompt {length}, decode {token}"
                );
            }
            team.exchange(Command::Evict(complete_id)).unwrap();
        }
        for chunk in input.chunks(team.rows) {
            forward(
                7,
                chunk.iter().map(|&v| (v + 1) % 129280).collect(),
                false,
                false,
            );
        }
        let changed = forward(7, tail, true, false);
        assert_ne!(
            actual[0].logits, changed[0].logits,
            "changed prefix must change output"
        );
        team.exchange(Command::Evict(length as u64)).unwrap();
        team.exchange(Command::Release(None)).unwrap();
        eprintln!(
            "A2 prompt replay prefix {length}: empty consumer, rebuilt final128 and decode agree after owner remap"
        );
    }
}
