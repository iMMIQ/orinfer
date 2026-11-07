use super::*;

#[test]
fn sampling_rows_keep_owner_positions_and_select_only_raw_outputs() {
    use orinfer_model_sdk::architecture::BatchSegment;
    let draw = crate::backend::ascend::sampling::Draw {
        seed: 20261002,
        step: 4097,
        temperature: 1.0,
    };
    let readout = Readout {
        full: true,
        draws: [(17, draw.clone())].into(),
        full_owners: Some([7].into()),
    };
    let layout = Layout::verification(
        vec![
            BatchSegment {
                slot: 17,
                tokens: 3,
            },
            BatchSegment { slot: 7, tokens: 2 },
        ],
        8,
        32,
    )
    .unwrap();
    let controls = readout.controls(&layout).unwrap();
    let draft_controls = draw.controls(7 * 16160, 3).unwrap();
    for (position, bytes) in draft_controls.as_chunks::<16>().0.iter().enumerate() {
        let actual: [i32; 4] = std::array::from_fn(|i| {
            i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        });
        let key = draw.control(position).unwrap();
        assert_eq!(actual, [key[0], key[1], 7 * 16160, key[2]]);
    }
    assert_eq!(
        controls[..3],
        [
            draw.control(0).unwrap(),
            draw.control(1).unwrap(),
            draw.control(2).unwrap()
        ]
    );
    assert_eq!(controls[3..], [[0; 3]; 2]);
    let owners = [(0, 17), (1, 17), (2, 17), (3, 7), (4, 7)];
    assert_eq!(readout.logit_ranges(owners.into_iter()), vec![3..5]);
    assert_eq!(
        Readout::from(true).logit_ranges(owners.into_iter()),
        vec![0..5]
    );
    let reversed = Layout::verification(
        vec![
            BatchSegment { slot: 7, tokens: 2 },
            BatchSegment {
                slot: 17,
                tokens: 3,
            },
        ],
        8,
        32,
    )
    .unwrap();
    assert_eq!(&readout.controls(&reversed).unwrap()[2..], &controls[..3]);
    let invalid = Readout {
        full_owners: None,
        ..readout
    };
    assert!(invalid.controls(&layout).is_err());
}
fn direct(uses: u8) -> CacheState {
    CacheState::Direct {
        uses,
        consecutive: uses,
    }
}
#[test]
fn promotion_waits_for_successful_uses_on_every_rank() {
    assert!(preparation(&[], false).is_err());
    for decode in [false, true] {
        assert_eq!(
            preparation(&[CacheState::Missing, CacheState::Graph], decode).unwrap(),
            (true, false)
        );
        assert_eq!(
            preparation(&[CacheState::Graph, direct(0)], decode).unwrap(),
            (true, true)
        );
        assert_eq!(
            preparation(&[CacheState::Graph; 2], decode).unwrap(),
            (false, true)
        );
    }
    assert_eq!(
        preparation(&[direct(0), direct(1)], true).unwrap(),
        (false, false)
    );
    assert_eq!(preparation(&[direct(1); 2], true).unwrap(), (false, false));
    assert_eq!(
        preparation(&[direct(1), direct(2)], true).unwrap(),
        (false, false)
    );
    assert_eq!(preparation(&[direct(2); 2], true).unwrap(), (false, true));
    let alternating = |uses| CacheState::Direct {
        uses,
        consecutive: 0,
    };
    assert_eq!(
        preparation(&[alternating(1), alternating(2)], true).unwrap(),
        (false, false)
    );
    assert_eq!(
        preparation(&[alternating(2); 2], true).unwrap(),
        (false, true)
    );
    assert_eq!(
        preparation(&[direct(7), direct(8)], false).unwrap(),
        (false, false)
    );
    assert_eq!(
        preparation(&[direct(8), direct(9)], false).unwrap(),
        (false, true)
    );
}
#[test]
fn collective_capture_is_coordinated_after_an_asymmetric_cache_eviction() {
    for statuses in [
        [CacheState::Graph, CacheState::Graph],
        [direct(1), direct(1)],
        [direct(2), direct(2)],
        [direct(7), direct(8)],
        [direct(8), direct(9)],
        [CacheState::Graph, CacheState::Missing],
        [direct(1), CacheState::Graph],
        [CacheState::Missing, CacheState::Missing],
    ]
    .into_iter()
    .flat_map(|states| [(states, false), (states, true)])
    .flat_map(|(states, repeated)| {
        [None, Some(5.), Some(64.)].map(|uses| {
            let budget = uses.map(|uses| CaptureBudget {
                uses,
                direct_s: Some(0.105),
                replay_s: Some(0.070),
                capture_s: Some(0.294),
                slack_s: 1.,
                trace: false,
            });
            (states, repeated, budget)
        })
    })
    .flat_map(|(states, repeated, budget)| {
        [false, true].map(|graphs_enabled| (states, repeated, budget, graphs_enabled))
    }) {
        let (statuses, repeated, budget, graphs_enabled) = statuses;
        std::thread::scope(|scope| {
            let mut peers = Transport::new(Duration::from_secs(5));
            let mut maintained = vec![];
            let plan = if graphs_enabled {
                preparation::plan(&statuses, repeated, budget)
            } else {
                preparation::direct_plan(&statuses)
            }
            .unwrap();
            let (recapture, capture) = (plan.invalidate, plan.capture());
            for hit in statuses {
                let (send, commands) = mpsc::channel();
                let (reply, receive) = reply_channel();
                let (maintenance, completed) = mpsc::channel();
                maintained.push(maintenance);
                peers.push(Peer { send, receive });
                scope.spawn(move || {
                    assert!(matches!(
                        commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                        Command::GrowCached(_)
                    ));
                    reply.send(Ok(Reply::Cached(hit))).unwrap();
                    if recapture {
                        assert!(matches!(
                            commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                            Command::Invalidate(_)
                        ));
                        reply.send(Ok(Reply::Ready)).unwrap();
                    }
                    let cached_prepare = !recapture
                        && statuses.iter().all(|state| {
                            matches!(state, CacheState::Graph) == capture
                        });
                    if !cached_prepare {
                        assert!(matches!(
                            commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                            Command::Prepare(_, mode) if mode == capture
                        ));
                        reply.send(Ok(Reply::Ready)).unwrap();
                    }
                    assert!(matches!(
                        commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                        Command::Forward(_, _, Readout { full: false, .. }, mode, _) if mode == cached_prepare.then_some(capture)
                    ));
                    // Workers cannot complete until host maintenance has run.
                    completed.recv_timeout(Duration::from_secs(5)).unwrap();
                    reply.send(Ok(Reply::Outputs(vec![], false))).unwrap();
                });
            }
            let layout = Layout::new(
                vec![orinfer_model_sdk::architecture::BatchSegment {
                    slot: 17,
                    tokens: if repeated { 1 } else { 2 },
                }],
                vec![],
                repeated,
                8,
                2,
            )
            .unwrap();
            assert!(
                forward_with(
                    &peers,
                    layout,
                    vec![vec![0; if repeated { 1 } else { 2 }]],
                    false.into(),
                    budget,
                    || {
                        for maintenance in maintained {
                            maintenance.send(()).unwrap();
                        }
                    },
                    graphs_enabled,
                )
                .unwrap()
                .0
                .is_empty()
            );
        });
    }
}
