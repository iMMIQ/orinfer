use super::*;
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
                        Command::Forward(_, _, false, mode, _) if mode == cached_prepare.then_some(capture)
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
                    false,
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
