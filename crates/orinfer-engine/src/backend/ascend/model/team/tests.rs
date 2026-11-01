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
        preparation(&[alternating(7); 2], true).unwrap(),
        (false, false)
    );
    assert_eq!(
        preparation(&[alternating(8); 2], true).unwrap(),
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
    ] {
        std::thread::scope(|scope| {
            let mut peers = Transport::new(Duration::from_secs(5));
            let mut maintained = vec![];
            let (recapture, capture) = preparation(&statuses, false).unwrap();
            for hit in statuses {
                let (send, commands) = mpsc::channel();
                let (reply, receive) = reply_channel();
                let (maintenance, completed) = mpsc::channel();
                maintained.push(maintenance);
                peers.push(Peer { send, receive });
                scope.spawn(move || {
                    assert!(matches!(
                        commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                        Command::Grow(_)
                    ));
                    reply.send(Ok(Reply::Ready)).unwrap();
                    assert!(matches!(
                        commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                        Command::Cached(_)
                    ));
                    reply.send(Ok(Reply::Cached(hit))).unwrap();
                    if recapture {
                        assert!(matches!(
                            commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                            Command::Invalidate(_)
                        ));
                        reply.send(Ok(Reply::Ready)).unwrap();
                    }
                    assert!(matches!(
                        commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                        Command::Prepare(_, mode) if mode == capture
                    ));
                    reply.send(Ok(Reply::Ready)).unwrap();
                    assert!(matches!(
                        commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                        Command::Forward(_, _, false)
                    ));
                    // Workers cannot complete until host maintenance has run.
                    completed.recv_timeout(Duration::from_secs(5)).unwrap();
                    reply.send(Ok(Reply::Outputs(vec![]))).unwrap();
                });
            }
            let layout = Layout::new(
                vec![orinfer_model_sdk::architecture::BatchSegment {
                    slot: 17,
                    tokens: 2,
                }],
                vec![],
                false,
                8,
                2,
            )
            .unwrap();
            assert!(
                forward_with(&peers, layout, vec![vec![0, 1]], false, || {
                    for maintenance in maintained {
                        maintenance.send(()).unwrap();
                    }
                })
                .unwrap()
                .0
                .is_empty()
            );
        });
    }
}
