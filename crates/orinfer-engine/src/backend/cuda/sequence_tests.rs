use super::*;

#[cfg(test)]
impl Executor {
    pub(crate) fn profile_events(&mut self, count: usize) -> Result<Vec<sys::CUevent>> {
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            let mut event = ptr::null_mut();
            // SAFETY: Live context; transfer ownership before any further error.
            unsafe {
                check(sys::cuEventCreate(&mut event, 0), "profile event")?;
            }
            self.session.events.push(event);
            events.push(event);
        }
        Ok(events)
    }
    pub(crate) fn profile_hot_graph(
        &mut self,
        key: &[(usize, usize)],
        repetitions: usize,
    ) -> Result<Vec<f32>> {
        let batch = {
            self.batch_graphs
                .borrow()
                .iter()
                .find(|(cached, _)| {
                    **cached == crate::backend::cuda::executor::BatchGraphKey::from(key.to_vec())
                        || cached.shared.is_some()
                            && cached.operation
                                == orinfer_model_sdk::packet::BatchOperation::Ordinary
                            && cached.rows == key.len()
                            && key.iter().all(|(_, tokens)| *tokens == 1)
                })
                .map(|(_, graph)| graph.0)
        };
        let graph = if let Some(graph) = batch {
            graph
        } else if key.len() == 1 {
            self.program_graph("decode")?
        } else {
            return Err("Missing hot batch graph".into());
        };
        self.profile_graph_time(graph, repetitions)
    }
    pub(crate) fn profile_invocations_hot(
        &mut self,
        operations: &[Invocation],
        repetitions: usize,
    ) -> Result<Vec<f32>> {
        self.sync()?;
        let graph = self.capture(operations)?;
        self.profile_graph_time(graph, repetitions)
    }
    fn profile_graph_time(&mut self, graph: Handle, repetitions: usize) -> Result<Vec<f32>> {
        let events = self.profile_events(2)?;
        let mut times = vec![];
        for _ in 0..repetitions {
            let mut ms = 0.;
            // SAFETY: Stable graph bindings, owned events and worker stream.
            unsafe {
                check(
                    sys::cuEventRecord(events[0], self.session.stream),
                    "profile start",
                )?;
                check(
                    sys::cuGraphLaunch(graph, self.session.stream),
                    "profile replay",
                )?;
                check(
                    sys::cuEventRecord(events[1], self.session.stream),
                    "profile end",
                )?;
                check(sys::cuEventSynchronize(events[1]), "profile completion")?;
                check(
                    sys::cuEventElapsedTime(&mut ms, events[0], events[1]),
                    "profile elapsed",
                )?;
            }
            times.push(ms);
        }
        Ok(times)
    }
    pub(crate) fn profile_graph_operations(
        &mut self,
        operations: &[Invocation],
    ) -> Result<Vec<Vec<f32>>> {
        self.profile_graph_operations_at_position(operations, None)
    }
    pub(crate) fn profile_stream_operations_at_position(
        &mut self,
        operations: &[Invocation],
        position: (&str, u32),
        exact_state: bool,
    ) -> Result<Vec<Vec<f32>>> {
        self.sync()?;
        self.ensure_invocation_kernels(operations)?;
        let events = self.profile_events(operations.len() + 1)?;
        let mut trials = vec![];
        for trial in 0..if exact_state { 1 } else { 4 } {
            self.upload_ids(position.0, &[position.1])?;
            for (i, operation) in operations.iter().enumerate() {
                // SAFETY: Owned events and live worker stream; ordinary launches.
                unsafe {
                    check(
                        sys::cuEventRecord(events[i], self.session.stream),
                        "profile stream boundary",
                    )?;
                }
                self.invoke(operation)?;
            }
            // SAFETY: Records completion after all ordinary launches.
            unsafe {
                check(
                    sys::cuEventRecord(*events.last().unwrap(), self.session.stream),
                    "profile stream end",
                )?;
            }
            self.sync()?;
            if trial == 0 && !exact_state {
                continue;
            }
            let mut times = vec![];
            for pair in events.windows(2) {
                let mut ms = 0.;
                // SAFETY: Completed events on the same stream, in launch order.
                unsafe {
                    check(
                        sys::cuEventElapsedTime(&mut ms, pair[0], pair[1]),
                        "profile stream elapsed",
                    )?;
                }
                times.push(ms);
            }
            trials.push(times);
        }
        Ok(trials)
    }
    pub(crate) fn profile_graph_operations_at_position(
        &mut self,
        operations: &[Invocation],
        position: Option<(&str, u32)>,
    ) -> Result<Vec<Vec<f32>>> {
        self.sync()?;
        self.ensure_invocation_kernels(operations)?;
        let events = self.profile_events(operations.len() + 1)?;
        // SAFETY: Sole stream owner; external event nodes retain timestamps in replay.
        unsafe {
            check(
                sys::cuStreamBeginCapture_v2(
                    self.session.stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                ),
                "profile capture",
            )?;
        }
        let result = (|| -> Result<()> {
            for (i, operation) in operations.iter().enumerate() {
                // SAFETY: Owned event, during capture; CU_EVENT_RECORD_EXTERNAL=1.
                unsafe {
                    check(
                        sys::cuEventRecordWithFlags(events[i], self.session.stream, 1),
                        "profile boundary",
                    )?;
                }
                self.invoke(operation)?;
            }
            // SAFETY: Last event records completion of all captured operations.
            unsafe {
                check(
                    sys::cuEventRecordWithFlags(*events.last().unwrap(), self.session.stream, 1),
                    "profile last boundary",
                )?;
            }
            Ok(())
        })();
        let mut graph = ptr::null_mut();
        // SAFETY: Always terminate capture after either success or failure.
        let end = unsafe {
            check(
                sys::cuStreamEndCapture(self.session.stream, &mut graph),
                "profile capture end",
            )
        };
        if let Err(error) = result.and(end) {
            if !graph.is_null() {
                // SAFETY: Failed graph has no other owner.
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(error);
        }
        let mut exec = ptr::null_mut();
        // SAFETY: Locally owned graph transferred to session on success.
        unsafe {
            if let Err(error) = check(
                sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
                "profile instantiate",
            ) {
                sys::cuGraphDestroy(graph);
                return Err(error);
            }
            if let Err(error) = check(
                sys::cuGraphUpload(exec, self.session.stream),
                "profile graph upload",
            ) {
                sys::cuGraphExecDestroy(exec);
                sys::cuGraphDestroy(graph);
                return Err(error);
            }
        }
        self.session.graphs.borrow_mut().push((graph, exec));
        let mut trials = vec![];
        for trial in 0..4 {
            if let Some((name, value)) = position {
                self.upload_ids(name, &[value])?;
            }
            // SAFETY: All request buffers are retained; bounded extra steps fit reservations.
            unsafe {
                check(
                    sys::cuGraphLaunch(exec, self.session.stream),
                    "profile operation replay",
                )?;
            }
            self.sync()?;
            if trial == 0 {
                continue;
            }
            let mut times = vec![];
            for pair in events.windows(2) {
                let mut ms = 0.;
                // SAFETY: Events completed on the same stream in replay order.
                unsafe {
                    check(
                        sys::cuEventElapsedTime(&mut ms, pair[0], pair[1]),
                        "profile operation elapsed",
                    )?;
                }
                times.push(ms);
            }
            trials.push(times);
        }
        Ok(trials)
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Session must see every inactive reservation before its cleanup.
        let buffers = self.session.virtual_buffers.get_mut();
        for (slot, sequence) in self.sequences.iter_mut().enumerate() {
            for (name, reservation) in std::mem::take(&mut sequence.reservations) {
                buffers.insert(format!("@{slot}/{name}"), reservation);
            }
        }
    }
}

#[cfg(test)]
mod graph_cache_tests {
    use super::*;

    fn calls(names: &[&str]) -> Vec<Invocation> {
        names
            .iter()
            .map(|name| Invocation {
                operation: Operation::Kernel {
                    name: (*name).into(),
                },
                sequence: None,
                launch: None,
                views: BTreeMap::new(),
            })
            .collect()
    }

    #[test]
    fn partial_shared_pieces_fork_only_complete_branches() {
        let region = ["root", "side", "side", "join"].map(str::to_owned);
        let ops = calls(&["root", "main", "side", "join", "tail"]);
        assert_eq!(
            parallel_regions(&ops, std::slice::from_ref(&region), &BTreeMap::new()).unwrap(),
            vec![[0, 2, 2, 3]]
        );
        for piece in [&ops[..3], &ops[1..], &ops[2..3]] {
            assert!(
                parallel_regions(piece, std::slice::from_ref(&region), &BTreeMap::new())
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn branch_boundaries_include_all_helpers_of_a_row_slot() {
        let ops = calls(&[
            "root/pack",
            "root/gemm",
            "main",
            "side/pack",
            "side/gemm",
            "join",
        ]);
        let slots = [
            ("root/pack", "root"),
            ("root/gemm", "root"),
            ("side/pack", "side"),
            ("side/gemm", "side"),
        ]
        .map(|(name, slot)| (name.into(), slot.into()))
        .into();
        let region = ["root", "side", "side", "join"].map(str::to_owned);
        assert_eq!(
            parallel_regions(&ops, std::slice::from_ref(&region), &slots).unwrap(),
            vec![[0, 3, 4, 5]]
        );
        // A repeated slot separated by another operation is ambiguous.
        let ops = calls(&["root/pack", "main", "root/gemm", "side/gemm", "join"]);
        assert!(
            parallel_regions(&ops, &[region], &slots)
                .unwrap()
                .is_empty()
        );
        let ops = calls(&["root", "private", "other", "private", "side", "join"]);
        let region = ["root", "side", "side", "join"].map(str::to_owned);
        assert_eq!(
            parallel_regions(&ops, &[region], &slots).unwrap(),
            vec![[0, 4, 4, 5]]
        );
    }

    #[test]
    fn malformed_and_overlapping_branches_are_rejected() {
        let ops = calls(&["a", "b", "c", "d", "e"]);
        assert!(
            parallel_regions(
                &ops,
                &[["a", "b", "b", "d"].map(str::to_owned)],
                &BTreeMap::new(),
            )
            .is_err()
        );
        assert!(
            parallel_regions(
                &ops,
                &[
                    ["a", "b", "b", "c"].map(str::to_owned),
                    ["c", "d", "d", "e"].map(str::to_owned),
                ],
                &BTreeMap::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn indirect_graph_identity_tracks_shape_and_workspace_not_request_slots() {
        use crate::backend::cuda::executor::{BatchGraphKey, SharedGraphKey};
        let graph = SharedGraphKey {
            operations: std::rc::Rc::new(vec![]),
            addresses: vec![42],
        };
        let first = BatchGraphKey {
            operation: orinfer_model_sdk::packet::BatchOperation::Ordinary,
            segments: vec![],
            rows: 2,
            shared: Some(graph.clone()),
        };
        let mut next = first.clone();
        assert_eq!(first, next);
        next.shared.as_mut().unwrap().addresses[0] = 43;
        assert_ne!(first, next);
        next = first.clone();
        next.rows = 3;
        assert_ne!(first, next);
        next = first.clone();
        next.operation = orinfer_model_sdk::packet::BatchOperation::Verification;
        assert_ne!(first, next);
    }

    #[test]
    fn cached_graph_plans_retain_storage_and_recheck_changed_launches_and_views() {
        let original = std::rc::Rc::new(vec![Invocation {
            operation: Operation::Kernel {
                name: "indirect".into(),
            },
            sequence: None,
            launch: Some(orinfer_model_sdk::operators::dynamic::RowLaunch {
                grid: [2, 1, 1],
                arguments: vec![(0, 2)],
            }),
            views: BTreeMap::from([(
                "Pointers".into(),
                crate::execution::BufferView {
                    buffer: "Pointers".into(),
                    offset: 16,
                },
            )]),
        }]);
        let prepared = BatchOperations {
            calls: &original,
            retained: Some(&original),
        };
        let retained = prepared.retain();
        assert!(std::rc::Rc::ptr_eq(&original, &retained));
        assert!(prepared.matches(&retained));
        let key = crate::backend::cuda::executor::SharedGraphKey {
            operations: retained.clone(),
            addresses: vec![42],
        };
        let mut equal = key.clone();
        assert_eq!(key.cmp(&equal), std::cmp::Ordering::Equal);
        equal.operations = std::rc::Rc::new(original.as_ref().clone());
        assert_eq!(key.cmp(&equal), std::cmp::Ordering::Equal);
        equal.addresses[0] = 43;
        assert_eq!(key.cmp(&equal), std::cmp::Ordering::Less);
        let borrowed = BatchOperations {
            calls: &original,
            retained: None,
        };
        assert!(borrowed.matches(&retained));
        assert!(!std::rc::Rc::ptr_eq(&borrowed.retain(), &retained));
        for field in 0..4 {
            let mut changed = original.as_ref().clone();
            match field {
                0 => changed[0].launch.as_mut().unwrap().grid[0] = 3,
                1 => changed[0].launch.as_mut().unwrap().arguments[0].1 = 3,
                2 => changed[0].views.get_mut("Pointers").unwrap().offset = 24,
                _ => changed[0].sequence = Some(7),
            }
            let changed = std::rc::Rc::new(changed);
            assert!(!prepared.matches(&changed));
            let changed_key = crate::backend::cuda::executor::SharedGraphKey {
                operations: changed,
                addresses: key.addresses.clone(),
            };
            assert_ne!(key.cmp(&changed_key), std::cmp::Ordering::Equal);
        }
    }

    #[test]
    fn verification_graphs_cannot_alias_ordinary_batches() {
        use crate::backend::cuda::executor::BatchGraphKey;
        use orinfer_model_sdk::packet::BatchOperation;
        let ordinary = BatchGraphKey::from(vec![(2, 4), (3, 4)]);
        let verification = BatchGraphKey {
            operation: BatchOperation::Verification,
            segments: ordinary.segments.clone(),
            rows: ordinary.rows,
            shared: None,
        };
        let mut cache = crate::backend::cuda::executor::BatchGraphCache::new();
        cache.insert(ordinary.clone(), (ptr::null_mut(), 1, 1));
        assert!(!cache.contains_key(&verification));
        cache.insert(verification, (ptr::null_mut(), 2, 1));
        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&ordinary));
    }
    #[test]
    fn node_budget_evicts_least_recently_used_before_entry_limit() {
        let mut cache = crate::backend::cuda::executor::BatchGraphCache::new();
        let large = BATCH_GRAPH_NODES * 2 / 3;
        let small = BATCH_GRAPH_NODES / 4;
        let remaining = 49_152 - large - small;
        cache.insert(vec![(2, 1)].into(), (ptr::null_mut(), 8, large));
        cache.insert(vec![(0, 1)].into(), (ptr::null_mut(), 3, small));
        assert_eq!(batch_graph_victim(&cache, remaining), None);
        assert_eq!(
            batch_graph_victim(&cache, remaining + 1),
            Some(vec![(0, 1)].into())
        );
        cache.get_mut(&vec![(0, 1)].into()).unwrap().1 = 10;
        assert_eq!(
            batch_graph_victim(&cache, remaining + 1),
            Some(vec![(2, 1)].into())
        );
        cache.remove(&vec![(0, 1)].into());
        assert_eq!(batch_graph_victim(&cache, 49_152 - large), None);
    }
    #[test]
    fn entry_limit_bounds_small_graphs() {
        let mut cache: crate::backend::cuda::executor::BatchGraphCache = (0..48)
            .map(|slot| (vec![(slot, 1)].into(), (ptr::null_mut(), slot as u64, 1)))
            .collect();
        assert_eq!(batch_graph_victim(&cache, 1), Some(vec![(0, 1)].into()));
        cache.remove(&vec![(47, 1)].into());
        assert_eq!(batch_graph_victim(&cache, 1), None);
    }
    #[test]
    fn fixed_resize_cost_preserves_small_shrinks_and_reclaims_large_excess() {
        let needed = 1024 * 1024;
        let shrink_threshold = 4 * 1024 * 1024;
        assert_eq!(fixed_resize_cost(needed - 1, needed, true), 1);
        for excess in [0, 64 * 1024, 512 * 1024, shrink_threshold - 1] {
            assert_eq!(fixed_resize_cost(needed + excess, needed, true), 0);
        }
        for excess in [shrink_threshold, shrink_threshold + 1] {
            assert_eq!(fixed_resize_cost(needed + excess, needed, true), needed);
            assert_eq!(fixed_resize_cost(needed + excess, needed, false), 0);
        }
    }
    #[test]
    fn commit_graph_shape_counts_owners_while_target_counts_token_rows() {
        use orinfer_model_sdk::packet::BatchOperation;
        assert_eq!(
            packet_graph_rows(BatchOperation::Commit, &[(7, 3), (31, 8)]),
            2
        );
        assert_eq!(
            packet_graph_rows(BatchOperation::Commit, &[(2, 2), (19, 4)]),
            2
        );
        assert_eq!(
            packet_graph_rows(BatchOperation::Verification, &[(7, 3), (31, 8)]),
            11
        );
    }
}
