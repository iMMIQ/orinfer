use super::{
    DriverLease, open_library,
    runtime::{Limits, Session},
};
use libloading::Library;
use std::{path::Path, sync::Arc};

pub(super) struct Fixture {
    _dir: tempfile::TempDir,
    library: Library,
    driver: Option<Arc<DriverLease>>,
}
impl Fixture {
    pub(super) fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("libacl-fixture.so");
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ascend_runtime.c");
        assert!(
            std::process::Command::new("cc")
                .args(["-shared", "-fPIC", "-Wall", "-Werror"])
                .arg(source)
                .arg("-o")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        // SAFETY: The test compiles this exact fixture and retains its code.
        let library = unsafe { Library::new(&path) }.unwrap();
        let driver = Arc::new(DriverLease {
            driver: Some(Arc::new(open_library(&path).unwrap())),
            registered: false,
        });
        Self {
            _dir: dir,
            library,
            driver: Some(driver),
        }
    }
    pub(super) fn session(&self) -> Session {
        Session::with_driver(
            Arc::clone(self.driver.as_ref().unwrap()),
            0,
            Limits::default(),
        )
        .unwrap()
    }
    pub(super) fn aot(&self) -> (std::path::PathBuf, super::kernels::Spec) {
        use super::kernels::{Access, Operand, Spec};
        use crate::artifact::{Dtype, FileIdentity};
        let library = self._dir.path().join("libacl-fixture.so");
        let spec = Spec {
            schema_version: 1,
            target: orinfer_device::Target {
                host_arch: std::env::consts::ARCH.into(),
                backend: orinfer_device::Backend::Ascend,
                device_arch: "ascend910b".into(),
            },
            op: "copy_fixture".into(),
            operands: vec![
                Operand {
                    name: "input".into(),
                    dtype: Dtype::I8,
                    shape: vec![4],
                    access: Access::Read,
                },
                Operand {
                    name: "output".into(),
                    dtype: Dtype::I8,
                    shape: vec![4],
                    access: Access::Write,
                },
            ],
            parameters: Default::default(),
            compiler: [("fixture".into(), "cc".into())].into(),
            source_sha256: "1".repeat(64),
            library: FileIdentity {
                file: "libacl-fixture.so".into(),
                sha256: crate::artifact::sha256(&std::fs::read(library).unwrap()),
                tensor: None,
            },
        };
        let text =
            std::ffi::CString::new(serde_json::to_string(&spec.identity()).unwrap()).unwrap();
        // SAFETY: The CPU fixture copies this string into its bounded static
        // descriptor; no borrowed host pointer survives this setter.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(*const std::ffi::c_char)>(b"test_kernel_contract\0")
                .unwrap()(text.as_ptr());
        }
        let path = self._dir.path().join("kernel.json");
        std::fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
        (path, spec)
    }
    pub(super) fn counter(&self, index: i32) -> i32 {
        // SAFETY: Fixture declares this function with the exact C signature.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"test_counter\0")
                .unwrap()(index)
        }
    }
    pub(super) fn failure(&self, index: i32) {
        // SAFETY: Fixture declares this function with the exact C signature.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(i32)>(b"test_failure\0")
                .unwrap()(index);
        }
    }
    pub(super) fn pending_jobs(&self) -> i32 {
        // SAFETY: The private CPU fixture exports this exact zero-argument ABI.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn() -> i32>(b"test_pending_jobs\0")
                .unwrap()()
        }
    }
    pub(super) fn assert_clean(&mut self) {
        self.driver.take();
        for index in 0..5 {
            assert_eq!(self.counter(index), 0, "resource counter {index}");
        }
        assert_eq!(self.counter(5), 1);
        assert_eq!(self.counter(6), 1);
        assert_eq!(self.counter(7), 0);
        assert_eq!(self.counter(8), 0);
    }
}

#[test]
fn aot_dynamic_rows_bind_real_extent_and_preserve_tail() {
    use super::kernels::{Kernel, Template};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let (path, mut spec) = fixture.aot();
    spec.parameters.insert("_row_capacity".into(), 4.into());
    let descriptor =
        std::ffi::CString::new(serde_json::to_string(&spec.identity()).unwrap()).unwrap();
    // SAFETY: The fixture copies the descriptor, as in Fixture::aot.
    unsafe {
        fixture
            .library
            .get::<unsafe extern "C" fn(*const std::ffi::c_char)>(b"test_kernel_contract\0")
            .unwrap()(descriptor.as_ptr());
    }
    std::fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    let template = Template::load(&session, &path).unwrap();
    let input = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    let buffers = [input.clone(), output.clone()];
    assert!(Kernel::bind_rows(&session, &template, &buffers, 0).is_err());
    assert!(Kernel::bind_rows(&session, &template, &buffers, 5).is_err());
    for rows in [1, 3, 4] {
        let kernel = Kernel::bind_rows(&session, &template, &buffers, rows).unwrap();
        session
            .upload(&input, 0, &[1, 2, 3, 4])
            .unwrap()
            .wait()
            .unwrap();
        session.upload(&output, 0, &[9; 4]).unwrap().wait().unwrap();
        kernel.enqueue(&session).unwrap().wait().unwrap();
        let actual = session.download(&output, 0, 4).unwrap();
        assert_eq!(&actual[..rows], &[1, 2, 3, 4][..rows]);
        assert_eq!(&actual[rows..], &vec![9; 4 - rows]);
    }
    drop((template, buffers, input, output, session));
    fixture.assert_clean();
}

#[test]
fn aot_binding_rejects_foreign_buffers_aliases_and_tampered_contract() {
    use super::kernels::Kernel;
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = fixture.session();
    let input = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    let foreign = other.allocate(4).unwrap();
    let short = session.allocate(2).unwrap();
    let (path, mut spec) = fixture.aot();
    assert!(Kernel::load(&session, &path, &[input.clone(), foreign.clone()]).is_err());
    assert!(Kernel::load(&session, &path, &[input.clone(), input.clone()]).is_err());
    assert!(Kernel::load(&session, &path, &[input.clone(), short.clone()]).is_err());
    assert!(Kernel::load(&session, &path, std::slice::from_ref(&input)).is_err());
    session
        .upload(&input, 0, &[1, 2, 3, 4])
        .unwrap()
        .wait()
        .unwrap();
    let kernel = Kernel::load(&session, &path, &[input.clone(), output.clone()]).unwrap();
    assert!(kernel.enqueue(&other).is_err());
    kernel.enqueue(&session).unwrap().wait().unwrap();
    assert_eq!(session.download(&output, 0, 4).unwrap(), [1, 2, 3, 4]);
    spec.op = "tampered".into();
    std::fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    assert!(Kernel::load(&session, &path, &[input.clone(), output.clone()]).is_err());
    drop((kernel, input, output, foreign, short, other, session));
    fixture.assert_clean();
}

#[test]
fn aot_graph_retains_operands_code_and_context_until_last_fence() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let (path, _) = fixture.aot();
    let input = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    session
        .upload(&input, 0, &[1, 2, 3, 4])
        .unwrap()
        .wait()
        .unwrap();
    let kernel =
        super::kernels::Kernel::load(&session, &path, &[input.clone(), output.clone()]).unwrap();
    let graph = kernel.capture(&session).unwrap();
    drop((kernel, input, output));
    assert_eq!(session.allocated_bytes(), 8);
    let ticket = graph.enqueue(&session).unwrap();
    drop(graph);
    assert_eq!(session.allocated_bytes(), 8);
    ticket.wait().unwrap();
    assert_eq!(session.allocated_bytes(), 0);
    drop((ticket, session));
    fixture.assert_clean();
}

#[test]
fn aot_program_orders_shared_operands_and_retains_the_whole_fragment() {
    use super::kernels::{Kernel, Program};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = fixture.session();
    let (path, _) = fixture.aot();
    let input = session.allocate(4).unwrap();
    let middle = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    session
        .upload(&input, 0, &[9, 8, 7, 6])
        .unwrap()
        .wait()
        .unwrap();
    let first = Kernel::load(&session, &path, &[input.clone(), middle.clone()]).unwrap();
    let second = Kernel::load(&session, &path, &[middle.clone(), output.clone()]).unwrap();
    assert!(Program::new(&session, &[]).is_err());
    assert!(Program::new(&other, &[&first]).is_err());
    let program = Program::new(&session, &[&first, &second]).unwrap();
    assert!(program.enqueue(&other).is_err());
    program.enqueue(&session).unwrap().wait().unwrap();
    assert_eq!(session.download(&output, 0, 4).unwrap(), [9, 8, 7, 6]);
    let graph = program.capture(&session).unwrap();
    drop((program, first, second, input, middle, output));
    assert_eq!(session.allocated_bytes(), 12);
    let ticket = graph.enqueue(&session).unwrap();
    drop(graph);
    ticket.wait().unwrap();
    assert_eq!(session.allocated_bytes(), 0);
    drop((ticket, other, session));
    fixture.assert_clean();
}

#[test]
fn input_slabs_split_buffers_and_backpressure_within_the_pinned_budget() {
    let mut fixture = Fixture::new();
    let session = Session::with_driver(
        Arc::clone(fixture.driver.as_ref().unwrap()),
        0,
        Limits {
            device_bytes: 64,
            pinned_bytes: 8,
            timeout_ms: 100,
        },
    )
    .unwrap();
    let a = session.allocate(4).unwrap();
    let b = session.allocate(5).unwrap();
    let c = session.allocate(11).unwrap();
    let output = session.allocate(11).unwrap();
    let mut inputs = vec![
        (a.clone(), vec![1; 4]),
        (b.clone(), vec![2; 5]),
        (c.clone(), vec![3; 11]),
    ];
    let uploaded = session.upload_inputs(&inputs).unwrap();
    assert!(session.pinned_bytes() <= 8);
    for (_, data) in &mut inputs {
        data.fill(99);
    }
    drop(inputs);
    let read = session.copy(&c, 0, &output, 0, 11).unwrap();
    let overwrite = session.upload_inputs(&[(c.clone(), vec![4; 11])]).unwrap();
    assert!(session.pinned_bytes() <= 8);
    assert_eq!(session.download(&a, 0, 4).unwrap(), [1; 4]);
    assert_eq!(session.download(&b, 0, 5).unwrap(), [2; 5]);
    assert_eq!(session.download(&output, 0, 8).unwrap(), [3; 8]);
    assert_eq!(session.download(&output, 8, 3).unwrap(), [3; 3]);
    assert_eq!(session.download(&c, 0, 8).unwrap(), [4; 8]);
    assert_eq!(session.download(&c, 8, 3).unwrap(), [4; 3]);
    drop((uploaded, read, overwrite, a, b, c, output, session));
    fixture.assert_clean();
}

#[test]
fn input_batch_copies_host_data_and_preserves_compute_reader_order() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = fixture.session();
    let input = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    let scale = session.allocate(2).unwrap();
    let mut inputs = vec![
        (input.clone(), vec![1, 2, 3, 4]),
        (scale.clone(), vec![7, 8]),
    ];
    let uploaded = session.upload_inputs(&inputs).unwrap();
    for (_, data) in &mut inputs {
        data.fill(0);
    }
    drop(inputs);
    assert_eq!(session.pinned_bytes(), 6);
    assert_eq!(fixture.counter(1), 1, "one pinned allocation for the batch");
    assert!(!uploaded.is_ready().unwrap());
    let copied = session.copy(&input, 0, &output, 0, 4).unwrap();
    let overwritten = session
        .upload_inputs(&[(input.clone(), vec![9; 4])])
        .unwrap();
    assert_eq!(session.download(&output, 0, 4).unwrap(), [1, 2, 3, 4]);
    assert_eq!(session.download(&input, 0, 4).unwrap(), [9; 4]);
    assert_eq!(session.download(&scale, 0, 2).unwrap(), [7, 8]);
    assert!(other.upload_inputs(&[(input.clone(), vec![0; 4])]).is_err());
    assert!(
        session
            .upload_inputs(&[(input.clone(), vec![0; 3])])
            .is_err()
    );
    drop((
        uploaded,
        copied,
        overwritten,
        input,
        output,
        scale,
        session,
        other,
    ));
    fixture.assert_clean();
}

#[test]
fn copy_batch_checks_every_span_before_queueing_and_retains_dropped_sources() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let source = session.allocate(8).unwrap();
    let target = session.allocate(8).unwrap();
    session
        .upload(&source, 0, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap()
        .wait()
        .unwrap();
    session.fill(&target, 0).unwrap().wait().unwrap();
    assert!(
        session
            .copy_many(&[
                (source.clone(), 0, target.clone(), 0, 4),
                (source.clone(), 7, target.clone(), 4, 2),
            ])
            .is_err()
    );
    assert_eq!(session.download(&target, 0, 8).unwrap(), [0; 8]);
    let ticket = session
        .copy_many(&[
            (source.clone(), 4, target.clone(), 0, 4),
            (source.clone(), 0, target.clone(), 4, 4),
        ])
        .unwrap();
    drop(source);
    assert!(!ticket.is_ready().unwrap());
    ticket.wait().unwrap();
    assert_eq!(
        session.download(&target, 0, 8).unwrap(),
        [5, 6, 7, 8, 1, 2, 3, 4]
    );
    drop((ticket, target, session));
    fixture.assert_clean();
}

#[test]
fn input_batch_validates_all_spans_before_submitting_and_fences_errors() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let first = session.allocate(4).unwrap();
    let second = session.allocate(2).unwrap();
    assert!(
        session
            .upload_inputs(&[(first.clone(), vec![1; 4]), (second.clone(), vec![1])])
            .is_err()
    );
    assert_eq!(fixture.counter(1), 0);
    assert_eq!(session.download(&first, 0, 4).unwrap(), [0; 4]);
    fixture.failure(4);
    assert!(
        session
            .upload_inputs(&[(first.clone(), vec![2; 4]), (second.clone(), vec![3; 2])])
            .is_err()
    );
    assert_eq!(session.pinned_bytes(), 10);
    fixture.failure(0);
    assert_eq!(session.download(&first, 0, 4).unwrap(), [2; 4]);
    assert_eq!(session.download(&second, 0, 2).unwrap(), [3; 2]);
    drop((first, second, session));
    fixture.assert_clean();
}

#[test]
fn packet_completion_retains_graphs_and_fences_partial_submission_errors() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    // SAFETY: Empty fixture capture has no operands and uses the owned stream.
    let graph = std::rc::Rc::new(unsafe {
        super::graph::Graph::capture(&session, Box::new(()), |_| Ok(())).unwrap()
    });
    // SAFETY: The enclosing completion retains the graph; the stream belongs
    // to the session that captured it.
    let ticket = unsafe {
        session
            .submit_compute(vec![], Box::new(graph.clone()), |stream| {
                graph.execute_on_stream(stream)
            })
            .unwrap()
    };
    drop(graph);
    assert_eq!(fixture.counter(7), 1);
    ticket.wait().unwrap();
    assert_eq!(fixture.counter(7), 0);
    // SAFETY: Same empty fixture capture on the owned stream.
    let graph = std::rc::Rc::new(unsafe {
        super::graph::Graph::capture(&session, Box::new(()), |_| Ok(())).unwrap()
    });
    // SAFETY: Even on failure, submit_compute fences queued work before its
    // retained owner can be released.
    let result = unsafe {
        session.submit_compute(vec![], Box::new(graph.clone()), |stream| {
            graph.execute_on_stream(stream)?;
            Err("partial packet failure".into())
        })
    };
    assert!(result.is_err());
    drop(graph);
    assert_eq!(fixture.counter(7), 0);
    drop((ticket, session));
    fixture.assert_clean();
}

#[test]
fn completion_cache_waits_for_the_last_ticket_and_reuses_fenced_generations() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(4).unwrap();
    let first = session.upload(&buffer, 0, &[1; 4]).unwrap();
    first.wait().unwrap();
    session.reap().unwrap();
    let second = session.upload(&buffer, 0, &[2; 4]).unwrap();
    assert_eq!(
        fixture.counter(10),
        2,
        "a live Ticket retains its generation"
    );
    assert!(first.is_ready().unwrap());
    assert!(!second.is_ready().unwrap());
    second.wait().unwrap();
    drop((first, second));
    session.reap().unwrap();
    let created = fixture.counter(10);
    for value in 3..35 {
        let ticket = session.upload(&buffer, 0, &[value; 4]).unwrap();
        assert!(
            !ticket.is_ready().unwrap(),
            "a fresh record resets readiness"
        );
        ticket.wait().unwrap();
        assert_eq!(session.download(&buffer, 0, 4).unwrap(), [value; 4]);
        drop(ticket);
        session.reap().unwrap();
    }
    assert_eq!(fixture.counter(10), created, "completed events are reused");
    drop((buffer, session));
    fixture.assert_clean();
}

#[test]
fn completion_cache_is_bounded_and_context_entry_preserves_the_previous_owner() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let tickets = (0..300)
        .map(|_| {
            // SAFETY: Empty submissions have no raw operands or external storage.
            unsafe { session.submit_compute(vec![], Box::new(()), |_| Ok(())) }.unwrap()
        })
        .collect::<Vec<_>>();
    session.synchronize().unwrap();
    drop(tickets);
    session.reap().unwrap();
    assert_eq!(
        fixture.counter(2),
        256,
        "idle completion handles stay bounded"
    );
    let other = fixture.session();
    let changes = fixture.counter(11);
    session.memory_info().unwrap();
    assert_eq!(fixture.counter(11), changes, "current context needs no set");
    other.memory_info().unwrap();
    assert_eq!(
        fixture.counter(11),
        changes + 2,
        "foreign context is restored"
    );
    drop((other, session));
    fixture.assert_clean();
}

#[test]
fn pending_record_is_not_completed_by_an_empty_waiter_set() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(4).unwrap();
    let ticket = session
        .upload_inputs(&[(buffer.clone(), vec![9, 8, 7, 6])])
        .unwrap();
    assert!(!ticket.is_ready().unwrap());
    // Polling must leave work queued rather than turn reap into a stream wait.
    session.reap().unwrap();
    assert!(!ticket.is_ready().unwrap());
    assert_eq!(session.pinned_bytes(), 4);
    ticket.wait().unwrap();
    assert!(ticket.is_ready().unwrap());
    assert_eq!(session.download(&buffer, 0, 4).unwrap(), [9, 8, 7, 6]);
    drop((ticket, buffer, session));
    fixture.assert_clean();
}

#[test]
fn ready_query_keeps_dma_backing_until_a_hard_completion_fence() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(4).unwrap();
    let ticket = session
        .upload_inputs(&[(buffer.clone(), vec![1, 2, 3, 4])])
        .unwrap();
    fixture.failure(11);
    assert!(ticket.is_ready().unwrap());
    assert_eq!(session.pinned_bytes(), 4);
    session.trim_pinned_cache();
    assert_eq!(session.pinned_bytes(), 4, "a query is not a hard DMA fence");
    drop(ticket);
    session.reap().unwrap();
    fixture.failure(0);
    assert_eq!(session.pinned_bytes(), 4);
    assert_eq!(session.download(&buffer, 0, 4).unwrap(), [1, 2, 3, 4]);
    drop((buffer, session));
    fixture.assert_clean();
}

#[test]
fn async_upload_retains_copied_host_data_and_dropped_device_handles() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(4).unwrap();
    let mut source = vec![1, 2, 3, 4];
    let ticket = session.upload(&buffer, 0, &source).unwrap();
    source.fill(0);
    assert!(!ticket.is_ready().unwrap());
    assert_eq!(session.pinned_bytes(), 4);
    drop(buffer);
    assert_eq!(fixture.counter(0), 1);
    ticket.wait().unwrap();
    assert_eq!(session.pinned_bytes(), 4);
    assert_eq!(fixture.counter(0), 0);
    drop(ticket);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn pinned_cache_preserves_pending_sources_and_reuses_completed_storage() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let first = session.allocate(4).unwrap();
    let second = session.allocate(4).unwrap();
    let a = session.upload(&first, 0, &[1, 2, 3, 4]).unwrap();
    let b = session.upload(&second, 0, &[5, 6, 7, 8]).unwrap();
    assert!(!a.is_ready().unwrap());
    assert!(!b.is_ready().unwrap());
    assert_eq!(
        fixture.counter(9),
        2,
        "pending DMA sources must be distinct"
    );
    assert_eq!(session.pinned_bytes(), 8);
    session.synchronize().unwrap();
    assert_eq!(session.download(&first, 0, 4).unwrap(), [1, 2, 3, 4]);
    assert_eq!(session.download(&second, 0, 4).unwrap(), [5, 6, 7, 8]);
    let c = session.upload(&first, 0, &[9, 10, 11, 12]).unwrap();
    c.wait().unwrap();
    assert_eq!(session.download(&first, 0, 4).unwrap(), [9, 10, 11, 12]);
    assert_eq!(fixture.counter(9), 2, "completed backing should be reused");
    // Completed Tickets may outlive the cache owner without keeping pinned
    // storage or making the next owner reuse an unfinished transfer.
    drop(session);
    assert_eq!(fixture.counter(1), 0);
    assert!(a.is_ready().unwrap());
    drop((a, b, c, first, second));
    fixture.assert_clean();
}

#[test]
fn pinned_cache_evicts_idle_capacity_and_preserves_inflight_budget() {
    let mut fixture = Fixture::new();
    let session = Session::with_driver(
        Arc::clone(fixture.driver.as_ref().unwrap()),
        0,
        Limits {
            device_bytes: 16,
            pinned_bytes: 8,
            timeout_ms: 100,
        },
    )
    .unwrap();
    let buffer = session.allocate(16).unwrap();
    let small = session.upload(&buffer, 0, &[1; 4]).unwrap();
    small.wait().unwrap();
    let large = session.upload(&buffer, 0, &[5; 8]).unwrap();
    large.wait().unwrap();
    assert_eq!(session.pinned_bytes(), 8);
    assert_eq!(fixture.counter(1), 1, "idle small buffer must be evicted");
    assert_eq!(fixture.counter(9), 2);
    let short = session.upload(&buffer, 0, &[8, 9, 10]).unwrap();
    session.trim_pinned_cache();
    assert_eq!(session.pinned_bytes(), 8);
    assert!(session.upload(&buffer, 8, &[7; 8]).is_err());
    assert!(session.upload(&buffer, 0, &[0; 9]).is_err());
    assert_eq!(fixture.counter(9), 2, "failed leases must not allocate");
    short.wait().unwrap();
    let changed = session.upload(&buffer, 8, &[7; 8]).unwrap();
    changed.wait().unwrap();
    assert_eq!(
        session.download(&buffer, 0, 8).unwrap(),
        [8, 9, 10, 5, 5, 5, 5, 5]
    );
    assert_eq!(session.download(&buffer, 8, 8).unwrap(), [7; 8]);
    assert_eq!(fixture.counter(9), 2);
    session.trim_pinned_cache();
    assert_eq!(session.pinned_bytes(), 0);
    drop((small, large, short, changed, buffer, session));
    fixture.assert_clean();
}

#[test]
fn pinned_cache_bounds_metadata_under_many_pending_uploads() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffers = (0..65)
        .map(|_| session.allocate(4).unwrap())
        .collect::<Vec<_>>();
    let tickets = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| session.upload(buffer, 0, &[i as u8; 4]).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(session.pinned_bytes(), 65 * 4);
    session.synchronize().unwrap();
    assert_eq!(session.pinned_bytes(), 64 * 4);
    for (i, buffer) in buffers.iter().enumerate() {
        assert_eq!(session.download(buffer, 0, 4).unwrap(), [i as u8; 4]);
    }
    assert_eq!(fixture.counter(9), 65);
    session.trim_pinned_cache();
    assert_eq!(session.pinned_bytes(), 0);
    drop((tickets, buffers, session));
    fixture.assert_clean();
}

#[test]
fn stream_dependency_roundtrip_and_session_ownership() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = fixture.session();
    let buffer = session.allocate(8).unwrap();
    let mut source = vec![3, 4, 5, 6];
    let ticket = session.upload(&buffer, 2, &source).unwrap();
    source.fill(0);
    session.wait_on_compute(&ticket).unwrap();
    assert!(other.upload(&buffer, 0, &[1]).is_err());
    assert!(other.wait_on_compute(&ticket).is_err());
    assert!(session.upload(&buffer, usize::MAX, &[1]).is_err());
    assert!(session.upload(&buffer, 7, &[1, 2]).is_err());
    let downloaded = session.download(&buffer, 2, 4).unwrap();
    assert_eq!(downloaded, [3, 4, 5, 6]);
    session.fill(&buffer, 9).unwrap().wait().unwrap();
    assert_eq!(session.download(&buffer, 0, 8).unwrap(), [9; 8]);
    drop((ticket, buffer, session, other));
    fixture.assert_clean();
}

#[test]
fn buffer_can_outlive_session_without_finalizing_its_context() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(8).unwrap();
    session.upload(&buffer, 0, &[7; 8]).unwrap();
    drop(session);
    assert_eq!(fixture.counter(1), 0);
    assert_eq!(fixture.counter(4), 1);
    drop(buffer);
    fixture.assert_clean();
}

#[test]
fn failed_record_fences_queued_dma_and_failed_creation_rolls_back() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(4).unwrap();
    fixture.failure(4);
    assert!(session.upload(&buffer, 0, &[1, 2, 3, 4]).is_err());
    session.trim_pinned_cache();
    assert_eq!(fixture.counter(1), 0);
    assert_eq!(fixture.counter(2), 0);
    fixture.failure(0);
    assert_eq!(session.download(&buffer, 0, 4).unwrap(), [1, 2, 3, 4]);
    drop((buffer, session));
    fixture.failure(5);
    assert!(
        Session::with_driver(
            Arc::clone(fixture.driver.as_ref().unwrap()),
            0,
            Limits::default()
        )
        .is_err()
    );
    fixture.assert_clean();
}

#[test]
fn tensor_views_retain_backing_and_reject_overlapping_device_copies() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let storage = session.allocate(32).unwrap();
    let left = storage.slice(4, 8).unwrap();
    let overlapping = storage.slice(8, 8).unwrap();
    let right = storage.slice(20, 8).unwrap();
    assert!(storage.slice(30, 4).is_err());
    assert!(storage.slice(0, 0).is_err());
    assert!(session.copy(&left, 0, &overlapping, 0, 8).is_err());
    session
        .upload(&left, 0, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap()
        .wait()
        .unwrap();
    session
        .copy(&left, 0, &right, 0, 8)
        .unwrap()
        .wait()
        .unwrap();
    drop(storage);
    assert_eq!(session.allocated_bytes(), 32);
    assert_eq!(
        session.download(&right, 0, 8).unwrap(),
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    drop((left, overlapping, right));
    assert_eq!(session.allocated_bytes(), 0);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn budgets_are_enforced_before_native_allocation() {
    let mut fixture = Fixture::new();
    let session = Session::with_driver(
        Arc::clone(fixture.driver.as_ref().unwrap()),
        0,
        Limits {
            device_bytes: 8,
            pinned_bytes: 4,
            timeout_ms: 100,
        },
    )
    .unwrap();
    let buffer = session.allocate(8).unwrap();
    assert!(session.allocate(1).is_err());
    assert!(session.allocate(0).is_err());
    assert!(session.upload(&buffer, 0, &[1; 5]).is_err());
    let ticket = session.upload(&buffer, 0, &[1; 4]).unwrap();
    assert!(session.upload(&buffer, 4, &[2; 4]).is_err());
    ticket.wait().unwrap();
    session.upload(&buffer, 4, &[2; 4]).unwrap().wait().unwrap();
    drop((buffer, ticket, session));
    fixture.assert_clean();
}

#[test]
fn device_copy_preserves_extents_and_retains_waited_events() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let source = session.allocate(8).unwrap();
    let destination = session.allocate(8).unwrap();
    let uploaded = session
        .upload(&source, 0, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    uploaded.wait().unwrap();
    let waited = session.wait_on_compute(&uploaded).unwrap();
    drop(uploaded);
    session.reap().unwrap();
    assert_eq!(
        fixture.counter(2),
        2,
        "queued compute wait retains its source event"
    );
    let copied = session.copy(&source, 2, &destination, 4, 4).unwrap();
    assert!(session.copy(&source, 1, &source, 2, 4).is_err());
    assert_eq!(session.download(&destination, 4, 4).unwrap(), [3, 4, 5, 6]);
    drop((waited, copied, source, destination, session));
    fixture.assert_clean();
}

#[test]
fn failed_completion_retires_and_quarantines_dma_backing() {
    let fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(8).unwrap();
    let ticket = session.upload(&buffer, 0, &[1; 8]).unwrap();
    fixture.failure(3);
    assert!(ticket.wait().is_err());
    assert!(session.allocate(1).is_err());
    drop((ticket, buffer, session));
    assert_eq!(fixture.counter(0), 1);
    assert_eq!(fixture.counter(1), 1);
    assert_eq!(fixture.counter(4), 1);
    assert_eq!(
        fixture.counter(6),
        0,
        "unfenced context must not finalize ACL"
    );
}

#[test]
fn retirement_during_enqueue_quarantines_backing_even_if_main_fence_succeeds() {
    let fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(8).unwrap();
    // SAFETY: This simulates an unfenced auxiliary stream by retiring the
    // context during submission; no unretained pointers are submitted.
    let result = unsafe {
        session.submit_compute(vec![buffer.clone()], Box::new(()), |_| {
            session.context.retire();
            Err("simulated auxiliary fence failure".into())
        })
    };
    assert!(result.is_err());
    assert!(session.allocate(1).is_err());
    drop((buffer, session));
    assert_eq!(fixture.counter(0), 1);
    assert_eq!(fixture.counter(4), 1);
    assert_eq!(fixture.counter(6), 0);
}

#[test]
fn graph_retains_its_resources_through_replay_and_rolls_back_capture_errors() {
    struct Guard(std::rc::Rc<std::cell::Cell<usize>>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let dropped = std::rc::Rc::new(std::cell::Cell::new(0));
    // SAFETY: Empty capture references only the retained Guard, with no raw data.
    let graph = unsafe {
        super::graph::Graph::capture(&session, Box::new(Guard(dropped.clone())), |_| Ok(()))
    }
    .unwrap();
    let ticket = graph.enqueue(&session).unwrap();
    drop(graph);
    assert_eq!(dropped.get(), 0);
    assert_eq!(fixture.counter(7), 1);
    ticket.wait().unwrap();
    assert_eq!(dropped.get(), 1);
    assert_eq!(fixture.counter(7), 0);
    // SAFETY: No pointer is submitted; failure exercises the capture rollback.
    let result = unsafe {
        super::graph::Graph::capture(&session, Box::new(()), |_| {
            Err("injected capture failure".into())
        })
    };
    assert!(result.is_err());
    assert_eq!(fixture.counter(7), 0);
    drop((ticket, session));
    fixture.assert_clean();
}

#[test]
fn packet_graph_keeps_collective_operands_and_communicator_until_completion() {
    use super::hccl::{Communicator, Element, RootInfo};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let root = RootInfo::fixture(&session, &fixture._dir.path().join("libacl-fixture.so")).unwrap();
    let comm = Communicator::join(&session, &root, 0).unwrap();
    let a = session.allocate(16).unwrap();
    let b = session.allocate(16).unwrap();
    // SAFETY: The graph retains every operand and the communicator used by
    // its captured collective, all belonging to this session.
    let graph = std::rc::Rc::new(unsafe {
        super::graph::Graph::capture(
            &session,
            Box::new((a.clone(), b.clone(), comm.clone())),
            |stream| comm.reduce_on_stream(stream, &a, &b, 4, Element::Float32),
        )
        .unwrap()
    });
    // SAFETY: The packet completion retains the graph on its owning stream.
    let ticket = unsafe {
        session
            .submit_compute(vec![], Box::new(graph.clone()), |stream| {
                graph.execute_on_stream(stream)
            })
            .unwrap()
    };
    drop((graph, a, b, comm));
    assert_eq!(fixture.counter(8), 1);
    assert_eq!(session.allocated_bytes(), 32);
    ticket.wait().unwrap();
    assert_eq!(fixture.counter(8), 0);
    assert_eq!(session.allocated_bytes(), 0);
    drop((ticket, root, session));
    fixture.assert_clean();
}

#[test]
fn collective_completion_retains_communicator_and_validates_byte_units() {
    use super::hccl::{Communicator, Counts, Element, RootInfo};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = Session::with_driver(
        Arc::clone(fixture.driver.as_ref().unwrap()),
        1,
        Limits::default(),
    )
    .unwrap();
    let root = {
        let _current = other.context.enter().unwrap();
        let root =
            RootInfo::fixture(&session, &fixture._dir.path().join("libacl-fixture.so")).unwrap();
        assert_eq!(
            root.as_bytes()[0],
            0,
            "bootstrap selects its own session context"
        );
        root
    };
    drop(other);
    let comm = Communicator::join(&session, &root, 0).unwrap();
    let a = session.allocate(16).unwrap();
    let b = session.allocate(16).unwrap();
    session.upload(&a, 0, &[7; 16]).unwrap().wait().unwrap();
    assert!(
        comm.all_reduce_sum(&session, &a, &b, 5, Element::Float32)
            .is_err()
    );
    let ticket = comm
        .all_reduce_sum(&session, &a, &b, 4, Element::Float32)
        .unwrap();
    drop(comm);
    assert_eq!(fixture.counter(8), 1);
    ticket.wait().unwrap();
    assert_eq!(fixture.counter(8), 0);
    assert_eq!(session.download(&b, 0, 16).unwrap(), [7; 16]);
    let comm = Communicator::join(&session, &root, 0).unwrap();
    comm.all_to_all_v(
        &session,
        &a,
        &b,
        Counts::new(vec![4]).unwrap(),
        Counts::new(vec![4]).unwrap(),
        Element::Float32,
    )
    .unwrap()
    .wait()
    .unwrap();
    assert_eq!(session.download(&b, 0, 16).unwrap(), [7; 16]);
    assert!(
        comm.all_gather(&session, &a, &a, 4, Element::Float32)
            .is_err()
    );
    assert!(
        comm.all_gather(&session, &a, &b, 5, Element::Float32)
            .is_err()
    );
    assert!(
        comm.all_gather(&session, &a, &b, u64::MAX, Element::Float32)
            .is_err()
    );
    session.fill(&b, 0).unwrap().wait().unwrap();
    let gathered = comm
        .all_gather(&session, &a, &b, 4, Element::Float32)
        .unwrap();
    drop(comm);
    assert_eq!(fixture.counter(8), 1);
    gathered.wait().unwrap();
    assert_eq!(fixture.counter(8), 0);
    assert_eq!(session.download(&b, 0, 16).unwrap(), [7; 16]);
    drop((gathered, ticket, a, b, root, session));
    fixture.assert_clean();
}

#[test]
fn captured_exchange_owns_counts_buffers_and_communicator_through_last_fence() {
    use super::hccl::{Communicator, Counts, Element, RootInfo};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let root = RootInfo::fixture(&session, &fixture._dir.path().join("libacl-fixture.so")).unwrap();
    let comm = Communicator::join(&session, &root, 0).unwrap();
    let source = session.allocate(16).unwrap();
    let destination = session.allocate(16).unwrap();
    assert!(
        comm.prepare_exchange(
            &session,
            &source,
            &source,
            Counts::new(vec![4]).unwrap(),
            Counts::new(vec![4]).unwrap(),
            Element::Float32
        )
        .is_err()
    );
    assert!(
        comm.prepare_exchange(
            &session,
            &source,
            &destination,
            Counts::new(vec![5]).unwrap(),
            Counts::new(vec![4]).unwrap(),
            Element::Float32
        )
        .is_err()
    );
    let operation = comm
        .prepare_exchange(
            &session,
            &source,
            &destination,
            Counts::new(vec![4]).unwrap(),
            Counts::new(vec![4]).unwrap(),
            Element::Float32,
        )
        .unwrap();
    // SAFETY: The prepared exchange validates all bindings and this graph owns
    // its count arrays, buffers and communicator until the last replay fence.
    let graph = unsafe {
        super::graph::Graph::capture(&session, Box::new(operation.clone()), |stream| {
            operation.execute_on_stream(stream)
        })
    }
    .unwrap();
    drop((operation, comm, root, source, destination));
    assert_eq!(session.allocated_bytes(), 32);
    assert_eq!(fixture.counter(8), 1);
    let completion = graph.enqueue(&session).unwrap();
    drop(graph);
    assert_eq!(session.allocated_bytes(), 32);
    assert_eq!(fixture.counter(8), 1);
    completion.wait().unwrap();
    assert_eq!(session.allocated_bytes(), 0);
    assert_eq!(fixture.counter(8), 0);
    drop((completion, session));
    fixture.assert_clean();
}

#[test]
fn reduce_scatter_checks_rank_extents_types_and_completion_ownership() {
    use super::hccl::{Communicator, Element, RootInfo};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let root = RootInfo::fixture_ranks(&session, &fixture._dir.path().join("libacl-fixture.so"), 8)
        .unwrap();
    let source = session.allocate(128).unwrap();
    let destination = session.allocate(16).unwrap();
    let short = session.allocate(8).unwrap();
    let comm = Communicator::join(&session, &root, 0).unwrap();
    let input = (0..128u8).collect::<Vec<_>>();
    session.upload(&source, 0, &input).unwrap().wait().unwrap();
    for count in [0, 5, u64::MAX] {
        assert!(
            comm.reduce_scatter_sum(&session, &source, &destination, count, Element::Float32)
                .is_err()
        );
    }
    assert!(
        comm.reduce_scatter_sum(&session, &source, &source, 4, Element::Float32)
            .is_err()
    );
    assert!(
        comm.reduce_scatter_sum(&session, &destination, &source, 4, Element::Float32)
            .is_err()
    );
    assert!(
        comm.reduce_scatter_sum(&session, &source, &short, 4, Element::Float32)
            .is_err()
    );
    for (element, count) in [
        (Element::Float32, 4),
        (Element::Bfloat16, 8),
        (Element::Int8, 16),
    ] {
        session.fill(&destination, 0).unwrap().wait().unwrap();
        comm.reduce_scatter_sum(&session, &source, &destination, count, element)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(session.download(&destination, 0, 16).unwrap(), input[..16]);
    }
    let ticket = comm
        .reduce_scatter_sum(&session, &source, &destination, 4, Element::Int32)
        .unwrap();
    drop((comm, source));
    assert_eq!(fixture.counter(8), 1);
    ticket.wait().unwrap();
    assert_eq!(fixture.counter(8), 0);
    assert_eq!(session.download(&destination, 0, 16).unwrap(), input[..16]);
    drop((ticket, destination, short, root, session));
    fixture.assert_clean();
}

#[test]
fn failed_collective_creation_never_destroys_an_unowned_output_handle() {
    use super::hccl::{Communicator, RootInfo};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let root = RootInfo::fixture(&session, &fixture._dir.path().join("libacl-fixture.so")).unwrap();
    fixture.failure(10);
    assert!(Communicator::join(&session, &root, 0).is_err());
    assert!(session.allocate(1).is_err());
    drop((root, session));
    fixture.assert_clean();
}
