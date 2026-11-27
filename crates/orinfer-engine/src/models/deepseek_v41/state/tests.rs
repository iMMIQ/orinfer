use super::*;
use crate::backend::ascend::tests::Fixture;

#[test]
fn compressed_snapshot_restores_after_crossing_a_physical_page_boundary() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let grain = 2 << 20;
    let source = session.reserve(grain * 2).unwrap();
    let buffers = BTreeMap::from([("kv".into(), source.clone())]);
    let layout = StateLayout {
        buffer: "kv".into(),
        row_bytes: 16,
        tokens_per_row: 2,
        window: None,
        circular: false,
        append_only: true,
    };
    session.grow(&source, 32).unwrap();
    session
        .upload(&source, 0, &[73; 32])
        .unwrap()
        .wait()
        .unwrap();
    let parent = Snapshot::capture(
        &session,
        std::iter::once(&layout),
        &buffers,
        &[9; 4],
        1,
        None,
    )
    .unwrap();
    session.grow(&source, grain + 32).unwrap();
    session
        .upload(&source, grain, &[91; 32])
        .unwrap()
        .wait()
        .unwrap();
    let history = vec![9; (grain / 16 + 2) * 2];
    let child = Snapshot::capture(
        &session,
        std::iter::once(&layout),
        &buffers,
        &history,
        2,
        Some(&parent),
    )
    .unwrap();
    drop(parent);
    let target = session.reserve(grain * 2).unwrap();
    session.grow(&target, grain + 32).unwrap();
    child
        .restore_into(&session, &BTreeMap::from([("kv".into(), target.clone())]))
        .unwrap();
    assert_eq!(session.download(&target, 0, 32).unwrap(), vec![73; 32]);
    assert_eq!(session.download(&target, grain, 32).unwrap(), vec![91; 32]);
    drop(child);
    drop(buffers);
    drop(source);
    drop(target);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn evicting_a_parent_requires_replanning_the_new_bank() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffers = BTreeMap::from([("kv".into(), session.allocate(128).unwrap())]);
    let layout = StateLayout {
        buffer: "kv".into(),
        row_bytes: 16,
        tokens_per_row: 2,
        window: None,
        circular: false,
        append_only: true,
    };
    let parent = Snapshot::capture(
        &session,
        std::iter::once(&layout),
        &buffers,
        &[0, 1, 2, 3],
        1,
        None,
    )
    .unwrap();
    let before = session.allocated_bytes();
    let child = Snapshot::<DeviceBuffer>::plan(
        std::iter::once(&layout),
        &buffers,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        Some(&parent),
    )
    .unwrap();
    assert_eq!(session.allocated_bytes(), before);
    assert_eq!(child.bytes, 32);
    assert!(
        child
            .allocations(2)
            .unwrap()
            .contains_key(&kv::checkpoint_id(1).unwrap())
    );
    drop(child);
    drop(parent);
    let replanned = Snapshot::<DeviceBuffer>::plan(
        std::iter::once(&layout),
        &buffers,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        None,
    )
    .unwrap();
    assert_eq!(replanned.bytes, 64);
    assert_eq!(replanned.allocations(2).unwrap().len(), 1);
    let saved = Snapshot::capture(
        &session,
        std::iter::once(&layout),
        &buffers,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        2,
        None,
    )
    .unwrap();
    assert_eq!(saved.allocations(), replanned.allocations(2).unwrap());
    drop(replanned);
    drop(saved);
    drop(buffers);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn inherited_closed_rows_survive_parent_eviction_and_keep_mutable_state_private() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let layouts = [
        StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: true,
        },
        StateLayout {
            buffer: "swa".into(),
            row_bytes: 4,
            tokens_per_row: 1,
            window: Some(3),
            circular: true,
            append_only: false,
        },
        StateLayout {
            buffer: "ring".into(),
            row_bytes: 64,
            tokens_per_row: 0,
            window: None,
            circular: false,
            append_only: false,
        },
    ];
    let buffers = [("kv", 128), ("swa", 32), ("ring", 64)]
        .into_iter()
        .map(|(name, bytes)| (name.to_owned(), session.allocate(bytes).unwrap()))
        .collect::<BTreeMap<_, _>>();
    let mut parent = None;
    let mut reference = None;
    for (index, length) in [3usize, 5, 9].into_iter().enumerate() {
        let kv = (0..128).map(|i| (i / 16 + 1) as u8).collect::<Vec<_>>();
        let mut swa = vec![0; 32];
        for token in 0..length {
            swa[token % 8 * 4..token % 8 * 4 + 4].fill(token as u8 + 1);
        }
        session
            .upload_inputs(&[
                (buffers["kv"].clone(), kv),
                (buffers["swa"].clone(), swa),
                (buffers["ring"].clone(), vec![length as u8; 64]),
            ])
            .unwrap()
            .wait()
            .unwrap();
        let history = (0..length as u32).collect::<Vec<_>>();
        let before_plan = session.allocated_bytes();
        let planned =
            Snapshot::<DeviceBuffer>::plan(layouts.iter(), &buffers, &history, parent.as_ref())
                .unwrap()
                .allocations(index as u64 + 1)
                .unwrap();
        assert_eq!(
            session.allocated_bytes(),
            before_plan,
            "Planning cannot allocate device memory"
        );
        let saved = Snapshot::capture(
            &session,
            layouts.iter(),
            &buffers,
            &history,
            index as u64 + 1,
            parent.as_ref(),
        )
        .unwrap();
        assert_eq!(saved.allocations(), planned);
        let kv_pieces = saved
            .pieces
            .iter()
            .filter(|p| p.name == "kv")
            .collect::<Vec<_>>();
        assert_eq!(kv_pieces.len(), index + 1);
        assert_eq!(kv_pieces[0].bank_id, kv::checkpoint_id(1).unwrap());
        assert_eq!(kv_pieces.last().unwrap().offset, [0, 16, 32][index]);
        assert!(
            saved
                .pieces
                .iter()
                .filter(|p| p.name != "kv")
                .all(|p| p.bank_id == kv::checkpoint_id(index as u64 + 1).unwrap())
        );
        assert!(
            Snapshot::capture(
                &session,
                layouts.iter(),
                &buffers,
                &[99; 9],
                20,
                Some(&saved)
            )
            .is_err()
        );
        let mut full_layouts = layouts.clone();
        full_layouts[0].append_only = false;
        reference = Some(
            Snapshot::capture(&session, full_layouts.iter(), &buffers, &history, 100, None)
                .unwrap(),
        );
        // Dropping the parent must preserve the child's inherited banks;
        // mutable spans still point exclusively to this capture's bank.
        parent = Some(saved);
    }
    drop(buffers);
    let saved = parent.unwrap();
    assert_eq!(saved.allocations().len(), 3);
    let target = || {
        [("kv", 128), ("swa", 32), ("ring", 64)]
            .into_iter()
            .map(|(name, bytes)| {
                let buffer = session.allocate(bytes).unwrap();
                session.fill(&buffer, 0).unwrap().wait().unwrap();
                (name.to_owned(), buffer)
            })
            .collect::<BTreeMap<_, _>>()
    };
    let actual = target();
    let expected = target();
    saved.restore_into(&session, &actual).unwrap();
    reference
        .as_ref()
        .unwrap()
        .restore_into(&session, &expected)
        .unwrap();
    for name in ["kv", "swa", "ring"] {
        assert_eq!(
            session
                .download(&actual[name], 0, actual[name].bytes())
                .unwrap(),
            session
                .download(&expected[name], 0, expected[name].bytes())
                .unwrap(),
            "{name}"
        );
    }
    assert_eq!(session.download(&actual["kv"], 64, 64).unwrap(), [0; 64]);
    assert_eq!(session.download(&actual["swa"], 4, 20).unwrap(), [0; 20]);
    drop((saved, reference, actual, expected, session));
    fixture.assert_clean();
}

#[test]
fn shared_rows_reject_mutable_geometry() {
    let mut layout = StateLayout {
        buffer: "kv".into(),
        row_bytes: 16,
        tokens_per_row: 2,
        window: None,
        circular: false,
        append_only: true,
    };
    assert_eq!(layout.spans(5, 128).unwrap(), vec![0..32]);
    layout.tokens_per_row = 0;
    assert!(layout.spans(5, 128).is_err());
    layout.tokens_per_row = 1;
    layout.window = Some(3);
    assert!(layout.spans(5, 128).is_err());
    layout.window = None;
    layout.circular = true;
    assert!(layout.spans(5, 128).is_err());
}

#[test]
fn state_spans_preserve_partial_compression_and_wrapped_window() {
    let mut layout = StateLayout {
        buffer: "kv".into(),
        row_bytes: 16,
        tokens_per_row: 2,
        window: None,
        circular: false,
        append_only: false,
    };
    assert_eq!(layout.spans(129, 2048).unwrap(), vec![0..1024]);
    assert!(layout.spans(258, 2048).is_err());
    layout.tokens_per_row = 1;
    layout.window = Some(3);
    layout.circular = true;
    assert_eq!(layout.spans(9, 128).unwrap(), vec![96..128, 0..16]);
    assert_eq!(layout.spans(8, 128).unwrap(), vec![80..128]);
    assert!(layout.spans(8, 0).is_err());
    layout.tokens_per_row = 0;
    assert_eq!(layout.spans(999, 128).unwrap(), vec![0..128]);
}
