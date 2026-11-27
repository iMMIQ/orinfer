use super::*;

#[test]
fn padded_history_ring_preserves_every_accepted_prefix_without_a_journal() {
    let layout = StateLayout {
        buffer: "swa".into(),
        row_bytes: 4,
        tokens_per_row: 1,
        window: Some(128),
        circular: true,
        append_only: false,
    };
    for start in [0usize, 127, 128, 255, 256, 257, 2047, 262111] {
        for count in [1usize, 6, 17, 32] {
            assert!(
                layout
                    .overwritten_spans(start, count, 256 * 4)
                    .unwrap()
                    .is_empty()
            );
            let mut ring = vec![usize::MAX; 256];
            for position in start.saturating_sub(128)..start + count {
                ring[position % 256] = position;
            }
            for keep in 1..=count {
                let end = start + keep;
                for position in end.saturating_sub(128)..end {
                    assert_eq!(ring[position % 256], position);
                }
                assert_eq!(
                    layout
                        .spans(end, 256 * 4)
                        .unwrap()
                        .iter()
                        .map(|s| s.len())
                        .sum::<usize>(),
                    end.min(128) * 4
                );
            }
        }
    }
}

#[test]
fn verification_journal_covers_only_overwritten_slots_across_wraps() {
    let layout = StateLayout {
        buffer: "swa".into(),
        row_bytes: 16,
        tokens_per_row: 1,
        window: Some(128),
        circular: true,
        append_only: false,
    };
    for count in [1, 6, 9, 17, 32] {
        for start in [0, 1, 122, 123, 126, 127, 128, 2047, 262138] {
            let spans = layout.overwritten_spans(start, count, 128 * 16).unwrap();
            assert_eq!(spans.iter().map(|s| s.len()).sum::<usize>(), count * 16);
            let rows = spans
                .into_iter()
                .flat_map(|s| (s.start / 16)..(s.end / 16))
                .collect::<Vec<_>>();
            assert_eq!(
                rows,
                (start..start + count).map(|p| p % 128).collect::<Vec<_>>()
            );
        }
    }
    let mut linear = layout.clone();
    linear.circular = false;
    assert!(
        linear
            .overwritten_spans(120, 6, 128 * 16)
            .unwrap()
            .is_empty()
    );
    assert!(
        layout
            .overwritten_spans(2047, 6, 4096 * 16)
            .unwrap()
            .is_empty()
    );
    let c2 = AppendWindow {
        offset: 32 * 4096,
        row_bytes: 4096,
        rows: 32,
        history: 1,
    };
    for start in [0, 31, 32, 2047, 2048, 2049] {
        for count in [6, 9, 17, 31] {
            assert!(
                c2.overwritten_spans(start, count, 64 * 4096)
                    .unwrap()
                    .is_empty()
            );
        }
        let spans = c2.overwritten_spans(start, 32, 64 * 4096).unwrap();
        assert_eq!(spans.iter().map(|s| s.len()).sum::<usize>(), 32 * 4096);
        assert!(c2.overwritten_spans(start, 33, 64 * 4096).is_err());
    }
    let recurrent = StateLayout {
        tokens_per_row: 0,
        window: None,
        circular: false,
        ..layout
    };
    assert_eq!(
        recurrent.overwritten_spans(2049, 6, 4096).unwrap(),
        vec![0..4096]
    );
}
