//! Pack disjoint input views into contiguous DMAs without reordering overlapping writes.
use super::*;

pub(super) fn ordered<'a>(inputs: &[(Buffer, usize, &'a [u8])]) -> Vec<(Buffer, usize, &'a [u8])> {
    let mut ordered = inputs.to_vec();
    ordered.retain(|(_, _, data)| !data.is_empty());
    ordered.sort_by_key(|(buffer, offset, _)| {
        (
            Rc::as_ptr(&buffer.allocation) as usize,
            buffer.offset + offset,
        )
    });
    if ordered.windows(2).any(|pair| {
        let (a, offset, data) = &pair[0];
        let (b, next, _) = &pair[1];
        Rc::ptr_eq(&a.allocation, &b.allocation) && a.offset + offset + data.len() > b.offset + next
    }) {
        // Some callers deliberately overwrite earlier inputs in one submission.
        return inputs.to_vec();
    }
    ordered
}

pub(super) fn append(
    pieces: &mut Vec<(Buffer, usize, usize, usize)>,
    buffer: &Buffer,
    offset: usize,
    source: usize,
    bytes: usize,
) {
    if bytes == 0 {
        return;
    }
    let offset = buffer.offset + offset;
    if let Some((last, _, start, count)) = pieces.last_mut()
        && Rc::ptr_eq(&last.allocation, &buffer.allocation)
        && last.offset + last.bytes == offset
        && *start + *count == source
    {
        // Each appended extent was validated against its original bounded view.
        last.bytes += bytes;
        *count += bytes;
        return;
    }
    pieces.push((
        Buffer {
            allocation: buffer.allocation.clone(),
            offset,
            bytes,
        },
        0,
        source,
        bytes,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ascend::tests::Fixture;

    #[test]
    fn disjoint_views_merge_without_crossing_gaps_or_allocations() {
        let mut fixture = Fixture::new();
        let session = fixture.session_with_limits(Limits {
            device_bytes: 64,
            pinned_bytes: 8,
            ..Default::default()
        });
        let a = session.allocate(12).unwrap();
        let b = session.allocate(4).unwrap();
        let mut data = vec![
            (a.slice(4, 4).unwrap(), vec![2; 4]),
            (b.clone(), vec![9; 4]),
            (a.slice(0, 4).unwrap(), vec![1; 4]),
            (a.slice(10, 2).unwrap(), vec![3; 2]),
        ];
        let regions = data
            .iter()
            .map(|(b, d)| (b.clone(), 0, d.as_slice()))
            .collect::<Vec<_>>();
        let ordered = ordered(&regions);
        let mut pieces = vec![];
        let mut start = 0;
        for (b, offset, d) in &ordered {
            append(&mut pieces, b, *offset, start, d.len());
            start += d.len();
        }
        assert_eq!(pieces.len(), 3);
        assert!(
            pieces
                .iter()
                .any(|(b, _, _, bytes)| Rc::ptr_eq(&b.allocation, &a.allocation)
                    && b.offset == 0
                    && *bytes == 8)
        );
        let uploaded = session.upload_inputs(&data).unwrap();
        drop((regions, ordered, pieces));
        for (_, d) in &mut data {
            d.fill(0);
        }
        drop(data);
        uploaded.wait().unwrap();
        assert_eq!(
            session.download(&a, 0, 8).unwrap(),
            [1, 1, 1, 1, 2, 2, 2, 2]
        );
        assert_eq!(session.download(&a, 10, 2).unwrap(), [3, 3]);
        assert_eq!(session.download(&b, 0, 4).unwrap(), [9; 4]);
        assert!(session.pinned_bytes() <= 8);
        drop((uploaded, a, b, session));
        fixture.assert_clean();
    }

    #[test]
    fn overlapping_inputs_keep_caller_order_and_prior_compute_readers() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let a = session.allocate(8).unwrap();
        let copy = session.allocate(8).unwrap();
        let inputs = vec![
            (a.slice(4, 4).unwrap(), vec![1; 4]),
            (a.slice(0, 6).unwrap(), vec![2; 6]),
            (a.slice(2, 2).unwrap(), vec![3; 2]),
        ];
        session.upload_inputs(&inputs).unwrap();
        let read = session.copy(&a, 0, &copy, 0, 8).unwrap();
        session.upload_inputs(&[(a.clone(), vec![4; 8])]).unwrap();
        assert_eq!(
            session.download(&copy, 0, 8).unwrap(),
            [2, 2, 3, 3, 2, 2, 1, 1]
        );
        assert_eq!(session.download(&a, 0, 8).unwrap(), [4; 8]);
        drop((inputs, read, a, copy, session));
        fixture.assert_clean();
    }

    #[test]
    #[ignore = "Real A2 input DMA cost; explicit ORINFER_INPUT_REPORT and hardware lock required"]
    fn owner_input_packet_cost_on_device() {
        use std::{fs::OpenOptions, os::fd::AsRawFd, path::PathBuf, time::Instant};
        let output = PathBuf::from(std::env::var_os("ORINFER_INPUT_REPORT").unwrap());
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("artifacts/gpu-experiment.lock")
            .unwrap();
        assert_eq!(
            // SAFETY: The retained file serializes hardware until session destruction.
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let session = Session::new(
            0,
            Limits {
                device_bytes: 64 << 20,
                ..Default::default()
            },
        )
        .unwrap();
        let mut reports = vec![];
        for mixed in [false, true] {
            let mut lengths = [6; 16];
            if mixed {
                lengths[15] = 730;
            }
            let rows = lengths.iter().sum::<usize>();
            let strides = [128, 128, 128, 128, 128, 128, 4, 4, 512, 512, 32, 32, 4, 4];
            let buffers = strides
                .iter()
                .map(|stride| session.allocate(rows * stride).unwrap())
                .collect::<Vec<_>>();
            let mut start = 0;
            let owners = lengths
                .iter()
                .enumerate()
                .map(|(owner, length)| {
                    let inputs = buffers
                        .iter()
                        .zip(strides)
                        .map(|(b, stride)| {
                            (
                                b.slice(start * stride, length * stride).unwrap(),
                                vec![owner as u8 + 1; length * stride],
                            )
                        })
                        .collect::<Vec<_>>();
                    start += length;
                    inputs
                })
                .collect::<Vec<_>>();
            let combined = owners.iter().flatten().cloned().collect::<Vec<_>>();
            let mut samples = [vec![], vec![]];
            for repeat in 0..11 {
                for step in 0..2 {
                    let mode = (repeat + step) % 2;
                    let at = Instant::now();
                    if mode == 0 {
                        let mut last = None;
                        for input in &owners {
                            last = Some(session.upload_inputs(input).unwrap());
                        }
                        last.unwrap().wait().unwrap();
                    } else {
                        session.upload_inputs(&combined).unwrap().wait().unwrap();
                    }
                    if repeat >= 2 {
                        samples[mode].push(at.elapsed().as_secs_f64());
                    }
                }
            }
            for (buffer, stride) in buffers.iter().zip(strides) {
                let expected = lengths
                    .iter()
                    .enumerate()
                    .flat_map(|(owner, n)| std::iter::repeat_n(owner as u8 + 1, n * stride))
                    .collect::<Vec<_>>();
                assert_eq!(
                    session.download(buffer, 0, buffer.bytes()).unwrap(),
                    expected
                );
            }
            for values in &mut samples {
                values.sort_by(f64::total_cmp);
            }
            reports.push(serde_json::json!({"mixed":mixed,"owners":16,"rows":rows,"bytes":combined.iter().map(|(_, d)|d.len()).sum::<usize>(),"per_owner_median_s":samples[0][samples[0].len()/2],"packet_median_s":samples[1][samples[1].len()/2],"samples_s":samples}));
        }
        std::fs::write(output, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
        eprintln!("INPUT_PACKET {}", serde_json::to_string(&reports).unwrap());
        assert_eq!(session.allocated_bytes(), 0);
    }
}
