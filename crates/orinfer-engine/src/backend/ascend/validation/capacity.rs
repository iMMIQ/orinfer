//! Compare exact and capacity descriptors on the same real W4 banks and pointers.
use super::super::runtime::Buffer;
use super::*;
use serde::Deserialize;
use std::{fs::File, io::Read, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    input: usize,
    output: usize,
    experts: usize,
    files: [PathBuf; 3],
}

fn upload_file(session: &Session, path: &PathBuf, bytes: usize) -> Buffer {
    assert_eq!(std::fs::metadata(path).unwrap().len(), bytes as u64);
    let buffer = session.allocate(bytes).unwrap();
    let mut file = File::open(path).unwrap();
    let mut chunk = vec![0; (4 << 20).min(bytes)];
    let mut offset = 0;
    while offset < bytes {
        let n = chunk.len().min(bytes - offset);
        file.read_exact(&mut chunk[..n]).unwrap();
        session
            .upload(&buffer, offset, &chunk[..n])
            .unwrap()
            .wait()
            .unwrap();
        offset += n;
    }
    buffer
}

fn upload_bytes(session: &Session, buffer: &Buffer, data: &[u8]) {
    for (index, chunk) in data.chunks(4 << 20).enumerate() {
        session
            .upload(buffer, index * (4 << 20), chunk)
            .unwrap()
            .wait()
            .unwrap();
    }
}

#[test]
#[ignore = "Needs the GPU experiment lock, explicit device and real W4 fixture paths"]
fn real_w4_capacity_descriptors_and_live_group_graph() {
    let device = std::env::var("ORINFER_ASCEND_TEST_DEVICE")
        .unwrap()
        .parse()
        .unwrap();
    let fixtures: Vec<Fixture> = serde_json::from_slice(
        &std::fs::read(std::env::var_os("ORINFER_A2_W4_CAPACITY_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(fixtures.len(), 2);
    let session = Session::new(
        device,
        Limits {
            device_bytes: 4 << 30,
            ..Limits::default()
        },
    )
    .unwrap();
    let nz_enabled = std::env::var_os("ORINFER_A2_W4_CAPACITY_NZ").is_some();
    let workspace = session
        .allocate(if nz_enabled { 320 << 20 } else { 256 << 20 })
        .unwrap();
    let small_workspace = session.allocate(32 << 20).unwrap();
    let hints: Vec<Option<usize>> = serde_json::from_str(
        &std::env::var("ORINFER_A2_W4_CAPACITY_HINTS").unwrap_or_else(|_| "[null]".into()),
    )
    .unwrap();
    assert!(!hints.is_empty());
    let distributions: Vec<usize> = serde_json::from_str(
        &std::env::var("ORINFER_A2_W4_CAPACITY_DISTRIBUTIONS").unwrap_or_else(|_| "[48]".into()),
    )
    .unwrap();
    assert!(distributions.iter().all(|&n| n > 0 && n <= 48));
    let mut reports = vec![];
    for fixture in fixtures {
        let Fixture {
            input: k,
            output: n,
            experts: e,
            files,
        } = fixture;
        assert_eq!(e, 48);
        let weights = upload_file(&session, &files[0], e * k * n / 2);
        let nz_dir = std::env::var_os("ORINFER_A2_W4_CAPACITY_NZ").map(PathBuf::from);
        let candidate = match &nz_dir {
            Some(dir) => upload_file(
                &session,
                &dir.join(format!("w4-{k}-{n}.bin")),
                e * k * n / 2,
            ),
            None => weights.clone(),
        };
        let scales = upload_file(&session, &files[1], e * n * 8);
        let assist = upload_file(&session, &files[2], e * n * 4);
        let maximum = 6 * 2048;
        let activation = session.allocate(maximum * k).unwrap();
        let token_scale = session.allocate(maximum * 4).unwrap();
        let output = session.allocate(maximum * n * 2 + 256).unwrap();
        let groups = session.allocate(e * 8).unwrap();
        let token_scales: Vec<u8> = (0..maximum)
            .flat_map(|i| [0.013f32, 0.039, 0.071][i % 3].to_le_bytes())
            .collect();
        session
            .upload(&token_scale, 0, &token_scales)
            .unwrap()
            .wait()
            .unwrap();
        let patterns: Vec<Vec<u8>> = (0..17)
            .map(|p| {
                (0..k)
                    .map(|i| {
                        if i % 19 < 7 {
                            [-128i8, -127, -1, 0, 1, 126, 127][i % 19] as u8
                        } else {
                            (((i * 3 + p * 7 + 20261002) % 251) as i16 - 125) as i8 as u8
                        }
                    })
                    .collect()
            })
            .collect();
        let mut source = vec![0; maximum * k];
        let prepare = |rows, hint, nz| {
            let a = activation.slice(0, rows * k).unwrap();
            let ts = token_scale.slice(0, rows * 4).unwrap();
            let y = output.slice(0, rows * n * 2).unwrap();
            let mut ends = vec![0; e];
            ends[e - 1] = rows as i64;
            GroupedW4A8::prepare_with_config(
                &session,
                GroupedShape {
                    rows,
                    input: k,
                    output: n,
                    experts: e,
                },
                &ends,
                GroupedInputs {
                    activation: &a,
                    weight: if nz { &candidate } else { &weights },
                    weight_scale: &scales,
                    assist: &assist,
                    token_scale: &ts,
                    output: &y,
                },
                Some(if nz && nz_enabled && rows <= 192 {
                    &small_workspace
                } else {
                    &workspace
                }),
                Some(&groups),
                super::super::operators::GroupedConfig {
                    expected_rows: hint,
                    weight_layout: if nz && nz_dir.is_some() {
                        super::super::operators::W4Layout::TransposedNz
                    } else {
                        super::super::operators::W4Layout::Nd
                    },
                },
            )
            .unwrap()
        };
        session.fill(&activation, 0).unwrap().wait().unwrap();
        for &hint in &hints {
            for (capacity, tokens) in [(32, vec![1, 16, 32]), (2048, vec![128, 800, 2048])] {
                let full_rows = capacity * 6;
                let start = Instant::now();
                let full = prepare(full_rows, hint, true);
                let full_prepare_ms = start.elapsed().as_secs_f64() * 1000.;
                let full_graph = full.capture(&session).unwrap();
                for tokens in tokens {
                    let rows = tokens * 6;
                    let start = Instant::now();
                    let exact = prepare(
                        if nz_dir.is_some() { full_rows } else { rows },
                        if nz_dir.is_some() { hint } else { None },
                        false,
                    );
                    let exact_prepare_ms = start.elapsed().as_secs_f64() * 1000.;
                    let exact_graph = exact.capture(&session).unwrap();
                    let live = rows.div_ceil(8);
                    let mut live_counts = Vec::new();
                    for pass in 0..if nz_enabled && tokens == 2048 { 4 } else { 3 } {
                        let active = match pass {
                            0 => live,
                            1 => live / 2,
                            3 => rows,
                            _ => 0,
                        };
                        let mut counts = vec![0i64; e];
                        for i in 0..active {
                            counts[if pass == 1 {
                                7
                            } else if pass == 3 {
                                i % 6
                            } else {
                                i % e
                            }] += 1;
                        }
                        let data: Vec<u8> = counts.iter().flat_map(|v| v.to_le_bytes()).collect();
                        session.upload(&groups, 0, &data).unwrap().wait().unwrap();
                        for (row, chunk) in source.chunks_exact_mut(k).enumerate() {
                            chunk.copy_from_slice(&patterns[(row + pass * 3) % patterns.len()]);
                        }
                        let mut reference = None;
                        for graph in [&exact_graph, &full_graph] {
                            upload_bytes(&session, &activation, &source[..full_rows * k]);
                            session.fill(&output, 0xa5).unwrap().wait().unwrap();
                            graph.enqueue(&session).unwrap().wait().unwrap();
                            let actual = if active == 0 {
                                vec![]
                            } else {
                                let mut data = Vec::with_capacity(active * n * 2);
                                for offset in (0..active * n * 2).step_by(4 << 20) {
                                    data.extend(
                                        session
                                            .download(
                                                &output,
                                                offset,
                                                (4 << 20).min(active * n * 2 - offset),
                                            )
                                            .unwrap(),
                                    );
                                }
                                assert!(
                                    data.iter().any(|&v| v != 0xa5),
                                    "Graph left output untouched"
                                );
                                data
                            };
                            if let Some(expected) = &reference {
                                if actual != *expected {
                                    let expected: &Vec<u8> = expected;
                                    let a: Vec<f32> = expected
                                        .as_chunks::<2>()
                                        .0
                                        .iter()
                                        .map(|b| {
                                            half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32()
                                        })
                                        .collect();
                                    let b: Vec<f32> = actual
                                        .as_chunks::<2>()
                                        .0
                                        .iter()
                                        .map(|b| {
                                            half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32()
                                        })
                                        .collect();
                                    let max_error = a
                                        .iter()
                                        .zip(&b)
                                        .map(|(a, b)| (a - b).abs())
                                        .fold(0f32, f32::max);
                                    let maximum = a.iter().map(|a| a.abs()).fold(0f32, f32::max);
                                    let changed = a.iter().zip(&b).filter(|(a, b)| a != b).count();
                                    let report = std::path::PathBuf::from(
                                        std::env::var_os("ORINFER_A2_W4_CAPACITY_REPORT").unwrap(),
                                    );
                                    std::fs::write(
                                        report.with_extension("reference.bin"),
                                        expected,
                                    )
                                    .unwrap();
                                    std::fs::write(report.with_extension("actual.bin"), &actual)
                                        .unwrap();
                                    panic!(
                                        "Capacity changed output M={rows} cap={full_rows} K={k} N={n} pass={pass} changed={changed}/{} maxerror={max_error} maximum={maximum} first={:?}/{:?}",
                                        a.len(),
                                        &a[..a.len().min(8)],
                                        &b[..b.len().min(8)]
                                    );
                                }
                            } else {
                                reference = Some(actual);
                            }
                            assert!(
                                session
                                    .download(&output, full_rows * n * 2, 256)
                                    .unwrap()
                                    .iter()
                                    .all(|&v| v == 0xa5),
                                "Capacity output overrun"
                            );
                        }
                        live_counts.push(active);
                    }
                    for &used_experts in &distributions {
                        let mut counts = vec![0i64; e];
                        for i in 0..live {
                            counts[i % used_experts] += 1;
                        }
                        let data: Vec<u8> = counts.iter().flat_map(|v| v.to_le_bytes()).collect();
                        session.upload(&groups, 0, &data).unwrap().wait().unwrap();
                        let mut timings = [vec![], vec![]];
                        for repeat in 0..7 {
                            for index in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                                upload_bytes(&session, &activation, &source[..full_rows * k]);
                                let start = Instant::now();
                                [&exact_graph, &full_graph][index]
                                    .enqueue(&session)
                                    .unwrap()
                                    .wait()
                                    .unwrap();
                                timings[index].push(start.elapsed().as_secs_f64() * 1000.);
                            }
                        }
                        for values in &mut timings {
                            values.sort_by(f64::total_cmp);
                        }
                        let report = serde_json::json!({"tokens": tokens, "used_experts": used_experts, "transposed_nz": nz_dir.is_some(), "expected_expert_rows": hint, "capacity_tokens": capacity, "live_counts": live_counts, "input": k, "output": n, "exact_prepare_ms": exact_prepare_ms, "capacity_prepare_ms": full_prepare_ms, "capacity_workspace_bytes": full.workspace_bytes(), "exact_graph_ms": timings[0][3], "capacity_graph_ms": timings[1][3], "change_pct": 100. * (timings[1][3] / timings[0][3] - 1.), "changed_input_and_groups_exact": true});
                        println!("{report}");
                        reports.push(report);
                    }
                }
            }
        }
    }
    assert_eq!(reports.len(), 12 * hints.len() * distributions.len());
    std::fs::write(
        std::env::var_os("ORINFER_A2_W4_CAPACITY_REPORT").unwrap(),
        serde_json::to_vec_pretty(&reports).unwrap(),
    )
    .unwrap();
}
