//! Complete TP8 requests with shared rows and owner-private causal state.
use super::{
    batch::Layout,
    load_spec,
    team::{Command, Team},
};
use crate::artifact::Result;
use orinfer_model_sdk::architecture::BatchSegment;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inputs {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    #[serde(default)]
    frames: Vec<Frame>,
    #[serde(default)]
    requests: Vec<Request>,
    chunk_tokens: Option<usize>,
    #[serde(default)]
    decode_steps: usize,
    #[serde(default)]
    compare_serial: bool,
    #[serde(default)]
    compare_whole_prompt: bool,
    #[serde(default = "one")]
    repeats: usize,
    /// Collect only during the untimed preparation pass, after each forward.
    #[serde(default)]
    record_routing: bool,
}
fn one() -> usize {
    1
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    #[serde(default)]
    decode: bool,
    requests: Vec<Request>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    owner: usize,
    tokens: Vec<u32>,
    #[serde(default)]
    output: bool,
}
#[derive(Clone, Serialize)]
struct Output {
    owner: usize,
    token: u32,
    #[serde(skip)]
    logits: Vec<f32>,
}
#[derive(Serialize)]
pub struct Report {
    layers: usize,
    tensor_parallel: usize,
    owner_capacity: usize,
    context_capacity: usize,
    load_s: f64,
    requests: Vec<Measurement>,
    preparations: Vec<Preparation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    routing: Vec<RoutingSample>,
    scope: &'static str,
}
#[derive(Serialize)]
struct RoutingSample {
    name: String,
    frame: usize,
    decode: bool,
    rows: usize,
    ranks: Vec<Vec<super::team::RoutingGroup>>,
}
#[derive(Serialize)]
struct Measurement {
    name: String,
    repeat: usize,
    batch: usize,
    input_tokens: usize,
    prefill_s: f64,
    prefill_tokens_per_s: f64,
    decode_s: f64,
    decode_tokens_per_s: f64,
    preparation_s: f64,
    reset_s: f64,
    request_s: f64,
    output_tokens: Vec<Vec<Output>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    serial: Option<Comparison>,
}
#[derive(Serialize)]
struct Preparation {
    name: String,
    owner_admission_s: f64,
    reset_s: f64,
    graph_preparation_s: f64,
    first_prefill_s: f64,
    first_decode_s: f64,
}
#[derive(Default)]
struct Timings {
    prefill_s: f64,
    decode_s: f64,
    input_tokens: usize,
    decode_tokens: usize,
}
impl Timings {
    fn record(&mut self, frame: &Frame, outputs: usize, seconds: f64) {
        if frame.decode {
            self.decode_s += seconds;
            self.decode_tokens += outputs;
        } else {
            self.prefill_s += seconds;
            self.input_tokens += frame.requests.iter().map(|r| r.tokens.len()).sum::<usize>();
        }
    }
}
#[derive(Serialize)]
struct Comparison {
    positions: usize,
    greedy_equal: bool,
    max_abs_logit: f32,
    max_probability_error: f64,
}

fn frame_layout(frame: &Frame, capacity: usize, owners: usize) -> Result<Layout> {
    Layout::new(
        frame
            .requests
            .iter()
            .map(|r| BatchSegment {
                slot: r.owner,
                tokens: r.tokens.len(),
            })
            .collect(),
        frame
            .requests
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.output.then_some(i))
            .collect(),
        frame.decode,
        capacity,
        owners,
    )
}
fn forward(
    team: &Team,
    frame: &Frame,
    cap: usize,
    owners: usize,
    full: bool,
) -> Result<(Vec<Output>, f64, f64)> {
    let (values, seconds, preparation) = team.forward(
        frame_layout(frame, cap, owners)?,
        frame.requests.iter().map(|r| r.tokens.clone()).collect(),
        full,
    )?;
    Ok((
        values
            .into_iter()
            .map(|value| Output {
                owner: value.owner,
                token: value.token,
                logits: value.logits,
            })
            .collect(),
        seconds,
        preparation,
    ))
}
fn plan_frames(requests: &[Request], budget: usize, preferred: usize) -> Result<Vec<Frame>> {
    if requests.is_empty()
        || budget < requests.len()
        || requests.iter().any(|r| r.tokens.is_empty())
        || requests
            .iter()
            .map(|r| r.owner)
            .collect::<BTreeSet<_>>()
            .len()
            != requests.len()
    {
        return Err("Invalid batch requests/chunk budget".into());
    }
    let mut positions = vec![0; requests.len()];
    let mut frames = vec![];
    let mut cursor = 0;
    while positions
        .iter()
        .zip(requests)
        .any(|(&p, r)| p < r.tokens.len())
    {
        if preferred > 1 {
            let mut remaining = budget;
            let mut selected = vec![];
            for step in 0..requests.len() {
                let i = (cursor + step) % requests.len();
                let left = requests[i].tokens.len() - positions[i];
                if left == 0 {
                    continue;
                }
                let count = left.min(preferred).min(budget);
                if count <= remaining {
                    selected.push((i, count));
                    remaining -= count;
                }
            }
            while remaining > 0 {
                let mut active = selected
                    .iter()
                    .filter(|(i, n)| positions[*i] + *n < requests[*i].tokens.len())
                    .count();
                if active == 0 {
                    break;
                }
                for (i, n) in &mut selected {
                    let left = requests[*i].tokens.len() - positions[*i] - *n;
                    if left == 0 {
                        continue;
                    }
                    let count = left.min(remaining.div_ceil(active));
                    *n += count;
                    remaining -= count;
                    active -= 1;
                }
            }
            cursor = (selected.last().ok_or("Empty prefill selection")?.0 + 1) % requests.len();
            let mut rows = vec![];
            for (i, count) in selected {
                let r = &requests[i];
                let end = positions[i] + count;
                rows.push(Request {
                    owner: r.owner,
                    tokens: r.tokens[positions[i]..end].to_vec(),
                    output: end == r.tokens.len(),
                });
                positions[i] = end;
            }
            frames.push(Frame {
                decode: false,
                requests: rows,
            });
            continue;
        }
        let mut active = positions
            .iter()
            .zip(requests)
            .filter(|(p, r)| **p < r.tokens.len())
            .count();
        let mut remaining = budget;
        let mut rows = vec![];
        for (i, r) in requests.iter().enumerate() {
            if positions[i] == r.tokens.len() {
                continue;
            }
            let count = (remaining / active).min(r.tokens.len() - positions[i]);
            let end = positions[i] + count;
            rows.push(Request {
                owner: r.owner,
                tokens: r.tokens[positions[i]..end].to_vec(),
                output: end == r.tokens.len(),
            });
            positions[i] = end;
            remaining -= count;
            active -= 1;
        }
        frames.push(Frame {
            decode: false,
            requests: rows,
        });
    }
    Ok(frames)
}

fn plan_prompt_frames(requests: &[Request], budget: usize, keep: usize) -> Result<Vec<Frame>> {
    // Validate owner identities and token inputs with the common planner.
    if requests.is_empty()
        || budget == 0
        || requests.iter().any(|r| r.tokens.is_empty())
        || requests
            .iter()
            .map(|r| r.owner)
            .collect::<BTreeSet<_>>()
            .len()
            != requests.len()
    {
        return Err("Invalid prompt replay requests/chunk budget".into());
    }
    if budget < keep && requests.iter().any(|r| r.tokens.len() > budget) {
        return Err("Prompt replay chunk budget must hold its final window".into());
    }
    let mut positions = vec![0; requests.len()];
    let mut frames = vec![];
    let mut cursor = 0;
    while positions
        .iter()
        .zip(requests)
        .any(|(&p, r)| p < r.tokens.len())
    {
        let mut remaining = budget;
        let mut rows = vec![];
        let mut next = cursor;
        for step in 0..requests.len() {
            let i = (cursor + step) % requests.len();
            let left = requests[i].tokens.len() - positions[i];
            if left == 0 || remaining == 0 {
                continue;
            }
            let Some(count) = super::replay::prompt_rows(
                left,
                remaining,
                keep,
                requests[i].tokens.len() > budget,
            ) else {
                continue;
            };
            let end = positions[i] + count;
            rows.push(Request {
                owner: requests[i].owner,
                tokens: requests[i].tokens[positions[i]..end].to_vec(),
                output: end == requests[i].tokens.len(),
            });
            positions[i] = end;
            remaining -= count;
            next = (i + 1) % requests.len();
        }
        if rows.is_empty() {
            return Err("Prompt replay planner made no progress".into());
        }
        frames.push(Frame {
            decode: false,
            requests: rows,
        });
        cursor = next;
    }
    Ok(frames)
}

fn compare(batch: &[Vec<Output>], serial: &[Vec<Output>]) -> Result<Comparison> {
    let mut result = Comparison {
        positions: 0,
        greedy_equal: true,
        max_abs_logit: 0.,
        max_probability_error: 0.,
    };
    for (frame, (expected, actual)) in batch.iter().zip(serial).enumerate() {
        for b in expected {
            let s = actual
                .iter()
                .find(|v| v.owner == b.owner)
                .ok_or("Serial output missing")?;
            if b.logits.len() != 129280 || s.logits.len() != b.logits.len() {
                return Err("Missing complete validation logits".into());
            }
            if b.logits.iter().chain(&s.logits).any(|x| !x.is_finite()) {
                return Err("Nonfinite complete validation logits".into());
            }
            result.positions += 1;
            result.greedy_equal &= s.token == b.token;
            let probability = |x: &[f32]| {
                let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e = x.iter().map(|v| f64::from(v - m).exp()).collect::<Vec<_>>();
                let sum = e.iter().sum::<f64>();
                e.into_iter().map(|v| v / sum).collect::<Vec<_>>()
            };
            let bp = probability(&b.logits);
            let sp = probability(&s.logits);
            let mut max_abs = 0f32;
            let mut max_probability = 0f64;
            for ((&a, &b), (&pa, &pb)) in b.logits.iter().zip(&s.logits).zip(bp.iter().zip(&sp)) {
                max_abs = max_abs.max((a - b).abs());
                max_probability = max_probability.max((pa - pb).abs());
                result.max_abs_logit = result.max_abs_logit.max((a - b).abs());
                result.max_probability_error = result.max_probability_error.max((pa - pb).abs());
            }
            eprintln!(
                "batch check frame {frame} owner {} tokens {}/{} max_logit {max_abs} max_probability {max_probability}",
                b.owner, b.token, s.token
            );
        }
    }
    Ok(result)
}

pub fn run(plan: &Path, model_dir: &Path, inputs: &Path) -> Result<Report> {
    let (model, _, _) = load_spec(plan, model_dir)?;
    let contract = model
        .batch
        .as_ref()
        .ok_or("bench-ascend-batch requires a native batch plan")?;
    let owners = contract.owners;
    let preferred = contract.preferred_span();
    let capacity = model.ranks[0]
        .packets
        .iter()
        .filter(|p| !p.batch_decode)
        .map(|p| p.rows)
        .next()
        .ok_or("Missing prefill family")?;
    let mut inputs: Inputs = crate::model::read(inputs)?;
    if inputs.cases.is_empty() {
        return Err("Empty batch workload".into());
    }
    for case in &mut inputs.cases {
        if !case.requests.is_empty() {
            if !case.frames.is_empty() {
                return Err("Supply requests or explicit frames".into());
            }
            let budget = case.chunk_tokens.unwrap_or(capacity).min(capacity);
            case.frames = if let Some(replay) = &contract.decoder_replay
                && replay.final_prompt_only
            {
                plan_prompt_frames(&case.requests, budget, replay.keep)?
            } else {
                plan_frames(&case.requests, budget, preferred)?
            };
        }
        if case.frames.is_empty() || case.repeats == 0 {
            return Err("Empty batch case/repeats".into());
        }
        let mut lengths = std::collections::BTreeMap::<usize, usize>::new();
        for frame in &case.frames {
            frame_layout(frame, if frame.decode { owners } else { capacity }, owners)?;
            for r in &frame.requests {
                if r.tokens.iter().any(|&t| t >= 129280) {
                    return Err("Invalid token ID".into());
                }
                let count = lengths.entry(r.owner).or_default();
                *count = count
                    .checked_add(r.tokens.len())
                    .ok_or("Context overflow")?;
                if count
                    .checked_add(case.decode_steps)
                    .is_none_or(|n| n > model.capacity)
                {
                    return Err("Owner exceeds context capacity".into());
                }
            }
        }
        if lengths.len() > owners {
            return Err("Case exceeds resident owner capacity".into());
        }
        let mut final_outputs = std::collections::BTreeMap::new();
        for frame in &case.frames {
            for r in &frame.requests {
                final_outputs.insert(r.owner, r.output);
            }
        }
        if case.decode_steps > 0 && final_outputs.values().any(|&output| !output) {
            return Err("Automatic decode requires each owner's final prefill output".into());
        }
    }
    let team = Team::load(plan, model_dir)?;
    let logits_directory =
        std::env::var_os("ORINFER_A2_BATCH_LOGITS_DIR").map(std::path::PathBuf::from);
    let load_s = team.load_s;
    eprintln!("A2 shared batch model loaded in {load_s:.3}s");
    (|| {
        let mut measurements = vec![];
        let mut preparations = vec![];
        let mut routing = vec![];
        for case in &inputs.cases {
            let ids = case
                .frames
                .iter()
                .flat_map(|f| f.requests.iter().map(|r| r.owner))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let admission = Instant::now();
            team.exchange(Command::Release(None))?;
            team.exchange(Command::Admit(ids.clone()))?;
            let owner_admission_s = admission.elapsed().as_secs_f64();
            let mut validation = None;
            for repeat in 0..=case.repeats {
                let reset = Instant::now();
                team.exchange(Command::Reset(ids.clone()))?;
                let reset_s = reset.elapsed().as_secs_f64();
                let mut timings = Timings::default();
                let mut preparation_s = 0.;
                let mut outputs = vec![];
                let mut frames = case.frames.clone();
                let mut last_outputs = std::collections::BTreeMap::<usize, Output>::new();
                for index in 0..case.frames.len() + case.decode_steps {
                    let f = &frames[index];
                    let (out, elapsed, prepare) = forward(
                        &team,
                        f,
                        if f.decode { owners } else { capacity },
                        owners,
                        (case.compare_serial || logits_directory.is_some()) && repeat == 0,
                    )?;
                    if repeat == 0
                        && let Some(directory) = logits_directory.as_deref()
                    {
                        let directory = directory.join(&case.name);
                        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
                        for value in &out {
                            let bytes = value
                                .logits
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>();
                            std::fs::write(
                                directory.join(format!("frame{index}-owner{}.f32", value.owner)),
                                bytes,
                            )
                            .map_err(|e| e.to_string())?;
                        }
                    }
                    preparation_s += prepare;
                    timings.record(f, out.len(), elapsed);
                    if repeat == 0 && case.record_routing {
                        let ranks = team
                            .exchange(Command::RoutingGroups(f.decode))?
                            .into_iter()
                            .map(|reply| match reply {
                                super::team::Reply::RoutingGroups(groups) => Ok(groups),
                                _ => Err("Invalid routing diagnostic reply".into()),
                            })
                            .collect::<Result<Vec<_>>>()?;
                        routing.push(RoutingSample {
                            name: case.name.clone(),
                            frame: index,
                            decode: f.decode,
                            rows: f.requests.iter().map(|r| r.tokens.len()).sum(),
                            ranks,
                        });
                    }
                    for output in &out {
                        last_outputs.insert(output.owner, output.clone());
                    }
                    if index + 1 >= case.frames.len()
                        && index + 1 < case.frames.len() + case.decode_steps
                    {
                        frames.push(Frame {
                            decode: true,
                            requests: last_outputs
                                .values()
                                .map(|o| Request {
                                    owner: o.owner,
                                    tokens: vec![o.token],
                                    output: true,
                                })
                                .collect(),
                        });
                    }
                    outputs.push(out);
                }
                let Timings {
                    prefill_s,
                    decode_s,
                    input_tokens,
                    decode_tokens,
                } = timings;
                if repeat == 0 {
                    preparations.push(Preparation {
                        name: case.name.clone(),
                        owner_admission_s,
                        reset_s,
                        graph_preparation_s: preparation_s,
                        first_prefill_s: prefill_s,
                        first_decode_s: decode_s,
                    });
                    if case.compare_serial {
                        team.exchange(Command::Release(None))?;
                        let mut serial = vec![vec![]; frames.len()];
                        for &owner in ids.iter().rev() {
                            team.exchange(Command::Admit(vec![owner]))?;
                            if case.compare_whole_prompt && !case.requests.is_empty() {
                                let request =
                                    case.requests.iter().find(|r| r.owner == owner).unwrap();
                                let last = case
                                    .frames
                                    .iter()
                                    .rposition(|f| f.requests.iter().any(|r| r.owner == owner))
                                    .unwrap();
                                for frame in
                                    plan_frames(std::slice::from_ref(request), capacity, 1)?
                                {
                                    let (out, _, _) =
                                        forward(&team, &frame, capacity, owners, true)?;
                                    serial[last].extend(out);
                                }
                                for (index, frame) in
                                    frames.iter().enumerate().skip(case.frames.len())
                                {
                                    let one = Frame {
                                        decode: frame.decode,
                                        requests: frame
                                            .requests
                                            .iter()
                                            .filter(|r| r.owner == owner)
                                            .cloned()
                                            .collect(),
                                    };
                                    let (out, _, _) = forward(&team, &one, owners, owners, true)?;
                                    serial[index].extend(out);
                                }
                                team.exchange(Command::Release(None))?;
                                continue;
                            }
                            for (index, frame) in frames.iter().enumerate() {
                                let f = Frame {
                                    decode: frame.decode,
                                    requests: frame
                                        .requests
                                        .iter()
                                        .filter(|r| r.owner == owner)
                                        .cloned()
                                        .collect(),
                                };
                                if f.requests.is_empty() {
                                    continue;
                                }
                                let (out, _, _) = forward(
                                    &team,
                                    &f,
                                    if f.decode { owners } else { capacity },
                                    owners,
                                    true,
                                )?;
                                serial[index].extend(out);
                            }
                            team.exchange(Command::Release(None))?;
                        }
                        validation = Some(compare(&outputs, &serial)?);
                        let c = validation.as_ref().unwrap();
                        eprintln!(
                            "{}: serial positions {}, greedy {}, max logits {}, max probability {}",
                            case.name,
                            c.positions,
                            c.greedy_equal,
                            c.max_abs_logit,
                            c.max_probability_error
                        );
                        // Default BF16 collectives can change reduction order
                        // across message lengths. Near-tied greedy choices
                        // remain diagnostic; fixed-history probabilities are
                        // the numerical acceptance condition.
                        if c.max_probability_error > 0.02 {
                            return Err("Batched/serial quality validation failed".into());
                        }
                        team.exchange(Command::Admit(ids.clone()))?;
                    }
                    continue;
                }
                measurements.push(Measurement {
                    name: case.name.clone(),
                    repeat: repeat - 1,
                    batch: ids.len(),
                    input_tokens,
                    prefill_s,
                    prefill_tokens_per_s: input_tokens as f64 / prefill_s,
                    decode_s,
                    decode_tokens_per_s: if decode_s > 0. {
                        decode_tokens as f64 / decode_s
                    } else {
                        0.
                    },
                    preparation_s,
                    reset_s,
                    request_s: reset_s + preparation_s + prefill_s + decode_s,
                    output_tokens: outputs,
                    serial: validation.take(),
                });
                eprintln!(
                    "{} B{} repeat {} prefill {:.2} tok/s",
                    case.name,
                    ids.len(),
                    repeat - 1,
                    input_tokens as f64 / prefill_s
                );
            }
        }
        Ok(Report {
            layers: 40,
            tensor_parallel: 8,
            owner_capacity: owners,
            context_capacity: model.capacity,
            load_s,
            requests: measurements,
            preparations,
            routing,
            scope: "Complete shared-row native TP8 requests with owner-private INT8 KV; warm forward timings exclude compilation, Graph preparation, admission and reset; preparation is reported separately",
        })
    })()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_planner_reserves_final_windows_without_losing_owner_tokens() {
        let requests = [17usize, 129, 2048, 2049, 4097, 8193]
            .into_iter()
            .enumerate()
            .map(|(owner, n)| Request {
                owner,
                tokens: (0..n as u32).collect(),
                output: true,
            })
            .collect::<Vec<_>>();
        let frames = plan_prompt_frames(&requests, 2048, 128).unwrap();
        for frame in &frames {
            assert!(frame.requests.iter().map(|r| r.tokens.len()).sum::<usize>() <= 2048);
        }
        for r in &requests {
            let parts = frames
                .iter()
                .flat_map(|f| &f.requests)
                .filter(|p| p.owner == r.owner)
                .collect::<Vec<_>>();
            assert_eq!(
                parts
                    .iter()
                    .flat_map(|r| &r.tokens)
                    .copied()
                    .collect::<Vec<_>>(),
                r.tokens
            );
            assert!(parts.last().unwrap().output);
            assert!(parts.last().unwrap().tokens.len() >= r.tokens.len().min(128));
            assert!(parts[..parts.len() - 1].iter().all(|p| !p.output));
        }
        assert!(plan_prompt_frames(&requests, 127, 128).is_err());
    }
    #[test]
    fn explicit_teacher_decode_frames_keep_their_phase_and_output_count() {
        let case: Case = serde_json::from_value(serde_json::json!({
            "name": "teacher", "frames": [
                {"requests":[{"owner":17,"tokens":[1,2,3],"output":true}]},
                {"decode":true,"requests":[{"owner":17,"tokens":[7],"output":true}]},
                {"decode":true,"requests":[{"owner":17,"tokens":[9]}]}
            ]
        }))
        .unwrap();
        let mut times = Timings::default();
        for (frame, (outputs, seconds)) in case.frames.iter().zip([(1, 1.), (1, 0.2), (0, 0.3)]) {
            times.record(frame, outputs, seconds);
        }
        assert_eq!(times.input_tokens, 3);
        assert_eq!(times.prefill_s, 1.);
        assert_eq!(times.decode_s, 0.5);
        assert_eq!(times.decode_tokens, 1);
    }
    #[test]
    fn model_chunk_quantum_fills_packets_without_short_long_request_tails() {
        for batch in [1, 2, 3, 4, 5, 8, 16, 31, 64, 127, 128] {
            let requests = (0..batch)
                .map(|i| Request {
                    owner: i * 17 + 9,
                    tokens: vec![i as u32; 2048],
                    output: false,
                })
                .collect::<Vec<_>>();
            let frames = plan_frames(&requests, 2048, 512).unwrap();
            assert_eq!(frames.len(), batch);
            for frame in &frames {
                assert_eq!(
                    frame.requests.iter().map(|r| r.tokens.len()).sum::<usize>(),
                    2048
                );
                assert!(frame.requests.iter().all(|r| r.tokens.len() >= 512));
            }
            for r in &requests {
                let parts = frames
                    .iter()
                    .flat_map(|f| &f.requests)
                    .filter(|p| p.owner == r.owner)
                    .collect::<Vec<_>>();
                assert_eq!(
                    parts
                        .iter()
                        .flat_map(|p| &p.tokens)
                        .copied()
                        .collect::<Vec<_>>(),
                    r.tokens
                );
                assert!(parts.last().unwrap().output);
                assert!(parts[..parts.len() - 1].iter().all(|p| !p.output));
            }
        }
    }
    #[test]
    fn unequal_requests_keep_identity_and_exact_tokens_across_chunks() {
        let requests = [
            Request {
                owner: 91,
                tokens: (0..17).collect(),
                output: false,
            },
            Request {
                owner: 8,
                tokens: (30..33).collect(),
                output: false,
            },
            Request {
                owner: 44,
                tokens: (80..111).collect(),
                output: false,
            },
        ];
        let frames = plan_frames(&requests, 11, 1).unwrap();
        for frame in &frames {
            assert!(frame.requests.iter().map(|r| r.tokens.len()).sum::<usize>() <= 11);
            assert!(frame.requests.iter().all(|r| !r.tokens.is_empty()));
        }
        for original in &requests {
            let segments = frames
                .iter()
                .flat_map(|f| &f.requests)
                .filter(|r| r.owner == original.owner)
                .collect::<Vec<_>>();
            let joined = segments
                .iter()
                .flat_map(|r| r.tokens.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(joined, original.tokens);
            assert!(segments.last().unwrap().output);
            assert!(segments[..segments.len() - 1].iter().all(|r| !r.output));
        }
        assert!(plan_frames(&requests, 2, 1).is_err());
    }
}
