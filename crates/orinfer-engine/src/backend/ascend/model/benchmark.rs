use super::{
    Limits, Model, RootInfo, Session, SharedTables, Worker, affinity, load_spec, packet::Best,
};
use crate::artifact::Result;
use orinfer_model_sdk::model::PrefillPlan;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
struct Inputs {
    cases: Vec<Input>,
}
#[derive(Deserialize)]
struct Input {
    name: String,
    tokens: Vec<u32>,
    tokens_sha256: String,
    output_tokens: Option<usize>,
    repeats: Option<usize>,
    trace: Option<bool>,
    prefill_chunk_tokens: Option<usize>,
}
enum Command {
    Prepare(Vec<usize>),
    Reset,
    Forward {
        tokens: Vec<u32>,
        head: bool,
        trace: Option<PathBuf>,
    },
    Stop,
}
struct Peer {
    send: mpsc::Sender<Command>,
    receive: mpsc::Receiver<Result<Option<Best>>>,
}
#[derive(Serialize)]
pub struct Report {
    model_layers: usize,
    tensor_parallel: usize,
    capacity: usize,
    load_s: f64,
    logits_dump: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker_cpus: Option<Vec<Vec<usize>>>,
    requests: Vec<Request>,
    preparations: Vec<Preparation>,
    scope: &'static str,
}
#[derive(Serialize)]
struct Request {
    name: String,
    repeat: usize,
    input_tokens: usize,
    output_tokens: usize,
    tokens_sha256: String,
    prefill_chunks: Vec<usize>,
    prepare_s: f64,
    reset_s: f64,
    request_ttft_s: f64,
    request_e2e_s: f64,
    ttft_s: f64,
    prefill_tokens_per_s_from_ttft: f64,
    decode_s: f64,
    decode_tokens_per_s: f64,
    e2e_s: f64,
    output_token_ids: Vec<u32>,
}
#[derive(Serialize)]
struct Preparation {
    name: String,
    rows: Vec<usize>,
    new_rows: Vec<usize>,
    prepare_s: f64,
    first_ttft_s: f64,
    reset_s: f64,
    first_request_ttft_s: f64,
}

pub fn run(
    plan: &Path,
    model_dir: &Path,
    inputs: &Path,
    output_tokens: usize,
    repeats: usize,
    trace: Option<&Path>,
) -> Result<Report> {
    if output_tokens < 2 || repeats == 0 {
        return Err("At least two output tokens and one repeat are required".into());
    }
    let (mut model, package, layout) = load_spec(plan, model_dir)?;
    let inputs: Inputs = crate::model::read(inputs)?;
    if inputs.cases.is_empty()
        || inputs.cases.iter().any(|c| {
            c.tokens.is_empty()
                || c.tokens.iter().any(|&t| t >= 129280)
                || c.output_tokens.unwrap_or(output_tokens) < 2
                || c.repeats.unwrap_or(repeats) == 0
                || c.prefill_chunk_tokens
                    .is_some_and(|n| !(1..=4096).contains(&n))
                || c.tokens
                    .len()
                    .checked_add(c.output_tokens.unwrap_or(output_tokens).max(16))
                    .is_none_or(|n| n > model.capacity)
        })
    {
        return Err("Invalid input tokens or context capacity".into());
    }
    let plans: Vec<_> = model.ranks[0]
        .packets
        .iter()
        .map(|p| p.prefill_plan())
        .collect();
    let ranks = std::mem::take(&mut model.ranks);
    if ranks.iter().enumerate().any(|(i, r)| r.rank != i as u32) {
        return Err("A2 ranks must be ordered 0..7".into());
    }
    let worker_cpus = affinity::from_env(ranks.len())?;
    let load = Instant::now();
    let bootstrap = Session::new(0, Limits::default())?;
    let root = RootInfo::new(&bootstrap, 8)?;
    let shared_tables = SharedTables::new();
    std::thread::scope(|scope| {
        let mut handles = vec![];
        let mut peers = vec![];
        let model_ref: &Model = &model;
        let shared_tables = &shared_tables;
        for rank in ranks {
            let cpus = worker_cpus
                .as_ref()
                .map(|all| all[rank.rank as usize].clone());
            let (send, commands) = mpsc::channel();
            let (reply, receive) = mpsc::channel();
            let root = root.clone();
            let layout = layout.clone();
            let package = &package;
            handles.push(scope.spawn(move || {
                let loaded = (|| {
                    if let Some(cpus) = cpus.as_deref() {
                        affinity::apply(cpus)?;
                    }
                    let session = Session::new(
                        rank.rank,
                        Limits {
                            device_bytes: 58 << 30,
                            pinned_bytes: 64 << 20,
                            timeout_ms: 30000,
                        },
                    )?;
                    Worker::load(
                        session,
                        &root,
                        rank,
                        model_ref,
                        model_dir,
                        package,
                        layout,
                        trace.is_some(),
                        shared_tables,
                    )
                })();
                let mut worker = match loaded {
                    Ok(worker) => worker,
                    Err(error) => {
                        let _ = reply.send(Err(error));
                        return;
                    }
                };
                if reply.send(Ok(None)).is_err() {
                    return;
                }
                let mut history = vec![];
                while let Ok(command) = commands.recv() {
                    let result = match command {
                        Command::Prepare(rows) => rows
                            .into_iter()
                            .try_for_each(|n| worker.prepare_rows(n))
                            .map(|_| None),
                        Command::Stop => {
                            worker.report_steps();
                            break;
                        }
                        Command::Reset => {
                            worker.report_steps();
                            history.clear();
                            worker.reset().map(|_| None)
                        }
                        Command::Forward {
                            tokens,
                            head,
                            trace,
                        } => {
                            let result = worker.forward(&history, &tokens, head, trace.as_deref());
                            if result.is_ok() {
                                history.extend(tokens);
                            }
                            result
                        }
                    };
                    let failed = result.is_err();
                    if reply.send(result).is_err() || failed {
                        break;
                    }
                }
                let _ = worker.session.synchronize();
            }));
            peers.push(Peer { send, receive });
        }
        let measured = (|| {
            for peer in &peers {
                peer.receive
                    .recv_timeout(Duration::from_secs(1800))
                    .map_err(|e| format!("A2 model load: {e}"))??;
            }
            let load_s = load.elapsed().as_secs_f64();
            eprintln!("A2 complete model loaded in {load_s:.3}s");
            let mut requests = vec![];
            let mut preparations = vec![];
            let mut prepared: std::collections::BTreeSet<_> =
                plans.iter().map(|p| p.chunk_tokens).collect();
            for input in &inputs.cases {
                let trace = trace.filter(|_| input.trace != Some(false));
                // Warm the same packet shapes and context extent, excluding
                // weights, first-use registration and allocation from timing.
                let warm = generate(&peers, input, 16, usize::MAX, trace, &plans)?;
                let rows: std::collections::BTreeSet<_> =
                    warm.prefill_chunks.iter().copied().collect();
                preparations.push(Preparation {
                    name: input.name.clone(),
                    rows: rows.iter().copied().collect(),
                    new_rows: rows.difference(&prepared).copied().collect(),
                    prepare_s: warm.prepare_s,
                    first_ttft_s: warm.ttft_s,
                    reset_s: warm.reset_s,
                    first_request_ttft_s: warm.request_ttft_s,
                });
                prepared.extend(rows);
                for repeat in 0..input.repeats.unwrap_or(repeats) {
                    let request = generate(
                        &peers,
                        input,
                        input.output_tokens.unwrap_or(output_tokens),
                        repeat,
                        trace,
                        &plans,
                    )?;
                    eprintln!(
                        "{} r{} TTFT={:.4}s decode={:.3} token/s",
                        input.name, repeat, request.ttft_s, request.decode_tokens_per_s
                    );
                    requests.push(request);
                }
            }
            Ok(Report {
                model_layers: 40,
                tensor_parallel: 8,
                capacity: model.capacity,
                load_s,
                logits_dump: std::env::var_os("ORINFER_A2_LOGITS_DIR").is_some(),
                worker_cpus,
                requests,
                preparations,
                scope: "Complete 40-layer original-checkpoint native Rust/TileLang + ACLNN W4A8; single sequence, no prefix reuse or speculation; legacy ttft_s excludes preparation/reset, request_ttft_s includes them; first-shape costs are reported in preparations",
            })
        })();
        for peer in &peers {
            let _ = peer.send.send(Command::Stop);
        }
        for handle in handles {
            handle.join().map_err(|_| "A2 model worker panicked")?;
        }
        measured
    })
}
fn receive(peers: &[Peer]) -> Result<Vec<Option<Best>>> {
    peers
        .iter()
        .map(|peer| {
            peer.receive
                .recv_timeout(Duration::from_secs(600))
                .map_err(|e| format!("A2 model step: {e}"))?
        })
        .collect()
}
fn forward(
    peers: &[Peer],
    tokens: &[u32],
    head: bool,
    trace: Option<&Path>,
) -> Result<Option<u32>> {
    for peer in peers {
        peer.send
            .send(Command::Forward {
                tokens: tokens.to_vec(),
                head,
                trace: trace.map(Path::to_owned),
            })
            .map_err(|e| e.to_string())?;
    }
    let values = receive(peers)?;
    if !head {
        return Ok(None);
    }
    let mut best: Option<Best> = None;
    for candidate in values {
        let candidate = candidate.ok_or("Model rank omitted logits")?;
        if best.is_none_or(|b| {
            candidate.value > b.value || (candidate.value == b.value && candidate.token < b.token)
        }) {
            best = Some(candidate);
        }
    }
    Ok(Some(best.ok_or("Model output missing")?.token))
}
fn generate(
    peers: &[Peer],
    input: &Input,
    output_tokens: usize,
    repeat: usize,
    trace: Option<&Path>,
    plans: &[PrefillPlan],
) -> Result<Request> {
    let request_start = Instant::now();
    let mut chunks = vec![];
    let mut remaining = input.tokens.len();
    while remaining > 0 {
        let rows = plans
            .iter()
            .filter_map(|p| {
                p.chunk(remaining.min(input.prefill_chunk_tokens.unwrap_or(usize::MAX)))
            })
            .max()
            .ok_or("No compatible prefill row template")?;
        chunks.push(rows);
        remaining -= rows;
    }
    let shapes = chunks
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let prepare = Instant::now();
    for peer in peers {
        peer.send
            .send(Command::Prepare(shapes.iter().copied().collect()))
            .map_err(|e| e.to_string())?;
    }
    receive(peers)?;
    let prepare_s = prepare.elapsed().as_secs_f64();
    let reset = Instant::now();
    // Shape registration/capture precedes reset: warmups may mutate C2 rings.
    for peer in peers {
        peer.send.send(Command::Reset).map_err(|e| e.to_string())?;
    }
    receive(peers)?;
    let reset_s = reset.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut position = 0;
    let mut token = None;
    for &rows in &chunks {
        let end = position + rows;
        token = forward(
            peers,
            &input.tokens[position..end],
            end == input.tokens.len(),
            trace,
        )?;
        position = end;
    }
    let ttft_s = start.elapsed().as_secs_f64();
    let request_ttft_s = request_start.elapsed().as_secs_f64();
    let mut tokens = vec![token.ok_or("Prefill omitted first token")?];
    let decode = Instant::now();
    for _ in 1..output_tokens {
        let token = forward(peers, &[*tokens.last().unwrap()], true, trace)?
            .ok_or("Decode omitted token")?;
        tokens.push(token);
    }
    let decode_s = decode.elapsed().as_secs_f64();
    Ok(Request {
        name: input.name.clone(),
        repeat,
        input_tokens: input.tokens.len(),
        output_tokens,
        tokens_sha256: input.tokens_sha256.clone(),
        ttft_s,
        prefill_tokens_per_s_from_ttft: input.tokens.len() as f64 / ttft_s,
        decode_s,
        decode_tokens_per_s: (output_tokens - 1) as f64 / decode_s,
        e2e_s: start.elapsed().as_secs_f64(),
        output_token_ids: tokens,
        prefill_chunks: chunks,
        prepare_s,
        reset_s,
        request_ttft_s,
        request_e2e_s: request_start.elapsed().as_secs_f64(),
    })
}
