//! Offline-reference fixtures executed through the production native AOT path.
use super::{Access, Kernel, Spec};
use crate::artifact::{Dtype, FileIdentity, Result, read_identity, resolve_file};
use crate::backend::ascend::runtime::{Limits, Session};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixtures {
    schema_version: u32,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    manifest: String,
    passes: Vec<Pass>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pass {
    inputs: Vec<Input>,
    outputs: Vec<Output>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    slot: usize,
    data: FileIdentity,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Output {
    slot: usize,
    data: FileIdentity,
    atol: f32,
    rtol: f32,
}
#[derive(Serialize)]
pub struct Report {
    device: u32,
    cases: Vec<CaseReport>,
    scope: &'static str,
}
#[derive(Serialize)]
struct CaseReport {
    op: String,
    passes: usize,
    graph_passes: usize,
    max_abs_error: f32,
}

pub fn run(device: u32, path: &Path) -> Result<Report> {
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    let base = path.parent().ok_or("AOT fixtures have no parent")?;
    let fixtures: Fixtures = crate::model::read(&path)?;
    if fixtures.schema_version != 1 || fixtures.cases.is_empty() {
        return Err("Expected nonempty AOT fixture schema 1".into());
    }
    let session = Session::new(device, Limits::default())?;
    let mut reports = vec![];
    for case in fixtures.cases {
        if case.passes.len() < 4 {
            return Err("AOT fixtures require four changed-input passes".into());
        }
        let manifest = resolve_file(base, &case.manifest)?;
        let spec: Spec = crate::model::read(&manifest)?;
        spec.validate()?;
        for pass in &case.passes {
            validate_contract(&spec, pass)?;
        }
        let buffers = spec
            .operands
            .iter()
            .map(|o| session.allocate(o.bytes()?))
            .collect::<Result<Vec<_>>>()?;
        let kernel = Kernel::load(&session, &manifest, &buffers)?;
        let mut seen = vec![false; buffers.len()];
        for input in &case.passes[0].inputs {
            let Some(buffer) = buffers.get(input.slot) else {
                return Err("AOT input slot outside ABI".into());
            };
            let data = read_identity(base, &input.data)?;
            if data.len() != buffer.bytes() {
                return Err("AOT input extent differs".into());
            }
            upload(&session, buffer, &data)?;
        }
        // Warm native registration before capture. No allocations occur inside.
        kernel.enqueue(&session)?.wait()?;
        let graph = kernel.capture(&session)?;
        let mut max_abs_error = 0f32;
        for (index, pass) in case.passes.iter().enumerate() {
            seen.fill(false);
            for input in &pass.inputs {
                let Some(operand) = spec.operands.get(input.slot) else {
                    return Err("AOT input slot outside ABI".into());
                };
                if matches!(operand.access, Access::Write | Access::Scratch) || seen[input.slot] {
                    return Err("Duplicate AOT input or input bound to write-only operand".into());
                }
                seen[input.slot] = true;
                let data = read_identity(base, &input.data)?;
                if data.len() != buffers[input.slot].bytes() {
                    return Err("AOT input extent differs".into());
                }
                upload(&session, &buffers[input.slot], &data)?;
            }
            if spec
                .operands
                .iter()
                .enumerate()
                .any(|(i, o)| matches!(o.access, Access::Read | Access::ReadWrite) && !seen[i])
            {
                return Err("AOT pass does not initialize every input".into());
            }
            for (buffer, operand) in buffers.iter().zip(&spec.operands) {
                if matches!(operand.access, Access::Write | Access::Scratch) {
                    session.fill(buffer, 0x55)?.wait()?;
                }
            }
            if index < 2 {
                kernel.enqueue(&session)?.wait()?;
            } else {
                graph.enqueue(&session)?.wait()?;
            }
            seen.fill(false);
            for output in &pass.outputs {
                let Some(operand) = spec.operands.get(output.slot) else {
                    return Err("AOT output slot outside ABI".into());
                };
                if matches!(operand.access, Access::Read | Access::Scratch)
                    || seen[output.slot]
                    || !output.atol.is_finite()
                    || !output.rtol.is_finite()
                    || output.atol < 0.
                    || output.rtol < 0.
                {
                    return Err("Invalid AOT output/tolerance".into());
                }
                seen[output.slot] = true;
                let expected = read_identity(base, &output.data)?;
                let actual =
                    session.download(&buffers[output.slot], 0, buffers[output.slot].bytes())?;
                max_abs_error = max_abs_error.max(compare(
                    &actual,
                    &expected,
                    operand.dtype,
                    output.atol,
                    output.rtol,
                )?);
            }
            if spec
                .operands
                .iter()
                .enumerate()
                .any(|(i, o)| matches!(o.access, Access::Write | Access::ReadWrite) && !seen[i])
            {
                return Err("AOT pass does not verify every output".into());
            }
        }
        reports.push(CaseReport {
            op: kernel.spec().op.clone(),
            passes: case.passes.len(),
            graph_passes: case.passes.len() - 2,
            max_abs_error,
        });
        drop((graph, kernel, buffers));
        session.synchronize()?;
    }
    session.synchronize()?;
    session.trim_pinned_cache();
    if session.allocated_bytes() != 0 || session.pinned_bytes() != 0 {
        return Err("AOT validation retained live allocations".into());
    }
    Ok(Report {
        device,
        cases: reports,
        scope: "TileLang A2 operators and changed-input native Graph replay; no full model inference",
    })
}

fn upload(session: &Session, buffer: &super::Buffer, data: &[u8]) -> Result<()> {
    const CHUNK: usize = 8 << 20;
    for (index, part) in data.chunks(CHUNK).enumerate() {
        session.upload(buffer, index * CHUNK, part)?.wait()?;
    }
    Ok(())
}

fn compare(actual: &[u8], expected: &[u8], dtype: Dtype, atol: f32, rtol: f32) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("AOT reference extent differs".into());
    }
    if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::Bf16) {
        return if actual == expected {
            Ok(0.)
        } else {
            Err("AOT integer result differs".into())
        };
    }
    let a = crate::backend::floats(actual, dtype);
    let b = crate::backend::floats(expected, dtype);
    let mut maximum = 0f32;
    for (index, (a, b)) in a.into_iter().zip(b).enumerate() {
        if a == b {
            continue;
        }
        let delta = (a - b).abs();
        if !a.is_finite() || !b.is_finite() || delta > atol + rtol * b.abs() {
            return Err(format!(
                "AOT mismatch element {index}: got {a}, expected {b}"
            ));
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}

fn validate_contract(spec: &Spec, pass: &Pass) -> Result<()> {
    let mut inputs = vec![false; spec.operands.len()];
    let mut outputs = inputs.clone();
    for input in &pass.inputs {
        let operand = spec
            .operands
            .get(input.slot)
            .ok_or("AOT input slot outside ABI")?;
        if !matches!(operand.access, Access::Read | Access::ReadWrite) || inputs[input.slot] {
            return Err("Invalid or duplicate AOT input slot".into());
        }
        inputs[input.slot] = true;
    }
    for output in &pass.outputs {
        let operand = spec
            .operands
            .get(output.slot)
            .ok_or("AOT output slot outside ABI")?;
        if !matches!(operand.access, Access::Write | Access::ReadWrite)
            || outputs[output.slot]
            || !output.atol.is_finite()
            || !output.rtol.is_finite()
            || output.atol < 0.
            || output.rtol < 0.
        {
            return Err("Invalid AOT output slot or tolerance".into());
        }
        outputs[output.slot] = true;
    }
    for (i, operand) in spec.operands.iter().enumerate() {
        if (matches!(operand.access, Access::Read | Access::ReadWrite) && !inputs[i])
            || (matches!(operand.access, Access::Write | Access::ReadWrite) && !outputs[i])
        {
            return Err("AOT pass must initialize every input and verify every output".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Requires GPU experiment lock and ORINFER_ASCEND_AOT_FIXTURES"]
    fn native_tilelang_changed_input_graph_replay() {
        let path =
            std::env::var("ORINFER_ASCEND_AOT_FIXTURES").expect("select current AOT fixtures");
        let device = std::env::var("ORINFER_ASCEND_TEST_DEVICE")
            .expect("select device")
            .parse()
            .unwrap();
        let report = super::run(device, std::path::Path::new(&path)).unwrap();
        println!("{}", serde_json::to_string(&report).unwrap());
    }
}
