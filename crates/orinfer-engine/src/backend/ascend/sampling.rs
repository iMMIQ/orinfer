//! One dynamic categorical program shared by all native target packets.
use super::{
    graph::Graph,
    kernels::{Access, Kernel, Program, Spec, Template},
    runtime::{Buffer, Session},
};
use crate::{
    artifact::{Dtype, Result, resolve_file},
    sampling::categorical_key,
};
use serde::Deserialize;
use std::path::Path;
mod processed;
pub(crate) use processed::{Parameters, Sample, Summary, report};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Contract {
    pub capacity: usize,
    pub vocab: usize,
    pub rng: String,
    pub local: String,
    pub merge: String,
    #[serde(default)]
    pub coupled_draft: bool,
    #[serde(default)]
    pub processed_local: Option<String>,
    #[serde(default)]
    pub processed_merge: Option<String>,
}
impl Contract {
    pub fn manifests(&self) -> impl Iterator<Item = &str> {
        [
            Some(self.local.as_str()),
            Some(self.merge.as_str()),
            self.processed_local.as_deref(),
            self.processed_merge.as_deref(),
        ]
        .into_iter()
        .flatten()
    }
    pub fn bytes(&self) -> Result<usize> {
        if !(2..=4096).contains(&self.capacity)
            || self.vocab == 0
            || self.vocab >= 1 << 24
            || self.rng != "philox2x32-10-v1"
            || self.processed_local.is_some() != self.processed_merge.is_some()
        {
            return Err("Invalid A2 categorical sampling contract".into());
        }
        let extra = if self.processed_local.is_some() {
            self.capacity
                * (processed::mask_stride(self.vocab)
                    + 8
                    + (self.vocab.div_ceil(1024) + 1) * processed::WIDTH * 4)
        } else {
            0
        };
        Ok(self.capacity * (self.vocab * 4 + 16 + self.vocab.div_ceil(1024) * 8 + 8) + 4 + extra)
    }
    pub fn preflight(&self, base: &Path) -> Result<()> {
        self.bytes()?;
        let parts = self.vocab.div_ceil(1024);
        let local = vec![
            (Dtype::F32, vec![self.capacity, self.vocab], Access::Read),
            (Dtype::I32, vec![self.capacity, 4], Access::Read),
            (Dtype::I32, vec![1], Access::Read),
            (Dtype::F32, vec![self.capacity, parts, 2], Access::Write),
        ];
        let merge = vec![
            (Dtype::F32, vec![self.capacity, parts, 2], Access::Read),
            (Dtype::I32, vec![1], Access::Read),
            (Dtype::F32, vec![self.capacity, 2], Access::Write),
        ];
        for (name, op, expected) in [
            (&self.local, "sampling.categorical", local),
            (&self.merge, "sampling.merge", merge),
        ] {
            let spec: Spec = crate::model::read(&resolve_file(base, name)?)?;
            spec.validate()?;
            if spec.op != op
                || spec.row_capacity()? != Some(self.capacity)
                || spec.operands.len() != expected.len()
                || spec
                    .operands
                    .iter()
                    .zip(expected)
                    .any(|(a, (dtype, shape, access))| {
                        a.dtype != dtype || a.shape != shape || a.access != access
                    })
            {
                return Err("A2 categorical operand/row contract differs".into());
            }
        }
        if let (Some(local), Some(merge)) = (&self.processed_local, &self.processed_merge) {
            let base_operands = vec![
                (Dtype::F32, vec![self.capacity, self.vocab], Access::Read),
                (Dtype::I32, vec![self.capacity, 4], Access::Read),
                (Dtype::I32, vec![1], Access::Read),
                (
                    Dtype::U8,
                    vec![self.capacity, processed::mask_stride(self.vocab)],
                    Access::Read,
                ),
                (Dtype::I32, vec![self.capacity, 2], Access::Read),
                (
                    Dtype::F32,
                    vec![self.capacity, parts, processed::WIDTH],
                    Access::Write,
                ),
            ];
            let merge_operands = vec![
                (
                    Dtype::F32,
                    vec![self.capacity, parts, processed::WIDTH],
                    Access::Read,
                ),
                (Dtype::I32, vec![1], Access::Read),
                (Dtype::I32, vec![self.capacity, 2], Access::Read),
                (
                    Dtype::F32,
                    vec![self.capacity, processed::WIDTH],
                    Access::Write,
                ),
            ];
            for (name, op, expected) in [
                (local, "sampling_processed.local", base_operands),
                (merge, "sampling_processed.merge", merge_operands),
            ] {
                let spec: Spec = crate::model::read(&resolve_file(base, name)?)?;
                spec.validate()?;
                if spec.op != op
                    || spec.row_capacity()? != Some(self.capacity)
                    || spec.operands.len() != expected.len()
                    || spec
                        .operands
                        .iter()
                        .zip(expected)
                        .any(|(a, (dtype, shape, access))| {
                            a.dtype != dtype || a.shape != shape || a.access != access
                        })
                {
                    return Err("Processed sampler operand/row contract differs".into());
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone)]
pub(crate) struct Draw {
    pub seed: u64,
    pub step: usize,
    pub temperature: f64,
}
impl Draw {
    pub fn control(&self, offset: usize) -> Result<[i32; 3]> {
        if self.temperature != 0.0 && !(1e-5..=2.0).contains(&self.temperature) {
            return Err("Invalid device categorical temperature".into());
        }
        let inverse = if self.temperature == 0.0 {
            0.0
        } else {
            (1.0 / self.temperature) as f32
        };
        if !inverse.is_finite() {
            return Err("Device categorical temperature exceeds FP32 range".into());
        }
        let step = self
            .step
            .checked_add(offset)
            .ok_or("Sampling position overflow")?;
        let key = categorical_key(self.seed, step as u64);
        Ok([key[0] as i32, key[1] as i32, inverse.to_bits() as i32])
    }
    pub fn controls(&self, origin: u32, count: usize) -> Result<Vec<u8>> {
        (0..count)
            .map(|offset| self.control(offset))
            .collect::<Result<Vec<_>>>()
            .map(|rows| {
                rows.into_iter()
                    .flat_map(|r| [r[0], r[1], origin as i32, r[2]])
                    .flat_map(i32::to_le_bytes)
                    .collect()
            })
    }
}
pub(crate) struct Categorical {
    contract: Contract,
    logits: Buffer,
    controls: Buffer,
    active: Buffer,
    output: Buffer,
    program: Program,
    graph: Option<Graph>,
    processed: Option<processed::Processed>,
}
impl Categorical {
    pub fn load(session: &Session, base: &Path, contract: &Contract) -> Result<Self> {
        contract.preflight(base)?;
        let local = Template::load(session, &resolve_file(base, &contract.local)?)?;
        let merge = Template::load(session, &resolve_file(base, &contract.merge)?)?;
        let logits = session.allocate(contract.capacity * contract.vocab * 4)?;
        let controls = session.allocate(contract.capacity * 16)?;
        let active = session.allocate(4)?;
        let partial = session.allocate(contract.capacity * contract.vocab.div_ceil(1024) * 8)?;
        let output = session.allocate(contract.capacity * 8)?;
        let first = Kernel::bind(
            session,
            &local,
            &[
                logits.clone(),
                controls.clone(),
                active.clone(),
                partial.clone(),
            ],
        )?;
        let second = Kernel::bind(session, &merge, &[partial, active.clone(), output.clone()])?;
        let program = Program::new(session, &[&first, &second])?;
        let processed = contract
            .processed_local
            .as_ref()
            .map(|_| {
                processed::Processed::load(session, base, contract, &logits, &controls, &active)
            })
            .transpose()?;
        Ok(Self {
            contract: contract.clone(),
            logits,
            controls,
            active,
            output,
            program,
            graph: None,
            processed,
        })
    }
    pub fn sample(
        &mut self,
        session: &Session,
        source: &Buffer,
        origin: u32,
        rows: &[[i32; 3]],
        parameters: &[Parameters],
        capture: bool,
    ) -> Result<Vec<Sample>> {
        if rows.is_empty()
            || rows.len() > self.contract.capacity
            || source.bytes() < rows.len() * self.contract.vocab * 4
            || origin as usize + self.contract.vocab >= 1 << 24
            || (!parameters.is_empty() && parameters.len() != rows.len())
        {
            return Err("Device sampling rows/vocabulary exceed the contract".into());
        }
        let processed = parameters.iter().any(Parameters::processed);
        if capture && !processed && self.graph.is_none() {
            session.upload_inputs(&[(self.active.clone(), 0i32.to_le_bytes().to_vec())])?;
            self.graph = Some(self.program.capture(session)?);
        }
        let controls = rows
            .iter()
            .flat_map(|r| [r[0], r[1], origin as i32, r[2]])
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        session.upload_inputs(&[
            (self.controls.slice(0, controls.len())?, controls),
            (
                self.active.clone(),
                (rows.len() as i32).to_le_bytes().to_vec(),
            ),
        ])?;
        session.copy(
            source,
            0,
            &self.logits,
            0,
            rows.len() * self.contract.vocab * 4,
        )?;
        if processed {
            return self
                .processed
                .as_mut()
                .ok_or("Execution package has no processed sampler")?
                .sample(
                    session,
                    &self.contract,
                    &self.active,
                    origin,
                    parameters,
                    capture,
                );
        }
        if capture {
            self.graph.as_ref().unwrap().enqueue(session)?.wait()?;
        } else {
            self.program.enqueue(session)?.wait()?;
        }
        let bytes = session.download(&self.output, 0, rows.len() * 8)?;
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                let score = f32::from_le_bytes(p[..4].try_into().unwrap());
                let index = f32::from_le_bytes(p[4..].try_into().unwrap());
                if !score.is_finite()
                    || !index.is_finite()
                    || index.fract() != 0.0
                    || index < 0.0
                    || index >= self.contract.vocab as f32
                {
                    return Err("Nonfinite/invalid device categorical output".into());
                }
                Ok(Sample {
                    score,
                    token: index as u32,
                    summary: None,
                })
            })
            .collect()
    }
}
