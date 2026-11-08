//! Device-resident proposal distributions and compact TP normalization.
use super::*;
use crate::backend::ascend::hccl::{Communicator, Element};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Contract {
    pub pool_rows: usize,
    pub stride: usize,
    pub ranks: usize,
    pub statistics: String,
    pub merge_statistics: String,
    pub normalize: String,
    pub select: String,
}
impl Contract {
    pub fn manifests(&self) -> [&str; 4] {
        [
            &self.statistics,
            &self.merge_statistics,
            &self.normalize,
            &self.select,
        ]
    }
    pub fn bytes(&self, c: &super::Contract) -> Result<usize> {
        if !(1..=16).contains(&self.stride)
            || self.pool_rows == 0
            || !self.pool_rows.is_multiple_of(self.stride)
            || self.pool_rows / self.stride > 128
            || !(1..=128).contains(&self.ranks)
        {
            return Err("Invalid rejection pool/rank contract".into());
        }
        Ok(self.pool_rows * c.vocab * 4
            + c.capacity * (16 + c.vocab.div_ceil(1024) * 16 + 32 + self.ranks * 32 + 16))
    }
    pub fn preflight(&self, c: &super::Contract, base: &Path) -> Result<()> {
        self.bytes(c)?;
        let logits = (Dtype::F32, vec![c.capacity, c.vocab], Access::Read);
        let draft = (Dtype::F32, vec![self.pool_rows, c.vocab], Access::Read);
        let controls = (Dtype::I32, vec![c.capacity, 4], Access::Read);
        let active = (Dtype::I32, vec![1], Access::Read);
        let partial = (
            Dtype::F32,
            vec![c.capacity, c.vocab.div_ceil(1024), 4],
            Access::Read,
        );
        let specifications = [
            (
                "statistics",
                &self.statistics,
                vec![
                    logits.clone(),
                    draft.clone(),
                    controls.clone(),
                    controls.clone(),
                    active.clone(),
                    (partial.0, partial.1.clone(), Access::Write),
                ],
            ),
            (
                "merge_statistics",
                &self.merge_statistics,
                vec![
                    partial,
                    logits.clone(),
                    draft.clone(),
                    controls.clone(),
                    controls.clone(),
                    active.clone(),
                    (Dtype::F32, vec![c.capacity, 8], Access::Write),
                ],
            ),
            (
                "normalize",
                &self.normalize,
                vec![
                    (Dtype::F32, vec![self.ranks, c.capacity, 8], Access::Read),
                    controls.clone(),
                    active.clone(),
                    (Dtype::F32, vec![c.capacity, 4], Access::Write),
                ],
            ),
            (
                "select",
                &self.select,
                vec![
                    logits,
                    draft,
                    controls.clone(),
                    controls,
                    (Dtype::F32, vec![c.capacity, 4], Access::Read),
                    active,
                    (
                        Dtype::F32,
                        vec![c.capacity, c.vocab.div_ceil(1024), 2],
                        Access::Write,
                    ),
                ],
            ),
        ];
        for (op, path, ports) in specifications {
            let spec: Spec = crate::model::read(&resolve_file(base, path)?)?;
            spec.validate()?;
            if spec.op != format!("sampling_rejection.{op}")
                || spec.row_capacity()? != Some(c.capacity)
                || spec.operands.len() != ports.len()
                || spec
                    .operands
                    .iter()
                    .zip(ports)
                    .any(|(p, (dtype, shape, access))| {
                        p.dtype != dtype || p.shape != shape || p.access != access
                    })
            {
                return Err("Rejection sampler operand/row contract differs".into());
            }
        }
        Ok(())
    }
}

pub(crate) fn requested() -> Result<bool> {
    match std::env::var("ORINFER_A2_DSPARK_SAMPLING").as_deref() {
        Ok("rejection") => Ok(true),
        Ok("coupled") | Err(std::env::VarError::NotPresent) => Ok(false),
        _ => Err("ORINFER_A2_DSPARK_SAMPLING must be coupled or rejection".into()),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Proposal {
    pub row: usize,
    pub token: u32,
    pub log_uniform: f32,
}

pub(super) struct Rejection {
    contract: Contract,
    pool: Buffer,
    proposals: Buffer,
    local: Buffer,
    gathered: Buffer,
    first: Program,
    second: Program,
    graphs: Option<(Graph, Graph)>,
    ready: Vec<usize>,
}
impl Rejection {
    pub fn load(
        session: &Session,
        base: &Path,
        c: &super::Contract,
        buffers: [&Buffer; 5],
    ) -> Result<Self> {
        let [logits, controls, active, partial, output] = buffers;
        let contract = c
            .rejection
            .as_ref()
            .ok_or("Execution package has no rejection sampler")?
            .clone();
        let pool = session.allocate(contract.pool_rows * c.vocab * 4)?;
        let proposals = session.allocate(c.capacity * 16)?;
        let statistics = session.allocate(c.capacity * c.vocab.div_ceil(1024) * 16)?;
        let local = session.allocate(c.capacity * 32)?;
        let gathered = session.allocate(contract.ranks * c.capacity * 32)?;
        let normalizers = session.allocate(c.capacity * 16)?;
        let bind = |path: &str, buffers: &[Buffer]| -> Result<Kernel> {
            Kernel::bind(
                session,
                &Template::load(session, &resolve_file(base, path)?)?,
                buffers,
            )
        };
        let stats = bind(
            &contract.statistics,
            &[
                logits.clone(),
                pool.clone(),
                controls.clone(),
                proposals.clone(),
                active.clone(),
                statistics.clone(),
            ],
        )?;
        let merge = bind(
            &contract.merge_statistics,
            &[
                statistics,
                logits.clone(),
                pool.clone(),
                controls.clone(),
                proposals.clone(),
                active.clone(),
                local.clone(),
            ],
        )?;
        let normalize = bind(
            &contract.normalize,
            &[
                gathered.clone(),
                proposals.clone(),
                active.clone(),
                normalizers.clone(),
            ],
        )?;
        let select = bind(
            &contract.select,
            &[
                logits.clone(),
                pool.clone(),
                controls.clone(),
                proposals.clone(),
                normalizers,
                active.clone(),
                partial.clone(),
            ],
        )?;
        let merge_samples = bind(&c.merge, &[partial.clone(), active.clone(), output.clone()])?;
        let first = Program::new(session, &[&stats, &merge])?;
        let second = Program::new(session, &[&normalize, &select, &merge_samples])?;
        let ready = vec![0; contract.pool_rows / contract.stride];
        Ok(Self {
            contract,
            pool,
            proposals,
            local,
            gathered,
            first,
            second,
            graphs: None,
            ready,
        })
    }
    /// Copy before the target may reuse draft scratch. Both copies and later
    /// consumers use the owning compute stream; the ticket retains all spans.
    pub fn store(
        &mut self,
        session: &Session,
        vocab: usize,
        source: &Buffer,
        stride: usize,
        owners: &[(usize, usize)],
    ) -> Result<()> {
        if stride != self.contract.stride || owners.is_empty() {
            return Err("Proposal pool physical stride differs".into());
        }
        let mut ready = vec![0; self.ready.len()];
        let mut copies = Vec::with_capacity(owners.len());
        for (lane, &(owner, count)) in owners.iter().enumerate() {
            if owner >= ready.len() || count == 0 || count > stride || ready[owner] != 0 {
                return Err("Invalid or repeated proposal owner".into());
            }
            ready[owner] = count;
            copies.push((
                source.clone(),
                lane * stride * vocab * 4,
                self.pool.clone(),
                owner * stride * vocab * 4,
                count * vocab * 4,
            ));
        }
        session.copy_many(&copies)?;
        self.ready = ready;
        Ok(())
    }
    pub fn sample(
        &mut self,
        session: &Session,
        comm: &Communicator,
        c: &super::Contract,
        proposals: &[Option<Proposal>],
        active: &Buffer,
        capture: bool,
    ) -> Result<()> {
        if comm.ranks() as usize != self.contract.ranks || proposals.len() > c.capacity {
            return Err("Rejection sampler rank/row count differs".into());
        }
        let mut packed = Vec::with_capacity(proposals.len() * 16);
        for proposal in proposals {
            let fields = if let Some(p) = proposal {
                if p.row >= self.contract.pool_rows
                    || p.row % self.contract.stride >= self.ready[p.row / self.contract.stride]
                    || p.token as usize >= c.vocab * self.contract.ranks
                    || !p.log_uniform.is_finite()
                    || p.log_uniform >= 0.
                {
                    return Err("Missing or invalid stored draft distribution".into());
                }
                [
                    p.row as i32,
                    p.token as i32,
                    p.log_uniform.to_bits() as i32,
                    0,
                ]
            } else {
                [-1, 0, 0, 0]
            };
            packed.extend(fields.into_iter().flat_map(i32::to_le_bytes));
        }
        if capture && self.graphs.is_none() {
            session.upload_inputs(&[(active.clone(), 0i32.to_le_bytes().to_vec())])?;
            self.graphs = Some((self.first.capture(session)?, self.second.capture(session)?));
        }
        session.upload_inputs(&[
            (self.proposals.slice(0, packed.len())?, packed),
            (
                active.clone(),
                (proposals.len() as i32).to_le_bytes().to_vec(),
            ),
        ])?;
        if let Some((first, _)) = self.graphs.as_ref().filter(|_| capture) {
            first.enqueue(session)?;
        } else {
            self.first.enqueue(session)?;
        }
        comm.all_gather(
            session,
            &self.local,
            &self.gathered,
            (c.capacity * 8) as u64,
            Element::Float32,
        )?;
        if let Some((_, second)) = self.graphs.as_ref().filter(|_| capture) {
            second.enqueue(session)?;
        } else {
            self.second.enqueue(session)?;
        }
        Ok(())
    }
}
