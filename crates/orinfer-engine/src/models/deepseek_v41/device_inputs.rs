//! Device-owned continuation tokens and coordinates, shared by target and draft.
//! CPU history remains authoritative for output parsing, host Engram and prefixes.
use super::{Instruction, Packet, PacketRole, Rank, Worker, batch::Layout, get};
use crate::{
    artifact::{Dtype, Result, resolve_file},
    backend::ascend::{
        hccl::Element,
        kernels::{Access, Kernel, Spec, Template},
        runtime::{Buffer, Session},
    },
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::Path,
    rc::Rc,
};

const VOCAB: usize = 129280;
const DRAFTS: usize = 16;
const WIDTH: usize = 5120;
const DRAFT_EMBEDDING: &str = "mtp.0.embed.weight";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Preparation {
    tokens: String,
    coordinates: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Contract {
    capacity: usize,
    preparations: BTreeMap<String, Preparation>,
    embeddings: BTreeMap<String, String>,
    merge: String,
    scatter_embedding: String,
    draft_embedding: String,
    commit: String,
    save_draft: String,
    hashes: String,
    hash_width: usize,
}

fn geometry(packet: &Packet, names: &super::packet::Coordinates) -> String {
    format!(
        "{}:{}:{}",
        packet.rows,
        names.draft_block.unwrap_or(0),
        usize::from(names.source_swa_indices.is_some())
    )
}
type Bank = (String, usize, usize);
fn banks(rank: &Rank) -> BTreeSet<Bank> {
    rank.packets
        .iter()
        .filter(|p| p.role != PacketRole::Draft)
        .flat_map(|p| &p.body)
        .filter_map(|i| match i {
            Instruction::Embed {
                tensor,
                first_column,
                columns,
                ..
            } => Some((tensor.clone(), *first_column, *columns)),
            _ => None,
        })
        .collect()
}
impl Contract {
    pub(super) fn manifests(&self) -> impl Iterator<Item = &str> {
        [
            &self.merge,
            &self.commit,
            &self.save_draft,
            &self.hashes,
            &self.scatter_embedding,
            &self.draft_embedding,
        ]
        .into_iter()
        .chain(self.embeddings.values())
        .chain(
            self.preparations
                .values()
                .flat_map(|p| [&p.tokens, &p.coordinates]),
        )
        .map(String::as_str)
    }
    pub(super) fn bytes(
        &self,
        rank: &Rank,
        context: usize,
        owners: usize,
        outputs: usize,
    ) -> Result<usize> {
        if !(1..=4096).contains(&self.capacity)
            || !(1..=128).contains(&owners)
            || !(1..=64 * 64 * 3).contains(&self.hash_width)
        {
            return Err("Invalid device continuation capacity".into());
        }
        let tables = banks(rank)
            .into_iter()
            .try_fold(0usize, |n, (_, first, columns)| {
                super::packet::embedding_span(first, columns)?;
                n.checked_add(VOCAB * columns * 2)
                    .ok_or_else(|| "Embedding bank budget overflow".to_string())
            })?;
        Ok(tables
            + VOCAB / 8 * WIDTH * 2
            + (owners + 1) * WIDTH * 2
            + owners * WIDTH * 2 * 2
            + context * 128 * 4
            + self.capacity * (32 + 8 + 512)
            + owners * (32 + 16 + DRAFTS * 4)
            + outputs * 8 * 9
            + owners * 16)
    }
    pub(super) fn preflight(&self, model: &super::Model, base: &Path) -> Result<()> {
        let batch = model
            .batch
            .as_ref()
            .ok_or("Device inputs require batch owners")?;
        if model.sampling.is_none()
            || model.ranks.len() != 8
            || self.capacity
                != model
                    .ranks
                    .iter()
                    .flat_map(|r| &r.packets)
                    .map(|p| p.rows)
                    .max()
                    .unwrap_or(0)
        {
            return Err("Device inputs differ from packet or TP sampling capacity".into());
        }
        let check = |name: &str,
                     op: &str,
                     rows: usize,
                     expected: Vec<(Dtype, Vec<usize>, Access)>|
         -> Result<()> {
            let spec: Spec = crate::model::read(&resolve_file(base, name)?)?;
            spec.validate()?;
            if spec.op != format!("frontier.{op}")
                || spec.row_capacity()? != Some(rows)
                || spec.operands.len() != expected.len()
                || spec
                    .operands
                    .iter()
                    .zip(expected)
                    .any(|(p, (dtype, shape, access))| {
                        p.dtype != dtype || p.shape != shape || p.access != access
                    })
            {
                return Err(format!("Device continuation {op} ABI differs"));
            }
            Ok(())
        };
        let (owners, outputs) = (batch.owners, batch.output_capacity());
        use Access::{Read as R, ReadWrite as RW, Write as W};
        use Dtype::{Bf16 as B, F32 as F, I32 as I, I64 as L, U64 as U};
        let layout: orinfer_model_sdk::conditional_memory::HashLayout =
            crate::model::read(&resolve_file(base, &model.engram.file)?)?;
        let layers = layout.layers.len();
        if self.hash_width != layers * layout.heads * 3
            || layout.token_map.len() != VOCAB
            || layout.barriers.len() != 2
        {
            return Err("Device hash geometry differs from conditional memory".into());
        }
        let mut checked = BTreeSet::new();
        for rank in &model.ranks {
            self.bytes(rank, model.capacity, owners, outputs)?;
            for p in &rank.packets {
                for c in [&p.coordinates]
                    .into_iter()
                    .chain(p.decoder_coordinates.as_ref())
                {
                    let key = geometry(p, c);
                    let prep = self
                        .preparations
                        .get(&key)
                        .ok_or("Missing device coordinate geometry")?;
                    if !c.device_indices && c.draft_block.is_none() {
                        return Err("Device inputs require device causal indices".into());
                    }
                    if !checked.insert(key) {
                        continue;
                    }
                    let n = p.rows;
                    check(
                        &prep.tokens,
                        "tokens",
                        n,
                        vec![
                            (I, vec![n, 8], R),
                            (I, vec![owners, 4], R),
                            (I, vec![owners, DRAFTS], R),
                            (I, vec![n], W),
                            (I, vec![n], W),
                        ],
                    )?;
                    let mut ports = vec![
                        (I, vec![n, 8], R),
                        (I, vec![n], R),
                        (B, vec![4, model.capacity, 64], R),
                    ];
                    ports.extend((0..6).map(|_| (B, vec![n, 64], W)));
                    ports.extend([
                        (I, vec![n], W),
                        (I, vec![n], W),
                        (I, vec![n, 128], W),
                        (I, vec![n, 128], W),
                        (I, vec![n, 8], W),
                        (F, vec![n, 8], W),
                    ]);
                    check(&prep.coordinates, "coordinates", n, ports)?;
                }
                for i in &p.body {
                    if let Instruction::Embed {
                        columns,
                        tensor,
                        first_column,
                        ..
                    } = i
                    {
                        if p.role == PacketRole::Draft {
                            if tensor != DRAFT_EMBEDDING || *columns != WIDTH || *first_column != 0
                            {
                                return Err("Cached draft embedding geometry differs".into());
                            }
                            check(
                                &self.draft_embedding,
                                "draft_embedding",
                                p.rows,
                                vec![
                                    (I, vec![p.rows, 8], R),
                                    (B, vec![owners, WIDTH], R),
                                    (B, vec![WIDTH], R),
                                    (B, vec![p.rows, WIDTH], W),
                                ],
                            )?;
                            continue;
                        }
                        let name = self
                            .embeddings
                            .get(&format!("{}:{columns}", p.rows))
                            .ok_or("Missing resident embedding geometry")?;
                        check(
                            name,
                            "embedding",
                            p.rows,
                            vec![
                                (B, vec![VOCAB, *columns], R),
                                (I, vec![p.rows], R),
                                (B, vec![p.rows, *columns], W),
                            ],
                        )?;
                    }
                }
            }
        }
        check(
            &self.merge,
            "merge",
            outputs,
            vec![(F, vec![8 * outputs * 2], R), (F, vec![outputs, 2], W)],
        )?;
        check(
            &self.commit,
            "commit",
            owners,
            vec![
                (I, vec![owners, 8], R),
                (F, vec![outputs, 2], R),
                (I, vec![self.capacity], R),
                (I, vec![owners, DRAFTS], R),
                (I, vec![owners, 4], W),
                (I, vec![self.capacity], R),
                (I, vec![owners, 4], RW),
                (B, vec![VOCAB / 8, WIDTH], R),
                (B, vec![owners, WIDTH], W),
            ],
        )?;
        check(
            &self.scatter_embedding,
            "scatter_embedding",
            owners,
            vec![
                (I, vec![owners, 8], R),
                (B, vec![owners, WIDTH], R),
                (B, vec![owners, WIDTH], RW),
            ],
        )?;
        check(
            &self.hashes,
            "hashes",
            self.capacity,
            vec![
                (I, vec![self.capacity, 8], R),
                (I, vec![self.capacity], R),
                (I, vec![owners, 4], R),
                (I, vec![VOCAB], R),
                (U, vec![layers, 4], R),
                (L, vec![layers, layout.heads * 3, 2], R),
                (L, vec![self.capacity, self.hash_width], W),
                (I, vec![self.capacity], W),
            ],
        )?;
        let strides = model
            .ranks
            .iter()
            .flat_map(|r| &r.packets)
            .filter_map(|p| p.draft_stride)
            .collect::<BTreeSet<_>>();
        if strides.len() != 1 || strides.first().copied() != Some(DRAFTS) {
            return Err("Device continuation requires the published physical draft stride".into());
        }
        check(
            &self.save_draft,
            "save_draft",
            owners,
            vec![
                (I, vec![owners, 8], R),
                (I, vec![owners * DRAFTS], R),
                (I, vec![owners, DRAFTS], W),
            ],
        )
    }
}

#[derive(Clone)]
pub(super) struct Prepared {
    pub(super) kernels: Vec<Kernel>,
    pub(super) ids: Buffer,
    pub(super) device_coordinates: bool,
    pub(super) device_embedding: bool,
}
type Binding = (String, usize, Vec<(usize, usize)>);
pub(super) struct Runtime {
    contract: Contract,
    templates: BTreeMap<String, Rc<Template>>,
    banks: BTreeMap<Bank, Buffer>,
    draft_bank: Buffer,
    draft_mask: Buffer,
    draft_cache: Buffer,
    draft_local: Buffer,
    draft_reduced: Buffer,
    rope: Buffer,
    state: Buffer,
    proposals: Buffer,
    mapping: Buffer,
    owners: Buffer,
    ids: Buffer,
    positions: Buffer,
    dummy_swa: Buffer,
    gathered: Buffer,
    output: Buffer,
    history: Buffer,
    epochs: Vec<i32>,
    ready: BTreeSet<usize>,
    drafted: BTreeSet<usize>,
    mapping_bytes: Vec<u8>,
    owner_bytes: Vec<u8>,
    bindings: BTreeMap<Binding, Kernel>,
    lru: VecDeque<Binding>,
}
impl Runtime {
    pub(super) fn load(
        worker: &mut Worker,
        c: &Contract,
        rank: &Rank,
        owners: usize,
        outputs: usize,
    ) -> Result<Self> {
        let session = &worker.session;
        let templates = c
            .manifests()
            // Keep the legacy hash asset in package validation. Host-resident
            // Engram tables use the shared CPU hasher without a device fence.
            .filter(|name| *name != c.hashes)
            .map(|name| {
                Ok((
                    name.to_owned(),
                    Template::load(session, &resolve_file(&worker.package, name)?)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut tables = BTreeMap::new();
        for (tensor, first, columns) in banks(rank) {
            let source = worker.checkpoint.read(&tensor)?;
            if source.dtype != safetensors::Dtype::BF16 || source.shape != [VOCAB, 5120] {
                return Err("Resident embedding checkpoint differs".into());
            }
            let bank = session.allocate(VOCAB * columns * 2)?;
            let span = super::packet::embedding_span(first, columns)?;
            let chunk = (8 << 20) / (columns * 2);
            for start in (0..VOCAB).step_by(chunk) {
                let end = (start + chunk).min(VOCAB);
                let mut bytes = Vec::with_capacity((end - start) * columns * 2);
                for row in start..end {
                    bytes.extend_from_slice(
                        &source.data[row * 10240 + span.start..row * 10240 + span.end],
                    );
                }
                session.upload(&bank, start * columns * 2, &bytes)?.wait()?;
            }
            tables.insert((tensor, first, columns), bank);
        }
        let rope = session.allocate(worker.capacity * 128 * 4)?;
        for (plane, name) in ["base_cos", "base_sin", "long_cos", "long_sin"]
            .iter()
            .enumerate()
        {
            let data = worker
                .rope
                .get(*name)
                .ok_or("Missing resident RoPE table")?;
            super::upload(
                session,
                &rope.slice(plane * worker.capacity * 128, worker.capacity * 128)?,
                data,
            )?;
        }
        let state = session.allocate(owners * 16)?;
        session.fill(&state, 0)?.wait()?;
        let source = worker.checkpoint.read(DRAFT_EMBEDDING)?;
        if source.dtype != safetensors::Dtype::BF16 || source.shape != [VOCAB, WIDTH] {
            return Err("Draft vocabulary embedding checkpoint differs".into());
        }
        let bytes = VOCAB / 8 * WIDTH * 2;
        let draft_bank = session.allocate(bytes)?;
        let start = rank.rank as usize * bytes;
        super::upload(session, &draft_bank, &source.data[start..start + bytes])?;
        let draft_mask = session.allocate(WIDTH * 2)?;
        let start = 128799 * WIDTH * 2;
        super::upload(session, &draft_mask, &source.data[start..start + WIDTH * 2])?;
        Ok(Self {
            contract: c.clone(),
            templates,
            banks: tables,
            draft_bank,
            draft_mask,
            draft_cache: session.allocate(owners * WIDTH * 2)?,
            draft_local: session.allocate(owners * WIDTH * 2)?,
            draft_reduced: session.allocate(owners * WIDTH * 2)?,
            rope,
            state,
            proposals: session.allocate(owners * DRAFTS * 4)?,
            mapping: session.allocate(c.capacity * 32)?,
            owners: session.allocate(owners * 32)?,
            ids: session.allocate(c.capacity * 4)?,
            positions: session.allocate(c.capacity * 4)?,
            dummy_swa: session.allocate(c.capacity * 512)?,
            gathered: session.allocate(8 * outputs * 8)?,
            output: session.allocate(outputs * 8)?,
            history: session.allocate(owners * 16)?,
            epochs: vec![1; owners],
            ready: BTreeSet::new(),
            drafted: BTreeSet::new(),
            mapping_bytes: vec![],
            owner_bytes: vec![],
            bindings: BTreeMap::new(),
            lru: VecDeque::new(),
        })
    }
    pub(super) fn invalidate(&mut self, owner: usize) -> Result<()> {
        let epoch = self
            .epochs
            .get_mut(owner)
            .ok_or("Invalid device frontier owner")?;
        *epoch = epoch
            .checked_add(1)
            .ok_or("Device frontier epoch exhausted")?;
        self.ready.remove(&owner);
        self.drafted.remove(&owner);
        Ok(())
    }
    pub(super) fn invalidate_all(&mut self) -> Result<()> {
        for owner in 0..self.epochs.len() {
            self.invalidate(owner)?;
        }
        Ok(())
    }
    fn bind(
        &mut self,
        session: &Session,
        name: &str,
        ports: &[Buffer],
        rows: usize,
    ) -> Result<Kernel> {
        let template = self
            .templates
            .get(name)
            .ok_or("Missing frontier template")?;
        if ports.len() != template.spec().operands.len() {
            return Err("Device input operand count differs".into());
        }
        // Shared storage has the largest packet capacity. The native ABI
        // consumes exact physical views; real rows remain a launch scalar.
        let ports = ports
            .iter()
            .zip(&template.spec().operands)
            .map(|(buffer, operand)| buffer.slice(0, operand.bytes()?))
            .collect::<Result<Vec<_>>>()?;
        let key = (
            name.to_owned(),
            rows,
            ports
                .iter()
                .map(|b| (b.pointer() as usize, b.bytes()))
                .collect(),
        );
        if let Some(k) = self.bindings.get(&key) {
            return Ok(k.clone());
        }
        let k = Kernel::bind_rows(
            session,
            self.templates
                .get(name)
                .ok_or("Missing frontier template")?,
            &ports,
            rows,
        )?;
        self.bindings.insert(key.clone(), k.clone());
        self.lru.push_back(key);
        while self.lru.len() > 256 {
            if let Some(key) = self.lru.pop_front() {
                self.bindings.remove(&key);
            }
        }
        Ok(k)
    }
    pub(super) fn prepare(
        &mut self,
        worker: &Worker,
        packet: &Packet,
        layout: &Layout,
        tokens: &[Vec<u32>],
        readout: &super::team::Readout,
    ) -> Result<Prepared> {
        // Bounded decoder replay uses a different row domain for prefill. Its
        // existing CPU coordinate path is retained; continuation rows agree in
        // both domains and use the same device positions.
        let device_coordinates = layout.role == PacketRole::Draft
            || layout
                .decoder_roles
                .iter()
                .all(|r| *r == super::replay::Role::Continue)
                && layout.segments.iter().all(|s| s.tokens <= DRAFTS + 1);
        let mut mapping = vec![];
        let mut seeds = vec![];
        for (lane, (segment, input)) in layout.segments.iter().zip(tokens).enumerate() {
            let owner = segment.slot;
            if !readout.continuations.contains(&owner) && self.ready.contains(&owner) {
                self.invalidate(owner)?;
            }
            let base = worker.batch.as_ref().unwrap().owners[&owner].history.len();
            if input.iter().any(|&token| token as usize >= VOCAB)
                || base
                    .checked_add(input.len())
                    .is_none_or(|end| end > worker.capacity)
            {
                return Err("Device input token or position exceeds the model contract".into());
            }
            let mode = if self.ready.contains(&owner)
                && (layout.role == PacketRole::Draft
                    || layout.decoder_roles[lane] == super::replay::Role::Continue)
            {
                if layout.role == PacketRole::Draft {
                    3
                } else if self.drafted.contains(&owner) && segment.tokens > 1 {
                    2
                } else if segment.tokens == 1 {
                    1
                } else {
                    0
                }
            } else {
                0
            };
            if mode == 0 {
                let history = &worker.batch.as_ref().unwrap().owners[&owner].history;
                let mut previous = [-1i32; 4];
                for (i, &token) in history.iter().rev().take(3).enumerate() {
                    previous[i] = token as i32;
                }
                previous[3] = self.epochs[owner];
                seeds.push((
                    self.history.slice(owner * 16, 16)?,
                    previous
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ));
            }
            for (offset, &token) in input.iter().enumerate() {
                // In continuation mode no CPU token or position is transferred.
                let row = [
                    owner as i32,
                    offset as i32,
                    input.len() as i32,
                    mode,
                    if mode == 0 { base as i32 } else { 0 },
                    if mode == 0 { token as i32 } else { 0 },
                    self.epochs[owner],
                    0,
                ];
                mapping.extend(row.into_iter().flat_map(i32::to_le_bytes));
            }
        }
        if mapping != self.mapping_bytes {
            worker
                .session
                .upload_inputs(&[(self.mapping.slice(0, mapping.len())?, mapping.clone())])?;
            self.mapping_bytes = mapping;
        }
        if !seeds.is_empty() {
            worker.session.upload_inputs(&seeds)?;
        }
        let prep = self
            .contract
            .preparations
            .get(&geometry(packet, &packet.coordinates))
            .ok_or("Missing device input preparation")?
            .clone();
        let mut kernels = vec![self.bind(
            &worker.session,
            &prep.tokens,
            &[
                self.mapping.clone(),
                self.state.clone(),
                self.proposals.clone(),
                self.ids.clone(),
                self.positions.clone(),
            ],
            layout.rows.total,
        )?];
        if device_coordinates {
            for names in [&packet.coordinates]
                .into_iter()
                .chain(packet.decoder_coordinates.as_ref())
            {
                let prep = self
                    .contract
                    .preparations
                    .get(&geometry(packet, names))
                    .ok_or("Missing device coordinate preparation")?
                    .clone();
                let mut ports = vec![
                    self.mapping.clone(),
                    self.positions.clone(),
                    self.rope.clone(),
                ];
                for name in [
                    &names.base_cos,
                    &names.base_sin,
                    &names.long_cos,
                    &names.long_sin,
                    &names.source_cos,
                    &names.source_sin,
                    &names.slots,
                    &names.c2_slots,
                    &names.swa_indices,
                ] {
                    ports.push(get(&worker.buffers, name)?);
                }
                ports.push(
                    names
                        .source_swa_indices
                        .as_ref()
                        .map(|n| get(&worker.buffers, n))
                        .transpose()?
                        .unwrap_or_else(|| self.dummy_swa.clone()),
                );
                ports.push(get(&worker.buffers, &names.ring)?);
                ports.push(get(&worker.buffers, &names.premix)?);
                kernels.push(self.bind(
                    &worker.session,
                    &prep.coordinates,
                    &ports,
                    layout.rows.total,
                )?);
            }
        }
        Ok(Prepared {
            kernels,
            ids: self.ids.clone(),
            device_coordinates,
            device_embedding: layout.role != PacketRole::Draft
                || layout.segments.iter().all(|s| {
                    self.ready.contains(&s.slot) && readout.continuations.contains(&s.slot)
                }),
        })
    }
    pub(super) fn embedding(
        &mut self,
        session: &Session,
        step: &super::Step,
        packet: &Packet,
        rows: usize,
    ) -> Result<Kernel> {
        let super::Step::Embed {
            tensor,
            first_column,
            columns,
            destination,
        } = step
        else {
            return Err("Device embedding requires an embedding step".into());
        };
        if packet.role == PacketRole::Draft {
            let name = self.contract.draft_embedding.clone();
            return self.bind(
                session,
                &name,
                &[
                    self.mapping.clone(),
                    self.draft_cache.clone(),
                    self.draft_mask.clone(),
                    destination.clone(),
                ],
                rows,
            );
        }
        let name = self
            .contract
            .embeddings
            .get(&format!("{}:{columns}", packet.rows))
            .ok_or("Missing device embedding template")?
            .clone();
        let bank = self
            .banks
            .get(&(tensor.to_owned(), *first_column, *columns))
            .ok_or("Missing resident embedding bank")?
            .clone();
        self.bind(
            session,
            &name,
            &[bank, self.ids.clone(), destination.clone()],
            rows,
        )
    }
    fn owner_mapping(&self, layout: &Layout) -> Vec<u8> {
        layout
            .segments
            .iter()
            .map(|s| {
                [
                    s.slot as i32,
                    0,
                    s.tokens as i32,
                    0,
                    0,
                    0,
                    self.epochs[s.slot],
                    0,
                ]
            })
            .flat_map(|r| r.into_iter().flat_map(i32::to_le_bytes))
            .collect()
    }
    fn upload_owners(&mut self, session: &Session, bytes: Vec<u8>) -> Result<()> {
        if bytes != self.owner_bytes {
            session.upload_inputs(&[(self.owners.slice(0, bytes.len())?, bytes.clone())])?;
            self.owner_bytes = bytes;
        }
        Ok(())
    }
    pub(super) fn save_draft(
        &mut self,
        worker: &Worker,
        packet: &Packet,
        layout: &Layout,
    ) -> Result<()> {
        let mapping = self.owner_mapping(layout);
        self.upload_owners(&worker.session, mapping)?;
        let name = self.contract.save_draft.clone();
        self.bind(
            &worker.session,
            &name,
            &[
                self.owners.clone(),
                get(&worker.buffers, &packet.best_index)?,
                self.proposals.clone(),
            ],
            layout.segments.len(),
        )?
        .enqueue(&worker.session)?;
        self.drafted.extend(layout.segments.iter().map(|s| s.slot));
        Ok(())
    }
    pub(super) fn commit(
        &mut self,
        worker: &Worker,
        layout: &Layout,
        samples: &Buffer,
        readout: &super::team::Readout,
    ) -> Result<()> {
        let rows = layout.outputs.len();
        if rows == 0 {
            return Ok(());
        }
        worker.comm.all_gather(
            &worker.session,
            samples,
            &self.gathered,
            (rows * 2) as u64,
            Element::Float32,
        )?;
        let name = self.contract.merge.clone();
        self.bind(
            &worker.session,
            &name,
            &[self.gathered.clone(), self.output.clone()],
            rows,
        )?
        .enqueue(&worker.session)?;
        let mut entries = vec![];
        let mut at = 0;
        while at < rows {
            let lane = layout.outputs[at];
            let segment = &layout.segments[lane];
            let mut end = at + 1;
            while end < rows && layout.outputs[end] == lane {
                end += 1;
            }
            let count = end - at;
            if count > DRAFTS + 1
                || count > 1 && !self.drafted.contains(&segment.slot)
                || !readout.draws.contains_key(&segment.slot)
                || readout.wants_logits(segment.slot)
            {
                self.invalidate(segment.slot)?;
                at = end;
                continue;
            }
            let position = layout.rows.starts[lane] + layout.output_positions[at];
            entries.extend(
                [
                    segment.slot as i32,
                    at as i32,
                    count as i32,
                    worker.rank as i32,
                    position as i32,
                    layout.rows.starts[lane] as i32,
                    self.epochs[segment.slot],
                    0,
                ]
                .into_iter()
                .flat_map(i32::to_le_bytes),
            );
            self.ready.insert(segment.slot);
            self.drafted.remove(&segment.slot);
            at = end;
        }
        if !entries.is_empty() {
            let count = entries.len() / 32;
            self.upload_owners(&worker.session, entries)?;
            let name = self.contract.commit.clone();
            self.bind(
                &worker.session,
                &name,
                &[
                    self.owners.clone(),
                    self.output.clone(),
                    self.positions.clone(),
                    self.proposals.clone(),
                    self.state.clone(),
                    self.ids.clone(),
                    self.history.clone(),
                    self.draft_bank.clone(),
                    self.draft_local.clone(),
                ],
                count,
            )?
            .enqueue(&worker.session)?;
            // Only the selected token's vocabulary owner contributes a value;
            // all other ranks contribute zero. One BF16 value plus zeros
            // preserves its value and keeps communication bounded by owners.
            worker.comm.all_reduce_sum(
                &worker.session,
                &self.draft_local,
                &self.draft_reduced,
                (count * WIDTH) as u64,
                Element::Bfloat16,
            )?;
            let name = self.contract.scatter_embedding.clone();
            self.bind(
                &worker.session,
                &name,
                &[
                    self.owners.clone(),
                    self.draft_reduced.clone(),
                    self.draft_cache.clone(),
                ],
                count,
            )?
            .enqueue(&worker.session)?;
        }
        Ok(())
    }
}
