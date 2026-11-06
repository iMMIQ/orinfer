//! Prepared physical weight loading and bounded stream-ordered transfers.
use super::*;
use std::collections::VecDeque;
pub(super) mod pipeline;

/// In-flight pinned storage stays within the session budget as DMA overlaps host copies.
const UPLOAD_CHUNK: usize = 8 << 20;
const UPLOAD_DEPTH: usize = 6;

pub(super) fn host_memory_budget(name: &str, default_mib: usize) -> Result<usize> {
    let mib = match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("Invalid {name} MiB"))?,
        Err(std::env::VarError::NotPresent) => default_mib,
        Err(_) => return Err(format!("Invalid {name} environment")),
    };
    mib.checked_mul(1024 * 1024)
        .ok_or_else(|| format!("{name} budget overflow"))
}
fn wait_in_flight(tickets: &mut VecDeque<Ticket>) -> Result<()> {
    while let Some(ticket) = tickets.pop_front() {
        ticket.wait()?;
    }
    Ok(())
}

/// Pipelined upload: submitted chunks overlap host copies with in-flight DMA.
pub(super) fn upload(session: &Session, buffer: &Buffer, data: &[u8]) -> Result<()> {
    if data.len() != buffer.bytes() {
        return Err("Model upload byte extent differs".into());
    }
    let mut tickets = VecDeque::new();
    for (chunk, bytes) in data.chunks(UPLOAD_CHUNK).enumerate() {
        tickets.push_back(session.upload(buffer, chunk * UPLOAD_CHUNK, bytes)?);
        if tickets.len() > UPLOAD_DEPTH {
            tickets.pop_front().unwrap().wait()?;
        }
    }
    wait_in_flight(&mut tickets)
}

pub(super) fn weight_layouts(
    definitions: &[Definition],
) -> Result<Vec<crate::backend::arena::ArenaLayout>> {
    let mut layouts = Vec::new();
    let mut group = Vec::new();
    let mut bytes = 0usize;
    for definition in definitions.iter().filter(|d| {
        d.view.is_none() && (d.prepared_tensor.is_some() || d.prepared_weight.is_some())
    }) {
        if !group.is_empty() && bytes.saturating_add(definition.bytes) > 1 << 30 {
            layouts.push(crate::backend::arena::ArenaLayout::from_sizes(
                group.drain(..),
            )?);
            bytes = 0;
        }
        bytes = bytes
            .checked_add(definition.bytes)
            .ok_or("Weight arena extent overflow")?;
        // HUGE_FIRST places large standalone weights on large-page boundaries.
        // Preserve that property inside the shared allocation as well.
        let alignment = if definition.bytes >= 2 << 20 {
            2 << 20
        } else {
            512
        };
        group.push((definition.name.as_str(), definition.bytes, alignment));
    }
    if !group.is_empty() {
        layouts.push(crate::backend::arena::ArenaLayout::from_sizes(
            group.into_iter(),
        )?);
    }
    Ok(layouts)
}

pub(super) fn weight_buffers(
    session: &Session,
    definitions: &[Definition],
    layouts: &[crate::backend::arena::ArenaLayout],
) -> Result<BTreeMap<String, Buffer>> {
    let sizes: BTreeMap<_, _> = definitions
        .iter()
        .map(|d| (d.name.as_str(), d.bytes))
        .collect();
    let mut buffers = BTreeMap::new();
    for layout in layouts {
        let owner = session.allocate(layout.allocation_bytes)?;
        let base = (layout.base(owner.pointer() as u64)? - owner.pointer() as u64) as usize;
        for (name, offset) in &layout.offsets {
            buffers.insert(
                name.clone(),
                owner.slice(base + offset, sizes[name.as_str()])?,
            );
        }
    }
    Ok(buffers)
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        session: Session,
        root: &RootInfo,
        rank: Rank,
        model: &Model,
        model_dir: &Path,
        package: &Path,
        layout: HashLayout,
        snapshots: bool,
        shared_tables: &SharedTables,
    ) -> Result<Self> {
        let start = Instant::now();
        let mut phases = LoadPhases::default();
        let t = Instant::now();
        let comm = Communicator::join(&session, root, rank.rank)?;
        phases.hccl_s = t.elapsed().as_secs_f64();
        let mut checkpoint = Checkpoint::open(model_dir, &model.index.file)?;
        let views = rank
            .buffers
            .iter()
            .filter_map(|definition| {
                definition
                    .view
                    .clone()
                    .map(|view| (definition.name.clone(), (view, definition.bytes)))
            })
            .collect();
        let mut buffers = BTreeMap::new();
        let needed = rank
            .buffers
            .iter()
            .filter(|b| b.view.is_none())
            .try_fold(0usize, |n, b| {
                n.checked_add(b.bytes).ok_or("Model byte estimate overflow")
            })?;
        let layouts = weight_layouts(&rank.buffers)?;
        let padding = layouts.iter().try_fold(0usize, |n, l| {
            n.checked_add(l.padding_bytes)
                .ok_or("Weight arena padding overflow")
        })?;
        let needed = needed
            .checked_add(padding)
            .ok_or("Weight arena budget overflow")?;
        let needed = needed
            .checked_add(
                model
                    .vision
                    .as_ref()
                    .map(vision::Program::device_bytes)
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or("Model and vision memory size overflow")?;
        if needed > session.memory_info()?.0.saturating_sub(1 << 30) {
            return Err(format!(
                "Rank {} needs {needed} bytes plus 1 GiB headroom",
                rank.rank
            ));
        }
        let t = Instant::now();
        let mut weight_buffers = weight_buffers(&session, &rank.buffers, &layouts)?;
        phases.alloc_s += t.elapsed().as_secs_f64();
        let mut fills: Vec<Ticket> = Vec::new();
        let mut prepared = Vec::new();
        let growth = model
            .batch
            .as_ref()
            .map(|contract| super::memory::growth_layouts(contract, &rank.buffers))
            .transpose()?
            .unwrap_or_default();
        for definition in rank.buffers {
            if buffers.contains_key(&definition.name) || definition.bytes == 0 {
                return Err("Duplicate/empty model buffer".into());
            }
            let buffer = if let Some(view) = definition.view {
                if definition.weight.is_some()
                    || definition.data.is_some()
                    || definition.prepared_weight.is_some()
                    || definition.prepared_tensor.is_some()
                {
                    return Err("Model view cannot own weight data".into());
                }
                get(&buffers, &view.parent)?.slice(view.offset, definition.bytes)?
            } else {
                let t = Instant::now();
                let demand = growth.contains_key(&definition.name);
                let buffer = if let Some(buffer) = weight_buffers.remove(&definition.name) {
                    if demand {
                        return Err("Resident weight cannot be a growing state buffer".into());
                    }
                    buffer
                } else if demand {
                    session.reserve(definition.bytes)?
                } else {
                    session.allocate(definition.bytes)?
                };
                phases.alloc_s += t.elapsed().as_secs_f64();
                if let Some(file) = definition.prepared_weight {
                    let path = resolve_file(package, &file)?;
                    let metadata = std::fs::metadata(&path).map_err(|e| e.to_string())?;
                    if !metadata.is_file() || metadata.len() != definition.bytes as u64 {
                        return Err("Prepared weight metadata differs".into());
                    }
                    prepared.push(pipeline::Upload {
                        source: crate::weights::TensorSource {
                            path,
                            offset: 0,
                            bytes: definition.bytes,
                        },
                        buffer: buffer.clone(),
                    });
                } else if let Some(tensor) = definition.prepared_tensor {
                    let source = checkpoint.source(&tensor.name, &tensor.dtype, &tensor.shape)?;
                    if source.bytes != definition.bytes {
                        return Err("Prepared tensor byte extent differs".into());
                    }
                    prepared.push(pipeline::Upload {
                        source,
                        buffer: buffer.clone(),
                    });
                } else if let Some(weight) = definition.weight {
                    let t = Instant::now();
                    let data = weights::load(&mut checkpoint, &weight)?;
                    upload(&session, &buffer, &data)?;
                    phases
                        .matrix
                        .add(definition.bytes, t.elapsed().as_secs_f64());
                } else if let Some(identity) = definition.data {
                    let t = Instant::now();
                    let data = read_asset(package, &identity)?;
                    upload(&session, &buffer, &data)?;
                    phases.data.add(definition.bytes, t.elapsed().as_secs_f64());
                } else if !demand {
                    // Fills are compute-stream ordered before every later use;
                    // deferring their waits keeps the transfer stream busy.
                    let t = Instant::now();
                    fills.push(session.fill(&buffer, 0)?);
                    phases.zero.add(definition.bytes, t.elapsed().as_secs_f64());
                }
                buffer
            };
            buffers.insert(definition.name, buffer);
        }
        let t = Instant::now();
        let loaded = pipeline::run(&session, prepared)?;
        phases.prepared.add(loaded.bytes, t.elapsed().as_secs_f64());
        eprintln!(
            "A2 rank {} weights ready after {:.3}s; arenas={}; padding_bytes={padding}; read_chunks={}; read_wait={:.3}s; upload={:.3}s",
            rank.rank,
            start.elapsed().as_secs_f64(),
            layouts.len(),
            loaded.chunks,
            loaded.read_wait_s,
            loaded.upload_s
        );
        let t = Instant::now();
        for ticket in &fills {
            ticket.wait()?;
        }
        phases.zero.seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let tables = shared_tables
            .get_or_init(|| tables(model_dir, &model.index.file, &layout))
            .as_ref()
            .map_err(Clone::clone)?;
        let tables = Arc::clone(tables);
        let rope = model
            .rope
            .iter()
            .map(|(name, id)| Ok((name.clone(), read_asset(package, id)?)))
            .collect::<Result<_>>()?;
        phases.tables_s = t.elapsed().as_secs_f64();
        let batch = model.batch.as_ref().map(|contract| {
            batch::Runtime::new(
                contract.clone(),
                model.packet_cache_capacity,
                &rank.packets,
                &model.graph_launches,
            )
        });
        let vision = model
            .vision
            .as_ref()
            .map(|program| {
                vision::Encoder::from_program(&session, program, package, &mut checkpoint)
            })
            .transpose()?;
        let mut worker = Self {
            vision,
            growth,
            session,
            comm,
            buffers,
            views,
            row_selections: model.row_selections.clone(),
            w4: BTreeMap::new(),
            routed_w4: BTreeMap::new(),
            packets: BTreeMap::new(),
            packet_order: row_contract::PacketLru::new(model.packet_cache_capacity),
            packet_templates: rank.packets.into_iter().map(Rc::new).collect(),
            package: package.to_owned(),
            snapshots,
            checkpoint,
            hasher: Hasher::new(layout)?,
            tables,
            rope,
            capacity: model.capacity,
            rank: rank.rank,
            templates: BTreeMap::new(),
            template_aliases: BTreeMap::new(),
            phases,
            steps: BTreeMap::new(),
            batch,
        };
        if model.batch.is_some() {
            return Ok(worker);
        }
        for packet in worker.packet_templates.clone() {
            let actual = packet.dispatch_max();
            let body = worker.compile(&packet.body, package, snapshots, (packet.rows, actual))?;
            let head = worker.compile(&packet.head, package, snapshots, (packet.rows, actual))?;
            if worker
                .packets
                .insert(
                    actual,
                    Rc::new(Execution {
                        packet,
                        body,
                        head,
                        captured: true,
                        direct_uses: Cell::new(0),
                    }),
                )
                .is_some()
            {
                return Err("Duplicate packet row count".into());
            }
            worker.touch_packet(actual);
        }
        let phases = &worker.phases;
        eprintln!(
            "A2 rank {} load {:.1}s: hccl={:.1}s alloc={:.1}s prepared={} matrix={} data={} zero={} tables={:.1}s templates={}/{:.1}s binds={}/{:.1}s aclnn={}/{:.1}s warmup={}/{:.1}s captures={}/{:.1}s",
            rank.rank,
            start.elapsed().as_secs_f64(),
            phases.hccl_s,
            phases.alloc_s,
            phases.prepared.report(),
            phases.matrix.report(),
            phases.data.report(),
            phases.zero.report(),
            phases.tables_s,
            phases.templates.0,
            phases.templates.1,
            phases.binds.0,
            phases.binds.1,
            phases.aclnn.0,
            phases.aclnn.1,
            phases.warmups.0,
            phases.warmups.1,
            phases.captures.0,
            phases.captures.1,
        );
        Ok(worker)
    }
}

pub(super) fn tables(
    model_dir: &Path,
    index: &str,
    layout: &HashLayout,
) -> Result<Arc<BTreeMap<usize, SharedInt8Rows>>> {
    let started = Instant::now();
    let checkpoint = Checkpoint::open(model_dir, index)?;
    let mut stores = BTreeMap::new();
    for layer in &layout.layers {
        let prefix = format!("layers.{}.engram.embed", layer.layer);
        let shard = Shard {
            first: 0,
            rows: layer.rows,
        };
        let store = if let Some(records) = checkpoint.row_records(&prefix)? {
            Int8Rows::open_records(&records, shard)?
        } else {
            Int8Rows::open_sharded(
                &checkpoint.row_sources(&format!("{prefix}.weight"))?,
                &checkpoint.row_sources(&format!("{prefix}.scale"))?,
                shard,
            )?
        };
        stores.insert(layer.layer, store);
    }
    let mut resident_budget = host_memory_budget("ORINFER_A2_HOST_TABLE_MIB", 0)?;
    let resident = resident_budget != 0;
    let required = stores
        .values()
        .try_fold(0usize, |total, store| -> Result<usize> {
            total
                .checked_add(store.required_resident_bytes()?)
                .ok_or("Host table total extent overflow".into())
        })?;
    if resident && required > resident_budget {
        return Err("A2 resident host tables exceed configured budget".into());
    }
    let table_budget = if resident {
        0
    } else {
        host_memory_budget("ORINFER_A2_HOST_ROW_CACHE_MIB", 2048)? / layout.layers.len().max(1)
    };
    let mut tables = BTreeMap::new();
    for (layer, mut store) in stores {
        if resident {
            store = store.into_resident(resident_budget)?;
            resident_budget -= store.resident_bytes();
        }
        tables.insert(layer, SharedInt8Rows::with_cache(store, table_budget)?);
    }
    eprintln!(
        "A2 HOST TABLES READY after {:.3}s; resident_bytes={}",
        started.elapsed().as_secs_f64(),
        tables
            .values()
            .map(|t| t.statistics().map(|s| s.resident_bytes).unwrap_or(0))
            .sum::<usize>()
    );
    Ok(Arc::new(tables))
}
