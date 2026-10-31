//! Native A2 plan identities, physical views and execution contract validation.
use super::*;

pub(super) fn load_spec(path: &Path, model_dir: &Path) -> Result<(Model, PathBuf, HashLayout)> {
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    let base = path.parent().ok_or("Model plan has no parent")?.to_owned();
    let mut model: Model = crate::model::read(&path)?;
    if model.schema_version != 1
        || model.layers != 40
        || model.ranks.len() != 8
        || !(128..=262144).contains(&model.capacity)
        || !model.capacity.is_multiple_of(32)
        || !(4..=4096).contains(&model.packet_cache_capacity)
    {
        return Err("Expected complete 40-layer TP8 A2 plan with capacity <=262144".into());
    }
    let layout: HashLayout =
        serde_json::from_slice(&read_asset(&base, &model.engram)?).map_err(|e| e.to_string())?;
    if layout.config_sha256 != model.config.sha256 {
        return Err("Engram/config identity differs".into());
    }
    preflight(&model, &base)?;
    let mut checkpoint = Checkpoint::open(model_dir, &model.index.file)?;
    let mut checked = std::collections::BTreeSet::new();
    for definition in model.ranks.iter().flat_map(|r| &r.buffers) {
        if let Some(tensor) = &definition.prepared_tensor
            && checked.insert((&tensor.name, &tensor.dtype, &tensor.shape, definition.bytes))
        {
            let source = checkpoint.source(&tensor.name, &tensor.dtype, &tensor.shape)?;
            if source.bytes != definition.bytes {
                return Err("Prepared tensor byte extent differs".into());
            }
        }
    }
    model.graph_launches = batch::graph_launches(&model);
    Ok((model, base, layout))
}

/// Reject incomplete or mismatched packages before allocating resident weights.
pub(super) fn preflight(model: &Model, base: &Path) -> Result<()> {
    batch::validate(model)?;
    let mut kernels = BTreeMap::<String, super::super::kernels::Spec>::new();
    let used: std::collections::BTreeSet<_> = model
        .ranks
        .iter()
        .flat_map(|r| &r.packets)
        .flat_map(|p| p.body.iter().chain(&p.head))
        .filter_map(|i| match i {
            Instruction::Kernel {
                manifest,
                uncompacted_manifest,
                ..
            } => Some(
                [Some(manifest.clone()), uncompacted_manifest.clone()]
                    .into_iter()
                    .flatten(),
            ),
            _ => None,
        })
        .flatten()
        .collect();
    let mut names = used.clone();
    for (slot, variants) in &model.row_selections {
        if !used.contains(slot) || variants.is_empty() {
            return Err("A2 row selection requires a used canonical slot".into());
        }
        let mut previous = 0;
        for variant in variants {
            if variant.min_rows <= previous
                || variant.min_rows > variant.max_rows
                || variant.max_rows > 4096
                || variant.multiple_of != 1
                || variant.kernels.len() != 1
            {
                return Err("Invalid or overlapping A2 row selections".into());
            }
            previous = variant.max_rows;
            names.extend(variant.kernels.iter().cloned());
        }
    }
    for name in names {
        let path = resolve_file(base, &name)?;
        let spec: super::super::kernels::Spec = crate::model::read(&path)?;
        spec.validate()?;
        spec.target.require_host(std::env::consts::ARCH)?;
        kernels.insert(name, spec);
    }
    for (slot, variants) in &model.row_selections {
        let source = &kernels[slot];
        let capacity = source
            .row_capacity()?
            .ok_or("A2 row selections require dynamic kernels")?;
        for variant in variants {
            let replacement = &kernels[&variant.kernels[0]];
            row_contract::validate_replacement(source, replacement)?;
            if variant.max_rows > capacity {
                return Err("A2 row selection exceeds its operator capacity".into());
            }
        }
    }
    for (rank_index, rank) in model.ranks.iter().enumerate() {
        if rank.rank as usize != rank_index {
            return Err("Rank order differs from TP topology".into());
        }
        let mut buffers = BTreeMap::new();
        for definition in &rank.buffers {
            if definition.bytes == 0 || buffers.contains_key(&definition.name) {
                return Err("Invalid/duplicate model buffer".into());
            }
            if let Some(view) = &definition.view {
                row_contract::view_offset(view, view.row_capacity.unwrap_or(1))?;
                let bytes = *buffers
                    .get(&view.parent)
                    .ok_or("Model view precedes its allocation")?;
                if view
                    .offset
                    .checked_add(definition.bytes)
                    .is_none_or(|end| end > bytes)
                    || definition.weight.is_some()
                    || definition.data.is_some()
                    || definition.prepared_weight.is_some()
                    || definition.prepared_tensor.is_some()
                {
                    return Err("Invalid model tensor view".into());
                }
            }
            let sources = usize::from(definition.weight.is_some())
                + usize::from(definition.data.is_some())
                + usize::from(definition.prepared_weight.is_some())
                + usize::from(definition.prepared_tensor.is_some());
            if sources > 1 {
                return Err("Model buffer has multiple weight sources".into());
            }
            buffers.insert(definition.name.clone(), definition.bytes);
            if let Some(file) = &definition.prepared_weight {
                let path = resolve_file(base, file)?;
                let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
                if !metadata.is_file()
                    || metadata.len() != definition.bytes as u64
                    || definition.weight.is_some()
                    || definition.data.is_some()
                {
                    return Err("Prepared weight metadata differs".into());
                }
            }
        }
        if model.batch.is_none() {
            validate_dispatch(&rank.packets)?;
        }
        if rank
            .packets
            .iter()
            .map(|p| (p.rows, p.min_rows, p.max_rows))
            .collect::<Vec<_>>()
            != model.ranks[0]
                .packets
                .iter()
                .map(|p| (p.rows, p.min_rows, p.max_rows))
                .collect::<Vec<_>>()
        {
            return Err("TP ranks have different row contracts".into());
        }
        for packet in &rank.packets {
            replay::validate_packet(packet, model.batch.as_ref())?;
            for coordinates in
                std::iter::once(&packet.coordinates).chain(packet.decoder_coordinates.as_ref())
            {
                if coordinates
                    .names()
                    .iter()
                    .any(|name| !buffers.contains_key(*name))
                {
                    return Err("Packet coordinate buffer is not defined".into());
                }
            }
            branches::validate_instructions(&packet.body)?;
            branches::validate_instructions(&packet.head)?;
            if packet.rows == 0
                || packet.rows > 4096
                || packet.rows > model.capacity
                || packet.dispatch_max() > packet.rows
            {
                return Err("Invalid packet row capacity".into());
            }
            let layers = packet
                .body
                .iter()
                .filter_map(|step| {
                    if let Instruction::Snapshot { name, .. } = step
                        && !model
                            .batch
                            .as_ref()
                            .and_then(|c| c.decoder_replay.as_ref())
                            .and_then(|c| c.source_ready.as_ref())
                            .is_some_and(|s| s == name)
                    {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
                .collect::<std::collections::BTreeSet<_>>();
            if layers != (0..40).map(|i| format!("layer{i:02}")).collect() {
                return Err("A2 execution packet does not cover all 40 layers".into());
            }
            let mut current_capacity = packet.rows;
            for instruction in packet.body.iter().chain(&packet.head) {
                if let Instruction::Embed {
                    destination,
                    first_column,
                    columns,
                } = instruction
                {
                    let span = packet::embedding_span(*first_column, *columns)?;
                    if buffers.get(destination).copied() != Some(current_capacity * span.len()) {
                        return Err("Embedding input geometry differs".into());
                    }
                }
                if let Instruction::LimitRows { rows } = instruction {
                    if packet.min_rows == 0 {
                        return Err("Decode packets cannot truncate their rows".into());
                    }
                    bounded_rows(current_capacity, current_capacity, *rows)?;
                    current_capacity = *rows;
                }
                if let Instruction::Kernel {
                    manifest,
                    uncompacted_manifest,
                    buffers: bindings,
                    fixed_rows,
                    row_expression,
                    owner_work,
                    owners,
                } = instruction
                {
                    let spec = &kernels[manifest];
                    if let Some(fallback) = uncompacted_manifest {
                        replay::validate_fallback(spec, &kernels[fallback])?;
                    }
                    let expected = if owner_work.is_some() {
                        4096
                    } else if *owners {
                        model
                            .batch
                            .as_ref()
                            .ok_or("Owner head requires batch contract")?
                            .owners
                    } else {
                        row_contract::kernel_rows(
                            current_capacity,
                            current_capacity,
                            *fixed_rows,
                            row_expression.as_ref(),
                        )?
                    };
                    if owner_work.is_some() != spec.op.starts_with("private_rows.") {
                        return Err("Owner dispatch and AOT identity differ".into());
                    }
                    if let Some(grid) = owner_work {
                        let c = model.batch.as_ref().ok_or("Missing owner contract")?;
                        let owner_capacity = bindings
                            .first()
                            .map(|table| c.token_capacity(table, packet.rows))
                            .transpose()?
                            .unwrap_or(packet.rows);
                        if *owners
                            || fixed_rows.is_some()
                            || row_expression.is_some()
                            || bindings.len() < 3
                            || c.work.get(&bindings[0]) != Some(grid)
                            || !c.pointers.contains_key(&bindings[1])
                            || bindings[2] != c.active
                            || spec.parameters["_owner_token_capacity"].as_u64()
                                != Some(owner_capacity as u64)
                        {
                            return Err("Owner kernel/table/row contract differs".into());
                        }
                        let accesses: Vec<batch::Access> =
                            serde_json::from_value(spec.parameters["_owner_access"].clone())
                                .map_err(|e| e.to_string())?;
                        let mapping = &c.pointers[&bindings[1]];
                        if accesses.is_empty()
                            || !accesses.iter().any(|a| a.write)
                            || accesses.iter().any(|a| {
                                a.column >= mapping.len()
                                    || c.columns[mapping[a.column]]
                                        .decode
                                        .is_some_and(|d| d != packet.batch_decode)
                            })
                        {
                            return Err("Invalid owner pointee access contract".into());
                        }
                    }

                    if spec
                        .row_capacity()?
                        .is_some_and(|cap| packet.min_rows == 0 || cap != expected)
                        || ((fixed_rows.is_some() || row_expression.is_some())
                            && spec.row_capacity()?.is_none())
                    {
                        return Err("Packet/AOT dynamic row capacities differ".into());
                    }
                    if bindings.len() != spec.operands.len() {
                        return Err("Packet/AOT operand count differs".into());
                    }
                    for (binding, operand) in bindings.iter().zip(&spec.operands) {
                        let bytes = buffers.get(binding).copied();
                        let required = operand.bytes()?;
                        let compatible = if operand.access == super::super::kernels::Access::Scratch
                        {
                            bytes.is_some_and(|size| size >= required)
                        } else {
                            bytes == Some(required)
                        };
                        if !compatible {
                            return Err(format!("{binding}: packet/AOT byte extent differs"));
                        }
                    }
                    for variant in model.row_selections.get(manifest).into_iter().flatten() {
                        for (binding, operand) in
                            bindings.iter().zip(&kernels[&variant.kernels[0]].operands)
                        {
                            if operand.access == super::super::kernels::Access::Scratch
                                && buffers[binding] < operand.bytes()?
                            {
                                return Err(format!(
                                    "{binding}: selected scratch exceeds allocation"
                                ));
                            }
                        }
                    }
                }
                if let Instruction::Gather { fp32, int32, .. }
                | Instruction::Exchange { fp32, int32, .. } = instruction
                {
                    gather_element(*fp32, *int32)?;
                }
                if let Instruction::Gather {
                    elements,
                    fixed_rows,
                    elements_expression,
                    ..
                }
                | Instruction::Reduce {
                    elements,
                    fixed_rows,
                    elements_expression,
                    ..
                }
                | Instruction::Exchange {
                    elements,
                    fixed_rows,
                    elements_expression,
                    ..
                } = instruction
                {
                    row_contract::elements(
                        usize::try_from(*elements).map_err(|_| "Collective count overflow")?,
                        current_capacity,
                        current_capacity,
                        *fixed_rows,
                        elements_expression.as_ref(),
                    )?;
                }
                if let Instruction::Exchange {
                    source,
                    destination,
                    elements,
                    fp32,
                    int32,
                    ..
                } = instruction
                {
                    let bytes = elements
                        .checked_mul(gather_element(*fp32, *int32)?.bytes())
                        .and_then(|v| v.checked_mul(model.ranks.len() as u64))
                        .ok_or("Model exchange extent overflow")?;
                    if *elements == 0
                        || buffers
                            .get(source.as_str())
                            .is_none_or(|&v| bytes > v as u64)
                        || buffers
                            .get(destination.as_str())
                            .is_none_or(|&v| bytes > v as u64)
                    {
                        return Err("Model exchange exceeds its buffers".into());
                    }
                }
                if let Instruction::W4 {
                    rows,
                    input,
                    output,
                    experts,
                    name,
                    buffers: bindings,
                    workspace,
                    groups,
                    expected_expert_rows,
                    weight_layout,
                    ..
                } = instruction
                {
                    if *experts != 48
                        || *rows != current_capacity * 6
                        || !input.is_multiple_of(256)
                        || !output.is_multiple_of(32)
                    {
                        return Err("W4 model geometry differs from the routed packet".into());
                    }
                    w4_plan_rows(
                        *rows,
                        *rows,
                        *expected_expert_rows,
                        groups.is_some(),
                        packet.body.iter().chain(&packet.head).any(|step| {
                            matches!(step, Instruction::Route { groups, .. } if groups.contains(name))
                        }),
                    )?;
                    if expected_expert_rows.is_some()
                        && bindings
                            .iter()
                            .chain(groups.iter())
                            .chain([workspace])
                            .any(|name| {
                                rank.buffers
                                    .iter()
                                    .any(|b| b.name == *name && b.view.is_some())
                            })
                    {
                        return Err("Capacity W4 operands require dedicated allocations".into());
                    }
                    let packed_shape = match weight_layout {
                        W4Layout::Nd => vec![*experts, *input, output / 2],
                        W4Layout::TransposedNz => {
                            vec![*experts, input / 64, output / 16, 16, 32]
                        }
                    };
                    let packed = rank.buffers.iter().find(|b| b.name == bindings[1]);
                    if packed.is_none_or(|b| {
                        b.prepared_tensor
                            .as_ref()
                            .map_or(*weight_layout != W4Layout::Nd, |t| {
                                t.dtype != "I8" || t.shape != packed_shape
                            })
                    }) {
                        return Err("W4 prepared tensor differs from its physical layout".into());
                    }
                    let extents = [
                        rows * input,
                        experts * input * output / 2,
                        experts * output * 8,
                        experts * output * 4,
                        rows * 4,
                        rows * output * 2,
                    ];
                    if bindings
                        .iter()
                        .zip(extents)
                        .any(|(name, bytes)| buffers.get(name).copied() != Some(bytes))
                        || buffers.get(workspace).is_none_or(|&bytes| {
                            bytes
                                < if *weight_layout == W4Layout::Nd {
                                    100663808
                                } else {
                                    1
                                }
                        })
                        || groups
                            .as_ref()
                            .is_some_and(|name| buffers.get(name).copied() != Some(experts * 8))
                    {
                        return Err("W4 packet operand/workspace extent differs".into());
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn gather_element(fp32: bool, int32: bool) -> Result<Element> {
    match (fp32, int32) {
        (true, false) => Ok(Element::Float32),
        (false, true) => Ok(Element::Int32),
        (false, false) => Ok(Element::Bfloat16),
        (true, true) => Err("AllGather has conflicting element types".into()),
    }
}

pub(super) fn w4_plan_rows(
    capacity: usize,
    actual: usize,
    hint: Option<usize>,
    device_groups: bool,
    host_routed: bool,
) -> Result<usize> {
    if hint.is_some_and(|n| n == 0 || i64::try_from(n).is_err()) {
        return Err("Invalid W4 expert row estimate".into());
    }
    if hint.is_some() {
        if !device_groups || host_routed {
            return Err("Capacity W4 execution requires device-owned routing counts".into());
        }
        Ok(capacity)
    } else {
        Ok(actual)
    }
}

pub(super) fn bounded_rows(actual: usize, capacity: usize, limit: usize) -> Result<usize> {
    if actual == 0 || actual > capacity || limit == 0 || limit > capacity {
        return Err("Compacted row extent exceeds its packet capacity".into());
    }
    Ok(actual.min(limit))
}

pub(super) fn validate_dispatch(packets: &[Packet]) -> Result<()> {
    let Some(decode) = packets.first() else {
        return Err("Missing A2 decode packet".into());
    };
    if decode.rows != 1 || decode.min_rows != 0 || decode.dispatch_max() != 1 || packets.len() < 2 {
        return Err("Expected single-row decode followed by prefill ranges".into());
    }
    validate_row_ranges(&packets[1..].iter().collect::<Vec<_>>(), 2, None)
}

pub(super) fn validate_row_ranges(
    packets: &[&Packet],
    first: usize,
    capacity: Option<usize>,
) -> Result<()> {
    let mut next = first;
    for packet in packets {
        if packet.min_rows != next
            || packet.dispatch_max() < next
            || packet.dispatch_max() > packet.rows
            || capacity.is_some_and(|c| packet.rows != c)
        {
            return Err("A2 prefill ranges must be ordered, contiguous and disjoint".into());
        }
        next = packet.dispatch_max() + 1;
    }
    if packets.is_empty() || capacity.is_some_and(|c| next != c + 1) {
        return Err("A2 row ranges do not cover their declared capacity".into());
    }
    Ok(())
}
