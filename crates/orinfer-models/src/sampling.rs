//! Shared sampling program registration; independent of model architecture.
use crate::{
    artifact::Result,
    model::{BatchedSampling, Manifest, Operation, Sampling},
};
use std::collections::BTreeSet;

pub(crate) fn register(m: &mut Manifest, bindings: &BTreeSet<String>) -> Result<()> {
    m.sampling = None;
    let mut names: Vec<String> = ["initialize", "history", "scores"]
        .into_iter()
        .map(|s| format!("sampling/{s}"))
        .collect();
    let deferred = bindings.contains("sampling/fallback_masses");
    if !deferred {
        names.push("sampling/masses".into());
    }
    let shortlist = [
        "sampling/shortlist_bound",
        "sampling/shortlist_collect",
        "sampling/shortlist_sample",
    ];
    if deferred || shortlist.iter().any(|name| bindings.contains(*name)) {
        names.extend(shortlist.map(str::to_owned));
    }
    let mut radix = Vec::new();
    if deferred {
        radix.push("sampling/fallback_masses".to_owned());
    }
    for mode in 0..3 {
        for digit in 0..12 {
            radix.extend([
                format!("sampling/{mode}/{digit}/hist"),
                format!("sampling/{mode}/{digit}/choose"),
            ]);
        }
    }
    if !deferred {
        names.append(&mut radix);
    }
    let count = bindings
        .iter()
        .filter(|n| n.starts_with("sampling/"))
        .count();
    if count == 0 {
        if bindings.iter().any(|n| n.starts_with("sampling_rows/")) {
            return Err("Batched sampler requires the scalar sampling contract".into());
        }
        return Ok(());
    }
    if count != names.len() + radix.len()
        || names.iter().chain(&radix).any(|n| !bindings.contains(n))
    {
        return Err("Incomplete shared GPU sampling bindings".into());
    }
    let programs = [("sampling", names), ("sampling_fallback", radix)];
    for (program, names) in &programs {
        if !names.is_empty() {
            m.programs.insert(
                (*program).into(),
                names
                    .iter()
                    .cloned()
                    .map(|name| Operation::Kernel { name })
                    .collect(),
            );
        }
    }
    m.sampling = Some(Sampling {
        program: "sampling".into(),
        history: "SamplingHistory".into(),
        bias_ids: "SamplingBiasIds".into(),
        bias_values: "SamplingBiasValues".into(),
        mask: "SamplingMask".into(),
        control: "SamplingControl".into(),
        parameters: "SamplingParameters".into(),
        output: "SamplingOutput".into(),
        batch: None,
    });
    let batch_programs: Vec<_> = programs
        .into_iter()
        .filter(|(_, names)| !names.is_empty())
        .map(|(program, names)| {
            (
                program.replacen("sampling", "sampling_rows", 1),
                names
                    .into_iter()
                    .map(|n| n.replacen("sampling/", "sampling_rows/", 1))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let count = bindings
        .iter()
        .filter(|n| n.starts_with("sampling_rows/"))
        .count();
    if count != 0 {
        let mut expected = batch_programs.iter().flat_map(|(_, names)| names);
        if count != expected.clone().count() || expected.any(|n| !bindings.contains(n)) {
            return Err("Incomplete batched GPU sampling bindings".into());
        }
        let rows = m
            .row_execution
            .get_or_insert_with(|| crate::operators::rows::RowExecution {
                capacity: 128,
                programs: BTreeSet::new(),
                program_capacities: Default::default(),
                variants: Default::default(),
            });
        rows.capacity = rows.capacity.max(128);
        for (program, names) in batch_programs {
            rows.programs.insert(program.clone());
            rows.program_capacities.insert(program.clone(), 128);
            m.programs.insert(
                program,
                names
                    .into_iter()
                    .map(|name| Operation::Kernel { name })
                    .collect(),
            );
        }
        let (offsets, stride) = BatchedSampling::layout(m.vocab, m.max_context)?;
        m.sampling.as_mut().unwrap().batch = Some(BatchedSampling {
            program: "sampling_rows".into(),
            capacity: 128,
            pointers: "SamplingRowsPointers".into(),
            scratch: "SamplingRowsScratch".into(),
            output: "SamplingRowsOutput".into(),
            stride,
            offsets,
        });
    }
    Ok(())
}
