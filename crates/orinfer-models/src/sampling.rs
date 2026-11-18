//! Shared sampling program registration; independent of model architecture.
use crate::{
    artifact::Result,
    model::{BatchedSampling, Manifest, Operation, Sampling},
};
use std::collections::BTreeSet;

pub(crate) fn register(m: &mut Manifest, bindings: &BTreeSet<String>) -> Result<()> {
    m.sampling = None;
    let mut names: Vec<String> = ["initialize", "history", "scores", "masses"]
        .into_iter()
        .map(|s| format!("sampling/{s}"))
        .collect();
    let shortlist = [
        "sampling/shortlist_bound",
        "sampling/shortlist_collect",
        "sampling/shortlist_sample",
    ];
    if shortlist.iter().any(|name| bindings.contains(*name)) {
        names.extend(shortlist.map(str::to_owned));
    }
    for mode in 0..3 {
        for digit in 0..12 {
            names.extend([
                format!("sampling/{mode}/{digit}/hist"),
                format!("sampling/{mode}/{digit}/choose"),
            ]);
        }
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
    if count != names.len() || names.iter().any(|n| !bindings.contains(n)) {
        return Err("Incomplete shared GPU sampling bindings".into());
    }
    m.programs.insert(
        "sampling".into(),
        names
            .into_iter()
            .map(|name| Operation::Kernel { name })
            .collect(),
    );
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
    let batch_names: Vec<_> = m.programs["sampling"]
        .iter()
        .map(|op| {
            let Operation::Kernel { name } = op else {
                unreachable!()
            };
            name.replacen("sampling/", "sampling_rows/", 1)
        })
        .collect();
    let count = bindings
        .iter()
        .filter(|n| n.starts_with("sampling_rows/"))
        .count();
    if count != 0 {
        if count != batch_names.len() || batch_names.iter().any(|n| !bindings.contains(n)) {
            return Err("Incomplete batched GPU sampling bindings".into());
        }
        m.programs.insert(
            "sampling_rows".into(),
            batch_names
                .into_iter()
                .map(|name| Operation::Kernel { name })
                .collect(),
        );
        let rows = m
            .row_execution
            .get_or_insert_with(|| crate::operators::rows::RowExecution {
                capacity: 128,
                programs: BTreeSet::new(),
                program_capacities: Default::default(),
                variants: Default::default(),
            });
        rows.capacity = rows.capacity.max(128);
        rows.programs.insert("sampling_rows".into());
        rows.program_capacities.insert("sampling_rows".into(), 128);
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
