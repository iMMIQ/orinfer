//! Runtime row specialization of one model-owned execution template.
//! Lengths change launch arguments, never compile identities or tensor roles.
use crate::{
    artifact::Result, execution::Invocation, model::Operation,
    operators::dynamic::DynamicBatchKernel,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Optional local algorithm/tile substitution at a canonical kernel slot.
/// An empty replacement omits a slot consumed by a fused implementation.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelVariant {
    pub min_rows: usize,
    pub max_rows: usize,
    #[serde(default = "unit_multiple")]
    pub multiple_of: usize,
    pub kernels: Vec<String>,
}

impl KernelVariant {
    pub fn accepts(&self, rows: usize) -> bool {
        self.multiple_of != 0
            && (self.min_rows..=self.max_rows).contains(&rows)
            && rows.is_multiple_of(self.multiple_of)
    }
}

fn unit_multiple() -> usize {
    1
}

/// Decode an offline-selected local implementation without model dispatch.
pub fn specialization(name: &str, capacity: usize) -> Result<Option<(&str, KernelVariant)>> {
    let (slot, range, skip) = if let Some((slot, range)) = name.split_once("/rows/") {
        (slot, range, false)
    } else if let Some((slot, range)) = name.split_once("/skip/") {
        (slot, range, true)
    } else {
        return Ok(None);
    };
    let (first, last) = range.split_once('-').ok_or("Invalid row selection")?;
    let min_rows: usize = first.parse().map_err(|_| "Invalid row minimum")?;
    let max_rows: usize = last.parse().map_err(|_| "Invalid row maximum")?;
    if min_rows == 0 || min_rows > max_rows || max_rows > capacity {
        return Err("Row selection exceeds its allocation".into());
    }
    Ok(Some((
        slot,
        KernelVariant {
            min_rows,
            max_rows,
            multiple_of: 1,
            kernels: if skip { vec![] } else { vec![name.into()] },
        },
    )))
}

/// Names are supplied by the native model adapter. Capacity bounds allocation;
/// it neither rounds the scheduler's rows nor selects an entire forward plan.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RowExecution {
    pub capacity: usize,
    pub programs: BTreeSet<String>,
    /// Per-program bounds let decode and prefill share the same specialization machinery.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub program_capacities: BTreeMap<String, usize>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub variants: BTreeMap<String, Vec<KernelVariant>>,
}

impl RowExecution {
    /// Register measured local choices against each program's actual allocation.
    pub fn register_variants<'a>(
        &mut self,
        programs: &BTreeMap<String, Vec<Operation>>,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<()> {
        let mut slots = BTreeMap::<&str, usize>::new();
        for program in &self.programs {
            let capacity = self.program_capacity(program);
            for operation in programs
                .get(program)
                .ok_or("Missing row execution template")?
            {
                if let Operation::Kernel { name } = operation {
                    slots
                        .entry(name)
                        .and_modify(|bound| *bound = (*bound).max(capacity))
                        .or_insert(capacity);
                }
            }
        }
        let mut variants = BTreeMap::<String, Vec<KernelVariant>>::new();
        for name in names {
            let Some((slot, choice)) = specialization(name, self.capacity)? else {
                continue;
            };
            let capacity = slots
                .get(slot)
                .ok_or("Local row choice has no computation slot")?;
            if choice.max_rows > *capacity {
                return Err("Local row choice exceeds its program allocation".into());
            }
            variants.entry(slot.into()).or_default().push(choice);
        }
        for choices in variants.values_mut() {
            choices.sort_by_key(|choice| choice.min_rows);
            if choices
                .windows(2)
                .any(|pair| pair[0].max_rows >= pair[1].min_rows)
            {
                return Err("Overlapping local row choices".into());
            }
        }
        if !self.variants.is_empty() {
            return Err("Local row choices were already registered".into());
        }
        self.variants = variants;
        Ok(())
    }

    pub fn validate(
        &self,
        programs: &BTreeMap<String, Vec<Operation>>,
        kernels: &BTreeSet<String>,
        dynamic: &BTreeMap<String, DynamicBatchKernel>,
    ) -> Result<()> {
        if !(2..=32768).contains(&self.capacity) || self.programs.is_empty() {
            return Err("Invalid row execution capacity/programs".into());
        }
        if self.program_capacities.iter().any(|(name, &capacity)| {
            !self.programs.contains(name) || capacity == 0 || capacity > self.capacity
        }) {
            return Err("Invalid per-program row capacity".into());
        }
        let mut slots = BTreeMap::new();
        for name_of_program in &self.programs {
            let operations = programs
                .get(name_of_program)
                .filter(|ops| !ops.is_empty())
                .ok_or("Missing row execution template")?;
            for operation in operations {
                if let Operation::Kernel { name } = operation {
                    slots
                        .entry(name)
                        .and_modify(|bound: &mut usize| {
                            *bound = (*bound).max(self.program_capacity(name_of_program))
                        })
                        .or_insert(self.program_capacity(name_of_program));
                }
            }
        }
        for (name, variants) in &self.variants {
            if !slots.contains_key(name) || variants.is_empty() {
                return Err("Row variants require a used canonical kernel slot".into());
            }
            let mut previous = 0;
            for variant in variants {
                if variant.min_rows <= previous
                    || variant.min_rows > variant.max_rows
                    || variant.max_rows > slots[name]
                    || variant.multiple_of == 0
                    || variant.multiple_of > variant.max_rows
                    || variant.min_rows.div_ceil(variant.multiple_of) * variant.multiple_of
                        > variant.max_rows
                    || variant.kernels.iter().any(|name| !kernels.contains(name))
                {
                    return Err("Invalid/overlapping local row kernel variants".into());
                }
                previous = variant.max_rows;
                for replacement in &variant.kernels {
                    let bound = variant.max_rows;
                    slots
                        .entry(replacement)
                        .and_modify(|v| *v = (*v).max(bound))
                        .or_insert(bound);
                }
            }
        }
        for (name, capacity) in slots {
            if !kernels.contains(name) {
                return Err("Unknown row execution kernel".into());
            }
            if let Some(contract) = dynamic.get(name) {
                if contract.capacity < capacity {
                    return Err("Row template exceeds a symbolic kernel's allocation".into());
                }
                contract.launch(capacity)?;
            }
        }
        Ok(())
    }

    /// Ambiguous multi-slot replacements cannot identify a parallel endpoint.
    pub fn kernel_slots(&self) -> BTreeMap<String, String> {
        let (mut slots, mut ambiguous) = (BTreeMap::new(), BTreeSet::new());
        for (slot, variants) in &self.variants {
            for name in variants.iter().flat_map(|v| &v.kernels) {
                if let Some(previous) = slots.insert(name.clone(), slot.clone())
                    && previous != *slot
                {
                    ambiguous.insert(name.clone());
                }
            }
        }
        slots.retain(|name, _| !ambiguous.contains(name));
        slots
    }

    pub fn program_capacity(&self, program: &str) -> usize {
        self.program_capacities
            .get(program)
            .copied()
            .unwrap_or(self.capacity)
    }

    pub fn specialize(
        &self,
        program: &str,
        operations: &[Operation],
        dynamic: &BTreeMap<String, DynamicBatchKernel>,
        rows: usize,
    ) -> Result<Vec<Invocation>> {
        if !self.programs.contains(program) || !(1..=self.program_capacity(program)).contains(&rows)
        {
            return Err("Invalid dynamic program/row extent".into());
        }
        let mut invocations = Vec::with_capacity(operations.len());
        let mut append = |operation: Operation| -> Result<()> {
            let launch = if let Operation::Kernel { name } = &operation {
                dynamic.get(name).map(|k| k.launch(rows)).transpose()?
            } else {
                None
            };
            invocations.push(Invocation {
                operation,
                sequence: None,
                launch,
                views: BTreeMap::new(),
            });
            Ok(())
        };
        for operation in operations {
            if let Operation::Kernel { name } = operation
                && let Some(variant) = self
                    .variants
                    .get(name)
                    .and_then(|variants| variants.iter().find(|v| v.accepts(rows)))
            {
                for name in &variant.kernels {
                    append(Operation::Kernel { name: name.clone() })?;
                }
            } else {
                append(operation.clone())?;
            }
        }
        Ok(invocations)
    }
}

#[cfg(test)]
#[path = "rows_tests.rs"]
mod tests;
