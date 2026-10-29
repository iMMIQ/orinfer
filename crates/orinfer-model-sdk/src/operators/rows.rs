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
mod tests {
    use super::*;
    use crate::operators::dynamic::{RowArgument, RowExpression};

    fn fixture() -> (
        RowExecution,
        BTreeMap<String, Vec<Operation>>,
        BTreeMap<String, DynamicBatchKernel>,
    ) {
        let kernel = |name: &str| Operation::Kernel { name: name.into() };
        let program = vec![kernel("projection"), kernel("mix"), kernel("state")];
        let execution = RowExecution {
            capacity: 4096,
            program_capacities: BTreeMap::new(),
            programs: BTreeSet::from(["prefill".into()]),
            variants: BTreeMap::from([
                (
                    "projection".into(),
                    vec![KernelVariant {
                        min_rows: 1,
                        max_rows: 128,
                        multiple_of: 1,
                        kernels: vec!["fused".into()],
                    }],
                ),
                (
                    "mix".into(),
                    vec![KernelVariant {
                        min_rows: 1,
                        max_rows: 128,
                        multiple_of: 1,
                        kernels: vec![],
                    }],
                ),
            ]),
        };
        let dynamic = ["projection", "fused", "state"]
            .into_iter()
            .map(|name| {
                (
                    name.into(),
                    DynamicBatchKernel {
                        name: name.into(),
                        capacity: 4096,
                        grid: [
                            RowExpression::Rows,
                            RowExpression::Constant { value: 1 },
                            RowExpression::Constant { value: 1 },
                        ],
                        arguments: vec![RowArgument {
                            index: 0,
                            value: RowExpression::Rows,
                        }],
                    },
                )
            })
            .collect();
        (
            execution,
            BTreeMap::from([("prefill".into(), program)]),
            dynamic,
        )
    }

    #[test]
    fn measured_choices_cover_large_packets_and_respect_each_program() {
        let (mut rows, programs, mut dynamic) = fixture();
        rows.capacity = 32768;
        rows.variants.clear();
        for launch in dynamic.values_mut() {
            launch.capacity = 32768;
        }
        let selected = "projection/rows/2049-32768";
        rows.register_variants(&programs, ["projection", selected])
            .unwrap();
        let mut launch = dynamic["projection"].clone();
        launch.capacity = 32768;
        dynamic.insert("projection".into(), launch.clone());
        launch.name = selected.into();
        dynamic.insert(selected.into(), launch);
        for (extent, expected) in [(2048, "projection"), (2049, selected), (32768, selected)] {
            let calls = rows
                .specialize("prefill", &programs["prefill"], &dynamic, extent)
                .unwrap();
            assert!(matches!(&calls[0].operation, Operation::Kernel { name } if name == expected));
            assert_eq!(calls[0].launch.as_ref().unwrap().grid[0], extent as u32);
        }
        rows.variants.clear();
        assert!(
            rows.register_variants(&programs, ["unknown/rows/1-2"])
                .is_err()
        );
        assert!(
            rows.register_variants(&programs, ["projection/rows/1-8", "projection/rows/8-16"])
                .is_err()
        );
        rows.program_capacities.insert("prefill".into(), 1024);
        assert!(rows.register_variants(&programs, [selected]).is_err());
        assert!(rows.variants.is_empty());
    }

    #[test]
    fn bounded_dynamic_variants_use_the_selected_interval_capacity() {
        let (rows, programs, mut dynamic) = fixture();
        dynamic.get_mut("fused").unwrap().capacity = 128;
        let kernels = BTreeSet::from([
            "projection".into(),
            "fused".into(),
            "mix".into(),
            "state".into(),
        ]);
        rows.validate(&programs, &kernels, &dynamic).unwrap();
        let short = rows
            .specialize("prefill", &programs["prefill"], &dynamic, 127)
            .unwrap();
        assert!(matches!(&short[0].operation, Operation::Kernel {name} if name == "fused"));
        let long = rows
            .specialize("prefill", &programs["prefill"], &dynamic, 129)
            .unwrap();
        assert!(matches!(&long[0].operation, Operation::Kernel {name} if name == "projection"));
        dynamic.get_mut("fused").unwrap().capacity = 127;
        assert!(rows.validate(&programs, &kernels, &dynamic).is_err());
    }

    #[test]
    fn per_program_capacity_limits_decode_without_restricting_prefill() {
        let (mut rows, mut programs, mut dynamic) = fixture();
        let op = Operation::Kernel {
            name: "decode_projection".into(),
        };
        programs.insert("decode_rows".into(), vec![op]);
        rows.programs.insert("decode_rows".into());
        rows.program_capacities.insert("decode_rows".into(), 128);
        dynamic.insert(
            "decode_projection".into(),
            DynamicBatchKernel {
                name: "decode_projection".into(),
                capacity: 128,
                grid: [
                    RowExpression::Rows,
                    RowExpression::Constant { value: 1 },
                    RowExpression::Constant { value: 1 },
                ],
                arguments: vec![],
            },
        );
        let kernels = dynamic
            .keys()
            .cloned()
            .chain([
                "fused".into(),
                "state".into(),
                "mix".into(),
                "projection".into(),
            ])
            .collect();
        rows.validate(&programs, &kernels, &dynamic).unwrap();
        assert!(
            rows.specialize("decode_rows", &programs["decode_rows"], &dynamic, 128)
                .is_ok()
        );
        assert!(
            rows.specialize("decode_rows", &programs["decode_rows"], &dynamic, 129)
                .is_err()
        );
        assert!(
            rows.specialize("prefill", &programs["prefill"], &dynamic, 4096)
                .is_ok()
        );
        rows.program_capacities.insert("missing".into(), 128);
        assert!(rows.validate(&programs, &kernels, &dynamic).is_err());
    }

    #[test]
    fn every_length_specializes_launches_without_new_programs() {
        let (execution, programs, dynamic) = fixture();
        let kernels = ["projection", "mix", "state", "fused"]
            .into_iter()
            .map(String::from)
            .collect();
        execution.validate(&programs, &kernels, &dynamic).unwrap();
        for rows in 1..=4096 {
            let ops = execution
                .specialize("prefill", &programs["prefill"], &dynamic, rows)
                .unwrap();
            let first = if rows <= 128 { "fused" } else { "projection" };
            assert_eq!(ops[0].operation, Operation::Kernel { name: first.into() });
            assert_eq!(ops[0].launch.as_ref().unwrap().grid, [rows as u32, 1, 1]);
            assert_eq!(ops.len(), if rows <= 128 { 2 } else { 3 });
        }
        assert_eq!(programs.len(), 1);
        for rows in [0, 4097] {
            assert!(
                execution
                    .specialize("prefill", &programs["prefill"], &dynamic, rows)
                    .is_err()
            );
        }
    }

    #[test]
    fn rejects_overlapping_variants_and_underallocated_kernels() {
        let (mut execution, programs, mut dynamic) = fixture();
        let kernels = ["projection", "mix", "state", "fused"]
            .into_iter()
            .map(String::from)
            .collect();
        dynamic.get_mut("state").unwrap().capacity = 512;
        assert!(execution.validate(&programs, &kernels, &dynamic).is_err());
        dynamic.get_mut("state").unwrap().capacity = 4096;
        execution
            .variants
            .get_mut("projection")
            .unwrap()
            .push(KernelVariant {
                min_rows: 128,
                max_rows: 512,
                multiple_of: 1,
                kernels: vec!["projection".into()],
            });
        assert!(execution.validate(&programs, &kernels, &dynamic).is_err());
    }

    #[test]
    fn alignment_selects_one_local_implementation_without_length_enumeration() {
        let (mut execution, programs, dynamic) = fixture();
        execution.variants.insert(
            "projection".into(),
            vec![KernelVariant {
                min_rows: 128,
                max_rows: 4096,
                multiple_of: 128,
                kernels: vec!["fused".into()],
            }],
        );
        for rows in [17, 129, 421, 513, 1025] {
            let ops = execution
                .specialize("prefill", &programs["prefill"], &dynamic, rows)
                .unwrap();
            assert_eq!(
                ops[0].operation,
                Operation::Kernel {
                    name: "projection".into()
                }
            );
        }
        for rows in [128, 256, 512, 2048, 4096] {
            let ops = execution
                .specialize("prefill", &programs["prefill"], &dynamic, rows)
                .unwrap();
            assert_eq!(
                ops[0].operation,
                Operation::Kernel {
                    name: "fused".into()
                }
            );
            assert_eq!(ops[0].launch.as_ref().unwrap().grid, [rows as u32, 1, 1]);
        }
    }
}
