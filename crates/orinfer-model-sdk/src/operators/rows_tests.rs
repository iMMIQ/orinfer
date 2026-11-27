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
