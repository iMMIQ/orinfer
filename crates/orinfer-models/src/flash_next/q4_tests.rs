use super::*;
use crate::artifact::Argument;
fn contract(name: &str, args: &[&str]) -> Kernel {
    let asset = serde_json::json!({"file":"test.cubin","sha256":"0".repeat(64)});
    serde_json::from_value(serde_json::json!({
        "name":name,"symbol":"main_kernel","module":asset,"source":asset,"host_abi":asset,
        "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
        "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
    })).unwrap()
}

#[test]
fn paired_qkv_z_rotate_once_before_the_first_consumer() {
    let mut m = manifest(0);
    let qkv = "decode/layer0/k6";
    let z = "decode/layer0/k7";
    let rotation = format!("{z}/q4_rotation");
    let kernels = vec![
        contract(
            qkv,
            &[
                "A",
                "AS",
                "QKV",
                "Q4_blk.0.attn_qkv_scale",
                "Q4_blk.0.attn_qkv_packed",
            ],
        ),
        contract(
            z,
            &[
                "A",
                "AS",
                "Z",
                "Q4_blk.0.attn_gate_scale",
                "Q4_blk.0.attn_gate_packed",
            ],
        ),
        contract(&rotation, &["A", "AS", "Q4_blk.0.attn_gate_signs", "Input"]),
    ];
    // Even a transform attached to the later projection is moved before
    // both consumers. Reconstructing a plan must remain idempotent.
    for program in ["decode", "alias"] {
        m.programs.insert(
            program.into(),
            [qkv, rotation.as_str(), z]
                .into_iter()
                .map(|name| Operation::Kernel { name: name.into() })
                .collect(),
        );
    }
    for _ in 0..2 {
        prerequisites(&mut m, &kernels).unwrap();
        for operations in m.programs.values() {
            let names: Vec<_> = operations
                .iter()
                .map(|op| match op {
                    Operation::Kernel { name } => name.as_str(),
                    _ => panic!("Unexpected copy"),
                })
                .collect();
            assert_eq!(names, [rotation.as_str(), qkv, z]);
        }
    }
}

#[test]
fn prerequisites_cover_every_program_alias_and_are_idempotent() {
    let mut m = manifest(0);
    m.programs.clear();
    let projection = "prefill_m4096/layer0/k6";
    let rotation = format!("{projection}/q4_rotation");
    let kernels = vec![
        contract(
            projection,
            &["A", "AS", "Out", "Q4_test_scale", "Q4_test_packed"],
        ),
        contract(&rotation, &["A", "AS", "Q4_test_signs", "Input"]),
    ];
    for preinstalled in [false, true] {
        m.programs.clear();
        // Cover both destructive lookup across aliases and the case where
        // just one alias already contains its prerequisite.
        for program in ["prefill", "prefill_m2048", "prefill_m4096"] {
            let mut ops = vec![];
            if preinstalled && program == "prefill" {
                ops.push(Operation::Kernel {
                    name: (*rotation).into(),
                });
            }
            ops.push(Operation::Kernel {
                name: projection.into(),
            });
            m.programs.insert(program.into(), ops);
        }
        for _ in 0..2 {
            prerequisites(&mut m, &kernels).unwrap();
            for ops in m.programs.values() {
                assert_eq!(ops.len(), 2);
                assert!(matches!(&ops[0], Operation::Kernel { name } if name == &rotation));
                assert!(matches!(&ops[1], Operation::Kernel { name } if name == projection));
            }
        }
    }
}

#[test]
fn shared_lowering_expands_target_draft_and_prefill_recipes() {
    for section in [
        "flash_decode/layer0/pre",
        "draft_rows/pre",
        "prefill_m4096/layer0",
    ] {
        let mut m = manifest(0);
        m.programs.clear();
        let projection = format!("{section}/k6");
        let rotation = format!("{projection}/q4_rotation");
        m.programs.insert(
            section.into(),
            vec![Operation::Kernel {
                name: projection.clone(),
            }],
        );
        let mut kernels = vec![
            contract(
                &projection,
                &[
                    "A",
                    "AS",
                    "Out",
                    "Q4_blk.0.attn_qkv_scale",
                    "Q4_blk.0.attn_qkv_packed",
                ],
            ),
            contract(&rotation, &["A", "AS", "Q4_blk.0.attn_gate_signs", "Input"]),
        ];
        prerequisites(&mut m, &kernels).unwrap();
        assert!(matches!(&m.programs[section][0], Operation::Kernel {name} if name == &rotation));
        assert!(matches!(&m.programs[section][1], Operation::Kernel {name} if name == &projection));
        prerequisites(&mut m, &kernels).unwrap();
        assert_eq!(m.programs[section].len(), 2);
        m.programs.get_mut(section).unwrap().remove(0);
        kernels[1].args[0] = Argument::Buffer {
            name: "wrong".into(),
        };
        assert!(
            prerequisites(&mut m, &kernels)
                .unwrap_err()
                .contains("workspace mismatch")
        );
        kernels[1].args[0] = Argument::Buffer { name: "A".into() };
        m.programs.clear();
        assert!(
            prerequisites(&mut m, &kernels)
                .unwrap_err()
                .contains("Unregistered")
        );
    }
}

fn manifest(_layer: usize) -> Manifest {
    serde_json::from_value(serde_json::json!({
        "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,
        "max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
        "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
        "logits":"Logits","position":"Position","weight_bytes":0,
        "weight_parameters":0,"weight_scope":"test"
    }))
    .unwrap()
}

#[test]
fn optional_q4_out_proj_enforces_single_resident_bank() {
    use crate::artifact::{Access, Buffer};
    use crate::model::TensorIdentity;
    let weight_buf = |name: String, dtype: Dtype, shape: Vec<usize>| Buffer {
        name: name.clone(),
        dtype,
        shape,
        layout: "native_contiguous".into(),
        alignment: 256,
        access: Access::Read,
        data: Some(TensorIdentity {
            tensor: name,
            sha256: "0".repeat(64),
        }),
    };
    let mut m = manifest(0);
    m.buffers.push(weight_buf(
        "Q4_output_packed".into(),
        Dtype::U16,
        vec![248320, 640],
    ));
    m.buffers.push(weight_buf(
        "Q4_output_scale".into(),
        Dtype::F16,
        vec![248320],
    ));
    m.buffers
        .push(weight_buf("Q4_output_signs".into(), Dtype::I8, vec![2560]));
    for layer in 0..48 {
        if layer % 4 == 3 {
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_q_packed"),
                Dtype::U16,
                vec![12288, 640],
            ));
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_q_scale"),
                Dtype::F16,
                vec![12288],
            ));
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_q_signs"),
                Dtype::I8,
                vec![2560],
            ));
        } else {
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_gate_packed"),
                Dtype::U16,
                vec![6144, 640],
            ));
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_gate_scale"),
                Dtype::F16,
                vec![6144],
            ));
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_gate_signs"),
                Dtype::I8,
                vec![2560],
            ));
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_qkv_packed"),
                Dtype::U16,
                vec![10240, 640],
            ));
            m.buffers.push(weight_buf(
                format!("Q4_blk.{layer}.attn_qkv_scale"),
                Dtype::F16,
                vec![10240],
            ));
        }
    }
    // Baseline without Q4 ssm_out/attn_output passes.
    validate(&m).unwrap();
    // Adding all 48 Q4 ssm_out/attn_output banks passes.
    for layer in 0..48 {
        let role = if layer % 4 == 3 {
            "attn_output"
        } else {
            "ssm_out"
        };
        m.buffers.push(weight_buf(
            format!("Q4_blk.{layer}.{role}_packed"),
            Dtype::U16,
            vec![2560, 1536],
        ));
        m.buffers.push(weight_buf(
            format!("Q4_blk.{layer}.{role}_scale"),
            Dtype::F16,
            vec![2560],
        ));
        m.buffers.push(weight_buf(
            format!("Q4_blk.{layer}.{role}_signs"),
            Dtype::I8,
            vec![6144],
        ));
    }
    validate(&m).unwrap();
    // Retaining a duplicate W8 bank must be rejected.
    m.buffers.push(weight_buf(
        "W_blk.0.ssm_out.weight_weight".into(),
        Dtype::I8,
        vec![2560, 6144],
    ));
    assert!(validate(&m).unwrap_err().contains("Duplicate resident"));
}
