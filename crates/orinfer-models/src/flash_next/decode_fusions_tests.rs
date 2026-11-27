use super::*;
use crate::artifact::Argument;
fn offline(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    build(manifest, kernels, true)
}
#[test]
fn in_place_history_removes_only_the_validated_private_copy() {
    let (mut m, mut ks) = hc_fixture(320, false);
    let defs = [
        ("History", "f16", 30720, false),
        ("K", "f16", 2048, false),
        ("Q", "f16", 2048, false),
        ("V", "f16", 6144, false),
        ("W", "f32", 40960, true),
        ("X", "f16", 10240, false),
        ("Length", "i32", 1, false),
        ("Position", "i32", 1, false),
        ("PositionOut", "i32", 1, false),
        ("OldHistory", "f16", 30720, false),
    ];
    m.buffers = defs
        .into_iter()
        .map(|(name, dtype, n, immutable)| {
            let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":[n],
            "layout":"native_contiguous","alignment":256,
            "access":if immutable {"read"}else{"read_write"}});
            if immutable {
                b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
            }
            serde_json::from_value(b).unwrap()
        })
        .collect();
    let mut k = ks.remove(0);
    k.name = "decode/conv".into();
    k.symbol = "gdn_conv_inplace_kernel".into();
    k.grid = [1, 80, 1];
    k.block = [16, 1, 1];
    k.args = defs[..9]
        .iter()
        .map(|(name, _, _, _)| Argument::Buffer {
            name: (*name).into(),
        })
        .collect();
    m.programs.insert(
        "decode".into(),
        vec![
            Operation::Kernel {
                name: k.name.clone(),
            },
            Operation::Copy {
                source: "OldHistory".into(),
                destination: "History".into(),
                bytes: 61440,
            },
        ],
    );
    let mut valid = m.clone();
    offline(&mut valid, std::slice::from_ref(&k)).unwrap();
    assert_eq!(valid.programs["decode"].len(), 1);
    let mut corrupt = m.clone();
    if let Operation::Copy { bytes, .. } = &mut corrupt.programs.get_mut("decode").unwrap()[1] {
        *bytes -= 1;
    }
    assert!(offline(&mut corrupt, std::slice::from_ref(&k)).is_err());
    let mut alias = k.clone();
    alias.args[8] = alias.args[7].clone();
    assert!(
        offline(&mut m.clone(), &[alias])
            .unwrap_err()
            .contains("Aliased")
    );
    let ops = m.programs.remove("decode").unwrap();
    m.programs.insert("verify_m4".into(), ops);
    assert!(offline(&mut m, &[k]).is_err());
}
fn hc_fixture(rank: usize, patches: bool) -> (Manifest, Vec<Kernel>) {
    let (mut m, _) = fixture(false);
    let definitions = [
        ("Partial", "f32", 4 * rank, false),
        ("Activation", "f16", rank, false),
        ("Normed", "f16", 10240, false),
        ("Output", "f16", 2560, false),
        ("Hi", "u32", 80 * (rank / 64) * 128 * 10, true),
        ("Lo", "u32", 80 * (rank / 64) * 128 * 16, true),
        ("Patches", "u64", 3, true),
    ];
    m.buffers = definitions
        .into_iter()
        .map(|(name, dtype, n, immutable)| {
            let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":[n],
            "layout":"native_contiguous","alignment":256,
            "access":if immutable {"read"} else {"read_write"}});
            if immutable {
                b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
            }
            serde_json::from_value(b).unwrap()
        })
        .collect();
    let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
    let mut up = vec!["Partial", "Hi", "Lo", "Normed", "Output"];
    if patches {
        up.push("Patches");
    }
    let specs = [
        (
            "main_kernel",
            vec!["Normed", "Partial", "Weight"],
            [1, rank.div_ceil(64) as u32, 4],
        ),
        (
            "main_kernel",
            vec!["Activation", "Partial"],
            [rank.div_ceil(128) as u32, 1, 1],
        ),
        ("hc_up13_finish_kernel", up, [1, 80, 1]),
    ];
    let k: Vec<Kernel> = specs.into_iter().enumerate().map(|(i,(symbol,args,grid))|
        serde_json::from_value(serde_json::json!({"name":format!("decode/body/k{i}"),
            "symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
            "grid":grid,"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()).collect();
    m.programs.insert(
        "decode".into(),
        k.iter()
            .map(|k| Operation::Kernel {
                name: k.name.clone(),
            })
            .collect(),
    );
    (m, k)
}
#[test]
fn hc_finish_fusion_requires_exact_producer_geometry_and_workspace() {
    for rank in [256, 320] {
        for patches in [false, true] {
            let (mut m, k) = hc_fixture(rank, patches);
            offline(&mut m, &k).unwrap();
            assert_eq!(m.programs["decode"].len(), 2);
            let (mut m, mut k) = hc_fixture(rank, patches);
            k[0].symbol = "bf16_single_kernel".into();
            k[0].grid = [rank.div_ceil(32) as u32, 4, 1];
            k[0].block = [64, 1, 1];
            offline(&mut m, &k).unwrap();
            assert_eq!(m.programs["decode"].len(), 2);
            let (mut m, mut k) = hc_fixture(rank, patches);
            k[2].args[0] = Argument::Buffer {
                name: "Output".into(),
            };
            assert!(offline(&mut m, &k).is_err());
            let (mut m, k) = hc_fixture(rank, patches);
            m.buffers[0].shape[0] -= 1;
            assert!(offline(&mut m, &k).is_err());
            let (mut m, k) = hc_fixture(rank, patches);
            let ops = m.programs.remove("decode").unwrap();
            m.programs.insert("verify_m4".into(), ops);
            assert!(offline(&mut m, &k).is_err());
        }
    }
}
fn combine_norm_fixture() -> (Manifest, Vec<Kernel>) {
    let (mut m, _) = fixture(false);
    let definitions = [
        ("Block", "f16", 2560, false),
        ("Inject", "f16", 4, false),
        ("Normed", "f16", 10240, false),
        ("Output", "f16", 10240, false),
        ("Residual", "f16", 10240, false),
        ("Weight", "f32", 10240, true),
    ];
    m.buffers = definitions
        .into_iter()
        .map(|(name, dtype, n, immutable)| {
            let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":[n],
            "layout":"native_contiguous","alignment":256,
            "access":if immutable {"read"}else{"read_write"}});
            if immutable {
                b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
            }
            serde_json::from_value(b).unwrap()
        })
        .collect();
    let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
    let specs = [
        (
            "hc_combine_norm_kernel",
            vec!["Block", "Inject", "Normed", "Output", "Residual", "Weight"],
            [1, 4, 1],
            [256, 1, 1],
        ),
        (
            "main_kernel",
            vec!["Normed", "Residual", "Weight"],
            [1, 4, 1],
            [256, 1, 1],
        ),
    ];
    let k: Vec<Kernel> = specs
        .into_iter()
        .enumerate()
        .map(|(i, (symbol, args, grid, block))| {
            serde_json::from_value(serde_json::json!({"name":format!("decode/body/k{i}"),
                "symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
                "grid":grid,"block":block,"shared_memory_bytes":0,"cooperative":false,
                "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
            })).unwrap()
        })
        .collect();
    m.programs.insert(
        "decode".into(),
        k.iter()
            .map(|k| Operation::Kernel {
                name: k.name.clone(),
            })
            .collect(),
    );
    (m, k)
}
#[test]
fn combine_norm_removes_only_the_validated_adjacent_norm() {
    let (mut m, k) = combine_norm_fixture();
    offline(&mut m, &k).unwrap();
    assert_eq!(m.programs["decode"].len(), 1);
    let (mut m, mut k) = combine_norm_fixture();
    k[1].args[2] = Argument::Buffer {
        name: "Output".into(),
    };
    assert!(offline(&mut m, &k).is_err());
    let (mut m, mut k) = combine_norm_fixture();
    k[0].block = [128, 1, 1];
    assert!(offline(&mut m, &k).is_err());
    let (mut m, k) = combine_norm_fixture();
    m.buffers[0].shape[0] -= 1;
    assert!(offline(&mut m, &k).is_err());
    let (mut m, k) = combine_norm_fixture();
    let ops = m.programs.remove("decode").unwrap();
    m.programs.insert("verify_m4".into(), ops);
    assert!(offline(&mut m, &k).is_err());
}
fn fixture(gu: bool) -> (Manifest, Vec<Kernel>) {
    let width = if gu { 2560 } else { 640 };
    let rows = if gu { 1 } else { 10 };
    let definitions = [
        ("High", "u32", 10 * width / 8, false),
        ("Low", "u32", 10 * width / 8, false),
        ("Sum", "i32", 10, false),
        ("Q", "i8", rows * width, false),
        ("S", "f16", rows, false),
        ("Signs", "i8", width, true),
        ("X", "f16", rows * width * if gu { 1 } else { 2 }, false),
        ("DS", "f16", 10, false),
        ("IDs", "i32", 10, false),
        ("RelativeSlot", "i32", 10, false),
        ("RowOffsets", "i32", 512, false),
        ("SlotMap", "i32", 10, false),
    ];
    let buffers: Vec<_> = definitions.into_iter().map(|(name,dtype,n,immutable)| {
        let mut b=serde_json::json!({"name":name,"dtype":dtype,"shape":[n],"layout":"native_contiguous","alignment":256,
            "access":if immutable {"read"}else{"read_write"}});
        if immutable {b["data"]=serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});}
        b
    }).collect();
    let names: &[(&str, &[&str])] = if gu {
        &[
            (
                "rotate_dispatch_a8_kernel",
                &[
                    "DS",
                    "High",
                    "IDs",
                    "Low",
                    "Q",
                    "RelativeSlot",
                    "RowOffsets",
                    "S",
                    "Signs",
                    "SlotMap",
                    "Sum",
                    "X",
                ],
            ),
            (
                "main_kernel",
                &[
                    "Q",
                    "D",
                    "DS",
                    "IDs",
                    "RelativeSlot",
                    "RowOffsets",
                    "S",
                    "SlotMap",
                ],
            ),
            ("pack_exact_a8_kernel", &["D", "High", "Low", "Sum"]),
            (
                "exact_e8p_shared_input_kernel",
                &[
                    "S", "Book", "Counts", "High", "Low", "Offsets", "Out", "P", "Sum", "TC", "TE",
                    "TR", "WS",
                ],
            ),
        ]
    } else {
        &[
            (
                "rotate_split_a8_kernel",
                &["High", "Low", "Q", "S", "Signs", "Sum", "X"],
            ),
            ("pack_exact_a8_kernel", &["Q", "High", "Low", "Sum"]),
            (
                "exact_e8p_int4_kernel",
                &[
                    "S", "Book", "Counts", "High", "Low", "Offsets", "Out", "P", "Sum", "TC", "TE",
                    "TR", "WS",
                ],
            ),
        ]
    };
    let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
    let kernels: Vec<Kernel> = names.iter().enumerate().map(|(i,(symbol,args))| serde_json::from_value(serde_json::json!({
        "name":format!("decode/layer0/k{i}"),"symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
        "grid":if gu && i==0 {[1,1,1]}else{[10,1,1]},"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
        "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
    })).unwrap()).collect();
    let manifest=serde_json::from_value(serde_json::json!({"schema_version":1,"target":"sm_87","model":"test","chunk_tokens":4096,
        "max_context":262144,"vocab":16,"toolchain":{},"buffers":buffers,"reset_buffers":[],"input":"Input","token":"Token",
        "status":"Status","logits":"Logits","position":"Position","weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
        "programs":{"decode":kernels.iter().map(|k|serde_json::json!({"kind":"kernel","name":k.name})).collect::<Vec<_>>()}
    })).unwrap();
    (manifest, kernels)
}
#[test]
fn removes_dispatch_and_split_only_when_producer_contract_matches() {
    for gu in [false, true] {
        let (mut m, k) = fixture(gu);
        offline(&mut m, &k).unwrap();
        assert_eq!(m.programs["decode"].len(), 2);
        assert!(
            matches!(&m.programs["decode"][1],Operation::Kernel{name} if name==&k.last().unwrap().name)
        );
    }
}
#[test]
fn rejects_corrupt_bindings_aliases_and_short_buffers() {
    let (mut m, mut k) = fixture(true);
    k[0].args[3] = Argument::Buffer {
        name: "High".into(),
    };
    assert!(offline(&mut m, &k).unwrap_err().contains("mismatch"));
    let (mut m, k) = fixture(false);
    m.buffers
        .iter_mut()
        .find(|b| b.name == "Low")
        .unwrap()
        .shape[0] -= 1;
    assert!(offline(&mut m, &k).unwrap_err().contains("buffer"));
    let (mut m, mut k) = fixture(false);
    k[0].args[1] = Argument::BufferSlice {
        name: "Low".into(),
        offset: 0,
    };
    assert!(offline(&mut m, &k).unwrap_err().contains("whole buffer"));
}
#[test]
fn rejects_batched_fusion_and_unregistered_shared_projection() {
    let (mut m, k) = fixture(true);
    let ops = m.programs.remove("decode").unwrap();
    m.programs.insert("verify_m4".into(), ops);
    assert!(offline(&mut m, &k).unwrap_err().contains("single-token"));
    let (mut m, mut k) = fixture(true);
    k[0].symbol = "main_kernel".into();
    assert!(offline(&mut m, &k).unwrap_err().contains("Unregistered"));
    let (mut m, mut k) = fixture(false);
    k[2].args[0] = Argument::Buffer { name: "DS".into() };
    assert!(offline(&mut m, &k).unwrap_err().contains("mismatch"));
}
