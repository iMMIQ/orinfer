use super::*;
use serde_json::json;

#[test]
fn joined_sections_follow_weight_banks_before_kernel_contracts_are_materialized() {
    let mut m = request().metadata;
    m.kernels.clear();
    for role in ["self_attn.qkv_proj", "mlp.gate_up_proj"] {
        let mut buffer = m.buffers.iter().find(|b| b.data.is_some()).unwrap().clone();
        buffer.name = format!("model.language_model.layers.3.{role}.weight.packed");
        m.buffers.push(buffer);
    }
    let pre = section(&mut m, "layer3/pre", PRE_ATTN);
    assert_eq!(
        pre,
        kernels("qwen_rows/layer3/pre", &["norm_rotate", "qkv"])
    );
    let post = section(
        &mut m,
        "layer3/post",
        &["norm_rotate", "gate", "up", "swiglu_rotate", "down"],
    );
    assert_eq!(
        post,
        kernels(
            "qwen_rows/layer3/post",
            &["norm_rotate", "gate_up", "swiglu_rotate", "down"]
        )
    );
    let unjoined = section(&mut m, "layer7/pre", PRE_ATTN);
    assert_eq!(unjoined, kernels("qwen_rows/layer7/pre", PRE_ATTN));
}

fn request() -> CreateRequest {
    request_geometry(false)
}

#[test]
fn decode_packet_boundaries_keep_constant_launches_and_owner_addresses() {
    for small in [false, true] {
        let mut request = request_geometry(small);
        let layers = request.config["text_config"]["num_hidden_layers"]
            .as_u64()
            .unwrap() as usize;
        let columns = request
            .metadata
            .batch_layout
            .as_ref()
            .unwrap()
            .state_columns
            .len();
        request
            .metadata
            .buffers
            .iter_mut()
            .find(|b| b.name == "QwenRowPointers")
            .unwrap()
            .shape[0] += LANES * PACKET_COLUMNS.len();
        for name in ["Input", "Token", "TokenStatus", "Step"] {
            request.metadata.buffers.push(
                serde_json::from_value(json!({
                    "name":name,"dtype":"i32","shape":[4096],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write"
                }))
                .unwrap(),
            );
        }
        request.kernel_contracts.push(
            serde_json::from_value(json!({
                "name":"qwen_rows/packet/tokens","symbol":"test","grid":[1,1,1],"block":[32,1,1],
                "shared_memory_bytes":0,"cooperative":false,"args":[],
                "module":{"file":"test","sha256":"0".repeat(64)},
                "source":{"file":"test","sha256":"0".repeat(64)},
                "host_abi":{"file":"test","sha256":"0".repeat(64)}
            }))
            .unwrap(),
        );
        let (adapter, _) = create(request).unwrap();
        let mut count = None;
        for batch in [1, 2, 8, 128] {
            let segments = (0..batch)
                .map(|i| BatchSegment {
                    slot: 200 - i,
                    tokens: 1,
                })
                .collect::<Vec<_>>();
            let (calls, bindings) = adapter.batch(&segments, true).unwrap();
            assert_eq!(*count.get_or_insert(calls.len()), calls.len());
            assert!(
                calls.iter().all(
                    |c| c.sequence.is_none() && matches!(c.operation, Operation::Kernel { .. })
                )
            );
            assert!(
                matches!(&calls[0].operation, Operation::Kernel { name } if name == "qwen_rows/packet/tokens")
            );
            assert!(
                matches!(&calls.last().unwrap().operation, Operation::Kernel { name } if name == "qwen_rows/packet/outputs")
            );
            let base = LANES * layers * columns;
            assert_eq!(bindings.len(), base + LANES * PACKET_COLUMNS.len());
            for (lane, segment) in segments.iter().enumerate() {
                assert_eq!(
                    bindings[base + lane * PACKET_COLUMNS.len() + 1],
                    Some((segment.slot, "Token".into()))
                );
            }
            if batch < LANES {
                assert!(
                    bindings[base + batch * PACKET_COLUMNS.len()..]
                        .iter()
                        .all(Option::is_none)
                );
            }
            assert_eq!(adapter.batch(&segments, false).unwrap().1, bindings);
            if batch == LANES {
                continue;
            }
            let mut mixed = segments.clone();
            mixed.insert(
                0,
                BatchSegment {
                    slot: 999,
                    tokens: 3,
                },
            );
            let (calls, mixed_bindings) = adapter.batch(&mixed, true).unwrap();
            assert_eq!(bindings, mixed_bindings);
            assert!(
                calls
                    .iter()
                    .all(|c| c.sequence.is_none_or(|slot| slot == 999))
            );
            let gather = calls
                .iter()
                .find(|c| {
                    matches!(&c.operation,
                Operation::Kernel { name } if name == "qwen_rows/gather/norm")
                })
                .unwrap();
            assert_eq!(gather.sequence, Some(999));
            assert_eq!(
                gather.views["HeadHidden"].offset,
                batch * if small { 2048 } else { 5120 } * 2
            );
        }
    }
}

fn deferred_request() -> CreateRequest {
    let mut r = request_geometry(true);
    r.metadata.gdn_execution.as_mut().unwrap().decode =
        orinfer_model_sdk::operators::gdn::Decode::Deferred { capacity: 8 };
    let layers = r.config["text_config"]["layer_types"].as_array().unwrap();
    for (layer, kind) in layers.iter().enumerate() {
        if kind != "linear_attention" {
            continue;
        }
        for (suffix, dtype, shape) in [
            ("GdnLogCount", "i32", vec![16, 128]),
            ("GdnLogDelta", "f32", vec![8, 16, 128]),
            ("GdnLogGate", "f32", vec![8, 16]),
            ("GdnLogKey", "f16", vec![8, 16, 128]),
        ] {
            let name = format!("L{layer}_{suffix}");
            r.metadata.reset_buffers.push(name.clone());
            r.metadata.buffers.push(
                serde_json::from_value(json!({
                    "name":name,"dtype":dtype,"shape":shape,"layout":"native_contiguous",
                    "alignment":256,"access":"read_write"
                }))
                .unwrap(),
            );
        }
    }
    let columns = &mut r.metadata.batch_layout.as_mut().unwrap().state_columns;
    columns.extend(DEFERRED_COLUMNS.iter().map(|s| (*s).to_owned()));
    columns.sort_unstable();
    r.metadata
        .buffers
        .iter_mut()
        .find(|b| b.name == "QwenRowPointers")
        .unwrap()
        .shape = vec![LANES * layers.len() * columns.len()];
    r
}

#[test]
fn pending_gdn_updates_are_private_and_materialized_before_dense_execution() {
    let (adapter, plan) = create(deferred_request()).unwrap();
    let m = &plan.metadata;
    let names = |program: &str| {
        m.programs[program]
            .iter()
            .filter_map(|o| match o {
                Operation::Kernel { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let private = names("qwen_rows/layer0/private");
    assert!(
        private
            .iter()
            .position(|n| n.ends_with("/recurrent_materialize"))
            .unwrap()
            < private
                .iter()
                .position(|n| n.ends_with("/recurrent"))
                .unwrap()
    );
    let parallel = names("qwen_rows/state/layer0");
    assert_eq!(
        parallel
            .iter()
            .filter(|n| n.ends_with("/recurrent"))
            .count(),
        1
    );
    let (_, state) = adapter
        .batch(&[BatchSegment { slot: 3, tokens: 1 }], true)
        .unwrap();
    assert!(state.iter().any(|s| {
        s.as_ref()
            .is_some_and(|(slot, name)| *slot == 3 && name == "L0_GdnLogDelta")
    }));
    let (_, speculative) = create(with_mtp(deferred_request())).unwrap();
    let verify = &speculative.metadata.programs["qwen_rows/verify/layer0/private"];
    let materialize = verify
        .iter()
        .position(|o| {
            matches!(o, Operation::Kernel { name }
        if name.ends_with("/recurrent_materialize"))
        })
        .unwrap();
    let recurrent = verify
        .iter()
        .position(|o| {
            matches!(o, Operation::Kernel { name }
        if name.ends_with("/recurrent"))
        })
        .unwrap();
    assert!(materialize < recurrent);
    let mut bad = deferred_request();
    bad.metadata.reset_buffers.retain(|n| n != "L0_GdnLogCount");
    assert!(create(bad).is_err());
    let mut bad = deferred_request();
    bad.metadata
        .buffers
        .iter_mut()
        .find(|b| b.name == "L0_GdnLogDelta")
        .unwrap()
        .dtype = Dtype::F16;
    assert!(create(bad).is_err());
}

fn request_geometry(small: bool) -> CreateRequest {
    let dimensions = if small {
        (2048, 6144, 24, 16, 8, 2, true)
    } else {
        (5120, 17408, 64, 48, 24, 4, false)
    };
    dense_request(dimensions)
}

#[test]
fn attention_staging_is_shared_scratch_with_request_owned_inputs() {
    let staged = || {
        let mut request = request_geometry(true);
        for name in ["AttentionDecodedK", "AttentionDecodedV"] {
            request.metadata.buffers.push(
                serde_json::from_value(json!({
                    "name":name,"dtype":"f16","shape":[request.metadata.max_context,512],
                    "layout":"native_contiguous","alignment":256,"access":"read_write"
                }))
                .unwrap(),
            );
        }
        request
    };
    let (_, plan) = create(staged()).unwrap();
    let private = &plan.metadata.programs["qwen_rows/layer3/private"];
    let roles = private
        .iter()
        .filter_map(|op| match op {
            Operation::Kernel { name } => name.rsplit('/').next(),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        [
            "prepare",
            "attention_decode",
            "attention",
            "attention_merge"
        ]
    );
    assert!(
        !plan.metadata.programs["qwen_rows/state/layer3"]
            .iter()
            .any(|op| matches!(op,Operation::Kernel{name} if name.ends_with("attention_decode")))
    );
    let mut missing = staged();
    missing
        .metadata
        .buffers
        .retain(|b| b.name != "AttentionDecodedV");
    assert!(create(missing).is_err());
    let mut request = staged();
    request
        .metadata
        .reset_buffers
        .push("AttentionDecodedK".into());
    assert!(create(request).is_err());
}

#[test]
fn gdn_execution_is_required_and_cannot_be_inferred_from_buffers() {
    let mut missing = request();
    missing.metadata.gdn_execution = None;
    assert!(create(missing).is_err());
    let mut mismatch = deferred_request();
    mismatch.metadata.gdn_execution.as_mut().unwrap().decode =
        orinfer_model_sdk::operators::gdn::Decode::Dense;
    assert!(create(mismatch).is_err());
    let mut capacity = deferred_request();
    capacity.metadata.gdn_execution.as_mut().unwrap().decode =
        orinfer_model_sdk::operators::gdn::Decode::Deferred { capacity: 4 };
    assert!(create(capacity).is_err());
}

#[test]
fn request_parallel_history_update_precedes_recurrence_when_bound() {
    for separate_history in [false, true] {
        let mut request = request();
        if separate_history {
            let asset = json!({"file":"test", "sha256":"0".repeat(64)});
            request.kernel_contracts.push(
                serde_json::from_value(json!({
                    "name":"qwen_rows/state/layer0/history",
                    "module":asset,"source":asset,"host_abi":asset,
                    "symbol":"history","grid":[1,1,1],"block":[1,1,1],
                    "shared_memory_bytes":0,"cooperative":false,"args":[]
                }))
                .unwrap(),
            );
        }
        let (adapter, _) = create(request).unwrap();
        let segments = [
            BatchSegment { slot: 7, tokens: 1 },
            BatchSegment { slot: 2, tokens: 1 },
        ];
        let (calls, _) = adapter.batch(&segments, true).unwrap();
        let state: Vec<_> = calls
            .iter()
            .filter_map(|call| match &call.operation {
                Operation::Kernel { name } => name.strip_prefix("qwen_rows/state/layer0/"),
                _ => None,
            })
            .collect();
        let expected = if separate_history {
            vec!["gates", "conv", "history", "recurrent", "gated_norm"]
        } else {
            vec!["gates", "conv", "recurrent", "gated_norm"]
        };
        assert_eq!(state, expected);
    }
}

#[test]
fn chunked_gdn_prefill_preserves_sequential_decode_and_verification() {
    let mut request = with_mtp(dense_request((2048, 6144, 24, 16, 8, 2, true)));
    request.metadata.gdn_execution.as_mut().unwrap().prefill =
        orinfer_model_sdk::operators::gdn::Prefill::Chunked;
    for (name, dtype, width) in [
        ("GdnChunkW", "f16", 128),
        ("GdnChunkU", "f16", 128),
        ("GdnChunkD", "f16", 32),
        ("GdnChunkG", "f32", 2),
    ] {
        request.metadata.buffers.push(
            serde_json::from_value(json!({
                "name":name,"dtype":dtype,"shape":[16,4096,width],
                "layout":"native_contiguous","alignment":256,"access":"read_write"
            }))
            .unwrap(),
        );
    }
    let mut invalid: CreateRequest =
        serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
    invalid.metadata.buffers.retain(|b| b.name != "GdnChunkG");
    assert!(create(invalid).is_err());
    let (adapter, plan) = create(request).unwrap();
    let names = |program: &str| {
        plan.metadata.programs[program]
            .iter()
            .filter_map(|op| match op {
                Operation::Kernel { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let prefill = names("prefill");
    assert_eq!(
        prefill
            .iter()
            .filter(|n| n.ends_with("/recurrent_prepare"))
            .count(),
        18
    );
    assert_eq!(
        prefill
            .iter()
            .filter(|n| n.ends_with("/recurrent_chunk"))
            .count(),
        18
    );
    for program in ["decode", "verify"] {
        let calls = names(program);
        assert!(
            calls
                .iter()
                .all(|n| !n.ends_with("/recurrent_prepare") && !n.ends_with("/recurrent_chunk"))
        );
    }
    let (calls, _) = adapter
        .batch(&[BatchSegment { slot: 0, tokens: 1 }], false)
        .unwrap();
    assert!(calls.iter().all(|call| match &call.operation {
        Operation::Kernel { name } => !name.ends_with("/recurrent_chunk"),
        _ => true,
    }));
}

#[test]
fn four_billion_reuses_tied_dense_row_and_mtp_programs() {
    let request = with_mtp(dense_request((2560, 9216, 32, 32, 16, 4, true)));
    let (adapter, plan) = create(request).unwrap();
    let layout = plan.metadata.batch_layout.as_ref().unwrap();
    assert_eq!(layout.hidden, 2560);
    assert_eq!(plan.metadata.prefill_plans[0].stages.len(), 32);
    assert!(plan.metadata.programs.contains_key("mtp_warm"));
    let (calls, _) = adapter
        .batch(
            &[
                BatchSegment { slot: 2, tokens: 1 },
                BatchSegment { slot: 5, tokens: 3 },
            ],
            true,
        )
        .unwrap();
    assert!(calls.iter().any(|c| c.sequence == Some(5)));
}

fn dense_request(
    (h, f, layers, hv, qh, kvh, tied): (usize, usize, usize, usize, usize, usize, bool),
) -> CreateRequest {
    let config = json!({"model_type":"qwen3_5","text_config":{"hidden_size":h,
        "intermediate_size":f,"num_hidden_layers":layers,"vocab_size":248320,
        "linear_num_key_heads":16,"linear_num_value_heads":hv,"linear_key_head_dim":128,
        "linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        "num_attention_heads":qh,"num_key_value_heads":kvh,"head_dim":256,"tie_word_embeddings":tied,
        "layer_types":(0..layers).map(|i|if i%4==3 {"full_attention"} else {"linear_attention"}).collect::<Vec<_>>()},
        "quantization_config":{"quant_method":"orinfer_e8p_q4_int8","version":1,"rotation":"signed-block128",
        "scale":"fp16-output-row","compute_dtype":"int8_quality","coefficients":[4,1],
        "basis_tensor":"orinfer.e8p_q4.basis","layout":"u32-low-coarse-high-residual","embedding":"e8p_q4"}});
    let signature = Configuration::parse(config.clone()).unwrap().signature;
    let mut buffers = vec![];
    let mut q4 = |name: String, n: usize, k: usize| {
        for (suffix, dtype, shape) in [
            ("packed", "u32", vec![k / 128, n, 16]),
            ("scale", "f16", vec![n]),
        ] {
            buffers.push(
                json!({"name":format!("{name}.{suffix}"),"dtype":dtype,"shape":shape,
                "layout":if suffix=="packed" {"e8p-q4-k128-u32"} else {"native_contiguous"},"alignment":256,"access":"read",
                "data":{"tensor":format!("{name}.{suffix}"),"sha256":"0".repeat(64)}}),
            );
        }
    };
    q4(
        if tied {
            "model.language_model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        }
        .into(),
        248320,
        h,
    );
    if !tied {
        q4("model.language_model.embed_tokens.weight".into(), 248320, h);
    }
    for i in 0..layers {
        let p = format!("model.language_model.layers.{i}");
        for (name, n, k) in [("gate_proj", f, h), ("up_proj", f, h), ("down_proj", h, f)] {
            q4(format!("{p}.mlp.{name}.weight"), n, k);
        }
        let projections: &[(&str, usize)] = if i % 4 == 3 {
            &[
                ("self_attn.q_proj", 2 * qh * 256),
                ("self_attn.k_proj", kvh * 256),
                ("self_attn.v_proj", kvh * 256),
            ]
        } else {
            &[
                ("linear_attn.in_proj_qkv", (32 + hv) * 128),
                ("linear_attn.in_proj_z", hv * 128),
            ]
        };
        for (name, n) in projections {
            q4(format!("{p}.{name}.weight"), *n, h);
        }
    }
    for (name, dtype, shape) in [
        ("basis", "i8", vec![256, 8]),
        ("selectors", "u32", vec![256, 1]),
    ] {
        buffers.push(
            json!({"name":format!("orinfer.e8p_q4.{name}"),"dtype":dtype,"shape":shape,
            "layout":"native_contiguous","alignment":256,"access":"read",
            "data":{"tensor":format!("orinfer.e8p_q4.{name}"),"sha256":"0".repeat(64)}}),
        );
    }
    buffers.extend([
        json!({"name":"Hidden","dtype":"f16","shape":[4096,h],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
        json!({"name":"R0","dtype":"f32","shape":[4096,h],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
        json!({"name":"QwenRowPointers","dtype":"u64","shape":[128*layers*STATE_COLUMNS.len()],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
        json!({"name":"QwenRowScratch","dtype":"u8","shape":[256],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
    ]);
    serde_json::from_value(json!({"config":config,"expected_signature":signature,
        "architecture":"qwen3_5","compute_policy":"int8_quality","build_bindings":true,
        "prefill_profiles":[{"tokens":4096,"min_tokens":1,"kind":"qwen_q4_rows"}],
        "metadata":{"gdn_execution":{"prefill":"sequential","decode":{"kind":"dense"}},"schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,"max_context":262144,
            "vocab":248320,"toolchain":{},"buffers":buffers,"reset_buffers":["Step"],"input":"Input","token":"Token",
            "status":"TokenStatus","logits":"Logits","position":"Step","weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
            "batch_layout":{"layers":[],"row_strides":{},"profiles":{},"hidden":h,"state_columns":STATE_COLUMNS},
            "kv_cache":{"direct_prefill":true,"demand_mapping":true,"buffers":{},"prefill_workspace":{}},
            "dynamic_batch_kernels":{"qwen_rows/layer0/pre/norm_rotate":{"name":"qwen_rows/layer0/pre/norm_rotate","capacity":4096,
                "grid":[{"op":"rows"},{"op":"constant","value":1},{"op":"constant","value":1}],"arguments":[]}}}})).unwrap()
}

#[test]
fn moe_reuses_dense_causal_rows_with_compressed_expert_banks() {
    let mut request = dense_request((2048, 512, 40, 32, 16, 2, false));
    request.config["model_type"] = json!("qwen3_5_moe");
    for (key, value) in [
        ("num_experts", 256),
        ("num_experts_per_tok", 8),
        ("moe_intermediate_size", 512),
        ("shared_expert_intermediate_size", 512),
    ] {
        request.config["text_config"][key] = json!(value);
    }
    request.expected_signature = Configuration::parse(request.config.clone())
        .unwrap()
        .signature;
    for b in &mut request.metadata.buffers {
        if b.name.contains(".mlp.") {
            b.name = b.name.replace(".mlp.", ".mlp.shared_expert.");
            b.data.as_mut().unwrap().tensor = b.name.clone();
        }
    }
    for layer in 0..40 {
        for (role, n, k) in [("gate_up_proj", 1024, 2048), ("down_proj", 2048, 512)] {
            let name = format!("model.language_model.layers.{layer}.mlp.experts.{role}");
            for (suffix, dtype, shape, layout) in [
                (
                    "packed",
                    "u32",
                    vec![256, k / 128, n, 16],
                    "e8p-q4-k128-u32",
                ),
                ("scale", "f16", vec![256, n], "native_contiguous"),
            ] {
                request.metadata.buffers.push(
                    serde_json::from_value(json!({"name":format!("{name}.{suffix}"),
                    "dtype":dtype,"shape":shape,"layout":layout,"alignment":256,"access":"read",
                    "data":{"tensor":format!("{name}.{suffix}"),"sha256":"0".repeat(64)}}))
                    .unwrap(),
                );
            }
        }
    }
    request.metadata.buffers.push(
        serde_json::from_value(json!({
            "name":"MoeLogits","dtype":"f32","shape":[4096,256],
            "layout":"native_contiguous","alignment":256,"access":"read_write"
        }))
        .unwrap(),
    );
    let copied = serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
    let (adapter, plan) = create(with_mtp(copied)).unwrap();
    assert_eq!(plan.metadata.prefill_plans[0].stages.len(), 40);
    let post = &plan.metadata.programs["qwen_rows/layer0/post"];
    assert!(
        post.iter()
            .any(|op| matches!(op,Operation::Kernel{name} if name.ends_with("/expert_down")))
    );
    assert!(
        post.iter()
            .any(|op| matches!(op,Operation::Kernel{name} if name.ends_with("/combine")))
    );
    let (calls, _) = adapter
        .batch_operation(
            &[
                BatchSegment { slot: 7, tokens: 3 },
                BatchSegment { slot: 2, tokens: 2 },
            ],
            packet::BatchOperation::Warm,
            true,
        )
        .unwrap();
    let experts: Vec<_> = calls
        .iter()
        .filter(|c| {
            matches!(&c.operation,Operation::Kernel{name}
        if name=="qwen_rows/mtp/expert_up")
        })
        .collect();
    assert_eq!(experts.len(), 1);
    assert_eq!(experts[0].sequence, None);
    let conditions: Vec<_> = calls
        .iter()
        .filter_map(|c| {
            matches!(&c.operation,
        Operation::Copy{destination,..} if destination=="MtpCondition")
            .then_some(c.sequence)
        })
        .collect();
    assert_eq!(conditions, [Some(7), Some(2)]);
    let mut planes: CreateRequest =
        serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
    for (name, dtype, size) in [
        ("MoeLow", "i8", 32768 * 2048 / 2),
        ("MoeHigh", "i8", 32768 * 2048 / 2),
        ("MoeSum", "i32", 32768),
    ] {
        planes.metadata.buffers.push(
            serde_json::from_value(json!({
                "name":name,"dtype":dtype,"shape":[size],"layout":"native_contiguous",
                "alignment":256,"access":"read_write"
            }))
            .unwrap(),
        );
    }
    let copied = serde_json::from_value(serde_json::to_value(&planes).unwrap()).unwrap();
    let (_, plan) = create(with_mtp(copied)).unwrap();
    let post = &plan.metadata.programs["qwen_rows/layer0/post"];
    for role in ["expert_up", "expert_down"] {
        let index = |suffix: &str| {
            post.iter()
                .position(|op| matches!(op,Operation::Kernel{name} if name.ends_with(suffix)))
                .unwrap()
        };
        assert!(index(&format!("/{role}_planes")) < index(&format!("/{role}")));
    }
    assert!(
        plan.metadata.programs["mtp_post"]
            .iter()
            .all(|op| { !matches!(op,Operation::Kernel{name} if name.ends_with("_planes")) })
    );
    let mut partition: CreateRequest =
        serde_json::from_value(serde_json::to_value(&planes).unwrap()).unwrap();
    for (name, size) in [
        ("MoeColdTileOffsets", 256),
        ("MoeColdTileCount", 1),
        ("MoeColdTileExpert", 1280),
        ("MoeColdTileRow", 1280),
    ] {
        partition.metadata.buffers.push(
            serde_json::from_value(json!({
                "name":name,"dtype":"i32","shape":[size],"layout":"native_contiguous",
                "alignment":256,"access":"read_write"
            }))
            .unwrap(),
        );
    }
    let copied = serde_json::from_value(serde_json::to_value(&partition).unwrap()).unwrap();
    let (_, plan) = create(with_mtp(copied)).unwrap();
    let post = &plan.metadata.programs["qwen_rows/layer0/post"];
    for role in ["expert_up", "expert_down"] {
        let index = |suffix: &str| {
            post.iter()
                .position(|op| matches!(op,Operation::Kernel{name} if name.ends_with(suffix)))
                .unwrap()
        };
        assert!(index(&format!("/{role}_cold_planes")) < index(&format!("/{role}_cold")));
        assert!(index(&format!("/{role}_cold")) < index(&format!("/{role}")));
    }
    assert!(
        plan.metadata.programs["mtp_post"]
            .iter()
            .all(|op| { !matches!(op,Operation::Kernel{name} if name.contains("_cold")) })
    );
    partition
        .metadata
        .buffers
        .retain(|b| b.name != "MoeColdTileRow");
    assert!(create(partition).is_err());
    planes.metadata.buffers.retain(|b| b.name != "MoeSum");
    assert!(create(planes).is_err());
    request
        .metadata
        .buffers
        .retain(|b| !b.name.ends_with("layers.0.mlp.experts.down_proj.scale"));
    assert!(create(request).is_err());
}

#[test]
fn medium_dense_reuses_rows_with_separate_embedding_and_head_banks() {
    let request = dense_request((4096, 12288, 32, 32, 16, 4, false));
    let (adapter, plan) = create(request).unwrap();
    let m = &plan.metadata;
    assert_eq!(m.prefill_plans[0].stages.len(), 32);
    let layout = m.batch_layout.as_ref().unwrap();
    assert_eq!(layout.hidden, 4096);
    assert_eq!(layout.row_strides["QKV"], 8192 * 2);
    assert_eq!(layout.row_strides["FullX"], 10240 * 2);
    for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
        assert!(m.buffers.iter().any(|b| b.name == format!("{name}.packed")));
    }
    let (calls, _) = adapter
        .batch(
            &[
                BatchSegment { slot: 2, tokens: 1 },
                BatchSegment { slot: 5, tokens: 3 },
            ],
            true,
        )
        .unwrap();
    assert!(calls.iter().any(|c| c.sequence == Some(5)));
    let mut wrong = dense_request((4096, 12288, 32, 32, 16, 4, false));
    wrong
        .metadata
        .buffers
        .retain(|b| b.name != "lm_head.weight.scale");
    assert!(create(wrong).is_err());
}

#[test]
fn small_dense_reuses_row_state_and_tied_head_without_duplicate_weight() {
    let (adapter, plan) = create(request_geometry(true)).unwrap();
    let m = &plan.metadata;
    assert_eq!(m.prefill_plans[0].stages.len(), 24);
    let layout = m.batch_layout.as_ref().unwrap();
    assert_eq!(layout.hidden, 2048);
    assert_eq!(layout.row_strides["QKV"], 6144 * 2);
    assert!(!m.buffers.iter().any(|b| b.name.starts_with("lm_head.")));
    let (calls, _) = adapter
        .batch(
            &[
                BatchSegment { slot: 3, tokens: 1 },
                BatchSegment { slot: 7, tokens: 2 },
            ],
            true,
        )
        .unwrap();
    assert!(calls.iter().any(|c| c.sequence == Some(7)));
    let mut wrong = request_geometry(true);
    wrong
        .metadata
        .buffers
        .retain(|b| b.name != "model.language_model.embed_tokens.weight.scale");
    assert!(create(wrong).is_err());
}

#[test]
fn quantized_vision_inserts_shared_projection_stages_before_the_epilogue() {
    let mut m = request_geometry(true).metadata;
    m.vision = Some(serde_json::from_value(json!({
        "hidden":2048,"patch_size":16,"temporal_patch_size":2,"merge_size":2,
        "max_patches":32768,"max_features":8192,"pixels":"Pixels","grid":"Grid",
        "length":"Length","output":"Output","features":"Features","feature_index":"Index",
        "mrope_positions":"MRope","image_token_id":248056,"vision_start_token_id":248053,
        "vision_end_token_id":248054,"plans":[{"patches":32768,"min_patches":4,"program":"vision"}]
    })).unwrap());
    let mut packed = m.buffers[0].clone();
    packed.name = "model.visual.blocks.0.attn.qkv.weight.packed".into();
    m.buffers.push(packed);
    vision(&mut m, 24, 2048).unwrap();
    let names: Vec<_> = m.programs["vision"]
        .iter()
        .filter_map(|op| match op {
            Operation::Kernel { name } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        &names[3..7],
        &[
            "qwen_rows/vision/layer0/qkv_rotate",
            "qwen_rows/vision/layer0/qkv_project",
            "qwen_rows/vision/layer0/qkv",
            "qwen_rows/vision/layer0/rope",
        ]
    );
    assert!(!names.contains(&"qwen_rows/vision/layer1/qkv_project"));
}

#[test]
fn two_stage_codec_reuses_rows_and_rejects_scalar_storage() {
    let (adapter, _) = create(request()).unwrap();
    assert!(
        adapter
            .batch(&[BatchSegment { slot: 0, tokens: 1 }], false)
            .is_ok()
    );
    let mut wrong = request();
    wrong.config["quantization_config"]["quant_method"] = json!("orinfer_q4_int8");
    assert!(create(wrong).is_err());
    let mut wrong = request();
    wrong.metadata.buffers[0].dtype = Dtype::U8;
    assert!(create(wrong).is_err());
    let mut wrong = request();
    let packed = wrong
        .metadata
        .buffers
        .iter_mut()
        .find(|b| b.name.ends_with(".packed"))
        .unwrap();
    packed.layout = "e8p-q4-row-u32".into();
    assert!(create(wrong).is_err());
    let mut wrong = request();
    let packed = wrong
        .metadata
        .buffers
        .iter_mut()
        .find(|b| b.name.ends_with(".packed"))
        .unwrap();
    packed.shape.swap(0, 1);
    assert!(create(wrong).is_err());
    let mut wrong = request();
    wrong.config["quantization_config"]["coefficients"] = json!([3, 1]);
    assert!(create(wrong).is_err());
}

#[test]
fn mixed_rows_share_computation_without_padding_and_keep_private_owners() {
    let (adapter, plan) = create(request()).unwrap();
    let segments = [
        BatchSegment { slot: 7, tokens: 1 },
        BatchSegment {
            slot: 2,
            tokens: 17,
        },
    ];
    let (calls, _) = adapter.batch(&segments, true).unwrap();
    let shared:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/pre/norm_rotate")).collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].sequence, None);
    assert_eq!(shared[0].launch.as_ref().unwrap().grid[0], 18);
    let conv:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/private/conv")).collect();
    assert_eq!(
        conv.iter().map(|call| call.sequence).collect::<Vec<_>>(),
        [Some(2)]
    );
    assert_eq!(conv[0].views["QKV"].offset, 10240 * 2);
    assert!(!conv[0].views.contains_key("Q")); // compact causal scratch is reused, never offset as token-major data
    assert!(calls.iter().any(|call| call.sequence.is_none() && matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/state/layer0/conv")));
    let gather:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/gather/norm")).collect();
    assert_eq!(gather[1].views["HeadHidden"].offset, 5120 * 2);
    assert_eq!(plan.metadata.prefill_plans.len(), 1);
    assert!(
        plan.metadata.programs["head"]
            .iter()
            .any(|op| matches!(op,Operation::Kernel{name} if name=="qwen_rows/head_last/project"))
    );
    assert!(
        !plan.metadata.programs["head"]
            .iter()
            .any(|op| matches!(op,Operation::Kernel{name} if name=="qwen_rows/head/project"))
    );
}

#[test]
fn private_pointer_columns_are_a_pinned_semantic_contract() {
    let mut r = request();
    r.metadata
        .batch_layout
        .as_mut()
        .unwrap()
        .state_columns
        .clear();
    assert!(create(r).is_err());
    let mut r = request();
    r.metadata.batch_layout.as_mut().unwrap().state_columns[0] = "DifferentHistory".into();
    assert!(create(r).is_err());
}

#[test]
fn stages_preserve_the_full_forward_and_only_shared_live_values() {
    let (_, plan) = create(request()).unwrap();
    let m = plan.metadata;
    let p = &m.prefill_plans[0];
    assert_eq!(p.stages.len(), 64);
    assert_eq!(p.live_buffers, ["Hidden", "R0"]);
    let joined: Vec<_> = p
        .stages
        .iter()
        .flat_map(|s| m.programs[s].clone())
        .collect();
    assert_eq!(joined, m.programs["prefill"]);
    for s in &p.stages {
        assert_eq!(m.row_execution.as_ref().unwrap().program_capacity(s), 4096);
        assert!(m.kv_cache.as_ref().unwrap().growth.contains_key(s));
    }
}

fn mtp_request() -> CreateRequest {
    with_mtp(request())
}

fn with_mtp(mut r: CreateRequest) -> CreateRequest {
    let configuration = Configuration::parse(r.config.clone()).unwrap();
    let t = configuration.text();
    for (name, width, dtype) in [
        ("MtpEmbedding", t.hidden_size, "f16"),
        ("MtpConcat", 2 * t.hidden_size, "f16"),
        ("MtpHidden", t.hidden_size, "f16"),
        ("MtpNorm", t.hidden_size, "f16"),
        ("MtpR0", t.hidden_size, "f32"),
        ("MtpR1", t.hidden_size, "f32"),
        ("MtpFullX", t.full_x(), "f16"),
        ("MtpFullQ", t.mixer(), "f16"),
        ("MtpFullGate", t.mixer(), "f16"),
        ("MtpMixer", t.mixer(), "f16"),
        ("MtpMix", t.hidden_size, "f16"),
        ("MtpGateUp", 2 * t.intermediate_size, "f16"),
    ] {
        r.metadata.buffers.push(
            serde_json::from_value(json!({
                "name":name,"dtype":dtype,"shape":[512,width],"layout":"native_contiguous",
                "alignment":256,"access":"read_write"
            }))
            .unwrap(),
        );
    }
    r.metadata.mtp=Some(serde_json::from_value(json!({
        "position":"MtpStep","input":"MtpInput","token":"MtpToken","status":"MtpStatus",
        "verification_tokens":"SequenceTokens","verification_status":"SequenceStatus",
        "draft_logits":"MtpLogits","verification_logits":"SequenceLogits","feature_index":"MtpFeatureIndex",
        "accepted_inputs":"AcceptedInputs","target_length":"SeqLength","draft_program":"mtp_draft",
        "hidden_ring":"MtpTargetHidden","commit_always":true,"shared_verification_state":true,
        "default_verification_tokens":4,
        "warm_plans":[{"tokens":512,"min_tokens":1,"program":"mtp_warm","head_program":"mtp_head"}],
        "capture_plans":[{"tokens":4096,"program":"capture"}],
        "verification_plans":[{"tokens":8,"min_tokens":2,"program":"verify","restore_program":"restore","capture_program":"capture"}]
    })).unwrap());
    r
}

#[test]
fn owner_parallel_commit_keeps_acceptance_and_compact_stride_private() {
    let mut r = mtp_request();
    let layers = r.config["text_config"]["num_hidden_layers"]
        .as_u64()
        .unwrap() as usize;
    for name in ["AcceptedInputs", "SegmentLength"] {
        if !r.metadata.buffers.iter().any(|b| b.name == name) {
            r.metadata.buffers.push(serde_json::from_value(json!({"name":name,"dtype":"i32","shape":[1],"layout":"native_contiguous","alignment":256,"access":"read_write"})).unwrap());
        }
    }
    r.metadata.buffers.push(serde_json::from_value(json!({"name":"QwenCommitPointers","dtype":"u64","shape":[128*layers*8],"layout":"native_contiguous","alignment":256,"access":"read_write"})).unwrap());
    let (a, plan) = create(r).unwrap();
    assert_eq!(
        plan.metadata
            .mtp
            .as_ref()
            .unwrap()
            .commit_state_table
            .as_deref(),
        Some("QwenCommitPointers")
    );
    let segments = [
        BatchSegment { slot: 7, tokens: 3 },
        BatchSegment {
            slot: 31,
            tokens: 8,
        },
    ];
    let (calls, bindings) = a
        .batch_operation(&segments, packet::BatchOperation::Commit, true)
        .unwrap();
    assert!(calls.iter().all(|c| c.sequence.is_none()));
    assert!(calls.iter().any(
        |c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/commit/layer0/state")
    ));
    assert_eq!(bindings.len(), 2 * layers * 8);
    assert_eq!(bindings[3], Some((7, "AcceptedInputs".into())));
    assert_eq!(bindings[4], Some((7, "SegmentLength".into())));
    assert_eq!(
        bindings[layers * 8 + 3],
        Some((31, "AcceptedInputs".into()))
    );
    assert_eq!(bindings[layers * 8 + 4], Some((31, "SegmentLength".into())));
}

#[test]
fn mixed_roles_are_explicit_and_verification_heads_keep_all_positions() {
    let (a, _) = create(mtp_request()).unwrap();
    // Equal lengths have different semantics; they cannot identify roles.
    let segments = [
        BatchSegment { slot: 7, tokens: 2 },
        BatchSegment { slot: 3, tokens: 2 },
    ];
    let (calls, _) = a
        .batch_operation(
            &segments,
            packet::BatchOperation::MixedRows { prefills: 1 },
            true,
        )
        .unwrap();
    let prefix:Vec<_>=calls.iter().filter(|c|matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/verify/layer0/history")).map(|c|c.sequence).collect();
    assert_eq!(prefix, [Some(3)]);
    for (owner, destination, count) in [(7, "Logits", 1), (3, "SequenceLogits", 2)] {
        assert!(calls.iter().any(|c| c.sequence==Some(owner) && matches!(&c.operation,Operation::Copy{destination:d,bytes,..} if d==destination && *bytes==count*248320*4)));
    }
    let shared: Vec<_> = calls
        .iter()
        .filter(
            |c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/pre/qkv"),
        )
        .collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].sequence, None);
    assert!(
        a.batch_operation(
            &segments,
            packet::BatchOperation::MixedRows { prefills: 0 },
            true
        )
        .is_err()
    );
    assert!(
        a.batch_operation(
            &[BatchSegment { slot: 1, tokens: 1 }],
            packet::BatchOperation::Verification,
            true
        )
        .is_err()
    );
}

#[test]
fn indirect_packet_rebinds_owners_and_preserves_scalar_launches_and_views() {
    let (_, built) = create(mtp_request()).unwrap();
    let mut m = built.metadata;
    m.packet_table = Some(serde_json::from_value(json!({
        "buffer":"PacketPointers", "columns":["State","SequenceLogits"],
        "kernels":{"private":"indirect"},
        "scalar_arguments":{"private":[[0,2]]},
        "copies":[{"source":"HeadLogits","destination":"SequenceLogits","kernel":"copy","row_bytes":400}]
    })).unwrap());
    let kernel = |name, args| {
        serde_json::from_value(json!({
            "name":name,"module":{"file":"x","sha256":"0".repeat(64)},
            "source":{"file":"x","sha256":"0".repeat(64)},
            "host_abi":{"file":"x","sha256":"0".repeat(64)},"symbol":"kernel",
            "grid":[1,1,1],"block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,"args":args
        }))
        .unwrap()
    };
    let contracts = vec![
        kernel(
            "private",
            json!([{"kind":"i32","value":8},{"kind":"buffer","name":"State"}]),
        ),
        kernel(
            "indirect",
            json!([{"kind":"buffer","name":"PacketPointers"},{"kind":"i32","value":0},{"kind":"i32","value":8}]),
        ),
    ];
    m.dynamic_batch_kernels.insert("copy".into(), serde_json::from_value(json!({
        "name":"copy","capacity":8,"grid":[{"op":"rows"},{"op":"constant","value":1},{"op":"constant","value":1}],
        "arguments":[{"index":2,"value":{"op":"rows"}}]
    })).unwrap());
    let lower = |m: &Manifest, slots: [usize; 2]| {
        let segments = [
            BatchSegment {
                slot: slots[0],
                tokens: 3,
            },
            BatchSegment {
                slot: slots[1],
                tokens: 8,
            },
        ];
        let plan = vec![
            Invocation {
                operation: Operation::Kernel {
                    name: "private".into(),
                },
                sequence: Some(slots[0]),
                views: BTreeMap::new(),
                launch: Some(orinfer_model_sdk::operators::dynamic::RowLaunch {
                    grid: [3, 1, 1],
                    arguments: vec![(0, 3)],
                }),
            },
            Invocation {
                operation: Operation::Copy {
                    source: "HeadLogits".into(),
                    destination: "SequenceLogits".into(),
                    bytes: 800,
                },
                sequence: Some(slots[1]),
                views: BTreeMap::from([(
                    "HeadLogits".into(),
                    BufferView {
                        buffer: "HeadLogits".into(),
                        offset: 1200,
                    },
                )]),
                launch: None,
            },
            Invocation {
                operation: Operation::Copy {
                    source: "SharedSource".into(),
                    destination: "SharedDestination".into(),
                    bytes: 32,
                },
                sequence: Some(slots[1]),
                views: BTreeMap::new(),
                launch: None,
            },
        ];
        crate::packet::indirect(
            m,
            &segments,
            &crate::packet::Operands::from_kernels(&contracts),
            (plan, vec![Some((slots[0], "ExistingState".into()))]),
        )
        .unwrap()
    };
    let (first, bindings) = lower(&m, [7, 31]);
    let (next, next_bindings) = lower(&m, [2, 19]);
    assert_eq!(first, next);
    assert_ne!(bindings, next_bindings);
    assert!(first.iter().all(|c| c.sequence.is_none()));
    assert_eq!(first[0].launch.as_ref().unwrap().arguments, [(2, 3)]);
    assert_eq!(first[1].launch.as_ref().unwrap().arguments, [(2, 2)]);
    assert_eq!(first[1].views["PacketPointers"].offset, 16);
    assert_eq!(first[1].views["HeadLogits"].offset, 1200);
    assert!(matches!(
        &first[2].operation,
        Operation::Copy { bytes: 32, .. }
    ));
    assert_eq!(
        bindings,
        [
            Some((7, "ExistingState".into())),
            Some((7, "State".into())),
            Some((7, "SequenceLogits".into())),
            Some((31, "State".into())),
            Some((31, "SequenceLogits".into()))
        ]
    );
    m.mtp = None;
    let (ordinary, ordinary_bindings) = lower(&m, [7, 31]);
    assert_eq!(ordinary, first);
    assert_eq!(ordinary_bindings, bindings);
}

#[test]
fn draft_warm_joins_projections_and_keeps_conditions_private() {
    let (a, _) = create(mtp_request()).unwrap();
    let segments = [
        BatchSegment { slot: 7, tokens: 3 },
        BatchSegment { slot: 3, tokens: 2 },
    ];
    let (calls, _) = a
        .batch_operation(&segments, packet::BatchOperation::Warm, true)
        .unwrap();
    let fc: Vec<_> = calls
        .iter()
        .filter(|c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/mtp/fc"))
        .collect();
    assert_eq!(fc.len(), 1);
    assert_eq!(fc[0].sequence, None);
    let conditions:Vec<_>=calls.iter().filter(|c| matches!(&c.operation,Operation::Copy{destination,..} if destination=="MtpCondition")).map(|c|c.sequence).collect();
    assert_eq!(conditions, [Some(7), Some(3)]);
    assert!(
        !calls.iter().any(
            |c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/head/project")
        )
    );
    let endpoints = [
        BatchSegment { slot: 7, tokens: 1 },
        BatchSegment { slot: 3, tokens: 1 },
    ];
    let (head, _) = a
        .batch_operation(&endpoints, packet::BatchOperation::DraftHead, true)
        .unwrap();
    assert!(!head.iter().any(
        |c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/mtp/final_norm")
    ));
    assert_eq!(head.iter().filter(|c|matches!(&c.operation,Operation::Copy{destination,..} if destination=="MtpLogits")).count(),2);
}

#[test]
fn invalid_packet_and_unquantized_operands_fail_before_execution() {
    let (adapter, _) = create(request()).unwrap();
    for segments in [
        vec![],
        vec![BatchSegment { slot: 1, tokens: 0 }],
        vec![BatchSegment {
            slot: 1,
            tokens: 4097,
        }],
        vec![
            BatchSegment { slot: 1, tokens: 1 },
            BatchSegment { slot: 1, tokens: 1 },
        ],
    ] {
        assert!(adapter.batch(&segments, true).is_err());
    }
    let mut wrong = request();
    wrong.metadata.buffers[0].dtype = Dtype::F16;
    assert!(create(wrong).is_err());
    let mut wrong = request();
    wrong.prefill_profiles[0].min_tokens = 0;
    assert!(create(wrong).is_err());
    let mut wrong = request();
    wrong.config["quantization_config"]["rotation"] = json!("none");
    assert!(create(wrong).is_err());
}
