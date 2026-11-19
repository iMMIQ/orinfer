//! Shared dense/MoE work with request-private recurrence and sparse attention.
use crate::{
    artifact::Result,
    execution::Invocation,
    model::{Manifest, Operation},
};
use orinfer_model_sdk::architecture::BatchSegment;
use std::collections::BTreeMap;

const PROFILES: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];
const POINTERS: &str = "FlashBatchPointers";

pub(super) fn stages() -> Vec<(String, usize)> {
    let pre = 12;
    let post = 29;
    let mut out = vec![("begin".into(), 1), ("head".into(), 9)];
    for layer in 0..48 {
        let attention = layer % 4 == 3;
        if layer == 1 {
            out.push((format!("layer{layer}/ple"), 7));
        }
        out.extend([
            (
                format!("layer{layer}/pre"),
                pre - usize::from(attention) + usize::from(layer == 1),
            ),
            (format!("layer{layer}/post"), post - usize::from(attention)),
            (
                format!("layer{layer}/private"),
                if attention {
                    19
                } else if layer == 1 {
                    4
                } else {
                    2
                },
            ),
        ]);
    }
    out
}

pub(super) fn register(m: &mut Manifest) -> Result<()> {
    if !super::unified_decode::present(m)
        || m.dynamic_batch_kernels["flash_decode/begin/k0"].capacity
            != super::unified_decode::COMPUTATION_CAPACITY
        || m.batch_profiles != PROFILES
    {
        return Err("Flash decode requires 2/4/8/16/32/64/128 profiles".into());
    }
    let layout = m
        .batch_layout
        .as_ref()
        .ok_or("Missing Flash batch layout")?;
    if layout.hidden != 2560
        || layout.layers
            != (0..48)
                .map(|i| {
                    if i % 4 == 3 {
                        "full_attention".into()
                    } else {
                        "linear_attention".into()
                    }
                })
                .collect::<Vec<String>>()
        || layout.row_strides.get("M1_Embedding") != Some(&5120)
        || layout.row_strides.get("M1_Ple") != Some(&5120)
        || layout.row_strides.get("Logits") != Some(&(m.vocab * 4))
        || layout.row_strides.get("Selected") != Some(&8)
    {
        return Err("Invalid Flash decode row/state layout".into());
    }
    let columns = layout.state_columns.clone();
    if !columns.is_empty() {
        if columns.windows(2).any(|p| p[0] >= p[1])
            || m.buffers
                .iter()
                .find(|b| b.name == POINTERS)
                .is_none_or(|b| {
                    b.dtype != crate::artifact::Dtype::U64
                        || { b.shape != [128 * (48 * columns.len() + 10)] }
                        || b.data.is_some()
                })
            || columns.iter().any(|name| {
                !(0..48).any(|layer| {
                    let target = private_role(name, layer);
                    m.buffers
                        .iter()
                        .any(|b| b.name == target && b.data.is_none())
                })
            })
        {
            return Err("Invalid Flash private address-table layout".into());
        }
        m.state_pointer_table = Some(crate::model::StatePointerTable {
            buffer: POINTERS.into(),
            max_rows: 128,
        });
    } else if m.buffers.iter().any(|b| b.name == POINTERS) {
        return Err("Flash private address table needs column identities".into());
    }
    super::unified_decode::register(m)
}

fn private_role(column: &str, layer: usize) -> String {
    column.strip_prefix("State_").map_or_else(
        || column.to_owned(),
        |suffix| format!("State_{layer}_{suffix}"),
    )
}

pub(super) fn state_bindings(
    m: &Manifest,
    segments: &[BatchSegment],
) -> Result<orinfer_model_sdk::packet::StateBindings> {
    let Some(layout) = &m.batch_layout else {
        return Ok(vec![]);
    };
    if layout.state_columns.is_empty() {
        return Ok(vec![]);
    }
    if segments.iter().any(|s| s.tokens != 1) {
        super::mixed::validate_segments(m, segments)?;
        let decoders: Vec<_> = segments.iter().filter(|s| s.tokens == 1).cloned().collect();
        return state_bindings(m, &decoders);
    }
    // A prefill-only packet clears every decoder pointer in the shared table.
    if !segments.is_empty() {
        crate::packet::Rows::packet(segments, 128, 128)?;
    }
    let roles = crate::packet::layer_roles(m, 48, &layout.state_columns, |role, layer| {
        [private_role(role, layer)]
    });
    let rows = super::unified_decode::CAPACITY;
    let mut result = crate::packet::private_bindings(segments, rows, &roles);
    let io = [
        Some("M1_Embedding".to_owned()),
        Some("M1_Ple".to_owned()),
        Some(m.position.clone()),
        Some(m.token.clone()),
        Some(m.status.clone()),
        Some(m.logits.clone()),
        m.vision.as_ref().map(|_| "Features".into()),
        m.vision.as_ref().map(|_| "FeatureIndex".into()),
        m.mtp.as_ref().map(|_| "MtpHiddenRing".into()),
        Some("State_ple".into()),
    ];
    result.extend(crate::packet::private_bindings(segments, rows, &io));
    Ok(result)
}
pub(super) fn plan(m: &Manifest, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
    if segments.iter().any(|s| s.tokens != 1) {
        return super::mixed::plan(m, segments);
    }
    crate::packet::Rows::packet(segments, 128, 128)?;
    unified_plan(m, segments)
}

fn unified_plan(m: &Manifest, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
    use super::unified_decode::PREFIX;
    let rows = m
        .row_execution
        .as_ref()
        .ok_or("Missing decode specialization")?;
    let mut out = Vec::new();
    let append_rows = |out: &mut Vec<Invocation>, program: &str| -> Result<()> {
        out.extend(rows.specialize(
            program,
            &m.programs[program],
            &m.dynamic_batch_kernels,
            segments.len(),
        )?);
        Ok(())
    };
    let append_private = |out: &mut Vec<Invocation>, program: &str| -> Result<()> {
        for op in &m.programs[program] {
            let mut call = crate::packet::invocation(op.clone(), None, &BTreeMap::new());
            if let Operation::Kernel { name } = &call.operation {
                call.launch = Some(
                    m.dynamic_batch_kernels
                        .get(name)
                        .ok_or("Missing decode state launch")?
                        .launch(segments.len())?,
                );
            }
            out.push(call);
        }
        Ok(())
    };
    append_rows(&mut out, "flash_decode/packet/inputs")?;
    append_rows(&mut out, &format!("{PREFIX}/begin"))?;
    for layer in 0..48 {
        if layer == 1 {
            append_rows(&mut out, &format!("{PREFIX}/layer1/ple"))?;
            append_private(&mut out, "flash_decode_ple")?;
        }
        append_rows(&mut out, &format!("{PREFIX}/layer{layer}/pre"))?;
        append_private(&mut out, &format!("flash_decode_state/layer{layer}"))?;
        append_rows(&mut out, &format!("{PREFIX}/layer{layer}/post"))?;
    }
    append_rows(&mut out, "flash_decode/packet/end")?;
    append_rows(&mut out, &format!("{PREFIX}/head"))?;
    append_rows(&mut out, "flash_decode/packet/outputs")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    fn manifest() -> Manifest {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,"weight_parameters":0,"weight_scope":"test"
        })).unwrap();
        m.batch_profiles = PROFILES.into();
        m.batch_layout = Some(orinfer_model_sdk::architecture::BatchLayout {
            hidden: 2560,
            layers: (0..48)
                .map(|i| {
                    if i % 4 == 3 {
                        "full_attention".into()
                    } else {
                        "linear_attention".into()
                    }
                })
                .collect(),
            row_strides: BTreeMap::from([
                ("M1_Embedding".into(), 5120),
                ("M1_Ple".into(), 5120),
                ("Logits".into(), 64),
                ("Selected".into(), 8),
                ("Residual".into(), 20480),
            ]),
            profiles: Default::default(),
            segment_tokens: 0,
            state_columns: vec![],
            preparation_programs: BTreeMap::new(),
        });
        m.batch_layout.as_mut().unwrap().state_columns = vec!["Position".into()];
        for (name, dtype, shape) in [
            ("Position".to_owned(), "i32", vec![1]),
            (POINTERS.to_owned(), "u64", vec![128 * (48 + 10)]),
        ] {
            m.buffers.push(
                serde_json::from_value(serde_json::json!({
                    "name":name,"dtype":dtype,"shape":shape,"layout":"native_contiguous",
                    "alignment":256,"access":"read_write","data":null
                }))
                .unwrap(),
            );
        }
        for (role, stride) in &m.batch_layout.as_ref().unwrap().row_strides {
            m.buffers.push(
                serde_json::from_value(serde_json::json!({
                    "name":format!("FlashDecode_{role}"),"dtype":"u8","shape":[if matches!(role.as_str(), "Logits" | "Selected") {128*stride} else {1024*stride}],
                    "layout":"native_contiguous","alignment":256,"access":"read_write","data":null
                }))
                .unwrap(),
            );
        }
        for packet in ["inputs", "end", "outputs"] {
            let name = format!("flash_decode/packet/{packet}");
            m.dynamic_batch_kernels
                .insert(name.clone(), dynamic(&name, 128));
        }
        let entry = "flash_decode/begin/k0";
        m.dynamic_batch_kernels
            .insert(entry.into(), dynamic(entry, 1024));
        let variant = format!("{entry}/rows/1-1024");
        m.dynamic_batch_kernels
            .insert(variant.clone(), dynamic(&variant, 1024));
        register(&mut m).unwrap();
        m.programs.remove("flash_decode/local");
        m.prefill_plans = vec![crate::model::PrefillPlan {
            chunk_tokens: 4096,
            min_tokens: 1,
            prefill_program: "prefill".into(),
            head_program: "head".into(),
            stages: vec![],
            live_buffers: vec![],
        }];
        let endpoint = "prefill_m4096/end/k0";
        m.dynamic_batch_kernels
            .insert(endpoint.into(), dynamic(endpoint, 4096));
        let programs = m
            .programs
            .keys()
            .filter(|name| name.starts_with("flash_decode/"))
            .cloned()
            .collect::<BTreeSet<_>>();
        m.row_execution = Some(orinfer_model_sdk::operators::rows::RowExecution {
            capacity: 4096,
            program_capacities: programs
                .iter()
                .map(|p| {
                    (
                        p.clone(),
                        if p.contains("/packet/") || p.ends_with("/head") {
                            128
                        } else {
                            1024
                        },
                    )
                })
                .collect(),
            programs,
            variants: BTreeMap::from([(
                entry.into(),
                vec![orinfer_model_sdk::operators::rows::KernelVariant {
                    min_rows: 1,
                    max_rows: 1024,
                    multiple_of: 1,
                    kernels: vec![variant],
                }],
            )]),
        });
        for ops in m.programs.values() {
            for op in ops {
                if let Operation::Kernel { name } = op {
                    m.dynamic_batch_kernels.insert(
                        name.clone(),
                        dynamic(
                            name,
                            if name.contains("/packet/")
                                || name.starts_with("flash_decode/head/")
                                || name.starts_with("flash_decode_state/")
                            {
                                128
                            } else {
                                1024
                            },
                        ),
                    );
                }
            }
        }
        m
    }
    fn dynamic(
        name: &str,
        capacity: usize,
    ) -> orinfer_model_sdk::operators::dynamic::DynamicBatchKernel {
        use orinfer_model_sdk::operators::dynamic::{
            DynamicBatchKernel, RowArgument, RowExpression,
        };
        DynamicBatchKernel {
            name: name.into(),
            capacity,
            grid: [
                RowExpression::Constant { value: 1 },
                RowExpression::Constant { value: 1 },
                RowExpression::Rows,
            ],
            arguments: vec![RowArgument {
                index: 0,
                value: RowExpression::Rows,
            }],
        }
    }
    fn mixed_manifest() -> Manifest {
        let mut m = manifest();
        let layout = m.batch_layout.as_mut().unwrap();
        layout.profiles = BTreeMap::from([(512, "flash_recurrent".into())]);
        layout.segment_tokens = 512;
        m.prefill_batch_profiles = vec![orinfer_model_sdk::architecture::PrefillProfile {
            min_tokens: 0,
            tokens: 1024,
            kind: "flash_mixed_grouped".into(),
        }];
        for (name, bytes) in [
            ("FlashMixedResidual", 1024 * 20480),
            ("FlashMixed_Residual", 1024 * 20480),
        ] {
            m.buffers.push(serde_json::from_value(serde_json::json!({"name":name,"dtype":"u8","shape":[bytes],"layout":"native_contiguous","alignment":256,"access":"read_write","data":null})).unwrap());
        }
        register(&mut m).unwrap();
        for name in m.programs.values().flatten().filter_map(|op| match op {
            Operation::Kernel { name } if name.starts_with("causal_rows/") => Some(name),
            _ => None,
        }) {
            m.dynamic_batch_kernels.insert(
                name.clone(),
                dynamic(
                    name,
                    if name.contains("/verify/") || name.contains("/restore/") {
                        8
                    } else {
                        512
                    },
                ),
            );
        }
        for suffix in ["advance", "selections"] {
            let name = format!("causal_rows/verify/{suffix}");
            m.dynamic_batch_kernels
                .insert(name.clone(), dynamic(&name, 8));
        }
        super::super::causal_rows::configure(&mut m).unwrap();

        m
    }
    #[test]
    fn shared_mixed_body_uses_live_rows_and_gathers_the_common_residual() {
        let m = mixed_manifest();
        let segments = [
            BatchSegment {
                slot: 7,
                tokens: 512,
            },
            BatchSegment { slot: 3, tokens: 1 },
            BatchSegment {
                slot: 11,
                tokens: 1,
            },
        ];
        let plan = plan(&m, &segments).unwrap();
        assert!(!plan.iter().any(|i|matches!(&i.operation,Operation::Kernel{name} if name.starts_with("flash_mixed_m") || name.starts_with("flash_mixed_head_m"))));
        let begin=plan.iter().find(|i|matches!(&i.operation,Operation::Kernel{name} if name.starts_with("flash_decode/begin/"))).unwrap();
        assert_eq!(begin.launch.as_ref().unwrap().arguments, [(0, 514)]);
        let head = plan
            .iter()
            .find(
                |i| matches!(&i.operation,Operation::Kernel{name} if name=="flash_decode/head/k0"),
            )
            .unwrap();
        assert_eq!(head.launch.as_ref().unwrap().arguments, [(0, 3)]);
        assert_eq!(
            head.views["FlashDecode_Residual"].buffer,
            "FlashMixedHeadResidual"
        );
        let endpoints:Vec<_>=plan.iter().filter(|i|matches!(&i.operation,Operation::Copy{source,destination,..} if source=="FlashMixedResidual" && destination=="FlashMixedHeadResidual")).collect();
        assert_eq!(endpoints.len(), 3);
        for endpoint in endpoints {
            assert_eq!(
                endpoint.views["FlashMixedResidual"].buffer,
                "FlashDecode_Residual"
            );
        }
        assert!(
            !plan
                .iter()
                .any(|i| matches!(i.operation, Operation::Zero { .. }))
        );
    }
    #[test]
    fn mixed_endpoints_use_dynamic_prefill_template_at_segment_extent() {
        use orinfer_model_sdk::operators::dynamic::{
            DynamicBatchKernel, RowArgument, RowExpression,
        };
        let mut m = mixed_manifest();
        m.prefill_plans = vec![crate::model::PrefillPlan {
            chunk_tokens: 4096,
            min_tokens: 2,
            prefill_program: "prefill".into(),
            head_program: "head".into(),
            stages: vec![],
            live_buffers: vec![],
        }];
        let name = "prefill_m4096/end/k0";
        m.dynamic_batch_kernels.insert(
            name.into(),
            DynamicBatchKernel {
                name: name.into(),
                capacity: 4096,
                grid: std::array::from_fn(|_| RowExpression::Constant { value: 1 }),
                arguments: vec![RowArgument {
                    index: 1,
                    value: RowExpression::Rows,
                }],
            },
        );
        let segments = [
            BatchSegment {
                slot: 7,
                tokens: 128,
            },
            BatchSegment {
                slot: 3,
                tokens: 16,
            },
        ];
        let plan = plan(&m, &segments).unwrap();
        for segment in segments {
            let endpoint = plan.iter().find(|invocation| {
                invocation.sequence == Some(segment.slot)
                    && matches!(&invocation.operation, Operation::Kernel { name: n } if n == name)
            }).unwrap();
            assert_eq!(
                endpoint.launch.as_ref().unwrap().arguments,
                [(1, segment.tokens as i32)]
            );
        }
        assert!(!plan.iter().any(|invocation| matches!(&invocation.operation, Operation::Kernel { name } if name.starts_with("prefill_m16/") || name.starts_with("prefill_m128/"))));
    }
    #[test]
    fn mixed_prefill_and_decode_share_layer_weights_and_gather_only_endpoints() {
        let m = mixed_manifest();
        let segments = [
            BatchSegment {
                slot: 7,
                tokens: 128,
            },
            BatchSegment { slot: 3, tokens: 1 },
        ];
        let plan = plan(&m, &segments).unwrap();
        let bindings = state_bindings(&m, &segments).unwrap();
        assert_eq!(bindings[0], Some((3, "Position".into())));
        assert!(bindings[48..128 * 48].iter().all(Option::is_none));
        for layer in 0..48 {
            let prefix = format!("flash_decode/layer{layer}/pre/");
            assert_eq!(plan.iter().filter(|i| matches!(&i.operation, Operation::Kernel { name } if name.starts_with(&prefix))).count(), if layer%4==3 {11} else if layer==1 {13} else {12});
            for segment in segments {
                let name = format!("causal_rows/target/layer{layer}/private/k0");
                let private = plan
                    .iter()
                    .find(|i| {
                        i.sequence == Some(segment.slot)
                            && matches!(&i.operation, Operation::Kernel { name:n } if n == &name)
                    })
                    .unwrap();
                assert_eq!(private.sequence, Some(segment.slot));
                assert_eq!(
                    private.views["FlashMixedResidual"].offset,
                    if segment.slot == 7 { 20480 } else { 0 }
                );
            }
        }
        let endpoints: Vec<_> = plan.iter().filter(|i| matches!(&i.operation, Operation::Copy { source,destination,bytes:20480 } if source=="FlashMixedResidual" && destination=="FlashMixedHeadResidual")).map(|i| i.views["FlashMixedResidual"].offset).collect();
        assert_eq!(endpoints, [0, 128 * 20480]);
        assert!(plan.iter().any(|i| matches!(&i.operation,Operation::Kernel {name} if name.starts_with("flash_decode/head/"))));
        assert!(!plan.iter().any(|i| matches!(&i.operation,Operation::Kernel {name} if name.starts_with("prefill_m128/layer") || name.starts_with("decode/layer"))));
    }
    #[test]
    fn mixed_decoder_subgroups_share_private_kernels_and_preserve_table_ownership() {
        let mut m = mixed_manifest();
        for (role, stride) in &m.batch_layout.as_ref().unwrap().row_strides {
            let name = format!("FlashMixed_{role}");
            if !m.buffers.iter().any(|b| b.name == name) {
                m.buffers.push(
                    serde_json::from_value(serde_json::json!({
                        "name":name,"dtype":"u8","shape":[1024*stride],"layout":"native_contiguous",
                        "alignment":256,"access":"read_write","data":null
                    }))
                    .unwrap(),
                );
            }
        }
        for count in [2usize, 3, 4, 8] {
            let segments: Vec<_> = std::iter::once(BatchSegment {
                slot: 99,
                tokens: 128,
            })
            .chain((0..count).map(|lane| BatchSegment {
                slot: 20 + lane,
                tokens: 1,
            }))
            .collect();
            let bindings = state_bindings(&m, &segments).unwrap();
            assert_eq!(bindings.len(), 128 * (48 + 10));
            for lane in 0..count {
                assert_eq!(bindings[lane * 48], Some((20 + lane, "Position".into())));
            }
            assert!(bindings[count * 48..128 * 48].iter().all(Option::is_none));
            let plan = plan(&m, &segments).unwrap();
            let private: Vec<_> = plan.iter().filter(|i| matches!(&i.operation, Operation::Kernel {name} if name.starts_with("flash_decode_state/"))).collect();
            assert_eq!(private.len(), 36 * 3 + 12 * 19);
            assert!(private.iter().all(|i| i.sequence.is_none()));
            assert!(private.iter().all(|i| { i.views.is_empty() }));
            assert!(
                private
                    .iter()
                    .all(|i| i.launch.as_ref().unwrap().grid[2] == count as u32)
            );
            assert!(!plan.iter().any(|i| matches!(&i.operation, Operation::Kernel {name} if name.starts_with("flash_segment_m1/layer") && name.contains("/private/"))));
            let prefill = plan.iter().find(|i| matches!(&i.operation, Operation::Kernel {name} if name=="causal_rows/target/layer0/private/k0")).unwrap();
            assert_eq!(prefill.sequence, Some(99));
            assert_eq!(prefill.views["FlashMixedResidual"].offset, count * 20480);
        }
    }
    fn verification_manifest() -> Manifest {
        let mut m = mixed_manifest();
        m.mtp=Some(serde_json::from_value(serde_json::json!({
            "position":"MtpPosition","input":"MtpInput","token":"MtpToken","status":"MtpStatus",
            "verification_tokens":"VerificationTokens","verification_status":"VerificationStatus",
            "draft_logits":"DraftLogits","verification_logits":"VerificationLogits","feature_index":null,
            "accepted_inputs":"AcceptedInputs","target_length":"PositionOut","draft_program":"mtp_draft",
            "default_verification_tokens":4,"warm_plans":[],"capture_plans":[],"verification_plans":[],
            "batch_verification_tokens":[2,3,4,5,6,7,8]
        })).unwrap());
        m.mtp.as_mut().unwrap().verification_plans =
            vec![orinfer_model_sdk::mtp::VerificationPlan {
                tokens: 8,
                min_tokens: 2,
                program: "mtp_verify".into(),
                restore_program: "mtp_restore".into(),
                capture_program: "mtp_capture".into(),
            }];
        m.mtp.as_mut().unwrap().mixed_verification_tokens = vec![2, 4];
        super::super::mixed::register(&mut m).unwrap();
        m
    }
    #[test]
    fn causal_mixed_controls_use_live_segment_lengths_and_owners() {
        let mut m = verification_manifest();
        m.mtp.as_mut().unwrap().mixed_verification_tokens = (2..=8).collect();
        let marker = "causal_rows/verify/layer0/private/k0";
        m.dynamic_batch_kernels
            .insert(marker.into(), dynamic(marker, 8));
        super::super::mixed::register(&mut m).unwrap();
        for name in m.programs.values().flatten().filter_map(|op| {
            if let Operation::Kernel { name } = op {
                name.starts_with("causal_rows/").then_some(name)
            } else {
                None
            }
        }) {
            m.dynamic_batch_kernels
                .insert(name.clone(), dynamic(name, 512));
        }
        for name in ["advance", "selections"] {
            let name = format!("causal_rows/verify/{name}");
            m.dynamic_batch_kernels
                .insert(name.clone(), dynamic(&name, 8));
        }
        super::super::causal_rows::configure(&mut m).unwrap();
        for tokens in 2..=8 {
            let segments = [
                BatchSegment {
                    slot: 9,
                    tokens: 16,
                },
                BatchSegment { slot: 3, tokens },
            ];
            let plan = super::super::mixed::mixed_verification_plan(&m, &segments, true).unwrap();
            for call in &plan {
                let Operation::Kernel { name } = &call.operation else {
                    continue;
                };
                assert!(!name.starts_with("verify_m") || name.contains("/begin/"));
                if name.starts_with("causal_rows/") {
                    let count = if call.sequence == Some(3) { tokens } else { 16 };
                    assert!(matches!(call.sequence, Some(3 | 9)));
                    assert_eq!(call.launch.as_ref().unwrap().arguments, [(0, count as i32)]);
                }
            }
            assert!(plan.iter().any(|call| matches!(&call.operation, Operation::Kernel {name} if name=="causal_rows/verify/selections")));
        }
        let layout = m.batch_layout.as_mut().unwrap();
        layout.segment_tokens = 512;
        layout.profiles = BTreeMap::from([(512, "flash_recurrent".into())]);
        m.prefill_batch_profiles.retain(|p| p.tokens == 1024);
        super::super::mixed::register(&mut m).unwrap();
        super::super::causal_rows::configure(&mut m).unwrap();
        for tokens in [1, 2, 3, 7, 17, 129, 257, 511, 512] {
            let segments = [
                BatchSegment { slot: 9, tokens },
                BatchSegment { slot: 3, tokens: 2 },
            ];
            let plan = super::super::mixed::mixed_rows_plan(&m, &segments, 1, true).unwrap();
            assert!(plan.iter().any(|call| call.sequence==Some(9) && matches!(&call.operation,Operation::Kernel {name} if name=="causal_rows/target/layer0/private/k0")));
            assert!(plan.iter().any(|call| call.sequence==Some(3) && matches!(&call.operation,Operation::Kernel {name} if name=="causal_rows/verify/layer0/private/k0")));
        }
    }
    #[test]
    fn mixed_verification_shares_body_but_keeps_endpoint_ranges_and_commit_private() {
        let m = verification_manifest();
        for segments in [
            vec![
                BatchSegment {
                    slot: 8,
                    tokens: 128,
                },
                BatchSegment { slot: 3, tokens: 4 },
            ],
            vec![
                BatchSegment { slot: 3, tokens: 4 },
                BatchSegment {
                    slot: 8,
                    tokens: 512,
                },
                BatchSegment { slot: 6, tokens: 2 },
            ],
        ] {
            let p = super::super::mixed::mixed_verification_plan(&m, &segments, true).unwrap();
            let mut start = 0;
            let mut head = 0;
            for s in &segments {
                let verify = s.tokens < 16;
                let prefix = if verify { "verify" } else { "target" };
                for layer in 0..48 {
                    let i = p.iter().find(|i| i.sequence == Some(s.slot) && matches!(&i.operation, Operation::Kernel {name} if name==&format!("causal_rows/{prefix}/layer{layer}/private/k0"))).unwrap();
                    assert_eq!(i.sequence, Some(s.slot));
                    assert_eq!(i.views["FlashMixedResidual"].offset, start * 20480);
                }
                let i = p.iter().find(|i| i.sequence==Some(s.slot) && matches!(&i.operation, Operation::Copy {destination,..} if destination=="FlashMixedHeadResidual")).unwrap();
                assert_eq!(
                    i.views["FlashMixedResidual"].offset,
                    (start + if verify { 0 } else { s.tokens - 1 }) * 20480
                );
                assert_eq!(i.views["FlashMixedHeadResidual"].offset, head * 20480);
                let logits = p.iter().find(|i| i.sequence==Some(s.slot) && matches!(&i.operation,Operation::Copy {destination,..} if destination==if verify {"VerificationLogits"} else {&m.logits})).unwrap();
                assert_eq!(
                    logits.views["FlashDecode_Logits"].offset,
                    head * m.vocab * 4
                );
                start += s.tokens;
                head += if verify { s.tokens } else { 1 };
            }
        }
        for segments in [
            vec![
                BatchSegment {
                    slot: 0,
                    tokens: 128,
                },
                BatchSegment { slot: 0, tokens: 4 },
            ],
            vec![
                BatchSegment {
                    slot: 0,
                    tokens: 513,
                },
                BatchSegment { slot: 1, tokens: 4 },
            ],
            vec![
                BatchSegment { slot: 0, tokens: 4 },
                BatchSegment { slot: 1, tokens: 2 },
            ],
            vec![
                BatchSegment {
                    slot: 0,
                    tokens: 128,
                },
                BatchSegment { slot: 1, tokens: 1 },
            ],
            (0..17)
                .map(|slot| BatchSegment { slot, tokens: 8 })
                .chain([BatchSegment {
                    slot: 99,
                    tokens: 16,
                }])
                .collect(),
        ] {
            assert!(super::super::mixed::mixed_verification_plan(&m, &segments, true).is_err());
        }
    }
    #[test]
    fn verification_batches_keep_all_candidate_heads_and_private_commit_factors() {
        let m = verification_manifest();
        let segments = [
            BatchSegment { slot: 7, tokens: 4 },
            BatchSegment { slot: 3, tokens: 2 },
        ];
        let p = super::super::mixed::verification_plan(&m, &segments, true).unwrap();
        let copies:Vec<_>=p.iter().filter(|i|matches!(&i.operation,Operation::Copy {destination,..} if destination=="FlashMixedHeadResidual")).collect();
        assert_eq!(copies.len(), 2);
        assert!(matches!(
            copies[0].operation,
            Operation::Copy { bytes: 81920, .. }
        ));
        assert!(matches!(
            copies[1].operation,
            Operation::Copy { bytes: 40960, .. }
        ));
        assert_eq!(copies[1].views["FlashMixedHeadResidual"].offset, 81920);
        assert!(p.iter().any(|i|matches!(&i.operation,Operation::Copy {destination,bytes,..} if destination=="VerificationLogits" && *bytes==4*m.vocab*4) && i.sequence==Some(7)));
        assert!(p.iter().any(
            |i| matches!(&i.operation,Operation::Kernel {name} if name=="causal_rows/verify/selections")
                && i.sequence == Some(3)
        ));
        let conv=p.iter().position(|i|matches!(&i.operation,Operation::Copy {destination,..} if destination=="State_0_conv") && i.sequence==Some(7)).unwrap();
        assert!(
            matches!(&p[conv-1].operation,Operation::Kernel {name} if name=="causal_rows/verify/layer0/private/k1")
        );
        assert!(p.iter().any(|i|matches!(&i.operation,Operation::Kernel {name} if name=="causal_rows/verify/layer0/private/k3") && i.sequence==Some(7)));
        for bad in [
            vec![segments[0], segments[0]],
            vec![
                BatchSegment {
                    slot: 0,
                    tokens: 16,
                },
                segments[0],
            ],
            (0..17)
                .map(|slot| BatchSegment { slot, tokens: 8 })
                .collect(),
        ] {
            assert!(super::super::mixed::verification_plan(&m, &bad, true).is_err());
        }
    }
    #[test]
    fn mixed_segments_reject_invalid_shapes_aliases_and_capacity() {
        let m = mixed_manifest();
        for segments in [
            vec![],
            vec![BatchSegment { slot: 0, tokens: 0 }],
            vec![BatchSegment {
                slot: 0,
                tokens: 513,
            }],
            vec![
                BatchSegment {
                    slot: 0,
                    tokens: 128,
                },
                BatchSegment { slot: 0, tokens: 1 },
            ],
            vec![
                BatchSegment {
                    slot: 0,
                    tokens: 512,
                },
                BatchSegment {
                    slot: 1,
                    tokens: 512,
                },
                BatchSegment { slot: 2, tokens: 1 },
            ],
        ] {
            assert!(super::super::mixed::validate_segments(&m, &segments).is_err());
        }
    }

    #[test]
    fn decode_uses_actual_rows_without_padding_or_per_request_copies() {
        let m = manifest();
        for count in [1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 64, 127, 128] {
            let segments: Vec<_> = (0..count)
                .map(|i| BatchSegment {
                    slot: i * 3 + 7,
                    tokens: 1,
                })
                .collect();
            let p = plan(&m, &segments).unwrap();
            assert!(p.iter().all(|i| i.sequence.is_none() && i.views.is_empty()));
            assert!(
                p.iter()
                    .all(|i| matches!(i.operation, Operation::Kernel { .. }))
            );
            assert!(
                p.iter()
                    .all(|i| i.launch.as_ref().unwrap().grid[2] == count as u32)
            );
            assert!(
                matches!(&p[0].operation, Operation::Kernel {name} if name=="flash_decode/packet/inputs")
            );
            assert!(
                matches!(&p.last().unwrap().operation, Operation::Kernel {name} if name=="flash_decode/packet/outputs")
            );
            assert_eq!(p.iter().filter(|i| matches!(&i.operation, Operation::Kernel {name} if name=="flash_decode/begin/k0/rows/1-1024")).count(),1);
            let bindings = state_bindings(&m, &segments).unwrap();
            for (lane, segment) in segments.iter().enumerate() {
                assert_eq!(bindings[lane * 48], Some((segment.slot, "Position".into())));
                assert_eq!(
                    bindings[128 * 48 + lane * 10],
                    Some((segment.slot, "M1_Embedding".into()))
                );
            }
            assert!(bindings[count * 48..128 * 48].iter().all(Option::is_none));
            assert!(
                bindings[128 * 48 + count * 10..]
                    .iter()
                    .all(Option::is_none)
            );
        }
    }
    #[test]
    fn invalid_decode_work_and_incomplete_packages_are_rejected() {
        let m = manifest();
        for segments in [
            vec![],
            vec![BatchSegment { slot: 0, tokens: 2 }],
            vec![BatchSegment { slot: 0, tokens: 1 }; 2],
            (0..129)
                .map(|slot| BatchSegment { slot, tokens: 1 })
                .collect(),
        ] {
            assert!(plan(&m, &segments).is_err());
        }
        let mut broken = manifest();
        broken
            .batch_layout
            .as_mut()
            .unwrap()
            .row_strides
            .insert("M1_Ple".into(), 20480);
        assert!(register(&mut broken).is_err());
        let mut incomplete = manifest();
        incomplete
            .dynamic_batch_kernels
            .remove("flash_decode/begin/k0");
        assert!(register(&mut incomplete).is_err());
    }
    #[test]
    fn old_fixed_capacity_inputs_are_rejected() {
        let mut m = manifest();
        m.row_execution = None;
        m.dynamic_batch_kernels.clear();
        m.programs.clear();
        assert!(register(&mut m).is_err());
    }

    #[test]
    fn request_table_keeps_recurrence_ownership_and_inactive_rows_separate() {
        let mut m = manifest();
        m.batch_layout.as_mut().unwrap().state_columns = ["Position", "State_conv", "State_gdn"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for layer in (0..48).filter(|l| l % 4 != 3) {
            for suffix in ["conv", "gdn"] {
                m.buffers.push(serde_json::from_value(serde_json::json!({
                    "name":format!("State_{layer}_{suffix}"),"dtype":"u8","shape":[4],
                    "layout":"native_contiguous","alignment":256,"access":"read_write","data":null
                })).unwrap());
            }
        }
        m.buffers
            .iter_mut()
            .find(|b| b.name == POINTERS)
            .unwrap()
            .shape = vec![128 * (48 * 3 + 10)];
        register(&mut m).unwrap();
        assert_eq!(m.state_pointer_table.as_ref().unwrap().max_rows, 128);
        for count in [0, 1, 2, 3, 8, 15, 128] {
            let segments: Vec<_> = (0..count)
                .map(|lane| BatchSegment {
                    slot: 129 - lane,
                    tokens: 1,
                })
                .collect();
            let bindings = state_bindings(&m, &segments).unwrap();
            assert_eq!(bindings.len(), 128 * (48 * 3 + 10));
            for layer in 0..48 {
                for (lane, s) in segments.iter().enumerate() {
                    let start = (lane * 48 + layer) * 3;
                    assert_eq!(bindings[start], Some((s.slot, "Position".into())));
                    assert_eq!(
                        bindings[start + 1],
                        (layer % 4 != 3).then(|| (s.slot, format!("State_{layer}_conv")))
                    );
                    assert_eq!(
                        bindings[start + 2],
                        (layer % 4 != 3).then(|| (s.slot, format!("State_{layer}_gdn")))
                    );
                }
            }
            assert!(
                bindings[count * 48 * 3..128 * 48 * 3]
                    .iter()
                    .all(Option::is_none)
            );
        }
        m.batch_layout.as_mut().unwrap().state_columns.swap(0, 1);
        assert!(register(&mut m).is_err());
    }

    #[test]
    fn prefill_only_packet_clears_previous_decode_owner_bindings() {
        let mut m = mixed_manifest();
        m.batch_layout.as_mut().unwrap().state_columns = vec!["Position".into()];
        let active = state_bindings(
            &m,
            &[BatchSegment {
                slot: 19,
                tokens: 1,
            }],
        )
        .unwrap();
        assert!(active.iter().any(Option::is_some));
        let cold = state_bindings(
            &m,
            &[
                BatchSegment {
                    slot: 7,
                    tokens: 128,
                },
                BatchSegment { slot: 3, tokens: 7 },
            ],
        )
        .unwrap();
        assert_eq!(cold.len(), active.len());
        assert!(cold.iter().all(Option::is_none));
    }
}
