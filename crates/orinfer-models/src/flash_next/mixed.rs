//! Concatenated dense/MoE rows with independent causal segments and endpoint heads.
use crate::packet::invocation as invoke;
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::architecture::BatchSegment;
use std::collections::{BTreeMap, BTreeSet};

const RESIDUAL: &str = "FlashMixedResidual";
const HEAD_RESIDUAL: &str = "FlashMixedHeadResidual";

pub(super) fn register(m: &mut Manifest) -> Result<()> {
    let layout = m.batch_layout.as_ref().ok_or("Missing Flash layout")?;
    if layout.profiles.is_empty() {
        if !m.prefill_batch_profiles.is_empty() {
            return Err("Incomplete Flash mixed batch layout".into());
        }
        return Ok(());
    }
    let shapes_valid = layout.segment_tokens == 512
        && layout.profiles.keys().copied().collect::<Vec<_>>() == [512]
        && m.prefill_batch_profiles
            .iter()
            .map(|p| p.tokens)
            .collect::<Vec<_>>()
            == [1024];
    if !shapes_valid
        || layout.profiles.values().any(|v| v != "flash_recurrent")
        || m.prefill_batch_profiles
            .iter()
            .any(|p| !matches!(p.kind.as_str(), "flash_mixed" | "flash_mixed_grouped"))
        || m.prefill_batch_profiles
            .windows(2)
            .any(|p| p[0].kind != p[1].kind)
        || m.buffers
            .iter()
            .find(|b| b.name == RESIDUAL)
            .is_none_or(|b| b.data.is_some() || b.bytes().ok() != Some(1024 * 20480))
    {
        return Err("Invalid Flash mixed causal/row layout".into());
    }
    super::causal_rows::register(m);
    Ok(())
}

pub(super) fn validate_segments(
    m: &Manifest,
    segments: &[BatchSegment],
) -> Result<crate::packet::Rows> {
    if m.batch_layout
        .as_ref()
        .is_none_or(|l| l.profiles.is_empty())
    {
        return Err("Flash package lacks mixed causal segments".into());
    }
    let rows = crate::packet::Rows::packet(segments, 128, 1024)?;
    if segments.iter().any(|s| !segment_supported(m, s.tokens)) {
        return Err("Invalid Flash causal segment".into());
    }
    Ok(rows)
}

fn segment_supported(m: &Manifest, tokens: usize) -> bool {
    m.batch_layout
        .as_ref()
        .is_some_and(|l| (1..=l.segment_tokens).contains(&tokens))
}

pub(super) fn mixed_rows_plan(
    m: &Manifest,
    segments: &[BatchSegment],
    prefills: usize,
    include_plan: bool,
) -> Result<Vec<Invocation>> {
    if prefills == 0 || prefills >= segments.len() {
        return Err("Invalid mixed role boundary".into());
    }
    plan_kind(m, segments, Kind::MixedRows(prefills), include_plan)
}

fn append(
    out: &mut Vec<Invocation>,
    m: &Manifest,
    program: &str,
    slot: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Result<()> {
    out.extend(
        m.programs
            .get(program)
            .ok_or_else(|| format!("Missing mixed stage {program}"))?
            .iter()
            .cloned()
            .map(|op| invoke(op, slot, views)),
    );
    Ok(())
}

pub(super) fn plan(m: &Manifest, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
    plan_kind(m, segments, Kind::Ordinary, true)
}

pub(super) fn verification_plan(
    m: &Manifest,
    segments: &[BatchSegment],
    include_plan: bool,
) -> Result<Vec<Invocation>> {
    plan_kind(m, segments, Kind::Verification, include_plan)
}

pub(super) fn mixed_verification_plan(
    m: &Manifest,
    segments: &[BatchSegment],
    include_plan: bool,
) -> Result<Vec<Invocation>> {
    if !m
        .mtp
        .as_ref()
        .is_some_and(|s| !s.mixed_verification_tokens.is_empty())
    {
        return Err("Package lacks mixed prefill/verification".into());
    }
    plan_kind(m, segments, Kind::MixedVerification, include_plan)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ordinary,
    Verification,
    MixedVerification,
    MixedRows(usize),
}

fn plan_kind(
    m: &Manifest,
    segments: &[BatchSegment],
    kind: Kind,
    include_plan: bool,
) -> Result<Vec<Invocation>> {
    let explicit_verify: BTreeSet<_> = match kind {
        Kind::MixedRows(n) => segments[n..].iter().map(|s| s.slot).collect(),
        _ => BTreeSet::new(),
    };
    let mixed = matches!(kind, Kind::MixedVerification | Kind::MixedRows(_));
    let verified = |s: &BatchSegment| {
        kind != Kind::Ordinary
            && m.mtp.as_ref().is_some_and(|spec| {
                if matches!(kind, Kind::MixedRows(_)) {
                    explicit_verify.contains(&s.slot)
                        && spec.mixed_verification_tokens.contains(&s.tokens)
                } else if kind == Kind::MixedVerification {
                    spec.mixed_verification_tokens.contains(&s.tokens)
                } else {
                    spec.batch_verification_tokens.contains(&s.tokens)
                }
            })
    };
    // Put single-token rows first so the existing request-table mixers can
    // consume a contiguous subgroup without gathering projections or state.
    let ordered: Vec<_> = segments
        .iter()
        .filter(|s| s.tokens == 1)
        .chain(segments.iter().filter(|s| s.tokens != 1))
        .cloned()
        .collect();
    let segments = ordered.as_slice();
    let heads = crate::packet::Rows::new(
        segments
            .iter()
            .map(|s| if verified(s) { s.tokens } else { 1 }),
        128,
    )?;
    let head_used = heads.total;
    let rows = if kind == Kind::Ordinary {
        validate_segments(m, segments)?
    } else {
        let rows = crate::packet::Rows::packet(segments, 128, 1024)?;
        let mut has_prefill = false;
        let mut has_verify = false;
        for s in segments {
            if verified(s) {
                has_verify = true;
            } else if mixed && !explicit_verify.contains(&s.slot) && segment_supported(m, s.tokens)
            {
                has_prefill = true;
            } else {
                return Err("Invalid mixed verification segment".into());
            }
        }
        if !has_verify || (mixed && !has_prefill) {
            return Err("Invalid mixed verification endpoints".into());
        }
        rows
    };
    let layout = m
        .batch_layout
        .as_ref()
        .filter(|l| !l.profiles.is_empty())
        .ok_or("Missing mixed row layout")?;
    if !include_plan {
        for segment in segments.iter().filter(|segment| verified(segment)) {
            m.mtp
                .as_ref()
                .and_then(|spec| spec.verification_plan(segment.tokens))
                .ok_or("Missing verification input template")?;
        }
        return Ok(vec![]);
    }
    let decoders = segments.iter().take_while(|s| s.tokens == 1).count();
    let parallel = kind == Kind::Ordinary && decoders >= 2 && !layout.state_columns.is_empty();

    let prefix = "flash_decode";
    let empty = BTreeMap::new();
    let mut out = vec![];
    let mut starts = vec![];
    for (s, &start) in segments.iter().zip(&rows.starts) {
        let width = if verified(s) {
            m.mtp
                .as_ref()
                .and_then(|spec| spec.verification_plans.iter().find(|p| p.accepts(s.tokens)))
                .map_or(super::plan::arena_width(s.tokens), |p| {
                    super::plan::arena_width(p.tokens)
                })
        } else if m
            .batch_layout
            .as_ref()
            .is_some_and(|l| l.segment_tokens > 0)
            && !verified(s)
            && s.tokens > 1
        {
            4096
        } else {
            super::plan::arena_width(s.tokens)
        };
        let program = if verified(s) {
            m.mtp
                .as_ref()
                .and_then(|spec| spec.verification_plan(s.tokens))
                .ok_or("Missing verification input template")?
                .program
        } else if s.tokens == 1 {
            "decode".into()
        } else {
            format!(
                "prefill_m{}",
                m.prefill_plan(s.tokens)
                    .ok_or("Missing mixed prefill template")?
                    .chunk_tokens
            )
        };
        if m.vision.is_some() {
            let mut input_views = empty.clone();
            if !verified(s) && s.tokens > 1 {
                input_views.insert(
                    "M4096_Embedding".into(),
                    BufferView {
                        buffer: format!("M{width}_Embedding"),
                        offset: 0,
                    },
                );
            }
            let name = format!("{program}/begin/k0");
            let mut invocation = invoke(
                Operation::Kernel { name: name.clone() },
                Some(s.slot),
                &input_views,
            );
            if m.dynamic_batch_kernels.contains_key(&name) {
                invocation.launch = Some(
                    m.dynamic_batch_kernels
                        .get(&name)
                        .ok_or("Missing mixed overlay template contract")?
                        .launch(s.tokens)?,
                );
            }
            out.push(invocation);
        }
        let mut views = BTreeMap::new();
        for (role, stride) in &layout.row_strides {
            if matches!(role.as_str(), "Logits" | "Selected") {
                continue;
            }
            let old = format!("FlashMixed_{role}");
            let name = format!("FlashDecode_{role}");
            let view = crate::packet::view(&name, start, *stride)?;
            views.insert(old, view.clone());
            views.insert(name, view);
        }
        views.insert(
            RESIDUAL.into(),
            BufferView {
                buffer: "FlashDecode_Residual".into(),
                offset: start * 20480,
            },
        );
        for (source, destination) in [
            (format!("M{width}_Embedding"), "FlashMixed_M1_Embedding"),
            (format!("M{width}_Ple"), "FlashMixed_M1_Ple"),
        ] {
            let destination = destination.replace("FlashMixed_", "FlashDecode_");
            out.push(invoke(
                Operation::Copy {
                    source,
                    destination,
                    bytes: s.tokens * 5120,
                },
                Some(s.slot),
                &views,
            ));
        }
        starts.push((start, program, views));
    }

    let append_body = |out: &mut Vec<Invocation>, program: &str| -> Result<()> {
        super::rows::append(m, out, program, rows.total, None, &empty)
    };
    append_body(&mut out, &format!("{prefix}/begin"))?;
    for layer in 0..48 {
        if layer == 1 {
            append_body(&mut out, &format!("{prefix}/layer1/ple"))?;
            for (s, (_, _, views)) in segments.iter().zip(&starts) {
                super::rows::append(
                    m,
                    &mut out,
                    &format!(
                        "causal_rows/{}/layer1/ple",
                        if verified(s) { "verify" } else { "target" }
                    ),
                    s.tokens,
                    Some(s.slot),
                    views,
                )?;
            }
        }
        append_body(&mut out, &format!("{prefix}/layer{layer}/pre"))?;
        if parallel {
            let program = format!("flash_decode_state/layer{layer}");
            append(&mut out, m, &program, None, &empty)?;
            for invocation in out.iter_mut().rev().take(m.programs[&program].len()) {
                if let Operation::Kernel { name } = &invocation.operation {
                    invocation.launch = Some(
                        m.dynamic_batch_kernels
                            .get(name)
                            .ok_or("Missing mixed decoder subgroup launch")?
                            .launch(decoders)?,
                    );
                }
            }
        }
        for (s, (_, _, views)) in segments
            .iter()
            .zip(&starts)
            .filter(|(s, _)| !parallel || s.tokens != 1)
        {
            let section = format!(
                "causal_rows/{}/layer{layer}/private",
                if verified(s) { "verify" } else { "target" }
            );
            let private = &m.programs[&section];
            for (i, op) in private.iter().enumerate() {
                let mut invocation = invoke(op.clone(), Some(s.slot), views);
                if let Operation::Kernel { name } = op {
                    invocation.launch = Some(
                        m.dynamic_batch_kernels
                            .get(name)
                            .ok_or("Missing causal row launch")?
                            .launch(s.tokens)?,
                    );
                }
                out.push(invocation);
                if layer % 4 != 3 && i == usize::from(verified(s)) {
                    out.push(invoke(
                        Operation::Copy {
                            source: format!("M{}_HistoryOut", if verified(s) { 8 } else { 4096 }),
                            destination: format!("State_{layer}_conv"),
                            bytes: 61440,
                        },
                        Some(s.slot),
                        &empty,
                    ));
                }
            }
        }
        append_body(&mut out, &format!("{prefix}/layer{layer}/post"))?;
    }

    for ((s, (first, program, views)), &head_lane) in
        segments.iter().zip(&starts).zip(&heads.starts)
    {
        let name = if verified(s) {
            "causal_rows/verify/advance".into()
        } else {
            format!("{program}/end/k0")
        };
        let mut invocation = invoke(
            Operation::Kernel { name: name.clone() },
            Some(s.slot),
            &empty,
        );
        if verified(s) || s.tokens > 1 {
            invocation.launch = Some(
                m.dynamic_batch_kernels
                    .get(&name)
                    .ok_or("Missing mixed position template contract")?
                    .launch(s.tokens)?,
            );
        }
        out.push(invocation);
        if m.mtp.is_some() {
            super::rows::append(
                m,
                &mut out,
                &format!(
                    "causal_rows/{}/capture",
                    if verified(s) { "verify" } else { "target" }
                ),
                s.tokens,
                Some(s.slot),
                views,
            )?;
        }
        let copy = BTreeMap::from([
            (
                RESIDUAL.into(),
                crate::packet::view(
                    "FlashDecode_Residual",
                    first + if verified(s) { 0 } else { s.tokens - 1 },
                    20480,
                )?,
            ),
            (
                HEAD_RESIDUAL.into(),
                crate::packet::view(HEAD_RESIDUAL, head_lane, 20480)?,
            ),
        ]);
        out.push(invoke(
            Operation::Copy {
                source: RESIDUAL.into(),
                destination: HEAD_RESIDUAL.into(),
                bytes: if verified(s) { s.tokens * 20480 } else { 20480 },
            },
            Some(s.slot),
            &copy,
        ));
    }
    let views = BTreeMap::from([(
        "FlashDecode_Residual".into(),
        BufferView {
            buffer: HEAD_RESIDUAL.into(),
            offset: 0,
        },
    )]);
    super::rows::append(m, &mut out, "flash_decode/head", head_used, None, &views)?;

    let logits = "FlashDecode_Logits".to_owned();
    let selected = "FlashDecode_Selected".to_owned();
    for (s, &head_lane) in segments.iter().zip(&heads.starts) {
        if verified(s) {
            let spec = m.mtp.as_ref().unwrap();
            let views = crate::packet::views(
                head_lane,
                [(logits.as_str(), m.vocab * 4), (selected.as_str(), 8)],
            )?;
            for (source, destination, bytes) in [
                (
                    logits.clone(),
                    spec.verification_logits.clone(),
                    s.tokens * m.vocab * 4,
                ),
                (selected.clone(), "VerificationPairs".into(), s.tokens * 8),
            ] {
                out.push(invoke(
                    Operation::Copy {
                        source,
                        destination,
                        bytes,
                    },
                    Some(s.slot),
                    &views,
                ));
            }
            let name = "causal_rows/verify/selections".to_owned();
            let mut selection = invoke(
                Operation::Kernel { name: name.clone() },
                Some(s.slot),
                &empty,
            );
            selection.launch = Some(
                m.dynamic_batch_kernels
                    .get(&name)
                    .ok_or("Missing causal selection launch")?
                    .launch(s.tokens)?,
            );

            out.push(selection);
            continue;
        }
        let mut views = crate::packet::views(head_lane, [(logits.as_str(), m.vocab * 4)])?;
        out.push(invoke(
            Operation::Copy {
                source: logits.clone(),
                destination: m.logits.clone(),
                bytes: m.vocab * 4,
            },
            Some(s.slot),
            &views,
        ));
        for (offset, destination) in [(0, &m.token), (4, &m.status)] {
            let name = selected.clone();
            views.insert(
                name.clone(),
                BufferView {
                    buffer: name.clone(),
                    offset: head_lane * 8 + offset,
                },
            );
            out.push(invoke(
                Operation::Copy {
                    source: name,
                    destination: destination.clone(),
                    bytes: 4,
                },
                Some(s.slot),
                &views,
            ));
        }
    }

    Ok(out)
}
