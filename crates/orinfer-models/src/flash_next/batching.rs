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
#[path = "batching_tests.rs"]
mod tests;
