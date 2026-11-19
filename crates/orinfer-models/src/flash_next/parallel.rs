//! Derive fork/join endpoints from the projection operands in execution order.
use super::contracts;
use crate::{
    artifact::{Argument, Kernel, Result},
    model::{Manifest, Operation},
};

fn hc(part: &str, select: &impl Fn(&str) -> Result<usize>) -> Result<[usize; 4]> {
    let root = select(&format!("{part}_down.weight_value"))?;
    let inject = select(&format!("{part}_inject.weight_value"))?;
    Ok([root, inject, inject, inject + 1])
}

fn ffn(select: &impl Fn(&str) -> Result<usize>) -> Result<[usize; 4]> {
    let expert = select("ffn_gate_up_exps.weight_packed")?;
    let shared = select("ffn_gate_shexp.weight_weight")?;
    let gate = select("ffn_gate_inp_shexp.weight_value")?;
    Ok([
        expert,
        shared.checked_sub(1).ok_or("Missing shared FFN norm")?,
        gate,
        gate + 1,
    ])
}

fn projection<'a>(
    candidates: impl Iterator<Item = (usize, &'a Kernel)>,
    weight: &str,
    matches: impl Fn(&str) -> bool,
) -> Result<usize> {
    let mut positions = candidates.filter_map(|(i, k)| {
        k.args
            .iter()
            .any(|a| matches!(a, Argument::Buffer { name } if matches(name)))
            .then_some(i)
    });
    let first = positions
        .next()
        .ok_or_else(|| format!("Missing branch projection {weight}"))?;
    if positions.next().is_some() {
        return Err(format!("Ambiguous branch projection {weight}"));
    }
    Ok(first)
}

fn endpoints(
    mut regions: Vec<[usize; 4]>,
    name_at: impl Fn(usize) -> Option<String>,
) -> Result<Vec<[String; 4]>> {
    regions.sort_by_key(|region| region[0]);
    regions
        .into_iter()
        .map(|region| {
            let [a, b, c, d] = region.map(|index| name_at(index).ok_or("Missing branch endpoint"));
            Ok([a?, b?, c?, d?])
        })
        .collect()
}

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    manifest.graph_branches.clear();
    if !kernels
        .iter()
        .any(|k| k.name.starts_with("decode/") && !k.readonly_arguments.is_empty())
    {
        return Ok(());
    }
    let ops = manifest
        .programs
        .get("decode")
        .ok_or("Missing decode program")?;
    let catalog = contracts::kernels(kernels);
    let mut regions = Vec::new();
    for layer in 0..48 {
        let prefix = format!("decode/layer{layer}/");
        let exact = |weight: &str| {
            let candidates = ops.iter().enumerate().filter_map(|(i, op)| match op {
                Operation::Kernel { name } if name.starts_with(&prefix) => {
                    Some((i, *catalog.get(name.as_str())?))
                }
                _ => None,
            });
            projection(candidates, weight, |name| name == weight)
        };
        let select = |suffix: &str| exact(&format!("W_blk.{layer}.{suffix}"));
        regions.extend([hc("hc_attn", &select)?, hc("hc_ffn", &select)?]);
        if layer % 4 != 3 {
            let qkv = exact(&format!("Q4_blk.{layer}.attn_qkv_packed"))?;
            let gate = exact(&format!("Q4_blk.{layer}.attn_gate_packed"))?;
            let ab = select("ssm_beta.weight_weight")? + 1;
            regions.push([qkv, gate, ab, ab + 1]);
        }
        regions.push(ffn(&select)?);
    }
    let branches = endpoints(regions, |index| match ops.get(index)? {
        Operation::Kernel { name } => Some(name.clone()),
        _ => None,
    })?;
    manifest.graph_branches.insert("decode".into(), branches);
    Ok(())
}

/// Canonical slots own dependency regions; AOT row substitutions retain them.
pub(super) fn build_rows(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let catalog = contracts::kernels(kernels);
    for (program, ops) in &manifest.programs {
        if !program.starts_with("flash_decode/layer") {
            continue;
        }
        let part = if program.ends_with("/pre") {
            "hc_attn"
        } else if program.ends_with("/post") {
            "hc_ffn"
        } else {
            continue;
        };
        let names: Vec<_> = ops
            .iter()
            .filter_map(|op| match op {
                Operation::Kernel { name } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let select = |suffix: &str| {
            let candidates = names
                .iter()
                .enumerate()
                .filter_map(|(i, name)| Some((i, *catalog.get(name)?)));
            projection(candidates, suffix, |name| name.ends_with(suffix))
        };
        let mut regions = vec![hc(part, &select)?];
        if part == "hc_ffn" {
            regions.push(ffn(&select)?);
        }
        manifest.graph_branches.insert(
            program.clone(),
            endpoints(regions, |i| names.get(i).map(|s| (*s).into()))?,
        );
    }
    Ok(())
}
