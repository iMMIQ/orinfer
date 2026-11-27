//! Exact single-token producer fusions retain the original rounding boundaries.
use super::contracts::{
    Storage, connect, distinct, fuse, kernel_at, kernel_buffer_arg as buffer, kernels, retire,
};
use crate::{
    artifact::{Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
struct Step {
    offset: isize,
    args: usize,
    symbols: &'static [&'static str],
    links: &'static [(usize, usize)],
    remove: bool,
}
struct Rule {
    symbol: &'static str,
    steps: &'static [Step],
    distinct: &'static [usize],
}
const fn step(
    offset: isize,
    args: usize,
    symbols: &'static [&'static str],
    links: &'static [(usize, usize)],
    remove: bool,
) -> Step {
    Step {
        offset,
        args,
        symbols,
        links,
        remove,
    }
}

const MAIN: &[&str] = &["main_kernel"];
const PACK: &[&str] = &["pack_exact_a8_kernel"];
const RULES: &[Rule] = &[
    Rule {
        symbol: "rotate_dispatch_a8_kernel",
        steps: &[
            step(
                1,
                8,
                MAIN,
                &[(0, 2), (2, 3), (4, 0), (5, 4), (6, 5), (7, 6), (9, 7)],
                true,
            ),
            step(2, 4, PACK, &[(1, 1), (3, 2), (10, 3)], true),
            step(
                3,
                13,
                &["exact_e8p_shared_input_kernel"],
                &[(1, 3), (3, 4), (10, 8), (7, 0)],
                false,
            ),
        ],
        distinct: &[1, 3, 10, 4, 7, 11],
    },
    Rule {
        symbol: "rotate_split_a8_kernel",
        steps: &[
            step(1, 4, PACK, &[(2, 0), (0, 1), (1, 2), (5, 3)], true),
            step(
                2,
                13,
                &["exact_e8p_int4_kernel"],
                &[(0, 3), (1, 4), (5, 8), (3, 0)],
                false,
            ),
        ],
        distinct: &[0, 1, 5, 2, 3, 6],
    },
    Rule {
        symbol: "hc_combine_norm_kernel",
        steps: &[step(1, 3, MAIN, &[(2, 0), (4, 1), (5, 2)], true)],
        distinct: &[],
    },
    Rule {
        symbol: "gdn_conv_inplace_kernel",
        steps: &[],
        distinct: &[0, 1, 2, 3, 4, 5, 6, 7, 8],
    },
    Rule {
        symbol: "hc_up13_finish_kernel",
        steps: &[
            step(
                -2,
                3,
                &["main_kernel", "bf16_single_kernel"],
                &[(0, 1), (3, 0)],
                false,
            ),
            step(-1, 2, MAIN, &[(0, 1)], true),
        ],
        distinct: &[0, 3, 4],
    },
];

pub(super) fn build(manifest: &mut Manifest, all: &[Kernel], exhaustive: bool) -> Result<()> {
    let contracts = kernels(all);
    let rules: BTreeMap<_, _> = RULES.iter().map(|rule| (rule.symbol, rule)).collect();
    let anchors = all.iter().filter(|k| rules.contains_key(k.symbol.as_str()));
    let mut pending: BTreeSet<_> = all
        .iter()
        .filter(|k| k.symbol == "exact_e8p_shared_input_kernel")
        .map(|k| k.name.as_str())
        .collect();
    let storage = exhaustive.then(|| Storage::new(&manifest.buffers));
    fuse(
        &mut manifest.programs,
        anchors,
        |program, operations, i, k, skip| {
            let rule = rules[k.symbol.as_str()];
            let finish = k.symbol == "hc_up13_finish_kernel";
            if !(program == "decode" || finish && program == "head_m1") {
                return Err("Fusion requires a single-token target program".into());
            }
            let mut adjacent = Vec::with_capacity(rule.steps.len());
            for step in rule.steps {
                let at = i
                    .checked_add_signed(step.offset)
                    .ok_or("Incomplete decode fusion chain")?;
                let other = kernel_at(&contracts, operations, at)?;
                if other.args.len() != step.args || !step.symbols.contains(&other.symbol.as_str()) {
                    return Err(format!(
                        "Unexpected decode fusion producer/consumer: {}",
                        other.name
                    ));
                }
                connect(k, other, step.links)?;
                if step.remove {
                    retire(skip, [at])?;
                }
                adjacent.push(other);
            }
            match k.symbol.as_str() {
                "rotate_dispatch_a8_kernel" => {
                    connect(adjacent[1], adjacent[0], &[(0, 1)])?;
                    pending.remove(adjacent[2].name.as_str());
                }
                "gdn_conv_inplace_kernel" => {
                    let Some(Operation::Copy {
                        source,
                        destination,
                        bytes,
                    }) = operations.get(i + 1)
                    else {
                        return Err("Missing adjacent convolution history copy".into());
                    };
                    if destination != buffer(k, 0)? || source == destination {
                        return Err("Convolution history copy mismatch".into());
                    }
                    if let Some(storage) = &storage {
                        storage.check(source, Dtype::F16, false, |b| {
                            Ok(*bytes == 61440 && b.bytes()? == *bytes)
                        })?;
                    }
                    retire(skip, [i + 1])?;
                }
                _ => {}
            }
            let finish_output = finish.then(|| buffer(adjacent[1], 0));
            distinct(
                rule.distinct
                    .iter()
                    .map(|&arg| buffer(k, arg))
                    .chain(finish_output),
            )?;
            if let Some(storage) = &storage {
                validate_storage(k, &adjacent, |name, dtype, bytes, immutable| {
                    storage
                        .check(name, dtype, immutable, |b| Ok(b.bytes()? == bytes))
                        .map(|_| ())
                })?;
            }
            Ok(())
        },
    )?;
    if !pending.is_empty() {
        return Err("Unregistered decode fusion".into());
    }
    Ok(())
}

fn validate_storage(
    k: &Kernel,
    adjacent: &[&Kernel],
    check: impl Fn(&str, Dtype, usize, bool) -> Result<()>,
) -> Result<()> {
    use Dtype::{F16, F32, I8, I32, U32};
    match k.symbol.as_str() {
        "hc_combine_norm_kernel"
            if adjacent[0].grid != [1, 4, 1] || adjacent[0].block != [256, 1, 1] =>
        {
            return Err("Invalid HC combine-norm geometry".into());
        }
        "rotate_dispatch_a8_kernel" if adjacent[0].grid != [10, 1, 1] => {
            return Err("Invalid single-token activation dispatch geometry".into());
        }
        _ => {}
    }
    let (grid, block, specs): (_, _, &[(Dtype, usize, bool)]) = match k.symbol.as_str() {
        "hc_combine_norm_kernel" => (
            [1, 4, 1],
            [256, 1, 1],
            &[
                (F16, 5120, false),
                (F16, 8, false),
                (F16, 20480, false),
                (F16, 20480, false),
                (F16, 20480, false),
                (F32, 40960, true),
            ],
        ),
        "gdn_conv_inplace_kernel" => (
            [1, 80, 1],
            [16, 1, 1],
            &[
                (F16, 61440, false),
                (F16, 4096, false),
                (F16, 4096, false),
                (F16, 12288, false),
                (F32, 163840, true),
                (F16, 20480, false),
                (I32, 4, false),
                (I32, 4, false),
                (I32, 4, false),
            ],
        ),
        "rotate_dispatch_a8_kernel" => (
            [1, 1, 1],
            k.block,
            &[
                (F16, 20, false),
                (U32, 12800, false),
                (I32, 40, false),
                (U32, 12800, false),
                (I8, 2560, false),
                (I32, 40, false),
                (I32, 2048, false),
                (F16, 2, false),
                (I8, 2560, true),
                (I32, 40, false),
                (I32, 40, false),
                (F16, 5120, false),
            ],
        ),
        "rotate_split_a8_kernel" => (
            [10, 1, 1],
            k.block,
            &[
                (U32, 3200, false),
                (U32, 3200, false),
                (I8, 6400, false),
                (F16, 20, false),
                (I8, 640, true),
                (I32, 40, false),
                (F16, 25600, false),
            ],
        ),
        "hc_up13_finish_kernel" => {
            let rank = adjacent[1].grid[0] as usize;
            let low = buffer(k, 0)?;
            for rank in [256_usize, 320]
                .into_iter()
                .filter(|n| n.div_ceil(128) == rank)
            {
                if check(low, F32, 16 * rank, false).is_err() {
                    continue;
                }
                for (i, dtype, bytes, immutable) in [
                    (1, U32, 80 * (rank / 64) * 128 * 10 * 4, true),
                    (2, U32, 80 * (rank / 64) * 128 * 16 * 4, true),
                    (3, F16, 20480, false),
                    (4, F16, 5120, false),
                ] {
                    check(buffer(k, i)?, dtype, bytes, immutable)?;
                }
                check(buffer(adjacent[1], 0)?, F16, rank * 2, false)?;
                if k.grid != [1, 80, 1] || k.block != [128, 1, 1] || !matches!(k.args.len(), 5 | 6)
                {
                    return Err("Unexpected HC finish ABI/geometry".into());
                }
                let (down_grid, down_block) = if adjacent[0].symbol == "bf16_single_kernel" {
                    ([rank.div_ceil(32) as u32, 4, 1], [64, 1, 1])
                } else {
                    ([1, rank.div_ceil(64) as u32, 4], [128, 1, 1])
                };
                if adjacent[0].grid != down_grid
                    || adjacent[0].block != down_block
                    || adjacent[1].grid != [rank.div_ceil(128) as u32, 1, 1]
                    || adjacent[1].block != [128, 1, 1]
                {
                    return Err("Invalid HC finish producer geometry".into());
                }
                if k.args.len() == 6 {
                    let name = buffer(k, 5)?;
                    if [8, 16, 24]
                        .into_iter()
                        .all(|bytes| check(name, Dtype::U64, bytes, true).is_err())
                    {
                        return Err("Invalid HC escape buffer".into());
                    }
                }
                return Ok(());
            }
            return Err("Invalid HC finish rank or buffer".into());
        }
        _ => unreachable!(),
    };
    if k.grid != grid || k.block != block || k.args.len() != specs.len() {
        return Err("Unexpected decode fusion ABI/geometry".into());
    }
    for (i, &(dtype, bytes, immutable)) in specs.iter().enumerate() {
        check(buffer(k, i)?, dtype, bytes, immutable)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "decode_fusions_tests.rs"]
mod tests;
