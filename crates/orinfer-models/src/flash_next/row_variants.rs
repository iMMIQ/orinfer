//! Local tiling choices share the canonical operand and row contracts.
use super::contracts::{Storage, connect, kernel_buffer_arg as buffer};
use crate::artifact::{Argument, Buffer, Dtype, Kernel, Result};
use orinfer_model_sdk::operators::rows::KernelVariant;
use std::collections::BTreeMap;

pub(super) fn build<Data>(
    kernels: &[Kernel],
    buffers: Option<&[Buffer<Data>]>,
) -> Result<BTreeMap<String, Vec<KernelVariant>>> {
    let contracts = super::contracts::kernels(kernels);
    let storage = buffers.map(Storage::new);
    let mut variants: BTreeMap<String, Vec<KernelVariant>> = BTreeMap::new();
    for kernel in kernels {
        let Some((slot, suffix)) = kernel.name.rsplit_once('/') else {
            continue;
        };
        if suffix == "hc_finish" && !contracts.contains_key(format!("{slot}/hc_partials").as_str())
        {
            return Err("Missing HC stream producer".into());
        }
        if suffix == "hc_partials" {
            let canonical = contracts.get(slot).ok_or("Missing HC projection slot")?;
            let finish = contracts
                .get(format!("{slot}/hc_finish").as_str())
                .ok_or("Missing HC stream finish")?;
            let extent = |k: &Kernel, index: usize| {
                matches!(k.args.get(index), Some(Argument::I32 { value: 4096 }))
            };
            if !slot.starts_with("prefill_m4096/")
                || canonical.symbol != "hc_prefill_projection_grouped_kernel"
                || kernel.symbol != "hc_prefill_partial_kernel"
                || finish.symbol != "hc_prefill_finish_kernel"
                || canonical.args.len() != 4
                || kernel.args.len() != 4
                || finish.args.len() != 3
                || !extent(canonical, 3)
                || !extent(kernel, 3)
                || !extent(finish, 2)
            {
                return Err("HC stream projection producer/consumer mismatch".into());
            }
            connect(canonical, kernel, &[(0, 0), (2, 2)])?;
            connect(canonical, finish, &[(1, 0)])?;
            connect(kernel, finish, &[(1, 1)])?;
            if let Some(storage) = &storage {
                storage.check(buffer(kernel, 1)?, Dtype::F32, false, |b| {
                    Ok(b.shape == [4, 4096, 320])
                })?;
                for (index, elements) in [4096 * 10240, 4096 * 320, 320 * 10240]
                    .into_iter()
                    .enumerate()
                {
                    storage.check(buffer(canonical, index)?, Dtype::Bf16, index == 2, |b| {
                        Ok(b.bytes()? == elements * Dtype::Bf16.bytes()
                            && b.layout == "native_contiguous")
                    })?;
                }
            }
            variants
                .entry(slot.to_owned())
                .or_default()
                .push(KernelVariant {
                    min_rows: 1,
                    max_rows: 4096,
                    multiple_of: 1,
                    kernels: vec![kernel.name.clone(), finish.name.clone()],
                });
            continue;
        }
    }
    let mut measured: BTreeMap<(String, usize, usize, usize), BTreeMap<usize, String>> =
        BTreeMap::new();
    for kernel in kernels {
        let Some((slot, choice)) = kernel.name.split_once("/choice/") else {
            continue;
        };
        if !slot.starts_with("prefill_m4096/") || !contracts.contains_key(slot) {
            return Err("Measured prefill choice lacks a canonical slot".into());
        }
        let mut parts = choice.split('/');
        let (Some(interval), Some(multiple), Some(step), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err("Invalid measured prefill choice".into());
        };
        let (low, high) = interval
            .split_once('-')
            .ok_or("Invalid measured prefill interval")?;
        let first = low
            .parse::<usize>()
            .map_err(|_| "Invalid measured row minimum")?;
        let last = high
            .parse::<usize>()
            .map_err(|_| "Invalid measured row maximum")?;
        let multiple = multiple
            .parse::<usize>()
            .map_err(|_| "Invalid measured row multiple")?;
        if first == 0 || first > last || last > 4096 || multiple == 0 {
            return Err("Invalid measured prefill bounds".into());
        }
        let steps = measured
            .entry((slot.into(), first, last, multiple))
            .or_default();
        if step == "skip" {
            if !steps.is_empty() {
                return Err("Mixed skip/kernel prefill choice".into());
            }
        } else {
            let at = step
                .strip_prefix('k')
                .ok_or("Invalid measured prefill step")?
                .parse::<usize>()
                .map_err(|_| "Invalid measured prefill step")?;
            if steps.insert(at, kernel.name.clone()).is_some() {
                return Err("Duplicate measured prefill step".into());
            }
        }
    }
    for ((slot, min_rows, max_rows, multiple_of), steps) in measured {
        if steps.keys().copied().ne(0..steps.len()) {
            return Err("Noncontiguous measured prefill steps".into());
        }
        variants.entry(slot).or_default().push(KernelVariant {
            min_rows,
            max_rows,
            multiple_of,
            kernels: steps.into_values().collect(),
        });
    }
    for choices in variants.values_mut() {
        choices.sort_by_key(|v| v.min_rows);
    }
    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::Access;
    fn kernel(name: &str, buffer: &str) -> Kernel {
        serde_json::from_value(serde_json::json!({
            "name":name,"module":{"file":"k.cubin","sha256":"0".repeat(64)},
            "source":{"file":"k.cu","sha256":"0".repeat(64)},
            "host_abi":{"file":"host.txt","sha256":"0".repeat(64)},
            "symbol":"main_kernel","grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":[{"kind":"buffer","name":buffer},{"kind":"i32","value":4096}]
        }))
        .unwrap()
    }
    #[test]
    fn measured_ranges_do_not_invent_runtime_boundaries() {
        let slot = "prefill_m4096/layer0/k0";
        let selections = [
            kernel(slot, "A"),
            kernel(&format!("{slot}/choice/19-93/1/k0"), "A"),
            kernel(&format!("{slot}/choice/94-281/1/k0"), "A"),
        ];
        let v = build::<()>(&selections, None).unwrap();
        assert_eq!((v[slot][0].min_rows, v[slot][0].max_rows), (19, 93));
        assert_eq!((v[slot][1].min_rows, v[slot][1].max_rows), (94, 281));
        assert!(build::<()>(&selections[1..], None).is_err());
    }
    #[test]
    fn hc_stream_pair_checks_workspace_and_endpoints() {
        let slot = "prefill_m4096/layer0/k3";
        let args = |names: &[&str]| -> Vec<Argument> {
            names
                .iter()
                .map(|name| Argument::Buffer {
                    name: (*name).into(),
                })
                .chain([Argument::I32 { value: 4096 }])
                .collect()
        };
        let mut canonical = kernel(slot, "A");
        canonical.symbol = "hc_prefill_projection_grouped_kernel".into();
        canonical.args = args(&["A", "C", "W"]);
        let mut partial = kernel(&format!("{slot}/hc_partials"), "A");
        partial.symbol = "hc_prefill_partial_kernel".into();
        partial.args = args(&["A", "P", "W"]);
        let mut finish = kernel(&format!("{slot}/hc_finish"), "A");
        finish.symbol = "hc_prefill_finish_kernel".into();
        finish.args = args(&["C", "P"]);
        let buffer = |name: &str, dtype, shape, data| Buffer {
            name: name.into(),
            dtype,
            shape,
            layout: "native_contiguous".into(),
            alignment: 256,
            access: if data {
                Access::Read
            } else {
                Access::ReadWrite
            },
            data: data.then_some(()),
        };
        let mut buffers = vec![
            buffer("P", Dtype::F32, vec![4, 4096, 320], false),
            buffer("A", Dtype::Bf16, vec![4096, 4, 2560], false),
            buffer("C", Dtype::Bf16, vec![4096, 320], false),
            buffer("W", Dtype::Bf16, vec![320, 10240], true),
        ];
        let kernels = vec![canonical, partial, finish];
        let v = build(&kernels, Some(&buffers)).unwrap();
        assert_eq!(
            v[slot][0].kernels,
            [format!("{slot}/hc_partials"), format!("{slot}/hc_finish")]
        );
        assert!(build(&kernels[..2], Some(&buffers)).is_err());
        assert!(build(&[kernels[0].clone(), kernels[2].clone()], Some(&buffers)).is_err());
        buffers[0].shape[0] = 3;
        assert!(build(&kernels, Some(&buffers)).is_err());
        buffers[0].shape[0] = 4;
        let mut invalid = kernels.clone();
        invalid[2].args[1] = Argument::Buffer {
            name: "Other".into(),
        };
        assert!(build(&invalid, Some(&buffers)).is_err());
    }
}
