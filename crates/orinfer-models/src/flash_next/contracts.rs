use crate::{
    artifact::{Access, Argument, Buffer, Dtype, Kernel, Result},
    model::Operation,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) type Kernels<'a> = BTreeMap<&'a str, &'a Kernel>;
pub(super) fn kernels(kernels: &[Kernel]) -> Kernels<'_> {
    kernels.iter().map(|k| (k.name.as_str(), k)).collect()
}
pub(super) fn kernel_at<'a>(
    contracts: &Kernels<'a>,
    operations: &[Operation],
    index: usize,
) -> Result<&'a Kernel> {
    match operations.get(index) {
        Some(Operation::Kernel { name }) => contracts
            .get(name.as_str())
            .copied()
            .ok_or_else(|| format!("Missing fusion contract for {name}")),
        _ => Err("Missing adjacent fusion kernel".into()),
    }
}
pub(super) fn kernel_buffer_arg(kernel: &Kernel, index: usize) -> Result<&str> {
    match kernel.args.get(index) {
        Some(Argument::Buffer { name }) => Ok(name),
        _ => Err(format!(
            "{} argument {index} requires a whole buffer",
            kernel.name
        )),
    }
}
pub(super) fn pointer(arg: &Argument) -> Option<(&str, usize)> {
    match arg {
        Argument::Buffer { name } => Some((name, 0)),
        Argument::BufferSlice { name, offset } => Some((name, *offset)),
        _ => None,
    }
}
pub(super) fn connect(a: &Kernel, b: &Kernel, links: &[(usize, usize)]) -> Result<()> {
    for &(from, to) in links {
        if kernel_buffer_arg(a, from)? != kernel_buffer_arg(b, to)? {
            return Err(format!(
                "{} / {} producer/consumer workspace mismatch",
                a.name, b.name
            ));
        }
    }
    Ok(())
}
fn remove(operations: &mut Vec<Operation>, skip: &BTreeSet<usize>) {
    let mut index = 0;
    operations.retain(|_| {
        let keep = !skip.contains(&index);
        index += 1;
        keep
    });
}
pub(super) fn distinct<'a>(names: impl IntoIterator<Item = Result<&'a str>>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        let name = name?;
        if !seen.insert(name) {
            return Err(format!("Aliased fusion buffer: {name}"));
        }
    }
    Ok(())
}

/// Rewrite each alias against the original operation indices. Every published
/// fusion anchor must be registered, and two rules cannot retire the same step.
pub(super) fn fuse<'a>(
    programs: &mut BTreeMap<String, Vec<Operation>>,
    anchors: impl Iterator<Item = &'a Kernel>,
    mut rule: impl FnMut(&str, &[Operation], usize, &'a Kernel, &mut BTreeSet<usize>) -> Result<()>,
) -> Result<()> {
    let anchors: Kernels<'_> = anchors.map(|k| (k.name.as_str(), k)).collect();
    let mut missing: BTreeSet<_> = anchors.keys().copied().collect();
    for (program, operations) in programs {
        let mut skip = BTreeSet::new();
        for (index, operation) in operations.iter().enumerate() {
            let Operation::Kernel { name } = operation else {
                continue;
            };
            if let Some(&kernel) = anchors.get(name.as_str()) {
                rule(program, operations, index, kernel, &mut skip)?;
                missing.remove(name.as_str());
            }
        }
        if !skip.is_empty() {
            for &index in &skip {
                if matches!(&operations[index], Operation::Kernel { name } if anchors.contains_key(name.as_str()))
                {
                    return Err("Fusion removes another fusion anchor".into());
                }
            }
            remove(operations, &skip);
        }
    }
    if let Some(name) = missing.first() {
        return Err(format!("Unregistered fusion: {name}"));
    }
    Ok(())
}

pub(super) fn retire(
    skip: &mut BTreeSet<usize>,
    steps: impl IntoIterator<Item = usize>,
) -> Result<()> {
    if steps.into_iter().any(|i| !skip.insert(i)) {
        return Err("Overlapping fusions".into());
    }
    Ok(())
}

/// Check architecture-specific storage in the exhaustive package path.
pub(super) struct Storage<'a, Data>(BTreeMap<&'a str, &'a Buffer<Data>>);

impl<'a, Data> Storage<'a, Data> {
    pub fn new(buffers: &'a [Buffer<Data>]) -> Self {
        Self(buffers.iter().map(|b| (b.name.as_str(), b)).collect())
    }

    pub fn names(&self) -> impl Iterator<Item = &'a str> {
        self.0.keys().copied()
    }

    pub fn check(
        &self,
        name: &str,
        dtype: Dtype,
        immutable: bool,
        dimensions: impl FnOnce(&Buffer<Data>) -> Result<bool>,
    ) -> Result<&'a Buffer<Data>> {
        let b = self
            .0
            .get(name)
            .copied()
            .ok_or_else(|| format!("Missing fusion buffer: {name}"))?;
        let access = if immutable {
            Access::Read
        } else {
            Access::ReadWrite
        };
        if b.dtype != dtype
            || b.access != access
            || b.data.is_some() != immutable
            || !dimensions(b)?
        {
            return Err(format!("Invalid fusion buffer shape or storage: {name}"));
        }
        Ok(b)
    }
}
