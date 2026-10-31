//! Generated, typed tensor-role bindings live in the native implementation.
use crate::artifact::Result;
use orinfer_model_sdk::{abi::plans::KernelBinding, catalog::Catalog};
mod generated {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub fn kernels(catalog: &Catalog, all: bool) -> Result<Vec<crate::artifact::Kernel>> {
    check(catalog)?;
    generated::REGISTRY
        .iter()?
        .filter(|b| all || b.variant.plan_contract)
        .map(|b| {
            catalog.bind(
                b.variant.definition,
                b.name(),
                b.variant.grid,
                b.arguments()?,
            )
        })
        .collect()
}
pub(crate) fn buffer_argument(name: &str, position: usize) -> Result<String> {
    match kernel_arguments(name)?.get(position) {
        Some(crate::artifact::Argument::Buffer { name }) => Ok(name.clone()),
        _ => Err("Continuation must bind a whole mutable buffer".into()),
    }
}
pub(crate) fn kernel_arguments(name: &str) -> Result<Vec<crate::artifact::Argument>> {
    generated::REGISTRY.find(name)?.arguments()
}

fn check(catalog: &Catalog) -> Result<()> {
    catalog.validate()?;
    if catalog.binding_signature != generated::SIGNATURE || generated::REGISTRY.templates.is_empty()
    {
        return Err("Kernel catalog does not match this native binding registry; rebuild the execution package".into());
    }
    Ok(())
}

pub(crate) fn verify(catalog: &Catalog) -> Result<()> {
    use sha2::{Digest, Sha256};
    check(catalog)?;
    let mut value = serde_json::to_value(catalog).map_err(|e| e.to_string())?;
    value
        .as_object_mut()
        .ok_or("Invalid catalog object")?
        .remove("binding_signature");
    let raw = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    if hex::encode(Sha256::digest(raw)) != generated::CATALOG_ID {
        return Err("Compiled catalog identity differs from native role registry".into());
    }
    Ok(())
}
pub fn export(names: Option<&std::collections::BTreeSet<String>>) -> Result<Vec<KernelBinding>> {
    generated::REGISTRY
        .iter()?
        .filter_map(|b| {
            let name = b.name();
            if names.is_some_and(|names| !names.contains(&name)) {
                return None;
            }
            Some(b.arguments().map(|args| KernelBinding {
                name,
                definition: b.variant.definition,
                grid: b.variant.grid,
                args,
                dynamic: b.variant.dynamic,
                plan_contract: b.variant.plan_contract,
            }))
        })
        .collect()
}
pub fn dynamic(
    catalog: &Catalog,
) -> Result<
    std::collections::BTreeMap<String, orinfer_model_sdk::operators::dynamic::DynamicBatchKernel>,
> {
    check(catalog)?;
    generated::REGISTRY
        .iter()?
        .filter_map(|b| b.variant.dynamic.map(|i| (b.name(), i)))
        .map(|(name, i)| {
            let template = catalog
                .dynamic
                .get(i)
                .ok_or("Unknown dynamic launch definition")?;
            Ok((name.clone(), template.bind(&name)))
        })
        .collect()
}
