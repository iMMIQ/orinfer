//! Source registration is independent of deployable package identity and ABI.
use crate::{adapter::Adapter, policy::Policy};
use orinfer_model_sdk::{
    abi::{CreateRequest, CreatedPlan},
    artifact::Result,
};

pub(crate) type Built = (Box<dyn Adapter>, CreatedPlan);
struct Factory {
    architecture: &'static str,
    policies: &'static [Policy],
    create: fn(CreateRequest, Policy) -> Result<Built>,
}
const FACTORIES: &[Factory] = &[
    Factory {
        architecture: "flash_next",
        policies: &[Policy::Int8Quality],
        create: crate::flash_next::create,
    },
    Factory {
        architecture: "qwen3_5",
        policies: &[Policy::Int8Quality],
        create: crate::qwen3_5::create,
    },
];
fn select(architecture: &str, policy: Policy) -> Result<&'static Factory> {
    let factory = FACTORIES
        .iter()
        .find(|f| f.architecture == architecture)
        .ok_or_else(|| format!("Unregistered model architecture: {architecture}"))?;
    if !factory.policies.contains(&policy) {
        return Err(format!(
            "{architecture} does not support compute policy {}",
            policy.name()
        ));
    }
    Ok(factory)
}
pub(crate) fn create(request: CreateRequest) -> Result<Built> {
    let policy = Policy::parse(&request.compute_policy)?;
    (select(&request.architecture, policy)?.create)(request, policy)
}
pub(crate) fn architectures() -> Vec<String> {
    FACTORIES
        .iter()
        .map(|f| f.architecture.to_owned())
        .collect()
}
pub(crate) fn policies() -> Vec<String> {
    FACTORIES
        .iter()
        .flat_map(|f| f.policies)
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|p| p.name().to_owned())
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn family_and_policy_are_validated_as_separate_axes() {
        assert!(select("qwen3_5", Policy::Int8Quality).is_ok());
        assert!(select("unknown", Policy::Int8Quality).is_err());
        assert!(Policy::parse("a4").is_err());
        assert!(Policy::parse("force_int8").is_err());
        assert_eq!(policies(), ["int8_quality"]);
    }
}
