use orinfer_model_sdk::{
    architecture::{ExpertBank, PrefillProfile},
    model::Manifest,
};

#[cfg_attr(test, derive(serde::Deserialize, serde::Serialize))]
pub struct CreateRequest {
    #[cfg_attr(test, serde(default))]
    pub expert_contracts: bool,
    /// Offline inspection retains contracts used only while lowering the model.
    #[cfg_attr(test, serde(default))]
    pub build_bindings: bool,
    /// Immutable bindings required by architecture-owned execution and residency contracts.
    #[cfg_attr(test, serde(default))]
    pub kernel_contracts: Vec<crate::artifact::Kernel>,
    #[cfg_attr(test, serde(default))]
    pub model_root: std::path::PathBuf,
    #[cfg_attr(test, serde(skip))]
    pub operands: crate::packet::Operands,
    /// Full CPU weight-table verification; structural checks are always required.
    #[cfg_attr(test, serde(default))]
    pub verify_weights: bool,
    pub config: serde_json::Value,
    pub architecture: String,
    pub compute_policy: String,
    pub expected_signature: serde_json::Value,
    pub metadata: Manifest,
    pub prefill_profiles: Vec<PrefillProfile>,
}
pub struct CreatedPlan {
    pub expert_banks: Vec<ExpertBank>,
    pub metadata: Manifest,
}
