//! Open package identifiers and generic execution profile metadata.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub type Architecture = String;
pub type ComputePolicy = String;
pub type PrefillKind = String;
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillProfile {
    pub tokens: usize,
    /// Zero means exact shape; otherwise stateful kernels accept logical rows
    /// down to this minimum while tensor strides retain the physical capacity.
    #[serde(default, skip_serializing_if = "crate::model::is_zero")]
    pub min_tokens: usize,
    pub kind: PrefillKind,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BatchLayout {
    pub layers: Vec<String>,
    pub row_strides: BTreeMap<String, usize>,
    pub profiles: BTreeMap<usize, PrefillKind>,
    pub hidden: usize,
    /// Nonzero means every real segment length through this capacity is supported.
    #[serde(default, skip_serializing_if = "crate::model::is_zero")]
    pub segment_tokens: usize,
    /// Architecture-owned address-table column order for private decode arenas.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub state_columns: Vec<String>,
    /// Native preparation programs that map packet-specific causal workspace.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preparation_programs: BTreeMap<usize, String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSegment {
    pub slot: usize,
    pub tokens: usize,
}

/// Whole-expert storage and the kernels whose histogram selects its consumers.
/// Each consumer argument is an I32[experts] count array produced on the stream.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertBank {
    pub experts: usize,
    pub buffers: Vec<String>,
    pub consumers: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Copy)]
pub struct ImageGrid {
    pub grid_height: usize,
    pub grid_width: usize,
}
