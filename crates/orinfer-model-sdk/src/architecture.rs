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
    pub kind: PrefillKind,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BatchLayout {
    pub layers: Vec<String>,
    pub row_strides: BTreeMap<String, usize>,
    pub profiles: BTreeMap<usize, PrefillKind>,
    pub hidden: usize,
    #[serde(default)]
    pub small_mixed_shapes: Vec<usize>,
    /// Architecture-owned address-table column order for private decode arenas.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub state_columns: Vec<String>,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BatchSegment {
    pub slot: usize,
    pub tokens: usize,
}
