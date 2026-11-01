//! Runtime execution policy, independent of checkpoint and operator packages.
use crate::artifact::Result;
use serde::Serialize;
use std::{fmt, str::FromStr};

pub use orinfer_device::GraphMode;
/// Compatibility name for existing callers and serialized load options.
pub type CudaGraphMode = GraphMode;

/// Public request bound; each execution package validates its supported depths.
pub const MAX_MTP_DRAFTS: usize = 31;

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    /// Routed experts per layer equivalent; finite budgets are shared across banks.
    pub resident_experts: ResidentExperts,
    pub cuda_graph: CudaGraphMode,
    /// Hash loaded weight payloads and CPU row tables in addition to structural checks.
    pub verify_weights: bool,
    /// None selects a bounded number of available CPU cores for weight I/O.
    pub load_workers: Option<usize>,
    /// Maximum extra resident prefix snapshots. Zero disables reuse.
    pub prefix_cache_bytes: usize,
    /// Total CPU prefix-state offload budget across all A2 ranks; zero disables.
    pub kv_offload_bytes: usize,
    /// None keeps the package default; zero disables speculative execution.
    pub mtp_drafts: Option<usize>,
    /// Opt in to request-local copied proposals for greedy speculation.
    pub prompt_lookup: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(into = "serde_json::Value")]
pub enum ResidentExperts {
    #[default]
    Full,
    Count(usize),
}
impl From<ResidentExperts> for serde_json::Value {
    fn from(value: ResidentExperts) -> Self {
        match value {
            ResidentExperts::Full => Self::String("full".into()),
            ResidentExperts::Count(n) => n.into(),
        }
    }
}
pub use crate::backend::report::ExpertCacheStatistics;
impl fmt::Display for ResidentExperts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => f.write_str("full"),
            Self::Count(n) => write!(f, "{n}"),
        }
    }
}
impl FromStr for ResidentExperts {
    type Err = String;
    fn from_str(value: &str) -> Result<Self> {
        if value == "full" {
            return Ok(Self::Full);
        }
        value
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .map(Self::Count)
            .ok_or_else(|| "Resident experts must be full or a positive integer".into())
    }
}

pub fn parse_load_workers(value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| (1..=32).contains(n))
        .ok_or_else(|| "Weight load workers must be within 1..32".into())
}

pub fn parse_mtp_drafts(value: &str) -> Result<Option<usize>> {
    if value == "auto" {
        return Ok(None);
    }
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n <= MAX_MTP_DRAFTS)
        .map(Some)
        .ok_or_else(|| "MTP drafts must be auto or an integer in 0..31".into())
}

impl LoadOptions {
    pub(crate) fn weight_workers(self) -> Result<usize> {
        match self.load_workers {
            Some(n) if (1..=32).contains(&n) => Ok(n),
            Some(_) => Err("Weight load workers must be within 1..32".into()),
            None => Ok(std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(4)),
        }
    }

    pub(crate) fn configure_mtp(self, manifest: &mut crate::model::Manifest) -> Result<()> {
        let Some(drafts) = self.mtp_drafts else {
            return Ok(());
        };
        if drafts == 0 {
            disable_mtp(manifest);
            return Ok(());
        }
        if drafts > MAX_MTP_DRAFTS {
            return Err("MTP drafts must be within 0..31".into());
        }
        let spec = manifest
            .mtp
            .as_mut()
            .ok_or("Model package has no MTP support")?;
        let rows = drafts + 1;
        if spec.verification_plan(rows).is_none() {
            let available: Vec<_> = spec
                .verification_widths()
                .into_iter()
                .map(|n| n - 1)
                .collect();
            return Err(format!(
                "Model package does not support {drafts} MTP drafts; available: {available:?}"
            ));
        }
        spec.default_verification_tokens = rows;
        Ok(())
    }
}

fn disable_mtp(manifest: &mut crate::model::Manifest) {
    use crate::model::Operation;
    use std::collections::BTreeSet;
    let Some(spec) = manifest.mtp.take() else {
        return;
    };
    // Keep inactive draft KV virtual, but exclude it from prefix snapshots.
    // Pruning its programs below also removes the growth records that identify
    // its ownership. No target program maps or advances these buffers.
    if let Some(kv) = &manifest.kv_cache {
        let shared: BTreeSet<_> = kv
            .growth
            .values()
            .filter(|growth| growth.position != spec.position)
            .flat_map(|growth| &growth.buffers)
            .collect();
        let draft_kv: BTreeSet<_> = kv
            .growth
            .values()
            .filter(|growth| growth.position == spec.position)
            .flat_map(|growth| &growth.buffers)
            .filter(|name| kv.buffers.contains_key(*name) && !shared.contains(name))
            .collect();
        manifest
            .reset_buffers
            .retain(|name| !draft_kv.contains(name));
    }
    let mut disabled = BTreeSet::from([spec.draft_program]);
    disabled.extend(spec.batch_programs);
    disabled.extend(spec.draft_snapshot_program);
    disabled.extend(spec.draft_restore_program);
    for p in spec.warm_plans {
        disabled.extend([p.program, p.head_program]);
    }
    for p in spec.verification_plans {
        disabled.extend([p.program, p.restore_program, p.capture_program]);
    }
    // Target capture is also a dependency of architecture-owned batch plans.
    let mut required: BTreeSet<_> = spec.capture_plans.into_iter().map(|p| p.program).collect();
    required.extend(["decode".into(), "prefill".into(), "head".into()]);
    for p in &manifest.prefill_plans {
        required.extend([p.prefill_program.clone(), p.head_program.clone()]);
    }
    if let Some(vision) = &manifest.vision {
        required.extend(vision.plans.iter().map(|p| p.program.clone()));
    }
    disabled.retain(|name| !required.contains(name));
    let mut dead = BTreeSet::new();
    manifest.programs.retain(|name, ops| {
        if !disabled.contains(name) {
            return true;
        }
        dead.extend(ops.iter().filter_map(|op| match op {
            Operation::Kernel { name } => Some(name.clone()),
            _ => None,
        }));
        false
    });
    for op in manifest.programs.values().flatten() {
        if let Operation::Kernel { name } = op {
            dead.remove(name);
        }
    }
    // The common operand table also serves ordinary decode. Its registration
    // program may have belonged to the MTP program set, but its indirect
    // bindings remain live when speculation is disabled.
    if let Some(table) = manifest.packet_table() {
        for name in table
            .kernels
            .values()
            .chain(table.copies.iter().map(|copy| &copy.kernel))
        {
            dead.remove(name);
        }
    }
    if let Some(rows) = &mut manifest.row_execution {
        rows.programs
            .retain(|name| manifest.programs.contains_key(name));
        rows.program_capacities
            .retain(|name, _| rows.programs.contains(name));
        let slots: BTreeSet<_> = rows
            .programs
            .iter()
            .flat_map(|name| &manifest.programs[name])
            .filter_map(|op| match op {
                Operation::Kernel { name } => Some(name.clone()),
                _ => None,
            })
            .collect();
        rows.variants.retain(|slot, variants| {
            if slots.contains(slot) {
                true
            } else {
                dead.extend(variants.iter().flat_map(|v| v.kernels.iter().cloned()));
                false
            }
        });
        for variants in rows.variants.values() {
            for name in variants.iter().flat_map(|v| &v.kernels) {
                dead.remove(name);
            }
        }
    }
    manifest
        .kernels
        .retain(|kernel| !dead.contains(&kernel.name));
    if let Some(kv) = &mut manifest.kv_cache {
        kv.growth
            .retain(|program, _| manifest.programs.contains_key(program));
    }
}

pub fn parse_cache_mib(value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_mul(1024 * 1024))
        .ok_or_else(|| "Prefix cache MiB must be a nonnegative integer within usize".into())
}

pub(crate) use orinfer_device::ExecutionPhase;
pub use orinfer_model_sdk::execution::{BufferView, Invocation};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_packet_bindings_survive_disabling_mtp() {
        use serde_json::json;
        let mut manifest: crate::model::Manifest = serde_json::from_value(json!({
            "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":1,
            "max_context":64,"vocab":32,"toolchain":{},"reset_buffers":[],
            "input":"i","token":"t","status":"s","logits":"l","position":"p",
            "weight_bytes":0,"weight_parameters":1,"weight_scope":"test",
            "buffers":[
                {"name":"State","dtype":"f32","shape":[1],"layout":"native_contiguous",
                 "alignment":256,"access":"read_write","data":null},
                {"name":"Pointers","dtype":"u64","shape":[128],"layout":"native_contiguous",
                 "alignment":256,"access":"read_write","data":null}
            ],
            "programs":{
                "decode":[{"kind":"kernel","name":"target"}],
                "draft":[{"kind":"kernel","name":"draft"}],
                "indirect":[{"kind":"kernel","name":"packet_target"}]
            },
            "packet_table":{"buffer":"Pointers","columns":["State"],
                            "kernels":{"target":"packet_target"},"copies":[]},
            "mtp":{
                "position":"p","input":"i","token":"t","status":"s",
                "verification_tokens":"v","verification_status":"vs",
                "draft_logits":"d","verification_logits":"vl","accepted_inputs":"a",
                "target_length":"tl","draft_program":"draft","default_verification_tokens":2,
                "warm_plans":[],"capture_plans":[],"verification_plans":[],
                "batch_programs":["indirect"]
            }
        }))
        .unwrap();
        for name in ["target", "draft", "packet_target"] {
            manifest.kernels.push(
                serde_json::from_value(json!({
                    "name":name,"symbol":"test","args":[],
                    "module":{"file":"kernel.cubin","sha256":"0".repeat(64)},
                    "source":{"file":"kernel.cu","sha256":"0".repeat(64)},
                    "host_abi":{"file":"host.txt","sha256":"0".repeat(64)}
                }))
                .unwrap(),
            );
        }
        manifest
            .packet_table()
            .unwrap()
            .validate(&manifest)
            .unwrap();
        LoadOptions {
            mtp_drafts: Some(0),
            ..Default::default()
        }
        .configure_mtp(&mut manifest)
        .unwrap();
        assert!(manifest.mtp.is_none());
        assert!(!manifest.programs.contains_key("draft"));
        assert!(!manifest.programs.contains_key("indirect"));
        assert!(!manifest.kernels.iter().any(|k| k.name == "draft"));
        manifest
            .packet_table()
            .unwrap()
            .validate(&manifest)
            .unwrap();
    }

    #[test]
    fn modes_roundtrip_and_invalid_values_fail() {
        for mode in [
            CudaGraphMode::DecodeOnly,
            CudaGraphMode::Full,
            CudaGraphMode::Off,
        ] {
            assert_eq!(mode.to_string().parse::<CudaGraphMode>().unwrap(), mode);
            assert_eq!(serde_json::to_string(&mode).unwrap(), format!("\"{mode}\""));
        }
        for invalid in ["", "on", "0", "1", "decode", "FULL"] {
            assert!(invalid.parse::<CudaGraphMode>().is_err());
        }
        assert_eq!(LoadOptions::default().cuda_graph, CudaGraphMode::DecodeOnly);
        assert_eq!(LoadOptions::default().mtp_drafts, None);
        assert!(!LoadOptions::default().prompt_lookup);
        assert_eq!(
            LoadOptions::default().resident_experts,
            ResidentExperts::Full
        );
        assert_eq!(
            serde_json::to_string(&ResidentExperts::Full).unwrap(),
            "\"full\""
        );
        for value in ["full", "1", "128", "256", "512"] {
            assert_eq!(value.parse::<ResidentExperts>().unwrap().to_string(), value);
        }
        for value in ["", "0", "-1", "1.5", "FULL", "18446744073709551616"] {
            assert!(value.parse::<ResidentExperts>().is_err());
        }
        assert_eq!(parse_mtp_drafts("auto").unwrap(), None);
        for n in 0..=MAX_MTP_DRAFTS {
            assert_eq!(parse_mtp_drafts(&n.to_string()).unwrap(), Some(n));
        }
        for invalid in ["", "-1", "32", "1.5", "AUTO", "18446744073709551615"] {
            assert!(parse_mtp_drafts(invalid).is_err());
        }
        for workers in [1, 4, 12, 32] {
            assert_eq!(parse_load_workers(&workers.to_string()).unwrap(), workers);
            assert_eq!(
                LoadOptions {
                    load_workers: Some(workers),
                    ..Default::default()
                }
                .weight_workers()
                .unwrap(),
                workers
            );
        }
        for invalid in ["0", "33", "-1", "", "auto", "1.5"] {
            assert!(parse_load_workers(invalid).is_err());
        }
        assert!(
            LoadOptions {
                load_workers: Some(0),
                ..Default::default()
            }
            .weight_workers()
            .is_err()
        );
        assert_eq!(parse_cache_mib("0").unwrap(), 0);
        assert_eq!(parse_cache_mib("12288").unwrap(), 12usize << 30);
        for invalid in ["-1", "1.5", "", "18446744073709551615"] {
            assert!(parse_cache_mib(invalid).is_err());
        }
    }

    #[test]
    fn decode_only_obeys_call_phase_even_for_a_reused_program() {
        for phase in [
            ExecutionPhase::Prefill,
            ExecutionPhase::Decode,
            ExecutionPhase::Vision,
        ] {
            assert_eq!(
                CudaGraphMode::DecodeOnly.uses_graph(phase),
                phase == ExecutionPhase::Decode
            );
            assert!(CudaGraphMode::Full.uses_graph(phase));
            assert!(!CudaGraphMode::Off.uses_graph(phase));
        }
    }
}
