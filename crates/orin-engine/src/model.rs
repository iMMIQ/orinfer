//! AOT model programs: architecture-specific plans, parameterized Rust runtime.
use crate::artifact::{Access, Argument, Buffer, Kernel, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub target: String,
    pub model: String,
    pub chunk_tokens: usize,
    #[serde(default)]
    pub prefill_plans: Vec<PrefillPlan>,
    pub max_context: usize,
    pub vocab: usize,
    pub toolchain: BTreeMap<String, String>,
    pub buffers: Vec<Buffer>,
    pub kernels: Vec<Kernel>,
    pub programs: BTreeMap<String, Vec<Operation>>,
    pub reset_buffers: Vec<String>,
    pub input: String,
    pub token: String,
    pub status: String,
    pub logits: String,
    pub position: String,
    pub weight_bytes: usize,
    pub weight_parameters: usize,
    pub weight_scope: String,
    #[serde(default)]
    pub vision: Option<crate::vision::VisionSpec>,
}

/// Fixed-shape graphs sharing one model's weights, workspace and private state.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillPlan {
    pub chunk_tokens: usize,
    pub prefill_program: String,
    pub head_program: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Kernel {
        name: String,
    },
    Copy {
        source: String,
        destination: String,
        bytes: usize,
    },
    Zero {
        destination: String,
        bytes: usize,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requests {
    pub requests: Vec<Request>,
    #[serde(default)]
    pub logits_output: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: String,
    pub input_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    #[serde(default)]
    pub forced_tokens: Vec<u32>,
    #[serde(default)]
    pub logits_steps: Vec<usize>,
}

impl Manifest {
    pub fn validate<'a>(&'a self) -> Result<usize> {
        if self.schema_version != 1
            || self.target != "sm_87"
            || self.model.is_empty()
            || self.toolchain.is_empty()
        {
            return Err("Expected populated model schema 1 for SM87".into());
        }
        if self.chunk_tokens == 0
            || self.chunk_tokens > self.max_context
            || self.max_context > i32::MAX as usize
            || self.vocab == 0
            || self.vocab > i32::MAX as usize
        {
            return Err("Invalid model dimensions".into());
        }
        let mut identities = BTreeMap::<&str, &str>::new();
        let mut identity = |id: &'a crate::artifact::FileIdentity| -> Result<()> {
            if id.sha256.len() != 64
                || !id
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return Err("Malformed file hash".into());
            }
            if identities
                .insert(&id.file, &id.sha256)
                .is_some_and(|old| old != id.sha256)
            {
                return Err("Conflicting identities for one artifact file".into());
            }
            Ok(())
        };
        let mut buffers = BTreeMap::new();
        let mut total = 0usize;
        let mut weights = 0usize;
        for b in &self.buffers {
            if let Some(id) = &b.data {
                identity(id)?;
            }
            let bytes = b.bytes()?;
            if b.name.is_empty()
                || b.layout.is_empty()
                || !b.alignment.is_power_of_two()
                || b.alignment > 256
                || buffers.insert(&b.name, b).is_some()
            {
                return Err("Invalid/duplicate model buffer".into());
            }
            total = total.checked_add(bytes).ok_or("Buffer total overflow")?;
            if b.access == Access::Read {
                weights = weights.checked_add(bytes).ok_or("Weight total overflow")?;
            }
        }
        if weights != self.weight_bytes || self.weight_parameters == 0 {
            return Err("Weight accounting mismatch".into());
        }
        let lookup = |name: &str| -> Result<&Buffer> {
            buffers
                .get(&name.to_string())
                .copied()
                .ok_or_else(|| format!("Unknown buffer {name}"))
        };
        if let Some(vision) = &self.vision {
            vision.validate(self)?;
        }
        for (name, minimum) in [
            (
                &self.input,
                self.chunk_tokens
                    .checked_mul(4)
                    .ok_or("Input size overflow")?,
            ),
            (&self.token, 4),
            (&self.status, 4),
            (&self.position, 4),
        ] {
            let b = lookup(name)?;
            if b.bytes()? < minimum
                || !matches!(b.dtype, crate::artifact::Dtype::I32)
                || b.access == Access::Read
            {
                return Err(format!("Invalid control buffer {name}"));
            }
        }
        let logits = lookup(&self.logits)?;
        if !matches!(
            logits.dtype,
            crate::artifact::Dtype::F16 | crate::artifact::Dtype::F32
        ) || logits.bytes()?
            != self
                .vocab
                .checked_mul(logits.dtype.bytes())
                .ok_or("Logits size overflow")?
        {
            return Err("Invalid full-vocabulary logits buffer".into());
        }
        let mut names = BTreeSet::new();
        for k in &self.kernels {
            identity(&k.module)?;
            identity(&k.source)?;
            identity(&k.host_abi)?;
            if k.name.is_empty()
                || !names.insert(&k.name)
                || k.symbol.is_empty()
                || k.symbol.contains('\0')
                || k.args.is_empty()
                || k.cooperative
                || k.grid.contains(&0)
                || k.block.contains(&0)
                || k.block
                    .iter()
                    .try_fold(1u32, |n, d| n.checked_mul(*d))
                    .is_none_or(|n| n > 1024)
            {
                return Err(format!("Invalid model kernel {}", k.name));
            }
            for a in &k.args {
                match a {
                    Argument::Buffer { name } => {
                        lookup(name)?;
                    }
                    Argument::F32 { value } if !value.is_finite() => {
                        return Err("Nonfinite scalar".into());
                    }
                    _ => (),
                }
            }
        }
        for required in ["prefill", "head", "decode"] {
            if self.programs.get(required).is_none_or(Vec::is_empty) {
                return Err(format!("Missing program {required}"));
            }
        }
        let mut plan_sizes = BTreeSet::new();
        for plan in &self.prefill_plans {
            if plan.chunk_tokens == 0
                || plan.chunk_tokens > self.chunk_tokens
                || !plan_sizes.insert(plan.chunk_tokens)
                || plan.prefill_program == "decode"
                || plan.head_program == "decode"
                || plan.prefill_program == plan.head_program
                || [&plan.prefill_program, &plan.head_program]
                    .iter()
                    .any(|name| self.programs.get(*name).is_none_or(Vec::is_empty))
            {
                return Err("Invalid/duplicate prefill plan or missing program".into());
            }
        }
        for ops in self.programs.values() {
            for op in ops {
                let writable = |name: &str, bytes: usize| -> Result<()> {
                    let b = lookup(name)?;
                    if bytes == 0 || bytes > b.bytes()? || b.access == Access::Read {
                        return Err(format!("Invalid destination range {name}"));
                    }
                    Ok(())
                };
                match op {
                    Operation::Kernel { name } if !names.contains(name) => {
                        return Err(format!("Unknown kernel {name}"));
                    }
                    Operation::Copy {
                        source,
                        destination,
                        bytes,
                    } => {
                        writable(destination, *bytes)?;
                        if source == destination || *bytes > lookup(source)?.bytes()? {
                            return Err("Invalid copy source/alias".into());
                        }
                    }
                    Operation::Zero { destination, bytes } => writable(destination, *bytes)?,
                    _ => (),
                }
            }
        }
        if !self.reset_buffers.contains(&self.position) {
            return Err("Position must reset between requests".into());
        }
        for name in &self.reset_buffers {
            if lookup(name)?.access == Access::Read {
                return Err("Reset of immutable weight".into());
            }
        }
        Ok(total)
    }
    pub fn validate_requests(&self, r: &Requests) -> Result<()> {
        if r.requests.is_empty() {
            return Err("No requests".into());
        }
        let mut ids = BTreeSet::new();
        for q in &r.requests {
            if q.id.is_empty()
                || !ids.insert(&q.id)
                || q.id.contains('/')
                || q.id.contains('\\')
                || q.id == "."
                || q.id == ".."
            {
                return Err("Invalid/duplicate request identity".into());
            }
            if q.input_tokens.is_empty()
                || self.select_prefill_plan(q.input_tokens.len()).is_err()
                || q.max_new_tokens == 0
                || q.input_tokens
                    .len()
                    .checked_add(q.max_new_tokens - 1)
                    .is_none_or(|n| n > self.max_context)
            {
                return Err(format!(
                    "{}: input must fit a prefill plan, with generation fitting context",
                    q.id
                ));
            }
            if q.input_tokens
                .iter()
                .chain(&q.forced_tokens)
                .any(|id| *id as usize >= self.vocab)
                || q.forced_tokens.len() > q.max_new_tokens - 1
                || q.logits_steps.iter().any(|n| *n >= q.max_new_tokens)
            {
                return Err(format!("{}: invalid token/history/logits index", q.id));
            }
            if !q.logits_steps.is_empty() && r.logits_output.is_none() {
                return Err("Logits directory required".into());
            }
        }
        Ok(())
    }

    pub fn select_prefill_plan(&self, tokens: usize) -> Result<(usize, &str, &str)> {
        if tokens == 0 {
            return Err("Empty prefill".into());
        }
        if self.prefill_plans.is_empty() {
            if self.chunk_tokens == 0 || !tokens.is_multiple_of(self.chunk_tokens) {
                return Err("Input is not a multiple of the legacy chunk size".into());
            }
            return Ok((self.chunk_tokens, "prefill", "head"));
        }
        let plan = self
            .prefill_plans
            .iter()
            .filter(|plan| plan.chunk_tokens > 0 && tokens.is_multiple_of(plan.chunk_tokens))
            .max_by_key(|plan| plan.chunk_tokens)
            .ok_or("No compatible prefill plan")?;
        Ok((plan.chunk_tokens, &plan.prefill_program, &plan.head_program))
    }
}

#[derive(Debug, Serialize)]
pub struct RequestReport {
    pub id: String,
    pub input_tokens: usize,
    pub prefill_chunk_tokens: usize,
    pub prefill_program: String,
    pub output_tokens: Vec<u32>,
    pub reset_s: f64,
    pub prefill_s: f64,
    pub head_s: f64,
    pub ttft_s: f64,
    pub decode_s: f64,
    pub prefill_tps: f64,
    pub decode_tps: Option<f64>,
    pub final_position: usize,
    pub logits_files: Vec<String>,
    pub diagnostic: bool,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub manifest_sha256: String,
    pub model: String,
    pub device: crate::cuda::DeviceInfo,
    pub load_to_ready_s: f64,
    pub weight_io_hash_s: f64,
    pub weight_upload_s: f64,
    pub module_load_bind_s: f64,
    pub graph_capture_s: f64,
    pub buffer_bytes: usize,
    pub weight_bytes: usize,
    pub effective_weight_bits: f64,
    pub weight_scope: String,
    pub requests: Vec<RequestReport>,
    pub seed: u64,
    pub timing_scope: &'static str,
}

pub fn run(manifest: &Path, requests: &Path) -> Result<Report> {
    crate::cuda::run_model(manifest, requests)
}
pub(crate) fn read<T: serde::de::DeserializeOwned>(p: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))
}

/// Thread-affine resident model. All CUDA resources remain on the creating thread.
pub struct Model(crate::cuda::ModelRuntime);
impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        crate::cuda::ModelRuntime::load(path).map(Self)
    }
    pub fn max_context(&self) -> usize {
        self.0.manifest.max_context
    }
    pub fn vocab(&self) -> usize {
        self.0.manifest.vocab
    }
    pub fn vision(&self) -> Option<&crate::vision::VisionSpec> {
        self.0.manifest.vision.as_ref()
    }
    pub fn generate(
        &mut self,
        input: &[u32],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.0
            .generate(input, None, limit, options, cancelled, emit)
    }
    pub fn generate_visual(
        &mut self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.0
            .generate(input, Some(images), limit, options, cancelled, emit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Manifest {
        let buffers = [
            ("Weight", "f16", 2, "read"), ("Input", "i32", 2, "read_write"),
            ("Token", "i32", 1, "read_write"), ("Status", "i32", 1, "read_write"),
            ("Step", "i32", 1, "read_write"), ("Logits", "f16", 4, "read_write"),
        ].map(|(name,dtype,rows,access)|serde_json::json!({"name":name,"dtype":dtype,"shape":[rows],"layout":"contiguous","alignment":256,"access":access,"data":null}));
        let id = serde_json::json!({"file":"a.cubin","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({"schema_version":1,"target":"sm_87","model":"test","chunk_tokens":2,"max_context":8,"vocab":4,"toolchain":{"test":"test"},
            "buffers":buffers,"kernels":[{"name":"k","module":id,"source":id,"host_abi":id,"symbol":"k","grid":[1,1,1],"block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,"args":[{"kind":"buffer","name":"Input"}]}],
            "programs":{"prefill":[{"kind":"kernel","name":"k"}],"head":[{"kind":"kernel","name":"k"}],"decode":[{"kind":"kernel","name":"k"}]},
            "reset_buffers":["Step"],"input":"Input","token":"Token","status":"Status","logits":"Logits","position":"Step","weight_bytes":4,"weight_parameters":2,"weight_scope":"test"})).unwrap()
    }
    #[test]
    fn rejects_state_aliases_out_of_bounds_and_weight_writes() {
        let mut m = fixture();
        assert!(m.validate().is_ok());
        m.programs.get_mut("decode").unwrap().push(Operation::Copy {
            source: "Input".into(),
            destination: "Input".into(),
            bytes: 4,
        });
        assert!(m.validate().is_err());
        m.programs.get_mut("decode").unwrap().pop();
        m.programs.get_mut("decode").unwrap().push(Operation::Zero {
            destination: "Token".into(),
            bytes: 8,
        });
        assert!(m.validate().is_err());
        m.programs.get_mut("decode").unwrap().pop();
        m.programs.get_mut("decode").unwrap().push(Operation::Zero {
            destination: "Weight".into(),
            bytes: 4,
        });
        assert!(m.validate().is_err());
    }
    #[test]
    fn rejects_conflicting_cached_module_identity_and_unreset_position() {
        let mut m = fixture();
        m.kernels[0].source.sha256 = "1".repeat(64);
        assert!(m.validate().is_err());
        m.kernels[0].source.sha256 = "0".repeat(64);
        m.reset_buffers.clear();
        assert!(m.validate().is_err());
    }
    #[test]
    fn rejects_invalid_histories_and_context_capacity() {
        let m = fixture();
        let mut r = Requests {
            requests: vec![Request {
                id: "one".into(),
                input_tokens: vec![0, 1],
                max_new_tokens: 2,
                forced_tokens: vec![2],
                logits_steps: vec![],
            }],
            logits_output: None,
        };
        assert!(m.validate_requests(&r).is_ok());
        r.requests[0].forced_tokens[0] = 4;
        assert!(m.validate_requests(&r).is_err());
        r.requests[0].forced_tokens[0] = 2;
        r.requests[0].max_new_tokens = 8;
        assert!(m.validate_requests(&r).is_err());
        r.requests[0].max_new_tokens = 2;
        r.requests[0].input_tokens.push(0);
        assert!(m.validate_requests(&r).is_err());
        r.requests[0].input_tokens.pop();
        r.requests[0].logits_steps.push(2);
        assert!(m.validate_requests(&r).is_err());
    }

    #[test]
    fn selects_largest_compatible_plan_and_rejects_invalid_metadata() {
        let mut m = fixture();
        assert_eq!(m.select_prefill_plan(4).unwrap(), (2, "prefill", "head"));
        m.buffers
            .iter_mut()
            .find(|b| b.name == "Input")
            .unwrap()
            .shape = vec![8];
        m.chunk_tokens = 8;
        m.prefill_plans = vec![
            PrefillPlan {
                chunk_tokens: 2,
                prefill_program: "prefill".into(),
                head_program: "head".into(),
            },
            PrefillPlan {
                chunk_tokens: 8,
                prefill_program: "prefill".into(),
                head_program: "head".into(),
            },
        ];
        assert!(m.validate().is_ok());
        assert_eq!(m.select_prefill_plan(8).unwrap().0, 8);
        assert_eq!(m.select_prefill_plan(6).unwrap().0, 2);
        assert!(m.select_prefill_plan(3).is_err());
        assert!(m.select_prefill_plan(0).is_err());
        m.prefill_plans[1].chunk_tokens = 2;
        assert!(m.validate().is_err());
        m.prefill_plans[1].chunk_tokens = 8;
        m.prefill_plans[1].prefill_program = "missing".into();
        assert!(m.validate().is_err());
        m.prefill_plans[1].prefill_program = "decode".into();
        assert!(m.validate().is_err());
    }
}
