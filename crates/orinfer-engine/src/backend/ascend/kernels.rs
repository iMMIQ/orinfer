//! Hash-verified TileLang AOT libraries. The online path has no Python/Torch ABI.
use super::{
    api::Handle,
    check,
    runtime::{Buffer, Context, Session, Ticket},
};
use crate::artifact::{Dtype, FileIdentity, Result, read_identity, resolve_file};
use libloading::Library;
use orinfer_device::{Backend, Target};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{CStr, c_char},
    path::{Path, PathBuf},
    rc::Rc,
};

pub mod validation;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
    Scratch,
}
impl Access {
    fn writes(self) -> bool {
        self != Self::Read
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Operand {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub access: Access,
}
impl Operand {
    pub fn bytes(&self) -> Result<usize> {
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err("AOT operand requires nonempty, positive dimensions".into());
        }
        self.shape
            .iter()
            .try_fold(self.dtype.bytes(), |bytes, &dim| {
                bytes
                    .checked_mul(dim)
                    .ok_or_else(|| "AOT operand extent overflow".into())
            })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub schema_version: u32,
    pub target: Target,
    pub op: String,
    pub operands: Vec<Operand>,
    pub parameters: BTreeMap<String, serde_json::Value>,
    pub compiler: BTreeMap<String, String>,
    pub source_sha256: String,
    pub library: FileIdentity,
}
impl Spec {
    pub fn row_capacity(&self) -> Result<Option<usize>> {
        self.parameters
            .get("_row_capacity")
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| (2..=4096).contains(n))
                    .ok_or_else(|| "Invalid AOT dynamic row capacity".into())
            })
            .transpose()
    }

    pub fn validate(&self) -> Result<usize> {
        self.row_capacity()?;
        self.target.validate()?;
        if self.schema_version != 1
            || self.target.backend != Backend::Ascend
            || self.op.is_empty()
            || self.operands.is_empty()
            || self.operands.len() > 32
            || self.compiler.is_empty()
            || self.library.tensor.is_some()
            || self.source_sha256.len() != 64
            || !self.source_sha256.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err("Invalid Ascend AOT kernel schema or identity".into());
        }
        let mut names = BTreeSet::new();
        let mut total = 0usize;
        for operand in &self.operands {
            if operand.name.is_empty() || !names.insert(&operand.name) {
                return Err("Duplicate/empty AOT operand name".into());
            }
            total = total
                .checked_add(operand.bytes()?)
                .ok_or("AOT total extent overflow")?;
        }
        if !self
            .operands
            .iter()
            .any(|o| matches!(o.access, Access::Write | Access::ReadWrite))
        {
            return Err("AOT kernel requires a declared output".into());
        }
        Ok(total)
    }
    pub(super) fn identity(&self) -> serde_json::Value {
        serde_json::json!({"target": self.target, "op": self.op,
            "operands": self.operands, "parameters": self.parameters,
            "compiler": self.compiler, "source_sha256": self.source_sha256})
    }
}

type Entry = unsafe extern "C" fn(*const Handle, u64, Handle) -> i32;
type RowEntry = unsafe extern "C" fn(*const Handle, u64, u64, Handle) -> i32;
enum Launch {
    Static(Entry),
    Rows(RowEntry),
}

/// One hash-verified AOT library with its validated manifest. Loading, hash
/// and ABI checks happen once per unique manifest; instances only bind
/// operand tables against this shared code.
pub struct Template {
    path: PathBuf,
    _library: Library,
    spec: Spec,
    entry: Launch,
}
impl Template {
    pub fn load(session: &Session, manifest: &Path) -> Result<Rc<Self>> {
        let manifest = manifest.canonicalize().map_err(|e| e.to_string())?;
        let base = manifest.parent().ok_or("AOT manifest has no parent")?;
        let spec: Spec = crate::model::read(&manifest)?;
        spec.validate()?;
        spec.target.require_host(std::env::consts::ARCH)?;
        let image = read_identity(base, &spec.library)?;
        crate::model_package::validate_library(&image, &spec.target.host_arch)?;
        let path = resolve_file(base, &spec.library.file)?;
        let _current = session.context.enter()?;
        let soc = super::inventory(&session.context.driver)?
            .soc
            .ok_or("Ascend SoC unavailable")?;
        if !soc.starts_with("Ascend910B") {
            return Err(format!("A2 kernel cannot execute on {soc}"));
        }
        // SAFETY: Explicitly selected, hash/host-verified AOT code. Loading occurs
        // in the owning context; the library remains live until its last fence.
        let library =
            unsafe { Library::new(path) }.map_err(|e| format!("A2 kernel library: {e}"))?;
        // SAFETY: The versioned entry returns a static NUL-terminated descriptor
        // in this hash-verified library. Metadata is checked before any launch.
        let descriptor = unsafe {
            let entry = library
                .get::<unsafe extern "C" fn() -> *const c_char>(b"orinfer_ascend_kernel_spec_v1\0")
                .map_err(|e| e.to_string())?;
            let pointer = entry();
            if pointer.is_null() {
                return Err("AOT library returned null contract".into());
            }
            CStr::from_ptr(pointer)
                .to_str()
                .map_err(|e| e.to_string())?
        };
        let compiled: serde_json::Value =
            serde_json::from_str(descriptor).map_err(|e| e.to_string())?;
        if compiled != spec.identity() {
            return Err("AOT operand/shape contract differs from the compiled library".into());
        }
        // SAFETY: Versioned C signature emitted by the offline compiler wrapper.
        let entry = unsafe {
            if spec.row_capacity()?.is_some() {
                Launch::Rows(
                    *library
                        .get::<RowEntry>(b"orinfer_ascend_kernel_v2\0")
                        .map_err(|e| e.to_string())?,
                )
            } else {
                Launch::Static(
                    *library
                        .get::<Entry>(b"orinfer_ascend_kernel_v1\0")
                        .map_err(|e| e.to_string())?,
                )
            }
        };
        Ok(Rc::new(Self {
            path: manifest,
            _library: library,
            spec,
            entry,
        }))
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn spec(&self) -> &Spec {
        &self.spec
    }
}
struct Bound {
    context: Rc<Context>,
    template: Rc<Template>,
    buffers: Vec<Buffer>,
    pointers: Vec<Handle>,
    rows: usize,
}
impl Bound {
    fn launch(&self, stream: Handle) -> Result<()> {
        // SAFETY: The hash-verified library implements pointer-table ABI v1.
        // Metadata, extents, session ownership and aliases were checked on bind;
        // Bound retains every allocation, native code and context to the fence.
        check(
            unsafe {
                match self.template.entry {
                    Launch::Static(entry) => {
                        entry(self.pointers.as_ptr(), self.pointers.len() as u64, stream)
                    }
                    Launch::Rows(entry) => entry(
                        self.pointers.as_ptr(),
                        self.pointers.len() as u64,
                        self.rows as u64,
                        stream,
                    ),
                }
            },
            "launch TileLang Ascend kernel",
        )
    }
}

#[derive(Clone)]
pub struct Kernel {
    bound: Rc<Bound>,
}

/// A model fragment submits one ordered native batch and records one completion
/// fence. Each bound kernel keeps its validated operands and AOT library alive.
pub struct Program {
    context: Rc<Context>,
    kernels: Rc<Vec<Rc<Bound>>>,
}
impl Program {
    pub fn new(session: &Session, kernels: &[&Kernel]) -> Result<Self> {
        if kernels.is_empty() {
            return Err("AOT program must contain a kernel".into());
        }
        for kernel in kernels {
            session.same_context(&kernel.bound.context)?;
        }
        Ok(Self {
            context: Rc::clone(&session.context),
            kernels: Rc::new(kernels.iter().map(|k| Rc::clone(&k.bound)).collect()),
        })
    }
    pub fn enqueue(&self, session: &Session) -> Result<Ticket> {
        session.same_context(&self.context)?;
        let kernels = Rc::clone(&self.kernels);
        // SAFETY: Every bound operand and library is retained through the one
        // batch fence; all launches use this session's ordered compute stream.
        unsafe {
            session.submit_compute(vec![], Box::new(Rc::clone(&kernels)), |stream| {
                for kernel in kernels.iter() {
                    kernel.launch(stream)?;
                }
                Ok(())
            })
        }
    }
    pub fn capture(&self, session: &Session) -> Result<super::graph::Graph> {
        session.same_context(&self.context)?;
        let kernels = Rc::clone(&self.kernels);
        // SAFETY: The complete native program has stable operands, host pointer
        // tables and libraries, all retained by the captured graph.
        unsafe {
            super::graph::Graph::capture(session, Box::new(Rc::clone(&kernels)), |stream| {
                for kernel in kernels.iter() {
                    kernel.launch(stream)?;
                }
                Ok(())
            })
        }
    }
}
impl Kernel {
    pub fn load(session: &Session, manifest: &Path, buffers: &[Buffer]) -> Result<Self> {
        let template = Template::load(session, manifest)?;
        Self::bind(session, &template, buffers)
    }
    /// Binds an operand table to an already verified AOT template. The
    /// manifest is not re-read here; cache owners keep the template alive.
    pub fn bind(session: &Session, template: &Rc<Template>, buffers: &[Buffer]) -> Result<Self> {
        let rows = template.spec.row_capacity()?.unwrap_or(1);
        Self::bind_rows(session, template, buffers, rows)
    }
    /// Bind a real extent to one capacity-backed AOT template. Captured graphs
    /// retain this immutable extent together with their operands and library.
    pub fn bind_rows(
        session: &Session,
        template: &Rc<Template>,
        buffers: &[Buffer],
        rows: usize,
    ) -> Result<Self> {
        let spec = &template.spec;
        if rows == 0 || spec.row_capacity()?.is_some_and(|cap| rows > cap) {
            return Err("AOT rows exceed the compiled capacity".into());
        }
        if buffers.len() != spec.operands.len() {
            return Err("AOT operand count differs from the compiled ABI".into());
        }
        for (i, (buffer, operand)) in buffers.iter().zip(&spec.operands).enumerate() {
            session.require(buffer)?;
            if buffer.bytes() != operand.bytes()? {
                return Err(format!("{}: AOT operand byte extent differs", operand.name));
            }
            for (j, previous) in buffers.iter().enumerate().take(i) {
                if buffer.overlaps(previous)
                    && (operand.access.writes() || spec.operands[j].access.writes())
                {
                    return Err("AOT output must not alias another operand".into());
                }
            }
        }
        Ok(Self {
            bound: Rc::new(Bound {
                context: Rc::clone(&session.context),
                template: Rc::clone(template),
                buffers: buffers.to_vec(),
                pointers: buffers.iter().map(Buffer::pointer).collect(),
                rows,
            }),
        })
    }
    pub fn spec(&self) -> &Spec {
        &self.bound.template.spec
    }
    pub(super) fn accesses(&self) -> impl Iterator<Item = (&Buffer, bool)> {
        self.bound
            .buffers
            .iter()
            .zip(&self.bound.template.spec.operands)
            .map(|(buffer, operand)| (buffer, operand.access.writes()))
    }
    pub fn template(&self) -> &Rc<Template> {
        &self.bound.template
    }
    /// Submits this kernel on an explicit stream without a completion fence.
    /// Used by captured segments that replay the launch inside an ACL Graph.
    pub(super) fn launch_on_stream(&self, stream: Handle) -> Result<()> {
        self.bound.launch(stream)
    }
    pub fn enqueue(&self, session: &Session) -> Result<Ticket> {
        session.same_context(&self.bound.context)?;
        let bound = Rc::clone(&self.bound);
        // SAFETY: Bound retains all stable operands and code. Every launch uses
        // this session's ordered compute stream; no shared executor is mutated.
        unsafe {
            session.submit_compute(
                bound.buffers.clone(),
                Box::new(Rc::clone(&bound)),
                |stream| bound.launch(stream),
            )
        }
    }
    pub fn capture(&self, session: &Session) -> Result<super::graph::Graph> {
        session.same_context(&self.bound.context)?;
        let bound = Rc::clone(&self.bound);
        // SAFETY: Bound retains the stable pointer table, operands and native
        // library throughout capture, replay and the final completion fence.
        unsafe {
            super::graph::Graph::capture(session, Box::new(Rc::clone(&bound)), |stream| {
                bound.launch(stream)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operand_extents_reject_zero_and_overflow() {
        let mut operand = Operand {
            name: "x".into(),
            dtype: Dtype::Bf16,
            shape: vec![3, 5120],
            access: Access::Read,
        };
        assert_eq!(operand.bytes().unwrap(), 30720);
        operand.shape = vec![usize::MAX, 2];
        assert!(operand.bytes().is_err());
        operand.shape = vec![0];
        assert!(operand.bytes().is_err());
    }
}
