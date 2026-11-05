//! Dynamic, replicated ViT/aligner execution on the worker's existing ACL session.
use super::{Buffer, Checkpoint, Definition, Kernel, Session, Template, load};
use crate::{
    artifact::{Result, resolve_file},
    backend::ascend::graph::Graph,
};
use half::bf16;
use orinfer_model_sdk::vision::{ImageInput, ImageLayout, Precision, VisionSpec};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    path::Path,
    rc::Rc,
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Program {
    schema_version: u32,
    pub spec: VisionSpec,
    rope_theta: f32,
    patch_columns: usize,
    features_capacity: usize,
    buffers: Vec<Definition>,
    body: Vec<Operation>,
    #[serde(default)]
    segment_ends: Vec<usize>,
}
impl Program {
    fn segments(&self) -> Result<Vec<usize>> {
        let ends = if self.segment_ends.is_empty() {
            vec![self.body.len()]
        } else {
            self.segment_ends.clone()
        };
        if self.body.is_empty()
            || ends.last() != Some(&self.body.len())
            || ends[0] == 0
            || ends.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("Invalid vision frame boundaries".into());
        }
        Ok(ends)
    }
    pub(super) fn startup_fraction(&self) -> f64 {
        let Some(&last) = self.segment_ends.last() else {
            return 1.;
        };
        let longest = self
            .segment_ends
            .iter()
            .copied()
            .zip(std::iter::once(0).chain(self.segment_ends.iter().copied()))
            .map(|(end, start)| end.saturating_sub(start))
            .max()
            .unwrap_or(last);
        longest as f64 / last.max(1) as f64
    }
    pub(super) fn advance_budget(&self, estimated_s: f64, quantum_s: f64) -> usize {
        let count = self.segment_ends.len().max(1);
        if estimated_s <= quantum_s {
            return count;
        }
        ((quantum_s / (estimated_s * self.startup_fraction())) as usize).clamp(1, count)
    }
    pub(super) fn manifests(&self) -> impl Iterator<Item = &str> {
        self.body.iter().map(|op| op.manifest.as_str())
    }
    pub(super) fn device_bytes(&self) -> Result<usize> {
        self.buffers
            .iter()
            .filter(|b| b.view.is_none())
            .try_fold(0usize, |sum, b| {
                sum.checked_add(b.bytes)
                    .ok_or_else(|| "Vision memory size overflow".into())
            })
    }
    pub(super) fn definitions(&self) -> &[Definition] {
        &self.buffers
    }
    pub(super) fn preflight(&self, base: &Path) -> Result<()> {
        self.segments()?;
        let mut names = BTreeMap::new();
        for definition in &self.buffers {
            if definition.bytes == 0
                || names
                    .insert(definition.name.as_str(), definition.bytes)
                    .is_some()
            {
                return Err("Invalid vision buffer inventory".into());
            }
        }
        let mut specs = BTreeMap::new();
        for op in &self.body {
            if !specs.contains_key(&op.manifest) {
                let spec: super::super::kernels::Spec =
                    crate::model::read(&resolve_file(base, &op.manifest)?)?;
                spec.validate()?;
                specs.insert(op.manifest.clone(), spec);
            }
            let spec = &specs[&op.manifest];
            let capacity = if matches!(op.rows, Rows::Features) {
                self.features_capacity
            } else {
                self.spec.max_patches
            };
            if spec.row_capacity()? != Some(capacity) || spec.operands.len() != op.buffers.len() {
                return Err("Vision operation row/operand contract differs".into());
            }
            for (operand, name) in spec.operands.iter().zip(&op.buffers) {
                if names.get(name.as_str()).copied() != Some(operand.bytes()?) {
                    return Err(format!("Vision operation buffer extent differs: {name}"));
                }
            }
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Operation {
    manifest: String,
    buffers: Vec<String>,
    rows: Rows,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Rows {
    Patches,
    Features,
}
struct Prepared {
    template: Rc<Template>,
    buffers: Vec<Buffer>,
    features: bool,
}
struct Frame {
    kernels: Rc<Vec<Kernel>>,
    graphs: Option<Rc<Vec<Graph>>>,
}
struct Running {
    key: (usize, usize),
    span: usize,
    width: usize,
    cursor: usize,
    upload_s: f64,
    compute_s: f64,
    max_advance_s: f64,
    graph_replay: bool,
}
pub(super) struct Encoder {
    spec: VisionSpec,
    theta: f32,
    features_capacity: usize,
    buffers: BTreeMap<String, Buffer>,
    operations: Vec<Prepared>,
    frames: BTreeMap<(usize, usize), Frame>,
    order: VecDeque<(usize, usize)>,
    markers: [Vec<u8>; 3],
    phase_grid: Option<(usize, usize)>,
    segment_ends: Vec<usize>,
    running: Option<Running>,
}
pub(super) struct Encoded {
    /// Row-major BF16 in the same basis as the language embedding table.
    pub data: Vec<u8>,
    pub span: usize,
    pub patches: usize,
    pub upload_s: f64,
    pub compute_s: f64,
    pub download_s: f64,
    pub graph_replay: bool,
    pub max_advance_s: f64,
}
impl Encoder {
    pub fn load(session: &Session, path: &Path, checkpoint: &mut Checkpoint) -> Result<Self> {
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        let program: Program = crate::model::read(&path)?;
        Self::from_program(
            session,
            &program,
            path.parent().ok_or("Vision program has no directory")?,
            checkpoint,
        )
    }
    pub fn from_program(
        session: &Session,
        program: &Program,
        base: &Path,
        checkpoint: &mut Checkpoint,
    ) -> Result<Self> {
        program.preflight(base)?;
        let spec = &program.spec;
        if program.schema_version != 1
            || spec.layout != ImageLayout::DeepseekV41
            || !matches!(spec.dtype, Precision::Bf16)
            || spec.hidden != 5120
            || spec.patch_size != 14
            || spec.temporal_patch_size != 1
            || spec.merge_size != 3
            || spec.encoder_patch_values()? != program.patch_columns
            || !program.rope_theta.is_finite()
            || program.rope_theta <= 0.0
            || spec.max_patches == 0
            || spec.max_patches > 9216
            || !spec.max_patches.is_multiple_of(128)
            || program.features_capacity > 1024
            || program.features_capacity == 0
            || program.body.is_empty()
        {
            return Err("Invalid native DeepSeek vision program".into());
        }
        let mut buffers: BTreeMap<String, Buffer> = BTreeMap::new();
        // Views are resolved after all owning allocations, independent of JSON order.
        for definition in &program.buffers {
            if buffers.contains_key(&definition.name)
                || definition.bytes == 0
                || definition.weight.is_some()
                || definition.prepared_weight.is_some()
                || definition.data.is_some()
            {
                return Err("Invalid or duplicate native vision buffer".into());
            }
            if definition.view.is_some() {
                continue;
            }
            let buffer = session.allocate(definition.bytes)?;
            if let Some(tensor) = &definition.prepared_tensor {
                let source = checkpoint.source(&tensor.name, &tensor.dtype, &tensor.shape)?;
                let file = File::open(&source.path).map_err(|e| e.to_string())?;
                load::upload_file_range(session, &buffer, &file, source.offset, source.bytes)?;
            } else {
                session.fill(&buffer, 0)?.wait()?;
            }
            buffers.insert(definition.name.clone(), buffer);
        }
        for definition in &program.buffers {
            if let Some(view) = &definition.view {
                if definition.prepared_tensor.is_some()
                    || view.row_capacity.is_some()
                    || view.offset_expression.is_some()
                    || buffers.contains_key(&definition.name)
                {
                    return Err("Invalid native vision view".into());
                }
                let buffer = buffers
                    .get(&view.parent)
                    .ok_or("Missing vision view parent")?
                    .slice(view.offset, definition.bytes)?;
                buffers.insert(definition.name.clone(), buffer);
            }
        }
        let mut templates = BTreeMap::new();
        let mut operations = Vec::with_capacity(program.body.len());
        for operation in &program.body {
            if !templates.contains_key(&operation.manifest) {
                templates.insert(
                    operation.manifest.clone(),
                    Template::load(session, &resolve_file(base, &operation.manifest)?)?,
                );
            }
            operations.push(Prepared {
                template: Rc::clone(&templates[&operation.manifest]),
                buffers: operation
                    .buffers
                    .iter()
                    .map(|name| {
                        buffers
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("Missing vision operand: {name}"))
                    })
                    .collect::<Result<_>>()?,
                features: matches!(operation.rows, Rows::Features),
            });
        }
        for (name, bytes) in [
            (&spec.pixels, spec.max_patches * program.patch_columns * 2),
            (&spec.grid, 32),
            (&spec.output, program.features_capacity * spec.hidden * 2),
            (&"vision.phase".into(), spec.max_patches * 64 * 4),
        ] {
            if buffers.get(name).map(Buffer::bytes) != Some(bytes) {
                return Err(format!("Vision control buffer differs: {name}"));
            }
        }
        let mut markers = [Vec::new(), Vec::new(), Vec::new()];
        for (destination, name) in
            markers
                .iter_mut()
                .zip(["image_start", "image_end", "image_newline"])
        {
            let tensor = checkpoint.read(name)?;
            if tensor.dtype != safetensors::Dtype::BF16 || tensor.data.len() != spec.hidden * 2 {
                return Err(format!("Invalid learned image delimiter: {name}"));
            }
            *destination = tensor.data.to_vec();
        }
        Ok(Self {
            spec: program.spec.clone(),
            theta: program.rope_theta,
            features_capacity: program.features_capacity,
            buffers,
            operations,
            frames: BTreeMap::new(),
            order: VecDeque::new(),
            markers,
            phase_grid: None,
            segment_ends: program.segments()?,
            running: None,
        })
    }
    pub fn encode(&mut self, session: &Session, image: &ImageInput) -> Result<Encoded> {
        self.begin(session, image)?;
        self.advance(session, usize::MAX)?
            .ok_or_else(|| "Full vision execution did not complete".into())
    }
    pub(super) fn cancel(&mut self) {
        // Every segment has completed before control returns to the scheduler.
        self.running = None;
    }
    pub(super) fn begin(&mut self, session: &Session, image: &ImageInput) -> Result<()> {
        if self.running.is_some() {
            return Err("Vision encoder already has an active frame".into());
        }
        let span = self.spec.feature_count(image)?;
        let patches = image
            .grid_height
            .checked_mul(image.grid_width)
            .ok_or("Vision grid overflow")?;
        let height = image.grid_height.div_ceil(3);
        let width = image.grid_width.div_ceil(3);
        let merged = height
            .checked_mul(width)
            .ok_or("Vision merged grid overflow")?;
        if merged > self.features_capacity {
            return Err("Vision aligner capacity exceeded".into());
        }
        let key = (patches, merged);
        let start = Instant::now();
        let input = crate::vision::encode_pixels(image, &self.spec)?;
        session
            .upload(&self.buffers[&self.spec.pixels], 0, &input)?
            .wait()?;
        let grid_key = (image.grid_height, image.grid_width);
        if self.phase_grid != Some(grid_key) {
            let mut grid = [0i32; 8];
            grid[0] = i32::try_from(image.grid_height).map_err(|e| e.to_string())?;
            grid[1] = i32::try_from(image.grid_width).map_err(|e| e.to_string())?;
            let grid: Vec<_> = grid.into_iter().flat_map(i32::to_le_bytes).collect();
            session
                .upload(&self.buffers[&self.spec.grid], 0, &grid)?
                .wait()?;
            let mut phase = Vec::with_capacity(patches * 64 * 4);
            let frequencies: [f32; 16] =
                std::array::from_fn(|i| 1.0 / self.theta.powf(i as f32 / 16.0));
            for y in 0..image.grid_height {
                for x in 0..image.grid_width {
                    let mut angles = [0.0f32; 32];
                    for (axis, position) in [y, x].into_iter().enumerate() {
                        for i in 0..16 {
                            angles[axis * 16 + i] = position as f32 * frequencies[i];
                        }
                    }
                    for value in angles.map(f32::cos).into_iter().chain(angles.map(f32::sin)) {
                        phase.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
            session
                .upload(&self.buffers["vision.phase"], 0, &phase)?
                .wait()?;
            self.phase_grid = Some(grid_key);
        }
        let upload_s = start.elapsed().as_secs_f64();
        if !self.frames.contains_key(&key) {
            if self.frames.len() >= 16 {
                let oldest = self.order.pop_front().ok_or("Empty vision frame LRU")?;
                self.frames.remove(&oldest);
            }
            let kernels = self
                .operations
                .iter()
                .map(|op| {
                    Kernel::bind_rows(
                        session,
                        &op.template,
                        &op.buffers,
                        if op.features { merged } else { patches },
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            self.frames.insert(
                key,
                Frame {
                    kernels: Rc::new(kernels),
                    graphs: None,
                },
            );
        }
        self.order.retain(|previous| *previous != key);
        self.order.push_back(key);
        self.running = Some(Running {
            key,
            span,
            width,
            cursor: 0,
            upload_s,
            compute_s: 0.,
            max_advance_s: 0.,
            graph_replay: self.frames[&key].graphs.is_some(),
        });
        Ok(())
    }
    pub(super) fn advance(
        &mut self,
        session: &Session,
        segments: usize,
    ) -> Result<Option<Encoded>> {
        if segments == 0 {
            return Err("Vision execution needs a positive frame budget".into());
        }
        let Some(run) = self.running.as_mut() else {
            return Ok(None);
        };
        let frame = self.frames.get_mut(&run.key).unwrap();
        let next = run
            .cursor
            .saturating_add(segments)
            .min(self.segment_ends.len());
        let begin = if run.cursor == 0 {
            0
        } else {
            self.segment_ends[run.cursor - 1]
        };
        let end = self.segment_ends[next - 1];
        let compute_start = Instant::now();
        if let Some(graphs) = &frame.graphs {
            let graphs = Rc::clone(graphs);
            let first = run.cursor;
            // SAFETY: The completion owns every graph and stable operand.
            unsafe {
                session.submit_compute(vec![], Box::new(Rc::clone(&graphs)), |stream| {
                    for graph in &graphs[first..next] {
                        graph.execute_on_stream(stream)?;
                    }
                    Ok(())
                })
            }?
            .wait()?;
        } else {
            let kernels = Rc::clone(&frame.kernels);
            // SAFETY: This completion owns every kernel, stable operand and AOT library.
            unsafe {
                session.submit_compute(vec![], Box::new(Rc::clone(&kernels)), |stream| {
                    for kernel in &kernels[begin..end] {
                        kernel.launch_on_stream(stream)?;
                    }
                    Ok(())
                })
            }?
            .wait()?;
        }
        let seconds = compute_start.elapsed().as_secs_f64();
        run.compute_s += seconds;
        run.max_advance_s = run.max_advance_s.max(seconds);
        run.cursor = next;
        if next != self.segment_ends.len() {
            return Ok(None);
        }
        let run = self.running.take().unwrap();
        let (patches, merged) = run.key;
        let download_start = Instant::now();
        let output = session.download(
            &self.buffers[&self.spec.output],
            0,
            merged * self.spec.hidden * 2,
        )?;
        let mut data = Vec::with_capacity(run.span * self.spec.hidden * 2);
        data.extend_from_slice(&self.markers[0]);
        for row in output.chunks_exact(run.width * self.spec.hidden * 2) {
            data.extend_from_slice(row);
            data.extend_from_slice(&self.markers[2]);
        }
        data.extend_from_slice(&self.markers[1]);
        if data.len() != run.span * self.spec.hidden * 2 {
            return Err("Vision output span differs".into());
        }
        let download_s = download_start.elapsed().as_secs_f64();
        // Capture only after the first real execution; capture itself is not an inference.
        if session.graphs_enabled() && frame.graphs.is_none() {
            let mut graphs = Vec::with_capacity(self.segment_ends.len());
            let mut begin = 0;
            for &end in &self.segment_ends {
                let kernels = Rc::clone(&frame.kernels);
                // SAFETY: Capture follows complete readback. Only the next
                // image's first segment can reuse these stable intermediate
                // buffers; each graph retains its kernels and AOT libraries.
                graphs.push(unsafe {
                    Graph::capture(session, Box::new(Rc::clone(&kernels)), |stream| {
                        for kernel in &kernels[begin..end] {
                            kernel.launch_on_stream(stream)?;
                        }
                        Ok(())
                    })
                }?);
                begin = end;
            }
            frame.graphs = Some(Rc::new(graphs));
        }
        Ok(Some(Encoded {
            data,
            span: run.span,
            patches,
            upload_s: run.upload_s,
            compute_s: run.compute_s,
            download_s,
            graph_replay: run.graph_replay,
            max_advance_s: run.max_advance_s,
        }))
    }
}

/// Full-checkpoint acceptance and timing, using the production encoder implementation.
pub fn validate(
    device: u32,
    program: &Path,
    model_dir: &Path,
    fixtures: &Path,
) -> Result<serde_json::Value> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Inputs {
        cases: Vec<Input>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Input {
        name: String,
        image: ImageInput,
        expected: String,
    }
    let fixtures = fixtures.canonicalize().map_err(|e| e.to_string())?;
    let inputs: Inputs = crate::model::read(&fixtures)?;
    if inputs.cases.is_empty() {
        return Err("Empty vision acceptance fixtures".into());
    }
    let base = fixtures
        .parent()
        .ok_or("Vision fixtures have no directory")?;
    let session = Session::new(
        device,
        super::Limits {
            device_bytes: 3usize << 30,
            pinned_bytes: 64 << 20,
            timeout_ms: 60_000,
        },
    )?;
    let mut checkpoint = Checkpoint::open(model_dir, "model.safetensors.index.json")?;
    let load_start = Instant::now();
    let mut encoder = Encoder::load(&session, program, &mut checkpoint)?;
    let load_s = load_start.elapsed().as_secs_f64();
    let mut results = Vec::new();
    for input in inputs.cases {
        let expected =
            std::fs::read(resolve_file(base, &input.expected)?).map_err(|e| e.to_string())?;
        // A cancelled partial image must not contribute state to the next one.
        encoder.begin(&session, &input.image)?;
        if encoder.segment_ends.len() > 1 && encoder.advance(&session, 1)?.is_some() {
            return Err("Partial vision frame completed before its boundary".into());
        }
        encoder.cancel();
        let mut previous = None;
        for repeat in 0..2 {
            let encoded = if repeat == 0 {
                encoder.encode(&session, &input.image)?
            } else {
                encoder.begin(&session, &input.image)?;
                loop {
                    if let Some(encoded) = encoder.advance(&session, 1)? {
                        break encoded;
                    }
                }
            };
            if previous.as_ref().is_some_and(|data| *data != encoded.data) {
                return Err("Vision frame replay changed the complete image features".into());
            }
            previous = Some(encoded.data.clone());
            if expected.len() != encoded.data.len() {
                return Err("Vision reference extent differs".into());
            }
            let mut square_error = 0.0f64;
            let mut square_reference = 0.0f64;
            let mut dot = 0.0f64;
            let mut square_actual = 0.0f64;
            let mut maximum = 0.0f64;
            for (actual, reference) in encoded
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .zip(expected.as_chunks::<2>().0)
            {
                let a = bf16::from_le_bytes(*actual).to_f64();
                let r = bf16::from_le_bytes(*reference).to_f64();
                if !a.is_finite() || !r.is_finite() {
                    return Err("Nonfinite vision output".into());
                }
                square_error += (a - r).powi(2);
                square_reference += r * r;
                dot += a * r;
                square_actual += a * a;
                maximum = maximum.max((a - r).abs());
            }
            let relative_rms = (square_error / square_reference.max(f64::MIN_POSITIVE)).sqrt();
            let cosine = dot
                / (square_actual * square_reference)
                    .sqrt()
                    .max(f64::MIN_POSITIVE);
            results.push(serde_json::json!({"name":input.name,"repeat":repeat,"span":encoded.span,"patches":encoded.patches,"max_abs":maximum,"relative_rms":relative_rms,"cosine":cosine,"upload_s":encoded.upload_s,"compute_s":encoded.compute_s,"download_s":encoded.download_s,"graph_replay":encoded.graph_replay,"max_advance_s":encoded.max_advance_s,"segments":encoder.segment_ends.len()}));
        }
    }
    session.synchronize()?;
    Ok(
        serde_json::json!({"load_s":load_s,"cases":results,"scope":"native full-checkpoint vision encoder and aligner; not language-model image quality"}),
    )
}

/// Learned delimiters and aligned image rows at absolute prompt positions.
#[derive(Clone)]
pub(super) struct Span {
    pub start: usize,
    pub data: std::sync::Arc<Vec<u8>>,
}
impl Span {
    pub fn end(&self) -> usize {
        self.start + self.data.len() / (5120 * 2)
    }
}
pub(super) struct Overlay {
    pub row: usize,
    pub source_row: usize,
    pub rows: usize,
    pub data: std::sync::Arc<Vec<u8>>,
}
impl Overlay {
    pub fn apply(&self, first: usize, columns: usize, output: &mut [u8]) -> Result<()> {
        if first.checked_add(columns).is_none_or(|n| n > 5120) {
            return Err("Vision embedding column range differs".into());
        }
        let width = columns * 2;
        for i in 0..self.rows {
            let source = (self.source_row + i) * 5120 * 2 + first * 2;
            let destination = (self.row + i) * width;
            output
                .get_mut(destination..destination + width)
                .ok_or("Vision destination row outside packet")?
                .copy_from_slice(
                    self.data
                        .get(source..source + width)
                        .ok_or("Vision source row outside features")?,
                );
        }
        Ok(())
    }
}

pub(super) fn write_mask(
    output: &mut [u8],
    row: usize,
    spans: &[Span],
    position: usize,
    rows: usize,
) -> Result<()> {
    let end = row
        .checked_add(rows)
        .and_then(|n| n.checked_mul(4))
        .ok_or("Image mask extent overflow")?;
    let output = output
        .get_mut(row * 4..end)
        .ok_or("Image mask exceeds packet rows")?;
    output.fill(0);
    let limit = position
        .checked_add(rows)
        .ok_or("Image mask position overflow")?;
    for span in spans {
        let start = position.max(span.start);
        let end = limit.min(span.end());
        if start < end {
            for word in output[(start - position) * 4..(end - position) * 4]
                .as_chunks_mut::<4>()
                .0
            {
                *word = 1i32.to_le_bytes();
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_and_retained_rows_share_owner_positions_and_embedding_columns() {
        let mask = |spans: &[Span], position, rows| {
            let mut output = vec![0u8; rows * 4];
            write_mask(&mut output, 0, spans, position, rows).unwrap();
            output
        };
        let values: Vec<_> = (0..4 * 5120)
            .flat_map(|i| bf16::from_f32((i % 5120) as f32).to_le_bytes())
            .collect();
        let span = Span {
            start: 3,
            data: std::sync::Arc::new(values),
        };
        let decode_mask = |bytes: Vec<u8>| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| i32::from_le_bytes(*v))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            decode_mask(mask(std::slice::from_ref(&span), 2, 6)),
            [0, 1, 1, 1, 1, 0]
        );
        assert_eq!(
            decode_mask(mask(std::slice::from_ref(&span), 5, 3)),
            [1, 1, 0]
        );
        assert_eq!(decode_mask(mask(&[], 3, 2)), [0, 0]);
        let mut packet = vec![0u8; 12 * 4];
        write_mask(&mut packet, 2, std::slice::from_ref(&span), 2, 6).unwrap();
        write_mask(&mut packet, 8, std::slice::from_ref(&span), 4, 2).unwrap();
        assert_eq!(
            decode_mask(packet.clone()),
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
        );
        write_mask(&mut packet, 2, &[], 2, 6).unwrap();
        assert_eq!(
            decode_mask(packet.clone()),
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
        );
        assert!(write_mask(&mut packet, 11, &[], 2, 2).is_err());
        let overlay = Overlay {
            row: 1,
            source_row: 2,
            rows: 2,
            data: span.data,
        };
        let mut output = vec![19; 4 * 640 * 2];
        overlay.apply(4480, 640, &mut output).unwrap();
        assert!(output[..1280].iter().all(|&v| v == 19));
        assert!(output[3840..].iter().all(|&v| v == 19));
        assert_eq!(&output[1280..2560], &output[2560..3840]);
        assert_eq!(
            bf16::from_le_bytes(output[1280..1282].try_into().unwrap()),
            bf16::from_f32(4480.)
        );
        assert!(overlay.apply(5000, 640, &mut output).is_err());
    }
}
