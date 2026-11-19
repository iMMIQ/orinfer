use super::*;

impl ModelRuntime {
    pub(super) fn benchmark(
        mut self,
        requests: crate::model::Requests,
    ) -> Result<crate::model::Report> {
        use crate::model::{Report, RequestReport};
        use std::fs;
        self.manifest.validate_requests(&requests)?;
        self.prepare_visual(&[], &[], &|| false)?;
        let manifest = &self.manifest;
        let execution = &self.execution;
        let stats = self.stats;
        let output_directory = requests
            .logits_output
            .as_ref()
            .map(std::path::PathBuf::from);
        if let Some(p) = &output_directory {
            fs::create_dir(p).map_err(|e| format!("New logits directory {}: {e}", p.display()))?;
        }
        let mut reports = vec![];
        let logits_spec = manifest
            .buffers
            .iter()
            .find(|b| b.name == manifest.logits)
            .ok_or("Logits missing")?;
        let dump_logits = |_session: &Executor,
                           q: &crate::model::Request,
                           step: usize,
                           files: &mut Vec<String>|
         -> Result<()> {
            if !q.logits_steps.contains(&step) {
                return Ok(());
            }
            let raw = execution.read_ready_bytes(&manifest.logits, logits_spec.bytes()?)?;
            let values = floats(&raw, logits_spec.dtype);
            if values.iter().any(|v| !v.is_finite()) {
                return Err("Nonfinite logits".into());
            }
            let path = output_directory
                .as_ref()
                .ok_or("Logits directory absent")?
                .join(format!("{}-{step}.f32", q.id));
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .map_err(|e| e.to_string())?;
            use std::io::Write;
            let encoded: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            file.write_all(&encoded).map_err(|e| e.to_string())?;
            files.push(path.display().to_string());
            Ok(())
        };
        let read_i32 = |execution: &Executor, name: &str| execution.read_ready_control(name);
        for q in &requests.requests {
            let (chunk_tokens, prefill_program, _) =
                manifest.select_prefill_plan(q.input_tokens.len())?;
            eprintln!(
                "request {}: {} input, {} output tokens",
                q.id,
                q.input_tokens.len(),
                q.max_new_tokens
            );
            let reset_start = Instant::now();
            self.execution.reset_sequence(&manifest.reset_buffers)?;
            if let Some(controls) = &manifest.segment_controls {
                self.execution
                    .upload_ids(&controls.length, &[chunk_tokens as u32])?;
                self.execution
                    .upload_ids(&controls.last_index, &[(chunk_tokens - 1) as u32])?;
            }
            let reset_s = reset_start.elapsed().as_secs_f64();
            let request_start = Instant::now();
            let mut tokens = vec![];
            let mut logits_files = vec![];
            for (chunk_index, chunk) in q.input_tokens.chunks(chunk_tokens).enumerate() {
                let (_, program, _) = manifest.select_prefill_plan(chunk.len())?;
                if let Some(controls) = &manifest.segment_controls {
                    self.execution
                        .upload_ids(&controls.length, &[chunk.len() as u32])?;
                    self.execution
                        .upload_ids(&controls.last_index, &[(chunk.len() - 1) as u32])?;
                }
                self.execution.sync()?;
                if q.input_tokens.len() > 8192 && chunk_index > 0 && chunk_index.is_multiple_of(8) {
                    eprintln!(
                        "request {}: prefill {}/{} tokens complete after {:.3}s",
                        q.id,
                        chunk_index * chunk_tokens,
                        q.input_tokens.len(),
                        request_start.elapsed().as_secs_f64()
                    );
                }
                let input = if manifest.prefill_plan(chunk.len()).is_some() {
                    &manifest.input
                } else {
                    &manifest.token
                };
                self.execution.upload_ids(input, chunk)?;
                self.execution.sync()?;
                self.batch_plans.prepare_inputs(
                    if manifest.row_execution.is_some() {
                        program
                    } else {
                        "target"
                    },
                    chunk,
                    &q.input_tokens[..chunk_index * chunk_tokens],
                    |name, data| self.execution.upload_bytes(name, data),
                )?;
                self.execution.submit_program_rows(
                    program,
                    chunk.len(),
                    ExecutionPhase::Prefill,
                )?;
            }
            self.execution.sync()?;
            let prefill_s = request_start.elapsed().as_secs_f64();
            let head_start = Instant::now();
            let last_rows = (q.input_tokens.len() - 1) % chunk_tokens + 1;
            let (_, _, head_program) = manifest.select_prefill_plan(last_rows)?;
            self.execution
                .submit_program_rows(head_program, last_rows, ExecutionPhase::Prefill)?;
            self.execution.sync()?;
            if read_i32(execution, &manifest.status)? != 0 {
                return Err("Token selection status failure".into());
            }
            let token = read_i32(execution, &manifest.token)?;
            if token < 0 || token as usize >= manifest.vocab {
                return Err("Invalid selected token".into());
            }
            tokens.push(token as u32);
            let head_s = head_start.elapsed().as_secs_f64();
            let ttft_s = request_start.elapsed().as_secs_f64();
            dump_logits(execution, q, 0, &mut logits_files)?;
            let decode_start = Instant::now();
            let mut history = q.input_tokens.clone();
            for step in 1..q.max_new_tokens {
                if let Some(id) = q.forced_tokens.get(step - 1) {
                    self.execution.upload_ids(&manifest.token, &[*id])?;
                    self.execution.sync()?;
                }
                let input = q
                    .forced_tokens
                    .get(step - 1)
                    .copied()
                    .unwrap_or(*tokens.last().ok_or("Missing pending token")?);
                self.batch_plans
                    .prepare_inputs("target", &[input], &history, |name, data| {
                        self.execution.upload_bytes(name, data)
                    })?;
                if let Some(controls) = &manifest.segment_controls {
                    self.execution.upload_ids(&controls.length, &[1])?;
                    self.execution.upload_ids(&controls.last_index, &[0])?;
                }
                self.execution
                    .submit_program("decode", ExecutionPhase::Decode)?;
                history.push(input);
                self.execution.sync()?;
                if read_i32(execution, &manifest.status)? != 0 {
                    return Err("Decode token status failure".into());
                }
                let token = read_i32(execution, &manifest.token)?;
                if token < 0 || token as usize >= manifest.vocab {
                    return Err("Invalid decode token".into());
                }
                tokens.push(token as u32);
                dump_logits(execution, q, step, &mut logits_files)?;
            }
            let decode_s = decode_start.elapsed().as_secs_f64();
            let final_position = read_i32(execution, &manifest.position)?;
            if final_position < 0
                || final_position as usize != q.input_tokens.len() + q.max_new_tokens - 1
            {
                return Err("Model position advancement mismatch".into());
            }
            let diagnostic = !q.logits_steps.is_empty() || !q.forced_tokens.is_empty();
            let r = RequestReport {
                prefill_chunk_tokens: chunk_tokens,
                prefill_program: prefill_program.to_string(),
                id: q.id.clone(),
                input_tokens: q.input_tokens.len(),
                output_tokens: tokens,
                reset_s,
                prefill_s,
                head_s,
                ttft_s,
                decode_s,
                prefill_tps: q.input_tokens.len() as f64 / prefill_s,
                decode_tps: (q.max_new_tokens > 1)
                    .then_some((q.max_new_tokens - 1) as f64 / decode_s),
                final_position: final_position as usize,
                logits_files,
                diagnostic,
            };
            eprintln!(
                "{} prefill {:.2} TPS, decode {:?} TPS",
                r.id, r.prefill_tps, r.decode_tps
            );
            reports.push(r);
            if let Ok(path) = std::env::var("ORINFER_MODEL_PROGRESS") {
                let temporary = format!("{path}.tmp");
                fs::write(
                    &temporary,
                    serde_json::to_vec_pretty(&reports).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                fs::rename(&temporary, &path).map_err(|e| e.to_string())?;
            }
        }
        self.execution.cleanup()?;
        let weight_bytes = manifest.weight_bytes + stats.load_stages.weight_arena_padding_bytes;
        Ok(Report {
            resident_experts: stats.resident_experts,
            expert_cache: self.execution.expert_statistics(),
            manifest_sha256: stats.manifest_sha256,
            model: manifest.model.clone(),
            device: stats.device,
            load_to_ready_s: stats.load_to_ready_s,
            load_stages: stats.load_stages,
            verify_weights: stats.verify_weights,
            load_workers: stats.load_workers,
            preloaded_modules: stats.preloaded_modules,
            registered_modules: stats.registered_modules,
            weight_io_hash_s: stats.weight_io_hash_s,
            weight_upload_s: stats.weight_upload_s,
            module_load_bind_s: stats.module_load_bind_s,
            graph_capture_s: stats.graph_capture_s,
            cuda_graph: stats.cuda_graph,
            captured_programs: stats.captured_programs,
            buffer_bytes: stats.buffer_bytes,
            buffer_capacity_bytes: stats.buffer_capacity_bytes,
            peak_kv_bytes: self.execution.peak_kv_bytes(),
            peak_prefill_workspace_bytes: self.execution.peak_prefill_workspace_bytes(),
            weight_bytes,
            effective_weight_bits: 8.0 * weight_bytes as f64 / manifest.weight_parameters as f64,
            weight_scope: manifest.weight_scope.clone(),
            requests: reports,
            seed: 20261002,
            timing_scope: "Rust wall clock; prefill includes input copies and synchronization; decode excludes first token, includes token D2H/synchronization; diagnostic requests include logits I/O; no MTP, fixed output length",
        })
    }
}
