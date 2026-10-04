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
        let s = &self.execution.session;
        let pointers = &self.execution.pointers;
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
        let dump_logits = |session: &Session,
                           q: &crate::model::Request,
                           step: usize,
                           files: &mut Vec<String>|
         -> Result<()> {
            if !q.logits_steps.contains(&step) {
                return Ok(());
            }
            let mut raw = vec![0u8; logits_spec.bytes()?];
            // SAFETY: Caller synchronized this stream, host range and device
            // allocation cover the complete logits tensor.
            unsafe {
                check(
                    (session.driver.download)(
                        raw.as_mut_ptr().cast(),
                        pointers[&manifest.logits],
                        raw.len(),
                    ),
                    "download logits",
                )?;
            }
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
        let read_i32 = |session: &Session, name: &str| -> Result<i32> {
            let mut value = 0i32;
            // SAFETY: Control allocations and stack destination are >=4 bytes.
            // Every caller has synchronized the producing stream.
            unsafe {
                check(
                    (session.driver.download)((&mut value as *mut i32).cast(), pointers[name], 4),
                    "download control",
                )?;
            }
            Ok(value)
        };
        for q in &requests.requests {
            let (chunk_tokens, prefill_program, head_program) =
                manifest.select_prefill_plan(q.input_tokens.len())?;
            eprintln!(
                "request {}: {} input, {} output tokens",
                q.id,
                q.input_tokens.len(),
                q.max_new_tokens
            );
            let reset_start = Instant::now();
            self.execution.reset_sequence(&manifest.reset_buffers)?;
            let reset_s = reset_start.elapsed().as_secs_f64();
            let request_start = Instant::now();
            let mut tokens = vec![];
            let mut logits_files = vec![];
            for chunk in q.input_tokens.chunks(chunk_tokens) {
                // SAFETY: Input IDs fit this allocation and were range checked.
                // Previous graph is complete before overwriting its input buffer.
                unsafe {
                    check((s.driver.stream_sync)(s.stream), "prefill input dependency")?;
                    check(
                        (s.driver.upload)(
                            pointers[&manifest.input],
                            chunk.as_ptr().cast(),
                            chunk.len() * 4,
                        ),
                        "prefill token upload",
                    )?;
                    check((s.driver.context_sync)(), "prefill upload dependency")?;
                }
                self.execution
                    .submit_program(prefill_program, ExecutionPhase::Prefill)?;
            }
            // SAFETY: Synchronize before head timing and output download.
            unsafe {
                check((s.driver.stream_sync)(s.stream), "prefill complete")?;
            }
            let prefill_s = request_start.elapsed().as_secs_f64();
            let head_start = Instant::now();
            self.execution
                .submit_program(head_program, ExecutionPhase::Prefill)?;
            // SAFETY: Head consumes last prefill graph's buffers on the same stream.
            unsafe {
                check((s.driver.stream_sync)(s.stream), "head complete")?;
            }
            if read_i32(s, &manifest.status)? != 0 {
                return Err("Token selection status failure".into());
            }
            let token = read_i32(s, &manifest.token)?;
            if token < 0 || token as usize >= manifest.vocab {
                return Err("Invalid selected token".into());
            }
            tokens.push(token as u32);
            let head_s = head_start.elapsed().as_secs_f64();
            let ttft_s = request_start.elapsed().as_secs_f64();
            dump_logits(s, q, 0, &mut logits_files)?;
            let decode_start = Instant::now();
            for step in 1..q.max_new_tokens {
                if let Some(id) = q.forced_tokens.get(step - 1) {
                    // SAFETY: Previous graph completed, valid token ID and 4-byte
                    // destination; explicit synchronization separates upload/replay.
                    unsafe {
                        check(
                            (s.driver.upload)(
                                pointers[&manifest.token],
                                (id as *const u32).cast(),
                                4,
                            ),
                            "teacher-force token",
                        )?;
                        check((s.driver.context_sync)(), "teacher-force dependency")?;
                    }
                }
                self.execution
                    .submit_program("decode", ExecutionPhase::Decode)?;
                // SAFETY: This session owns the live stream and all queued work.
                unsafe {
                    check((s.driver.stream_sync)(s.stream), "decode complete")?;
                }
                if read_i32(s, &manifest.status)? != 0 {
                    return Err("Decode token status failure".into());
                }
                let token = read_i32(s, &manifest.token)?;
                if token < 0 || token as usize >= manifest.vocab {
                    return Err("Invalid decode token".into());
                }
                tokens.push(token as u32);
                dump_logits(s, q, step, &mut logits_files)?;
            }
            let decode_s = decode_start.elapsed().as_secs_f64();
            let final_position = read_i32(s, &manifest.position)?;
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
            if let Ok(path) = std::env::var("ORIN_MODEL_PROGRESS") {
                let temporary = format!("{path}.tmp");
                fs::write(
                    &temporary,
                    serde_json::to_vec_pretty(&reports).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                fs::rename(&temporary, &path).map_err(|e| e.to_string())?;
            }
        }
        self.execution.session.cleanup()?;
        Ok(Report {
            manifest_sha256: stats.manifest_sha256,
            model: manifest.model.clone(),
            device: stats.device,
            load_to_ready_s: stats.load_to_ready_s,
            weight_io_hash_s: stats.weight_io_hash_s,
            weight_upload_s: stats.weight_upload_s,
            module_load_bind_s: stats.module_load_bind_s,
            graph_capture_s: stats.graph_capture_s,
            cuda_graph: stats.cuda_graph,
            captured_programs: stats.captured_programs,
            buffer_bytes: stats.buffer_bytes,
            buffer_capacity_bytes: stats.buffer_capacity_bytes,
            peak_kv_bytes: self.execution.peak_kv_bytes.get(),
            weight_bytes: manifest.weight_bytes,
            effective_weight_bits: 8.0 * manifest.weight_bytes as f64
                / manifest.weight_parameters as f64,
            weight_scope: manifest.weight_scope.clone(),
            requests: reports,
            seed: 20261002,
            timing_scope: "Rust wall clock; prefill includes input copies and synchronization; decode excludes first token, includes token D2H/synchronization; diagnostic requests include logits I/O; no MTP, fixed output length",
        })
    }
}
