//! Resume a large forward pass at model-registered layer boundaries. Tokens
//! become committed only after its target body and shifted draft warm complete.
use super::*;
use crate::backend::snapshot::{Range, Workspace};
use std::{cell::RefCell, rc::Rc};

pub(super) struct Frame {
    pub plan: crate::model::PrefillPlan,
    pub tokens: usize,
    next: usize,
    workspace: Option<Rc<RefCell<Workspace>>>,
    regular_s: f64,
    key: cost::Key,
    active: cost::Measurement,
}

impl ModelRuntime {
    pub(super) fn release_prefill_frame(&mut self, req: &mut RequestState) -> Result<()> {
        drop(req.prefill_frame.take());
        self.execution.collect_snapshots()
    }
    pub(super) fn prefill_stage_cost(&self, req: &RequestState, chunk: usize) -> f64 {
        let plan = req
            .prefill_frame
            .as_ref()
            .map(|f| &f.plan)
            .or_else(|| self.manifest.prefill_plan(chunk));
        if let Some(plan) = plan.filter(|p| !p.stages.is_empty()) {
            if let Some(frame) = &req.prefill_frame {
                return self.prefill_costs.stage_upper(
                    chunk,
                    frame.next,
                    req.offset + chunk,
                    plan.stages.len(),
                );
            }
            // A new tile must fit its slowest measured layer, not just its
            // first layer. An in-flight tile keeps its original extent.
            (0..plan.stages.len())
                .map(|stage| {
                    self.prefill_costs.stage_upper(
                        chunk,
                        stage,
                        req.offset + chunk,
                        plan.stages.len(),
                    )
                })
                .fold(0., f64::max)
        } else {
            self.prefill_costs.chunk_upper(chunk, req.offset + chunk)
        }
    }

    #[cfg(test)]
    pub(super) fn advance_prefill_step(
        &mut self,
        req: &mut RequestState,
        chunk: usize,
        bounded: bool,
        budget_s: f64,
    ) -> Result<(usize, Vec<u32>)> {
        let timer = cost::Timer::start(self.execution.batch_statistics());
        self.advance_prefill_step_timed(req, chunk, bounded, budget_s, timer)
    }
    pub(super) fn advance_prefill_step_timed(
        &mut self,
        req: &mut RequestState,
        chunk: usize,
        bounded: bool,
        budget_s: f64,
        timer: cost::Timer,
    ) -> Result<(usize, Vec<u32>)> {
        let key = Self::iteration_key(&[&mut *req], &[(0, chunk)]);
        let plan = req
            .prefill_frame
            .as_ref()
            .map(|f| f.plan.clone())
            .or_else(|| self.manifest.prefill_plan(chunk).cloned());
        let Some(plan) = plan.filter(|p| !p.stages.is_empty()) else {
            let result = self.advance_prefill(req, chunk, bounded)?;
            self.observe_iteration(key, timer.finish(self.execution.batch_statistics()));
            return Ok(result);
        };
        if !plan.accepts(chunk)
            || req
                .prefill_frame
                .as_ref()
                .is_some_and(|f| f.tokens != chunk)
            || chunk > Self::prefill_boundary(req) - req.offset
        {
            return Err("Invalid resumed prefill extent".into());
        }
        self.prepare_pending_audio(req)?;
        let at = Instant::now();
        let mut frame = if let Some(frame) = req.prefill_frame.take() {
            frame
        } else {
            self.upload_ids(
                &self.manifest.input,
                &req.input[req.offset..req.offset + chunk],
            )?;
            self.prepare_prefill_inputs(
                &plan.prefill_program,
                &req.input[req.offset..req.offset + chunk],
                &req.input[..req.offset],
            )?;
            Frame {
                plan,
                tokens: chunk,
                next: 0,
                workspace: None,
                regular_s: at.elapsed().as_secs_f64(),
                key,
                active: cost::Measurement::default(),
            }
        };
        if let Some(workspace) = &frame.workspace {
            frame.regular_s += self.execution.resume_workspace(workspace)?;
        }
        // OOM while saving a continuation falls back to completing the same
        // forward pass, never to recomputing already-mutated recurrent layers.
        loop {
            let began = Instant::now();
            let before = self.execution.batch_statistics();
            let stage = frame.next;
            self.scheduler_statistics.layered_prefill_iterations += 1;
            if stage < frame.plan.stages.len() {
                self.upload_segment_controls(chunk)?;
                self.launch_rows(&frame.plan.stages[stage], chunk, ExecutionPhase::Prefill)?;
                frame.next += 1;
                frame.regular_s += began.elapsed().as_secs_f64()
                    - (self.execution.batch_statistics().initialization_s()
                        - before.initialization_s());
                if frame.next == frame.plan.stages.len() {
                    if let Some(spec) = &self.manifest.mtp {
                        self.mtp_capture(spec, chunk, ExecutionPhase::Prefill)?;
                    }
                    if req.offset + chunk == req.input.len()
                        || req.checkpoints.contains(&(req.offset + chunk))
                    {
                        self.launch_rows(&frame.plan.head_program, chunk, ExecutionPhase::Prefill)?;
                    }
                }
            } else if let Some(spec) = self.manifest.mtp.clone() {
                let end = req.offset + chunk;
                let warm_end =
                    end + usize::from(end < req.input.len() && !req.checkpoints.contains(&end));
                let remaining = warm_end.saturating_sub(req.warm.tokens + 1);
                if remaining > 0 {
                    let (_, tokens) = spec.warm_plan(remaining)?;
                    let offset = req.warm.tokens + 1;
                    self.mtp_warm_state(
                        &spec,
                        &req.input[offset..offset + tokens],
                        &|| false,
                        ExecutionPhase::Prefill,
                        false,
                    )?;
                    req.warm.tokens += tokens;
                    req.warm.seconds += began.elapsed().as_secs_f64();
                }
            }
            if bounded {
                self.scheduler_statistics.bounded_prefill_iterations += 1;
            }
            let after = self.execution.batch_statistics();
            let cold = after.initialization_s() - before.initialization_s()
                + after.sequence_capture_s
                - before.sequence_capture_s;
            self.prefill_costs.observe_stage(
                chunk,
                stage,
                req.offset + chunk,
                (began.elapsed().as_secs_f64() - cold).max(0.000001),
            );
            if frame.next < frame.plan.stages.len() {
                // Short layers share one latency window. Returning after each
                // one would buy a whole decode burst for only a few ms of
                // prefill progress. The token tile and layer order stay fixed.
                if at.elapsed().as_secs_f64()
                    + self.prefill_costs.stage_upper(
                        chunk,
                        frame.next,
                        req.offset + chunk,
                        frame.plan.stages.len(),
                    )
                    <= budget_s
                {
                    continue;
                }
                if frame.workspace.is_none() {
                    let reserve = Instant::now();
                    let ranges = frame
                        .plan
                        .live_buffers
                        .iter()
                        .map(|name| {
                            (
                                name.clone(),
                                Range {
                                    offset: 0,
                                    bytes: self.execution.buffer_size(name),
                                },
                            )
                        })
                        .collect();
                    frame.workspace = self.execution.reserve_workspace(ranges)?;
                    frame.regular_s += reserve.elapsed().as_secs_f64();
                }
                if let Some(workspace) = &frame.workspace {
                    self.execution.retain_workspace(workspace)?;
                    self.scheduler_statistics.peak_prefill_frame_bytes = self
                        .scheduler_statistics
                        .peak_prefill_frame_bytes
                        .max(workspace.borrow().bytes());
                    frame
                        .active
                        .add(timer.finish(self.execution.batch_statistics()));
                    req.prefill_frame = Some(frame);
                    self.observe_prefill_window(at, bounded);
                    return Ok((0, vec![]));
                }
                continue;
            }
            // The target body no longer needs the shared continuation buffers.
            // MTP warm and endpoint commit operate on request-private state.
            if frame.workspace.take().is_some() {
                self.execution.collect_snapshots()?;
            }
            let end = req.offset + chunk;
            let warm_end =
                end + usize::from(end < req.input.len() && !req.checkpoints.contains(&end));
            if self.manifest.mtp.is_some() && req.warm.tokens + 1 < warm_end {
                if at.elapsed().as_secs_f64()
                    + self.prefill_costs.stage_upper(
                        chunk,
                        frame.next,
                        end,
                        frame.plan.stages.len(),
                    )
                    <= budget_s
                {
                    continue;
                }
                frame
                    .active
                    .add(timer.finish(self.execution.batch_statistics()));
                req.prefill_frame = Some(frame);
                self.observe_prefill_window(at, bounded);
                return Ok((0, vec![]));
            }
            self.prefill_costs
                .observe_at(chunk, end, frame.regular_s.max(0.000001));
            let tokens = self.after_prefill(req, chunk)?;
            frame
                .active
                .add(timer.finish(self.execution.batch_statistics()));
            self.observe_iteration(frame.key, frame.active);
            self.observe_prefill_window(at, bounded);
            self.scheduler_statistics.prefill_tokens += chunk;
            *self
                .scheduler_statistics
                .prefill_chunk_histogram
                .entry(chunk)
                .or_default() += 1;
            return Ok((chunk, tokens));
        }
    }
    fn observe_prefill_window(&mut self, began: Instant, bounded: bool) {
        let seconds = began.elapsed().as_secs_f64();
        if bounded {
            self.scheduler_statistics.max_bounded_prefill_s =
                self.scheduler_statistics.max_bounded_prefill_s.max(seconds);
        }
    }
}
