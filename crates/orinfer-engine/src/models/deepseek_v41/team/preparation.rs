//! One all-rank decision after workspace growth, shared by every packet role.
use super::*;

pub(in crate::models::deepseek_v41) fn capture_credit(
    previous: f64,
    target: f64,
    elapsed: f64,
    committed: usize,
) -> f64 {
    (previous + target * committed as f64 - elapsed).clamp(0., target)
}

pub(in crate::models::deepseek_v41) fn capture_interval(target: f64, progress: f64) -> f64 {
    // Account for a speculative burst without allowing its entire depth to
    // become an unbounded pause for another owner.
    target * progress.clamp(1., 2.)
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(in crate::models::deepseek_v41) struct CaptureBudget {
    pub uses: f64,
    /// A temporary calibration path cannot spend forecast cross-cohort reuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<f64>,
    pub direct_s: Option<f64>,
    pub replay_s: Option<f64>,
    pub capture_s: Option<f64>,
    pub slack_s: f64,
    pub trace: bool,
}
impl CaptureBudget {
    pub fn profitable(self, warmup_uses: usize) -> bool {
        let uses = self
            .max_uses
            .map_or(self.uses, |limit| self.uses.min(limit));
        if self.capture_s.is_some_and(|capture| {
            capture + self.replay_s.or(self.direct_s).unwrap_or(0.) > self.slack_s
        }) {
            return false;
        }
        match (self.direct_s, self.replay_s, self.capture_s) {
            (Some(direct), Some(replay), Some(capture)) => {
                let useful = (uses - warmup_uses as f64).max(0.);
                useful * (direct - replay).max(0.) > capture * 1.1
                    && capture + replay <= self.slack_s
            }
            (Some(direct), None, Some(capture)) => {
                // The first replay is a bounded measurement, not a predicted
                // speedup. Limit it to a quarter of the remaining direct work:
                // a fixed reuse count can abandon a costly fused packet before
                // it ever measures the execution mode it was tuned for.
                let useful = (uses - warmup_uses as f64).max(0.);
                (uses >= 16. || useful * direct > capture * 4.) && capture + direct <= self.slack_s
            }
            // Seed one missing family on a long, repeated packet. Observations
            // then price new row counts from the same shared sparse surface.
            _ => uses >= 16. && self.slack_s > self.direct_s.unwrap_or(0.),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    BuildDirect,
    Direct,
    CaptureReplay,
    Replay,
}
pub(super) struct Plan {
    pub invalidate: bool,
    pub mode: Mode,
    pub reason: &'static str,
}
impl Plan {
    pub fn capture(&self) -> bool {
        matches!(self.mode, Mode::CaptureReplay | Mode::Replay)
    }
}

#[cfg(test)]
pub(super) fn plan(
    states: &[CacheState],
    repeated: bool,
    budget: Option<CaptureBudget>,
) -> Result<Plan> {
    plan_with_warmup(states, repeated, budget, if repeated { 2 } else { 8 })
}

pub(super) fn plan_with_warmup(
    states: &[CacheState],
    repeated: bool,
    budget: Option<CaptureBudget>,
    min_warmup: u8,
) -> Result<Plan> {
    if states.is_empty() {
        return Err("Missing A2 TP cache states".into());
    }
    let missing = states.contains(&CacheState::Missing);
    let all_missing = states.iter().all(|s| *s == CacheState::Missing);
    let graphs = states.iter().filter(|s| **s == CacheState::Graph).count();
    if graphs == states.len() {
        return Ok(Plan {
            invalidate: false,
            mode: Mode::Replay,
            reason: "resident_graph",
        });
    }
    let invalidate = (!all_missing && missing) || graphs > 0;
    let ready = !missing
        && (graphs > 0
            || states
                .iter()
                .all(|s| matches!(s, CacheState::Direct { uses, .. } if *uses >= min_warmup)));
    let budget = budget.map(|mut budget| {
        // Short tails and final prompt windows have little request-local reuse.
        // Actual hits of the same retained packet also establish reuse across
        // cohorts; bound that forecast so cold or changing shapes stay direct.
        let observed = states
            .iter()
            .map(|s| match s {
                CacheState::Direct { uses, .. } => *uses,
                _ => 0,
            })
            .min()
            .unwrap_or(0);
        // Consecutive hits are stronger evidence than scattered tail reuse.
        // A verification family can finish calibration before eight hits;
        // use four consecutive executions to price its first replay trial.
        let stable = repeated
            && states.iter().all(|state| {
                matches!(state, CacheState::Direct { consecutive, .. } if *consecutive >= 4)
            });
        if observed >= 8 || stable {
            budget.uses = budget.uses.max(f64::from(observed.min(16)) * 2.);
        }
        budget
    });
    let profitable = budget.is_none_or(|b| b.profitable(0));
    let capture = ready && profitable;
    Ok(Plan {
        invalidate,
        mode: if capture {
            Mode::CaptureReplay
        } else if invalidate || all_missing {
            Mode::BuildDirect
        } else {
            Mode::Direct
        },
        reason: if missing {
            "missing"
        } else if !ready {
            "warmup"
        } else if !profitable {
            "capture_budget"
        } else {
            "capture_payback"
        },
    })
}

pub(super) fn direct_plan(states: &[CacheState]) -> Result<Plan> {
    if states.is_empty() {
        return Err("Missing A2 TP cache states".into());
    }
    let missing = states.contains(&CacheState::Missing);
    let all_missing = states.iter().all(|s| *s == CacheState::Missing);
    let graphs = states.contains(&CacheState::Graph);
    let invalidate = (!all_missing && missing) || graphs;
    Ok(Plan {
        invalidate,
        mode: if invalidate || all_missing {
            Mode::BuildDirect
        } else {
            Mode::Direct
        },
        reason: "graph_disabled",
    })
}

#[cfg(test)]
#[path = "preparation_tests.rs"]
mod tests;
