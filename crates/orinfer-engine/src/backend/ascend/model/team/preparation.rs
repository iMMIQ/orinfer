//! One all-rank decision after workspace growth, shared by every packet role.
use super::*;

pub(in crate::backend::ascend::model) fn capture_credit(
    previous: f64,
    target: f64,
    elapsed: f64,
) -> f64 {
    (previous + target - elapsed).clamp(0., target)
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(in crate::backend::ascend::model) struct CaptureBudget {
    pub uses: f64,
    pub direct_s: Option<f64>,
    pub replay_s: Option<f64>,
    pub capture_s: Option<f64>,
    pub slack_s: f64,
    pub trace: bool,
}
impl CaptureBudget {
    pub fn profitable(self, warmup_uses: usize) -> bool {
        if self.capture_s.is_some_and(|capture| {
            capture + self.replay_s.or(self.direct_s).unwrap_or(0.) > self.slack_s
        }) {
            return false;
        }
        match (self.direct_s, self.replay_s, self.capture_s) {
            (Some(direct), Some(replay), Some(capture)) => {
                let useful = (self.uses - warmup_uses as f64).max(0.);
                useful * (direct - replay).max(0.) > capture * 1.1
                    && capture + replay <= self.slack_s
            }
            // Seed one missing family on a long, repeated packet. Observations
            // then price new row counts from the same shared sparse surface.
            _ => self.uses >= 16. && self.slack_s > self.direct_s.unwrap_or(0.),
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

pub(super) fn plan(
    states: &[CacheState],
    repeated: bool,
    budget: Option<CaptureBudget>,
) -> Result<Plan> {
    if states.is_empty() {
        return Err("Missing A2 TP cache states".into());
    }
    let missing = states.contains(&CacheState::Missing);
    let graphs = states.iter().filter(|s| **s == CacheState::Graph).count();
    if graphs == states.len() {
        return Ok(Plan {
            invalidate: false,
            mode: Mode::Replay,
            reason: "resident_graph",
        });
    }
    let invalidate = missing || graphs > 0;
    let ready = !missing && (graphs > 0 || states.iter().all(|s| {
        matches!(s, CacheState::Direct { uses, .. } if *uses >= if repeated { 2 } else { 8 })
    }));
    let profitable = budget.is_none_or(|b| b.profitable(0));
    let capture = ready && profitable;
    Ok(Plan {
        invalidate,
        mode: if capture {
            Mode::CaptureReplay
        } else if invalidate {
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

#[cfg(test)]
mod tests {
    use super::*;
    fn budget(uses: f64) -> CaptureBudget {
        CaptureBudget {
            uses,
            direct_s: Some(0.105),
            replay_s: Some(0.070),
            capture_s: Some(0.294),
            slack_s: 1.,
            trace: false,
        }
    }
    #[test]
    fn latency_credit_is_bounded_and_repays_capture_overrun() {
        assert!((capture_credit(0., 0.4, 0.18) - 0.22).abs() < 1e-12);
        assert!((capture_credit(0.22, 0.4, 0.42) - 0.20).abs() < 1e-12);
        assert_eq!(capture_credit(0.4, 0.4, 0.01), 0.4);
        assert_eq!(capture_credit(0.2, 0.4, 1.0), 0.);
    }
    #[test]
    fn short_tail_keeps_direct_and_long_packet_can_capture() {
        let states = [CacheState::Direct {
            uses: 2,
            consecutive: 2,
        }; 2];
        assert_eq!(
            plan(&states, true, Some(budget(5.))).unwrap().mode,
            Mode::Direct
        );
        assert_eq!(
            plan(&states, true, Some(budget(64.))).unwrap().mode,
            Mode::CaptureReplay
        );
        let no_slack = CaptureBudget {
            slack_s: 0.1,
            ..budget(64.)
        };
        assert_eq!(
            plan(&states, true, Some(no_slack)).unwrap().mode,
            Mode::Direct
        );
        assert_eq!(
            plan(&[CacheState::Graph; 2], true, Some(no_slack))
                .unwrap()
                .mode,
            Mode::Replay
        );
    }
    #[test]
    fn asymmetric_cache_defer_invalidates_every_rank() {
        let states = [
            CacheState::Graph,
            CacheState::Direct {
                uses: 3,
                consecutive: 3,
            },
        ];
        let p = plan(&states, true, Some(budget(2.))).unwrap();
        assert!(p.invalidate);
        assert_eq!(p.mode, Mode::BuildDirect);
        assert_eq!(
            plan(
                &[CacheState::Missing, CacheState::Graph],
                true,
                Some(budget(64.))
            )
            .unwrap()
            .mode,
            Mode::BuildDirect
        );
    }
}
