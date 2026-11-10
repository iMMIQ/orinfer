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
            (Some(direct), None, Some(capture)) => {
                // The first replay is a bounded measurement, not a predicted
                // speedup. Limit it to a quarter of the remaining direct work:
                // a fixed reuse count can abandon a costly fused packet before
                // it ever measures the execution mode it was tuned for.
                let useful = (self.uses - warmup_uses as f64).max(0.);
                (self.uses >= 16. || useful * direct > capture * 4.)
                    && capture + direct <= self.slack_s
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
        if observed >= 8 {
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

pub(super) fn direct_plan(states: &[CacheState]) -> Result<Plan> {
    if states.is_empty() {
        return Err("Missing A2 TP cache states".into());
    }
    let invalidate = states
        .iter()
        .any(|state| !matches!(state, CacheState::Direct { .. }));
    Ok(Plan {
        invalidate,
        mode: if invalidate {
            Mode::BuildDirect
        } else {
            Mode::Direct
        },
        reason: "graph_disabled",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eager_mode_never_promotes_hot_packets_and_retires_any_captured_rank() {
        let hot = CacheState::Direct {
            uses: 255,
            consecutive: 255,
        };
        let plan = direct_plan(&[hot, hot]).unwrap();
        assert!(!plan.capture());
        assert!(!plan.invalidate);
        assert_eq!(plan.reason, "graph_disabled");
        for states in [[hot, CacheState::Graph], [hot, CacheState::Missing]] {
            let plan = direct_plan(&states).unwrap();
            assert!(plan.invalidate);
            assert!(!plan.capture());
        }
        assert!(direct_plan(&[]).is_err());
    }
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
        assert!((capture_credit(0., 0.4, 0.18, 1) - 0.22).abs() < 1e-12);
        assert!((capture_credit(0.22, 0.4, 0.42, 1) - 0.20).abs() < 1e-12);
        assert_eq!(capture_credit(0.4, 0.4, 0.01, 1), 0.4);
        assert_eq!(capture_credit(0.2, 0.4, 1.0, 1), 0.);
        assert_eq!(capture_credit(0., 0.4, 0.54, 6), 0.4);
        assert!((capture_credit(0.2, 0.4, 0.1, 0) - 0.1).abs() < 1e-12);
    }
    #[test]
    fn speculative_progress_can_pay_for_capture_without_unbounded_burst_slack() {
        let budget = CaptureBudget {
            uses: 64.,
            direct_s: Some(0.54),
            replay_s: None,
            capture_s: Some(0.30),
            slack_s: capture_interval(0.4, 6.) - 0.30 + capture_credit(0., 0.4, 0.54, 6),
            trace: false,
        };
        assert!(budget.profitable(0));
        assert!(
            !CaptureBudget {
                slack_s: capture_interval(0.4, 1.) - 0.30,
                ..budget
            }
            .profitable(0)
        );
        assert_eq!(capture_interval(0.4, 6.), 0.8);
        assert_eq!(capture_interval(0.4, 1.), 0.4);
        assert_eq!(capture_interval(0.4, 0.), 0.4);
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
    fn first_replay_trial_uses_measured_cost_instead_of_a_fixed_reuse_count() {
        let measured = CaptureBudget {
            uses: 8.9409,
            direct_s: Some(0.21065),
            replay_s: None,
            capture_s: Some(0.35755),
            slack_s: 1.1494,
            trace: false,
        };
        let ready = [CacheState::Direct {
            uses: 2,
            consecutive: 2,
        }; 8];
        assert!(plan(&ready, true, Some(measured)).unwrap().capture());
        assert!(
            !CaptureBudget {
                uses: 4.,
                ..measured
            }
            .profitable(0)
        );
        assert!(
            !CaptureBudget {
                slack_s: 0.4,
                ..measured
            }
            .profitable(0)
        );
        assert!(
            !CaptureBudget {
                replay_s: Some(0.22),
                ..measured
            }
            .profitable(0)
        );
        assert!(!measured.profitable(3));
        assert!(
            !plan(&[CacheState::Missing; 8], true, Some(measured))
                .unwrap()
                .capture()
        );
    }
    #[test]
    fn retained_prompt_tail_can_learn_reuse_across_requests() {
        let unknown = CaptureBudget {
            uses: 1.,
            direct_s: Some(0.14),
            replay_s: None,
            capture_s: Some(0.30),
            slack_s: f64::INFINITY,
            trace: false,
        };
        let state = |uses| CacheState::Direct {
            uses,
            consecutive: 1,
        };
        assert_eq!(
            plan(&[state(7); 8], false, Some(unknown)).unwrap().mode,
            Mode::Direct
        );
        assert_eq!(
            plan(&[state(8); 8], false, Some(unknown)).unwrap().mode,
            Mode::CaptureReplay
        );
        for budget in [
            CaptureBudget {
                replay_s: Some(0.20),
                ..unknown
            },
            CaptureBudget {
                slack_s: 0.10,
                ..unknown
            },
        ] {
            assert_eq!(
                plan(&[state(16); 8], false, Some(budget)).unwrap().mode,
                Mode::Direct
            );
        }
        assert_eq!(
            plan(&[CacheState::Missing, state(16)], false, Some(unknown))
                .unwrap()
                .mode,
            Mode::BuildDirect
        );
    }
    #[test]
    fn retained_draft_shape_can_learn_reuse_across_short_cohorts() {
        let budget = CaptureBudget {
            uses: 1.,
            direct_s: Some(0.014),
            replay_s: None,
            capture_s: Some(0.04),
            slack_s: 0.4,
            trace: false,
        };
        let state = |uses| CacheState::Direct {
            uses,
            consecutive: 1,
        };
        assert_eq!(
            plan(&[state(7); 8], true, Some(budget)).unwrap().mode,
            Mode::Direct
        );
        assert_eq!(
            plan(&[state(8); 8], true, Some(budget)).unwrap().mode,
            Mode::CaptureReplay
        );
        assert_eq!(
            plan(
                &[state(16); 8],
                true,
                Some(CaptureBudget {
                    slack_s: 0.01,
                    ..budget
                })
            )
            .unwrap()
            .mode,
            Mode::Direct
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
