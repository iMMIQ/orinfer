//! Time-budgeted chunks use the same measured target cells as DSpark pricing.
use super::*;

#[derive(Clone, Copy)]
pub(in crate::models::deepseek_v41::provider) struct Budget {
    pub fallback: usize,
    pub capacity: usize,
    pub quantum: usize,
    pub quota_s: f64,
    pub deadline_s: f64,
    pub probe_credit_s: f64,
}
impl Model {
    pub(in crate::models::deepseek_v41::provider) fn prefill_rows(
        &self,
        full: bool,
        work: Work,
        budget: Budget,
    ) -> usize {
        let Budget {
            fallback,
            capacity,
            quantum,
            quota_s,
            deadline_s,
            probe_credit_s,
        } = budget;
        let decoders = work.rows.saturating_sub(work.prefill_rows);
        if work.prefill_rows == 0
            || decoders == 0
            || decoders >= capacity
            || quantum == 0
            || !quota_s.is_finite()
            || quota_s <= 0.
        {
            return fallback;
        }
        // Only live mixed executions in this context tier train chunk choice.
        // Offline prices and pure prefill cannot establish mixed readiness.
        let points = self
            .cells
            .iter()
            .filter(|(Key(f, rows, replay, shared), cell)| {
                let d = cell.work.rows.saturating_sub(cell.work.prefill_rows);
                f.0 == Stage::Target
                    && f.1 == 0
                    && f.2 == full
                    && f.3
                    && f.4 == bucket(work.context)
                    && d > 0
                    && cell.work.rows.max(work.rows)
                        <= cell.work.rows.min(work.rows).saturating_mul(2)
                    && d.max(decoders) <= d.min(decoders).saturating_mul(2)
                    && cell.complete.samples() >= crate::mtp::adaptive::HOT_SAMPLES
                    && self.modes.get(&(*f, *rows)) == Some(replay)
                    && self.shared_modes.contains(&(*f, *rows)) == *shared
            })
            .collect::<Vec<_>>();
        let Some(&(_, reference)) = points.iter().max_by_key(|(_, cell)| cell.tick) else {
            return fallback;
        };
        let fixed = |w: Work| {
            // Full mixed readout also covers first-token prompt heads. The
            // decoder-only component may still use the compact device head.
            [full, false].into_iter().find_map(|readout| {
                [Stage::Target, Stage::Decode]
                    .into_iter()
                    .find_map(|stage| {
                        self.predict_mode(
                            stage,
                            0,
                            readout,
                            Work {
                                rows: w.rows - w.prefill_rows,
                                prefill_rows: 0,
                                context: w.context,
                            },
                            false,
                        )
                        .filter(|p| p.nearby && p.samples >= crate::mtp::adaptive::HOT_SAMPLES)
                        .map(|p| p.seconds)
                    })
            })
        };
        let reference_base = fixed(reference.work)
            .unwrap_or(0.)
            .min(reference.complete.mean() * 0.9);
        let reference_decoders = reference.work.rows - reference.work.prefill_rows;
        let current_base = fixed(work)
            .unwrap_or_else(|| reference_base * decoders as f64 / reference_decoders as f64);
        let slope =
            (reference.complete.mean() - reference_base) / reference.work.prefill_rows as f64;
        if !slope.is_finite() || slope <= 0. {
            return fallback;
        }
        // Existing package geometry supplies the row quantum. Grow at most
        // 50% or one published quantum from a measured packet. The local
        // twofold window matches the existing sparse price extrapolation.
        let next_tile = (reference.work.rows / quantum + 1).saturating_mul(quantum);
        let maximum = capacity.min(work.rows.saturating_mul(2)).min(
            reference
                .work
                .rows
                .saturating_add(reference.work.rows / 2)
                .max(next_tile),
        );
        // Keep the prior floor: an unmeasured small chunk must not trade away
        // weight-pass efficiency merely to satisfy a noisy tail estimate.
        let minimum = fallback.max(decoders + 1);
        let mut candidates = vec![fallback, reference.work.rows];
        candidates.extend((quantum..=maximum).step_by(quantum));
        candidates.extend(points.iter().map(|(_, cell)| cell.work.rows));
        candidates.retain(|&rows| rows >= minimum && rows <= maximum);
        candidates.sort_unstable();
        candidates.dedup();
        let known_cost = |rows| {
            points
                .iter()
                .filter(|(_, cell)| cell.work.rows == rows)
                .max_by_key(|(_, cell)| cell.tick)
                .map(|(_, cell)| {
                    let base = fixed(cell.work)
                        .unwrap_or_else(|| {
                            reference_base * (cell.work.rows - cell.work.prefill_rows) as f64
                                / reference_decoders as f64
                        })
                        .min(cell.complete.mean() * 0.9);
                    (
                        cell.complete.mean() - base + current_base,
                        cell.complete.upper() - base + current_base,
                    )
                })
        };
        let best = candidates
            .iter()
            .copied()
            .rfind(|&rows| {
                if let Some((seconds, tail)) = known_cost(rows) {
                    // A larger measured tier can fit the deadline yet lose
                    // prefill throughput. Keep an efficient smaller tier.
                    let dominated = candidates
                        .iter()
                        .copied()
                        .filter(|&other| other < rows)
                        .filter_map(|other| known_cost(other).map(|(cost, _)| (other, cost)))
                        .any(|(other, cost)| {
                            cost <= quota_s
                                && (rows - decoders) as f64 / seconds
                                    < 0.95 * (other - decoders) as f64 / cost
                        });
                    return seconds <= quota_s && tail <= deadline_s && !dominated;
                }
                let candidate = Work {
                    rows,
                    prefill_rows: rows - decoders,
                    ..work
                };
                let prepare = self
                    .cold_estimate_scoped(Stage::Target, 0, candidate, false, true)
                    .unwrap_or(0.);
                let seconds = current_base + slope * candidate.prefill_rows as f64 + prepare;
                let tail =
                    seconds * (reference.complete.upper() / reference.complete.mean()).max(1.);
                let reserve = (seconds - quota_s * 0.95).max(0.)
                    * crate::mtp::adaptive::CALIBRATION_ROUNDS as f64;
                (seconds <= quota_s * 0.95 || reserve <= probe_credit_s) && tail <= deadline_s
            })
            .unwrap_or_else(|| candidates.first().copied().unwrap_or(fallback));
        // Keep the current measured tier through small timing changes.
        if best.abs_diff(reference.work.rows) * 10 < reference.work.rows
            && known_cost(reference.work.rows)
                .is_some_and(|(mean, tail)| mean <= quota_s && tail <= deadline_s)
        {
            reference.work.rows.max(fallback)
        } else {
            best.max(fallback)
        }
    }
    pub(in crate::models::deepseek_v41::provider) fn prefill_loss(
        &self,
        point: &Point,
        fallback: usize,
    ) -> Option<f64> {
        let decoders = point.work.rows.saturating_sub(point.work.prefill_rows);
        if point.stage != Stage::Target
            || point.work.rows <= fallback
            || point.work.prefill_rows == 0
            || decoders == 0
            || decoders >= fallback
        {
            return None;
        }
        let reference = self
            .cells
            .iter()
            .filter(|(Key(f, rows, mode, shared), cell)| {
                let d = cell.work.rows.saturating_sub(cell.work.prefill_rows);
                f.0 == Stage::Target
                    && f.1 == 0
                    && f.2 == point.full
                    && f.3
                    && *rows == fallback
                    && f.4 == bucket(point.work.context)
                    && d > 0
                    && d.max(decoders) <= d.min(decoders).saturating_mul(2)
                    && cell.complete.samples() >= crate::mtp::adaptive::HOT_SAMPLES
                    && self.modes.get(&(*f, *rows)) == Some(mode)
                    && self.shared_modes.contains(&(*f, *rows)) == *shared
            })
            .max_by_key(|(_, cell)| cell.tick)?
            .1;
        let normal = reference.complete.mean() * point.work.prefill_rows as f64
            / reference.work.prefill_rows as f64;
        // Charge losses only; speculative commits already earn the common
        // account's credit, so prefill must not count the same earnings twice.
        Some((normal - point.seconds - point.setup_s).min(0.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget() -> Budget {
        Budget {
            fallback: 800,
            capacity: 2048,
            quantum: 256,
            quota_s: 0.2,
            deadline_s: 0.4,
            probe_credit_s: 0.,
        }
    }
    fn work(rows: usize) -> Work {
        Work {
            rows,
            prefill_rows: rows - 96,
            context: 100_000,
        }
    }
    fn observe(model: &mut Model, rows: usize, seconds: f64, setup_s: f64, replay: bool) {
        for _ in 0..4 {
            model.observe(Point {
                stage: Stage::Target,
                block: 0,
                full: false,
                replay,
                shared: false,
                shared_setup_s: 0.,
                work: work(rows),
                seconds,
                setup_s,
            });
        }
    }
    #[test]
    fn prefill_choice_requires_live_mixed_samples_and_preserves_other_domains() {
        let mut m = Model::default();
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
        let p = Point {
            stage: Stage::Target,
            block: 0,
            full: false,
            replay: false,
            shared: false,
            shared_setup_s: 0.,
            work: work(800),
            seconds: 0.12,
            setup_s: 0.,
        };
        m.insert(p.clone(), false);
        m.observe(p.clone());
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
        observe(&mut m, 800, 0.12, 0., false);
        assert_eq!(m.prefill_rows(true, work(800), budget()), 800);
        assert_eq!(
            m.prefill_rows(
                false,
                Work {
                    context: 200_000,
                    ..work(800)
                },
                budget()
            ),
            800
        );
        assert_eq!(
            m.prefill_rows(
                false,
                Work {
                    prefill_rows: 0,
                    ..work(800)
                },
                budget()
            ),
            800
        );
        assert_eq!(
            m.prefill_rows(
                false,
                Work {
                    prefill_rows: 800,
                    ..work(800)
                },
                budget()
            ),
            800
        );
    }
    #[test]
    fn prefill_choice_grows_boundedly_and_remembers_a_slow_tier() {
        let mut m = Model::default();
        observe(&mut m, 800, 0.12, 0., false);
        assert_eq!(m.prefill_rows(false, work(800), budget()), 1024);
        observe(&mut m, 1024, 0.28, 0., false);
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
        observe(&mut m, 800, 0.12, 0., false);
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
    }
    #[test]
    fn prefill_choice_rejects_a_larger_but_less_efficient_measured_tier() {
        let mut m = Model::default();
        observe(&mut m, 800, 0.12, 0., false);
        observe(&mut m, 1024, 0.195, 0., false);
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
    }

    #[test]
    fn prefill_choice_does_not_learn_a_long_packet_from_a_short_tail() {
        let mut m = Model::default();
        observe(&mut m, 160, 0.1, 0., false);
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
        observe(&mut m, 512, 0.07, 0., false);
        assert!(
            m.prefill_rows(
                false,
                work(800),
                Budget {
                    quantum: 512,
                    ..budget()
                }
            ) >= 800
        );
    }

    #[test]
    fn prefill_probe_needs_shared_credit_and_learns_the_actual_loss() {
        let mut m = Model::default();
        observe(&mut m, 800, 0.18, 0., false);
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
        assert_eq!(
            m.prefill_rows(
                false,
                work(800),
                Budget {
                    probe_credit_s: 0.4,
                    ..budget()
                }
            ),
            1024
        );
        let point = Point {
            stage: Stage::Target,
            block: 0,
            full: false,
            replay: false,
            shared: false,
            shared_setup_s: 0.,
            work: work(1024),
            seconds: 0.3,
            setup_s: 0.,
        };
        assert!(m.prefill_loss(&point, 800).unwrap() < 0.);
        observe(&mut m, 1024, 0.3, 0., false);
        assert_eq!(
            m.prefill_rows(
                false,
                work(800),
                Budget {
                    probe_credit_s: 0.4,
                    ..budget()
                }
            ),
            800
        );
    }

    #[test]
    fn prefill_selection_does_not_spend_and_one_completed_probe_charges_once() {
        let mut policy = super::super::super::Policy::default();
        policy.start_iteration(0.4);
        observe(&mut policy.costs, 800, 0.18, 0., false);
        assert_eq!(policy.prefill_rows(false, work(800), budget()), 1024);
        assert_eq!(policy.ledger.startup_spent, 0.);
        assert_eq!(policy.ledger.balance, 0.);
        policy.prepare_prefill(Some(budget()));
        let point = Point {
            stage: Stage::Target,
            block: 0,
            full: false,
            replay: false,
            shared: false,
            shared_setup_s: 0.,
            work: work(1024),
            seconds: 0.3,
            setup_s: 0.,
        };
        policy.record_prefill(&point);
        let spent = policy.ledger.startup_spent;
        assert!(spent > 0. && spent <= 0.4);
        policy.record_prefill(&point);
        assert_eq!(policy.ledger.startup_spent, spent);
    }

    #[test]
    fn prefill_choice_prices_preparation_and_keeps_decode_rows_inside_capacity() {
        let mut m = Model::default();
        observe(&mut m, 800, 0.12, 0.09, false);
        assert_eq!(m.prefill_rows(false, work(800), budget()), 800);
        let mut m = Model::default();
        observe(&mut m, 800, 0.1, 0., true);
        // Replay observations do not establish a direct price or eliminate setup.
        assert_eq!(m.prefill_rows(false, work(800), budget()), 1024);
        let b = Budget {
            capacity: 900,
            ..budget()
        };
        let chosen = m.prefill_rows(false, work(800), b);
        assert!(chosen > 96 && chosen <= 900);
        assert_eq!(
            m.prefill_rows(
                false,
                work(800),
                Budget {
                    quota_s: f64::NAN,
                    ..budget()
                }
            ),
            800
        );
    }
}
