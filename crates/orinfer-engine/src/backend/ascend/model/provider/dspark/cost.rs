//! Sparse measured priors and bounded online corrections, independent of owners.
use crate::{artifact::Result, scheduler::Estimate};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::backend::ascend::model::provider) struct Work {
    pub rows: usize,
    pub prefill_rows: usize,
    pub context: usize,
}
impl Work {
    pub fn extend(self, extra: usize) -> Self {
        let displaced = extra.min(self.prefill_rows);
        Self {
            rows: self.rows + extra - displaced,
            prefill_rows: self.prefill_rows - displaced,
            ..self
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Stage {
    Decode,
    Target,
    Draft,
    Restore,
}
#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::backend::ascend::model::provider) struct Identity {
    pub schema_version: u32,
    pub execution_package: String,
    pub target: String,
    pub cann: String,
    pub ranks: usize,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Point {
    pub stage: Stage,
    pub block: usize,
    pub full: bool,
    pub replay: bool,
    pub work: Work,
    pub seconds: f64,
    pub setup_s: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    identity: Identity,
    points: Vec<Point>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Family(Stage, usize, bool, bool, usize, usize);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key(Family, usize, bool);
struct Cell {
    time: Estimate,
}
#[derive(Default)]
struct Cold {
    build: Estimate,
    capture: Estimate,
}
#[derive(Default)]
pub(super) struct Model {
    cells: BTreeMap<Key, Cell>,
    modes: BTreeMap<(Family, usize), bool>,
    trace: bool,
    cold: BTreeMap<(Family, usize), Cold>,
}
#[derive(Clone, Copy)]
pub(super) struct Prediction {
    pub seconds: f64,
    pub replay: bool,
}
fn bucket(value: usize) -> usize {
    value.max(1).next_power_of_two()
}
fn family(stage: Stage, block: usize, full: bool, work: Work) -> Family {
    Family(
        stage,
        block,
        full,
        work.prefill_rows > 0,
        bucket(work.context),
        // Keep prompt work separate without one curve per exact mixed layout.
        work.prefill_rows.div_ceil(128),
    )
}
impl Model {
    pub fn load(identity: Identity, trace: bool) -> Result<Self> {
        let mut model = Self {
            trace,
            ..Self::default()
        };
        if let Some(path) = std::env::var_os("ORINFER_A2_DSPARK_COST_MODEL") {
            model.read(Path::new(&path), &identity)?;
        }
        if trace {
            eprintln!(
                "DSPARK_COST_IDENTITY {}",
                serde_json::to_string(&identity).unwrap()
            );
        }
        Ok(model)
    }
    fn read(&mut self, path: &Path, identity: &Identity) -> Result<()> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        if file.metadata().map_err(|e| e.to_string())?.len() > 2 * 1024 * 1024 {
            return Err("DSpark cost profile exceeds 2 MiB".into());
        }
        let profile: Profile = serde_json::from_reader(file).map_err(|e| e.to_string())?;
        if profile.identity != *identity || profile.points.is_empty() || profile.points.len() > 2048
        {
            return Err("DSpark cost profile identity/point count differs".into());
        }
        for point in profile.points {
            if !valid(&point) {
                return Err("Invalid DSpark cost profile point".into());
            }
            self.insert(point, false);
        }
        Ok(())
    }
    pub fn observe(&mut self, point: Point) {
        if !valid(&point) {
            return;
        }
        if self.trace {
            eprintln!("DSPARK_COST {}", serde_json::to_string(&point).unwrap());
        }
        self.insert(point, true);
    }
    fn insert(&mut self, point: Point, online: bool) {
        let group = family(point.stage, point.block, point.full, point.work);
        let key = Key(group, point.work.rows, point.replay);
        // Offline priors describe costs, never residency in a live Graph cache.
        if online {
            if self.modes.len() >= 2048 && !self.modes.contains_key(&(group, point.work.rows)) {
                self.modes.clear();
            }
            self.modes.insert((group, point.work.rows), point.replay);
        }
        if self.cells.len() >= 2048 && !self.cells.contains_key(&key) {
            // Retain the bounded sparse surface; unseen shapes can still use it.
            return;
        }
        let cell = self.cells.entry(key).or_insert(Cell {
            time: Estimate::default(),
        });
        cell.time.observe(point.seconds);
        if point.setup_s > 0. {
            // Host readback mode does not change the captured launch geometry.
            let cold_group = family(point.stage, point.block, false, point.work);
            if self.cold.len() < 2048 || self.cold.contains_key(&(cold_group, point.work.rows)) {
                let cold = self.cold.entry((cold_group, point.work.rows)).or_default();
                if point.replay {
                    cold.capture.observe(point.setup_s);
                } else {
                    cold.build.observe(point.setup_s);
                }
            }
        }
    }
    pub fn predict(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
    ) -> Option<Prediction> {
        let requested = family(stage, block, full, work);
        let replay = self
            .modes
            .get(&(requested, work.rows))
            .copied()
            .unwrap_or(false);
        self.predict_mode(stage, block, full, work, replay)
    }
    pub fn predict_mode(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
        replay: bool,
    ) -> Option<Prediction> {
        self.predict_mode_scoped(stage, block, full, work, replay, false)
    }
    fn predict_mode_scoped(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
        replay: bool,
        comparable: bool,
    ) -> Option<Prediction> {
        let requested = family(stage, block, full, work);
        // Select a context/prompt-work neighborhood before interpolating rows.
        let group = self
            .cells
            .keys()
            .filter(|Key(f, rows, mode)| {
                f.0 == stage
                    && f.1 == block
                    && f.2 == full
                    && f.3 == requested.3
                    && *mode == replay
                    && (!comparable
                        || f.4.ilog2().abs_diff(requested.4.ilog2()) <= 1
                            && f.5.abs_diff(requested.5) <= 1
                            // Producer and final-window packets can differ in
                            // decoder work even in the same prompt-size bin.
                            && (!requested.3 || *rows == work.rows)
                            && (*rows).max(work.rows)
                                <= (*rows).min(work.rows).max(1).saturating_mul(2))
            })
            .min_by_key(|Key(f, rows, _)| {
                (
                    rows.ilog2().abs_diff(work.rows.max(1).ilog2()) * 2
                        + f.4.ilog2().abs_diff(requested.4.ilog2()),
                    f.5.abs_diff(requested.5),
                )
            })
            .map(|Key(f, _, _)| *f)?;
        let points = self
            .cells
            .iter()
            .filter(|(Key(f, _, mode), _)| *f == group && *mode == replay);
        let mut lower = None;
        let mut previous = None;
        let mut upper = None;
        for (Key(_, rows, _), cell) in points {
            if *rows <= work.rows {
                previous = lower;
                lower = Some((*rows, cell));
            }
            if *rows >= work.rows && upper.is_none() {
                upper = Some((*rows, cell));
            }
        }
        let value = |cell: &Cell| Prediction {
            seconds: cell.time.mean,
            replay,
        };
        match (lower, upper) {
            // Replay pays the captured tier; no invented savings within a tier.
            (_, Some((_, cell))) if replay => Some(value(cell)),
            (Some((lo, a)), Some((hi, b))) if lo < hi => {
                let fraction = (work.rows - lo) as f64 / (hi - lo) as f64;
                Some(Prediction {
                    seconds: a.time.mean + (b.time.mean - a.time.mean) * fraction,
                    replay,
                })
            }
            (Some((rows, cell)), _) => Some(Prediction {
                seconds: cell.time.mean
                    + (work.rows - rows) as f64
                        * previous.map_or(
                            cell.time.mean / rows as f64,
                            |(before_rows, before): (usize, &Cell)| {
                                ((cell.time.mean - before.time.mean) / (rows - before_rows) as f64)
                                    .max(0.)
                            },
                        ),
                replay,
            }),
            (_, Some((_, cell))) => Some(value(cell)),
            _ => None,
        }
    }
    fn cold_estimate(&self, stage: Stage, block: usize, work: Work, capture: bool) -> Option<f64> {
        let requested = family(stage, block, false, work);
        self.cold
            .iter()
            .filter(|((f, _), c)| {
                f.0 == stage
                    && f.1 == block
                    && if capture {
                        c.capture.samples > 0
                    } else {
                        c.build.samples > 0
                    }
            })
            .min_by_key(|((f, rows), _)| {
                (
                    f.3 != requested.3,
                    f.4.ilog2().abs_diff(requested.4.ilog2()),
                    f.5.abs_diff(requested.5),
                    rows.abs_diff(work.rows),
                )
            })
            .map(|(_, c)| {
                if capture {
                    c.capture.upper()
                } else {
                    c.build.upper()
                }
            })
    }
    pub fn budget(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
        uses: f64,
        slack_s: f64,
    ) -> super::super::super::super::team::CaptureBudget {
        super::super::super::super::team::CaptureBudget {
            uses,
            direct_s: self
                .predict_mode(stage, block, full, work, false)
                .map(|p| p.seconds),
            replay_s: self
                // A distant prior still prices unseen work, but cannot prove
                // that a new local Graph would be slower. Leave it unknown so
                // repeated packets can use the existing bounded capture probe.
                .predict_mode_scoped(stage, block, full, work, true, true)
                .map(|p| p.seconds),
            capture_s: self.cold_estimate(stage, block, work, true),
            slack_s,
            trace: self.trace,
        }
    }
    pub fn price(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
        uses: f64,
        warmup: usize,
    ) -> Option<f64> {
        let current = self.predict(stage, block, full, work)?;
        if current.replay {
            return Some(current.seconds);
        }
        let budget = self.budget(stage, block, full, work, uses, f64::INFINITY);
        let build = self.cold_estimate(stage, block, work, false).unwrap_or(0.);
        let direct = current.seconds
            + if warmup >= 2 {
                build / uses.max(1.)
            } else {
                0.
            };
        if budget.profitable(warmup)
            && let (Some(replay), Some(capture)) = (budget.replay_s, budget.capture_s)
        {
            let debt = capture * 1.1 + warmup as f64 * (current.seconds - replay).max(0.) + build;
            return Some(direct.min(replay + debt / uses.max(1.)));
        }
        Some(direct)
    }
}
fn valid(point: &Point) -> bool {
    point.work.rows > 0
        && point.work.rows <= 65536
        && point.work.prefill_rows <= point.work.rows
        && point.work.context <= 1 << 24
        && point.block <= crate::execution::MAX_MTP_DRAFTS
        && point.seconds.is_finite()
        && point.seconds > 0.
        && point.setup_s.is_finite()
        && point.setup_s >= 0.
}

#[cfg(test)]
mod tests {
    use super::*;
    fn point(rows: usize, replay: bool, seconds: f64) -> Point {
        Point {
            stage: Stage::Target,
            block: 0,
            full: false,
            replay,
            work: Work {
                rows,
                context: 2048,
                prefill_rows: 0,
            },
            seconds,
            setup_s: 0.,
        }
    }
    #[test]
    fn captures_survive_zero_setup_replays_and_price_direct_transition() {
        let mut model = Model::default();
        model.observe(Point {
            setup_s: 0.020,
            ..point(71, false, 0.105)
        });
        model.observe(Point {
            setup_s: 0.294,
            ..point(71, true, 0.070)
        });
        for _ in 0..32 {
            model.observe(point(71, true, 0.070));
        }
        let work = point(71, false, 0.).work;
        let short = model.budget(Stage::Target, 0, false, work, 5., 1.);
        assert_eq!(short.capture_s, Some(0.294));
        assert!(!short.profitable(0));
        assert!(
            model
                .budget(Stage::Target, 0, false, work, 64., 1.)
                .profitable(0)
        );
        assert_eq!(
            model.cold_estimate(Stage::Target, 0, work, false),
            Some(0.020)
        );
    }
    #[test]
    fn distant_replay_priors_do_not_block_local_capture_probes() {
        for distant in [
            Work {
                rows: 20,
                context: 131072,
                ..Work::default()
            },
            Work {
                rows: 2048,
                context: 1024,
                ..Work::default()
            },
        ] {
            let mut model = Model::default();
            let work = Work {
                rows: 20,
                context: 1024,
                ..Work::default()
            };
            model.observe(Point {
                work,
                ..point(20, false, 0.11)
            });
            model.insert(
                Point {
                    work: distant,
                    setup_s: 0.25,
                    ..point(distant.rows, true, 0.51)
                },
                false,
            );
            let prior = model
                .predict_mode(Stage::Target, 0, false, work, true)
                .unwrap();
            assert_eq!(prior.seconds, 0.51);
            let budget = model.budget(Stage::Target, 0, false, work, 64., 1.);
            assert!(budget.replay_s.is_none());
            assert!(budget.profitable(0));
            assert!(
                !model
                    .budget(Stage::Target, 0, false, work, 5., 1.)
                    .profitable(0)
            );

            // Nearby observations retain their cost and payback decision.
            model.insert(
                Point {
                    work: Work {
                        rows: 32,
                        context: 2048,
                        ..Work::default()
                    },
                    ..point(32, true, 0.06)
                },
                false,
            );
            let budget = model.budget(Stage::Target, 0, false, work, 64., 1.);
            assert_eq!(budget.replay_s, Some(0.06));
            assert!(budget.profitable(0));
        }
    }
    #[test]
    fn prompt_capture_does_not_price_a_producer_from_a_larger_final_window() {
        let mut model = Model::default();
        let work = Work {
            rows: 1921,
            prefill_rows: 1921,
            context: 0,
        };
        model.observe(Point {
            work,
            ..point(1921, false, 0.16)
        });
        model.insert(
            Point {
                work: Work {
                    rows: 2048,
                    prefill_rows: 2048,
                    context: 0,
                },
                setup_s: 0.25,
                ..point(2048, true, 0.20)
            },
            false,
        );
        assert!(
            model
                .budget(Stage::Target, 0, false, work, 16., 1.)
                .replay_s
                .is_none()
        );
    }
    #[test]
    fn large_verification_uses_nearby_rows_and_marginal_extrapolation() {
        let mut model = Model::default();
        model.observe(Point {
            stage: Stage::Decode,
            ..point(32, true, 0.113)
        });
        model.observe(point(24, true, 0.050));
        for (rows, seconds) in [(78, 0.065), (96, 0.067)] {
            model.observe(Point {
                work: Work {
                    context: 8192,
                    ..point(rows, true, seconds).work
                },
                setup_s: 0.295,
                ..point(rows, true, seconds)
            });
        }
        let prediction = model
            .predict_mode(
                Stage::Target,
                0,
                false,
                Work {
                    rows: 192,
                    context: 2048,
                    prefill_rows: 0,
                },
                true,
            )
            .unwrap();
        assert!(prediction.seconds > 0.067 && prediction.seconds < 0.080);
    }
    #[test]
    fn priors_do_not_imply_graph_residency_and_modes_do_not_mix() {
        let mut model = Model::default();
        model.insert(point(8, false, 0.04), false);
        model.insert(point(8, true, 0.02), false);
        let work = point(8, false, 0.).work;
        assert_eq!(
            model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds,
            0.04
        );
        model.observe(point(8, true, 0.02));
        assert_eq!(
            model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds,
            0.02
        );
        model.observe(point(8, false, 0.20));
        model.observe(point(8, true, 0.02));
        assert_eq!(
            model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds,
            0.02
        );
    }
    #[test]
    fn unseen_batch_uses_shared_prior_and_replay_preserves_tiers() {
        let mut model = Model::default();
        for replay in [false, true] {
            model.insert(point(8, replay, 0.02), false);
            model.insert(point(16, replay, 0.04), false);
        }
        let work = point(12, false, 0.).work;
        assert!(
            (model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds
                - 0.03)
                .abs()
                < 1e-12
        );
        model
            .modes
            .insert((family(Stage::Target, 0, false, work), 12), true);
        assert_eq!(
            model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds,
            0.04
        );
    }
}
