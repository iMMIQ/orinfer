//! Sparse measured priors and bounded online corrections, independent of owners.
use crate::{artifact::Result, scheduler::Estimate};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::models::deepseek_v41::provider) struct Work {
    pub rows: usize,
    pub prefill_rows: usize,
    pub context: usize,
}
#[cfg(test)]
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
pub(in crate::models::deepseek_v41::provider) struct Identity {
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
    #[serde(default)]
    pub shared: bool,
    #[serde(default)]
    pub shared_setup_s: f64,
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
struct Key(Family, usize, bool, bool);
struct Cell {
    time: crate::mtp::adaptive::Timing,
    prior: f64,
}
impl Cell {
    fn mean(&self) -> f64 {
        if self.time.observations() > 0 {
            self.time.mean()
        } else {
            self.prior
        }
    }
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
    shared_modes: std::collections::BTreeSet<(Family, usize)>,
    trace: bool,
    cold: BTreeMap<(Family, usize), Cold>,
}
#[derive(Clone, Copy)]
pub(super) struct Prediction {
    pub seconds: f64,
    pub replay: bool,
    pub samples: usize,
    pub observations: usize,
    nearby: bool,
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
    pub fn load(identity: Identity, package_root: &Path, trace: bool) -> Result<Self> {
        let mut model = Self {
            trace,
            ..Self::default()
        };
        if let Some(path) = std::env::var_os("ORINFER_A2_DSPARK_COST_MODEL") {
            model.read(Path::new(&path), &identity)?;
        } else if let Some(cache) = package_root.parent() {
            // Device timings are local tuning data, separate from immutable
            // operators and model weights. A stale optional profile is inert.
            let path = cache
                .join("profiles")
                .join(format!("{}.json", identity.execution_package));
            if path.is_file()
                && let Err(error) = model.read(&path, &identity)
            {
                eprintln!("A2 DSpark cost profile ignored: {error}");
            }
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
        // Validate the whole profile before publishing any optional prior.
        if profile.points.iter().any(|point| !valid(point)) {
            return Err("Invalid DSpark cost profile point".into());
        }
        for point in profile.points {
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
        let key = Key(group, point.work.rows, point.replay, point.shared);
        // Offline priors describe costs, never residency in a live Graph cache.
        if online {
            if self.modes.len() >= 2048 && !self.modes.contains_key(&(group, point.work.rows)) {
                self.modes.clear();
                self.shared_modes.clear();
            }
            self.modes.insert((group, point.work.rows), point.replay);
            if point.shared {
                self.shared_modes.insert((group, point.work.rows));
            } else {
                self.shared_modes.remove(&(group, point.work.rows));
            }
        }
        if self.cells.len() >= 2048 && !self.cells.contains_key(&key) {
            // Retain the bounded sparse surface; unseen shapes can still use it.
            return;
        }
        let cell = self.cells.entry(key).or_insert(Cell {
            time: Default::default(),
            prior: 0.,
        });
        if online {
            cell.time.observe(point.seconds);
        } else {
            // Imported timings price execution; only this process's real
            // calls may consume its primer and establish hot observations.
            cell.prior = if cell.prior > 0. {
                cell.prior * 0.8 + point.seconds * 0.2
            } else {
                point.seconds
            };
        }
        let local_setup = (point.setup_s - point.shared_setup_s).max(0.);
        if local_setup > 0. {
            // Host readback mode does not change the captured launch geometry.
            let cold_group = family(point.stage, point.block, false, point.work);
            if self.cold.len() < 2048 || self.cold.contains_key(&(cold_group, point.work.rows)) {
                let cold = self.cold.entry((cold_group, point.work.rows)).or_default();
                if point.replay {
                    cold.capture.observe(local_setup);
                } else {
                    cold.build.observe(local_setup);
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
    pub fn profiled(&self, stage: Stage, block: usize, full: bool, work: Work) -> bool {
        let group = family(stage, block, full, work);
        self.cells
            .iter()
            .any(|(Key(f, rows, _, _), cell)| *f == group && *rows == work.rows && cell.prior > 0.)
    }
    /// Measured neighboring rows can price another logical prefix in the same
    /// physical draft family; they do not establish Graph residency.
    pub(super) fn evidence(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
    ) -> (usize, usize) {
        // Preparation reserves use real observation counts; calibration uses
        // only samples retained by the shared primer in each execution mode.
        [false, true]
            .into_iter()
            .filter_map(|replay| {
                self.predict_mode(stage, block, full, work, replay)
                    .filter(|p| p.nearby)
            })
            .fold((0, 0), |(hot, seen), p| {
                (hot + p.samples, seen + p.observations)
            })
    }
    pub fn predict_mode(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
        replay: bool,
    ) -> Option<Prediction> {
        let requested = family(stage, block, full, work);
        // Select a context/prompt-work neighborhood before interpolating rows.
        let shared = !replay && self.shared_modes.contains(&(requested, work.rows));
        let group = self
            .cells
            .keys()
            .filter(|Key(f, rows, mode, shared_mode)| {
                f.0 == stage
                    && f.1 == block
                    && f.2 == full
                    && f.3 == requested.3
                    && *mode == replay
                    && *shared_mode == shared
                    // Native verification changes its execution strategy at
                    // 128 rows. Never interpolate across that boundary.
                    && (*rows >= 128) == (work.rows >= 128)
                    // Small packets include a large fixed launch cost. Do not
                    // multiply a single-owner observation across a distant
                    // batch; leave that cost unknown for bounded calibration.
                    && work.rows <= rows.saturating_mul(2)
            })
            .min_by_key(|Key(f, rows, _, _)| {
                (
                    rows.ilog2().abs_diff(work.rows.max(1).ilog2()) * 2
                        + f.4.ilog2().abs_diff(requested.4.ilog2()),
                    f.5.abs_diff(requested.5),
                )
            })
            .map(|Key(f, _, _, _)| *f)?;
        let points = self
            .cells
            .iter()
            .filter(|(Key(f, rows, mode, shared_mode), _)| {
                *f == group
                    && *mode == replay
                    && *shared_mode == shared
                    && (*rows >= 128) == (work.rows >= 128)
            });
        let mut lower = None;
        let mut previous = None;
        let mut upper = None;
        for (Key(_, rows, _, _), cell) in points {
            if *rows <= work.rows {
                previous = lower;
                lower = Some((*rows, cell));
            }
            if *rows >= work.rows && upper.is_none() {
                upper = Some((*rows, cell));
            }
        }
        // An upward row extrapolation is a prior, not local calibration.
        // Downward reuse keeps the measured upper packet's whole cost.
        let nearby = upper.is_some_and(|(rows, _)| rows <= work.rows.saturating_mul(2))
            && group.4.ilog2().abs_diff(requested.4.ilog2()) <= 1
            && group.5.abs_diff(requested.5) <= 1;
        let value = |cell: &Cell| Prediction {
            seconds: cell.mean(),
            replay,
            samples: cell.time.samples(),
            observations: cell.time.observations(),
            nearby,
        };
        match (lower, upper) {
            // Replay pays the captured tier; no invented savings within a tier.
            (_, Some((_, cell))) if replay => Some(value(cell)),
            (Some((lo, a)), Some((hi, b))) if lo < hi => {
                let fraction = (work.rows - lo) as f64 / (hi - lo) as f64;
                Some(Prediction {
                    seconds: a.mean() + (b.mean() - a.mean()) * fraction,
                    replay,
                    samples: a.time.samples().min(b.time.samples()),
                    observations: a.time.observations().min(b.time.observations()),
                    nearby,
                })
            }
            (Some((rows, cell)), _) => Some(Prediction {
                seconds: cell.mean()
                    + (work.rows - rows) as f64
                        * previous.map_or(
                            cell.mean() / rows as f64,
                            |(before_rows, before): (usize, &Cell)| {
                                ((cell.mean() - before.mean()) / (rows - before_rows) as f64)
                                    .max(0.)
                            },
                        ),
                replay,
                samples: cell.time.samples(),
                observations: cell.time.observations(),
                nearby,
            }),
            (_, Some((_, cell))) => Some(value(cell)),
            _ => None,
        }
    }
    fn cold_estimate(&self, stage: Stage, block: usize, work: Work, capture: bool) -> Option<f64> {
        self.cold_estimate_scoped(stage, block, work, capture, false)
    }
    /// Exploration reserves preparation only from a comparable packet. A cold
    /// prefill plan is not evidence that a tiny verification plan costs as much.
    pub(super) fn probe_preparation(
        &self,
        stage: Stage,
        block: usize,
        work: Work,
        capture: bool,
    ) -> Option<f64> {
        self.cold_estimate_scoped(stage, block, work, capture, true)
    }
    fn cold_estimate_scoped(
        &self,
        stage: Stage,
        block: usize,
        work: Work,
        capture: bool,
        comparable: bool,
    ) -> Option<f64> {
        let requested = family(stage, block, false, work);
        self.cold
            .iter()
            .filter(|((f, rows), c)| {
                f.0 == stage
                    && f.1 == block
                    && (!comparable
                        || f.3 == requested.3
                            && f.4.ilog2().abs_diff(requested.4.ilog2()) <= 1
                            && f.5.abs_diff(requested.5) <= 1
                            && (*rows).max(work.rows)
                                <= (*rows).min(work.rows).max(1).saturating_mul(2))
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
            // A larger graph's whole cost remains a conservative pricing prior.
            // It cannot prove that capturing this smaller work key is slower
            // than direct execution. Only the same work key prices payback.
            replay_s: self
                .cells
                .get(&Key(
                    family(stage, block, full, work),
                    work.rows,
                    true,
                    false,
                ))
                .map(Cell::mean),
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
        && !(point.replay && point.shared)
        && point.shared_setup_s.is_finite()
        && point.shared_setup_s >= 0.
        && point.shared_setup_s <= point.setup_s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cheap_profiled_prefix_waits_for_full_acceptance_calibration() {
        use super::super::{Budget, Feedback, History, Policy};
        let mut policy = Policy::default();
        for _ in 0..4 {
            policy.baseline(16, false, false, 0.08);
        }
        let state = Feedback {
            blocks: [(
                5,
                History {
                    previous: vec![0.01; 5],
                    ..Default::default()
                },
            )]
            .into(),
        };
        for (rows, seconds) in [(32, 0.005), (96, 0.1)] {
            for replay in [false, true] {
                policy.costs.insert(
                    Point {
                        work: Work {
                            context: 0,
                            ..point(rows, replay, seconds).work
                        },
                        ..point(rows, replay, seconds)
                    },
                    false,
                );
            }
        }
        for (stage, block) in [(Stage::Draft, 5), (Stage::Restore, 5)] {
            policy.costs.insert(
                Point {
                    stage,
                    block,
                    work: Work {
                        rows: 16,
                        ..Default::default()
                    },
                    ..point(16, false, 0.001)
                },
                false,
            );
        }
        let select = |p: &mut Policy| {
            p.select_requests(
                false,
                false,
                Budget {
                    rows: 96,
                    remaining: 256,
                    uniform: true,
                    work: Work {
                        rows: 16,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                &[5; 16],
                &[&state; 16],
                &[5],
            )
            .unwrap()
        };
        assert_eq!(select(&mut policy), [5; 16]);
        for _ in 0..crate::mtp::adaptive::CALIBRATION_ROUNDS {
            policy.observe(
                16,
                (false, false, true),
                [0.001, 0.1, 0.001, 0.],
                &[(5, 0); 16],
                5,
            );
        }
        let selected = select(&mut policy);
        assert!(selected.iter().any(|&depth| depth > 0 && depth < 5));
    }
    #[test]
    fn imported_timings_do_not_consume_the_runtime_primer() {
        let mut model = Model::default();
        let work = Work {
            rows: 96,
            context: 2048,
            ..Work::default()
        };
        model.insert(point(96, true, 0.29), false);
        assert!(model.profiled(Stage::Target, 0, false, work));
        assert!(!model.profiled(Stage::Target, 0, false, Work { rows: 80, ..work }));
        assert!(!model.profiled(
            Stage::Target,
            0,
            false,
            Work {
                context: 100256,
                ..work
            }
        ));
        assert!(
            !model
                .predict(Stage::Target, 0, false, work)
                .is_some_and(|p| p.replay)
        );
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (0, 0));
        assert!(
            (model
                .predict_mode(Stage::Target, 0, false, work, true)
                .unwrap()
                .seconds
                - 0.29)
                .abs()
                < 1e-12
        );
        model.observe(point(96, true, 0.4));
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (0, 1));
        model.observe(point(96, true, 0.29));
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (1, 2));
        assert!(
            (model
                .predict_mode(Stage::Target, 0, false, work, true)
                .unwrap()
                .seconds
                - 0.29)
                .abs()
                < 1e-12
        );
    }
    #[test]
    fn capture_primer_is_not_a_stable_replay_cost() {
        let mut model = Model::default();
        for (replay, seconds) in [(false, 0.45), (false, 0.44), (true, 0.4)] {
            model.observe(point(96, replay, seconds));
        }
        let work = Work {
            rows: 96,
            context: 2048,
            ..Work::default()
        };
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (1, 3));
        model.observe(point(96, true, 0.29));
        assert!(
            (model
                .predict_mode(Stage::Target, 0, false, work, true)
                .unwrap()
                .seconds
                - 0.29)
                .abs()
                < 1e-12
        );
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (2, 4));
    }
    #[test]
    fn distant_large_packets_are_priors_not_small_packet_calibration() {
        let mut model = Model::default();
        for _ in 0..5 {
            model.observe(point(96, false, 0.15));
        }
        let work = Work {
            rows: 6,
            context: 2048,
            ..Work::default()
        };
        let prior = model.predict(Stage::Target, 0, false, work).unwrap();
        assert!((prior.seconds - 0.15).abs() < 1e-12);
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (0, 0));
        assert_eq!(
            model.evidence(Stage::Target, 0, false, Work { rows: 48, ..work }),
            (4, 5)
        );
        assert_eq!(
            model.evidence(Stage::Target, 0, false, Work { rows: 47, ..work }),
            (0, 0)
        );
        for _ in 0..5 {
            model.observe(point(6, false, 0.02));
        }
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (4, 5));
    }
    #[test]
    fn old_round_samples_do_not_block_calibration_in_a_new_context() {
        use super::super::{Budget, Feedback, Policy};
        let mut policy = Policy::default();
        for _ in 0..5 {
            policy.baseline(1, false, false, 0.02);
            policy.observe(
                1,
                (false, false, false),
                [0.008, 0.02, 0.001, 0.],
                &[(5, 4)],
                5,
            );
            policy.costs.observe(Point {
                stage: Stage::Decode,
                ..point(1, false, 0.02)
            });
            policy.costs.observe(point(96, false, 0.15));
        }
        let work = Work {
            rows: 1,
            context: 2048,
            ..Work::default()
        };
        let state = Feedback::default();
        assert!(
            policy.curves[&(1, false, false)].rounds[&(5, 6, 1, false)]
                .time
                .samples()
                >= 3
        );
        assert_eq!(
            policy
                .costs
                .evidence(Stage::Target, 0, false, Work { rows: 6, ..work }),
            (0, 0)
        );
        let allocation = policy
            .select_requests_with_forecast(
                (false, false),
                Budget {
                    rows: 6,
                    remaining: 256,
                    work,
                    ..Budget::default()
                },
                &[5],
                &[&state],
                &[5],
                |depths, _| {
                    Ok(Some(Work {
                        rows: 1 + depths[0],
                        ..work
                    }))
                },
            )
            .unwrap();
        assert_eq!(allocation.depths, [5]);
    }
    #[test]
    fn calibration_combines_direct_and_replay_without_claiming_graph_residency() {
        let mut model = Model::default();
        for replay in [false, false, true, true] {
            model.observe(point(6, replay, if replay { 0.03 } else { 0.1 }));
        }
        let work = Work {
            rows: 5,
            context: 2048,
            ..Work::default()
        };
        assert_eq!(model.evidence(Stage::Target, 0, false, work), (2, 4));
        assert_eq!(
            model
                .evidence(Stage::Target, 0, false, Work { rows: 24, ..work })
                .0,
            0
        );
        let prediction = model.predict(Stage::Target, 0, false, work).unwrap();
        assert!(!prediction.replay);
        assert!((prediction.seconds - 0.1).abs() < 1e-12);
    }
    fn point(rows: usize, replay: bool, seconds: f64) -> Point {
        Point {
            shared: false,
            shared_setup_s: 0.,
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
    fn shared_execution_and_capture_cost_do_not_pollute_direct_or_full_modes() {
        let mut model = Model::default();
        let work = point(800, false, 0.).work;
        model.observe(point(800, false, 0.10));
        model.observe(Point {
            shared: true,
            shared_setup_s: 0.03,
            setup_s: 0.05,
            ..point(800, false, 0.07)
        });
        assert_eq!(
            model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds,
            0.07
        );
        assert!((model.cold_estimate(Stage::Target, 0, work, false).unwrap() - 0.02).abs() < 1e-12);
        model.observe(point(800, true, 0.05));
        assert_eq!(
            model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds,
            0.05
        );
        model.observe(point(800, false, 0.10));
        assert_eq!(
            model
                .predict(Stage::Target, 0, false, work)
                .unwrap()
                .seconds,
            0.10
        );
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
            let prior = model.predict_mode(Stage::Target, 0, false, work, true);
            if distant.rows >= 128 {
                assert!(prior.is_none());
            } else {
                assert_eq!(prior.unwrap().seconds, 0.51);
            }
            let budget = model.budget(Stage::Target, 0, false, work, 64., 1.);
            assert!(budget.replay_s.is_none());
            assert!(budget.profitable(0));
            assert!(
                !model
                    .budget(Stage::Target, 0, false, work, 5., 1.)
                    .profitable(0)
            );

            // Nearby rows still price execution, without denying a local probe.
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
            assert_eq!(budget.replay_s, None);
            assert!(budget.profitable(0));
        }
    }
    #[test]
    fn larger_draft_graph_does_not_block_a_smaller_local_capture() {
        let mut model = Model::default();
        let work = Work {
            rows: 16,
            ..Work::default()
        };
        for (rows, replay, seconds, setup_s) in [(16, false, 0.014, 0.), (32, true, 0.022, 0.04)] {
            model.observe(Point {
                shared: false,
                shared_setup_s: 0.,
                stage: Stage::Draft,
                block: 5,
                full: false,
                replay,
                work: Work { rows, ..work },
                seconds,
                setup_s,
            });
        }
        assert_eq!(
            model
                .predict_mode(Stage::Draft, 5, false, work, true)
                .unwrap()
                .seconds,
            0.022
        );
        let budget = model.budget(Stage::Draft, 5, false, work, 40., 0.4);
        assert_eq!(budget.replay_s, None);
        assert!(budget.profitable(0));
        assert!(
            !model
                .budget(Stage::Draft, 5, false, work, 4., 0.4)
                .profitable(0)
        );
        model.observe(Point {
            shared: false,
            shared_setup_s: 0.,
            stage: Stage::Draft,
            block: 5,
            full: false,
            replay: true,
            work,
            seconds: 0.018,
            setup_s: 0.04,
        });
        assert!(
            !model
                .budget(Stage::Draft, 5, false, work, 40., 0.4)
                .profitable(0)
        );
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
    fn verification_priors_stay_within_the_native_row_strategy() {
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
        assert!(
            model
                .predict_mode(
                    Stage::Target,
                    0,
                    false,
                    Work {
                        rows: 192,
                        context: 2048,
                        prefill_rows: 0,
                    },
                    true
                )
                .is_none()
        );
        for (rows, seconds) in [(128, 0.10), (160, 0.11)] {
            model.observe(point(rows, true, seconds));
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
        assert!((prediction.seconds - 0.12).abs() < 1e-12);
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
