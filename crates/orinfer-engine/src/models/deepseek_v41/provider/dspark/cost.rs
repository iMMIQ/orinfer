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
    time: crate::scheduler::Timing,
    prior: f64,
    last: usize,
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
    last: usize,
}
#[derive(Default)]
pub(super) struct Model {
    cells: BTreeMap<Key, Cell>,
    modes: BTreeMap<(Family, usize), (bool, usize)>,
    shared_modes: std::collections::BTreeSet<(Family, usize)>,
    trace: bool,
    cold: BTreeMap<(Family, usize), Cold>,
    tick: usize,
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
        self.tick = self.tick.saturating_add(1);
        let tick = self.tick;
        let group = family(point.stage, point.block, point.full, point.work);
        let key = Key(group, point.work.rows, point.replay, point.shared);
        let mode_key = (group, point.work.rows);
        // Offline priors describe costs, never residency in a live Graph cache.
        if online {
            if self.modes.len() >= 2048
                && !self.modes.contains_key(&mode_key)
                && let Some((&victim, _)) = self.modes.iter().min_by_key(|(_, (_, last))| *last)
            {
                self.modes.remove(&victim);
                self.shared_modes.remove(&victim);
            }
            self.modes.insert(mode_key, (point.replay, tick));
            if point.shared {
                self.shared_modes.insert(mode_key);
            } else {
                self.shared_modes.remove(&mode_key);
            }
        }
        crate::mtp::adaptive::evict_oldest(&mut self.cells, key, 2048, |cell| cell.last);
        let cell = self.cells.entry(key).or_insert(Cell {
            time: Default::default(),
            prior: 0.,
            last: tick,
        });
        cell.last = tick;
        if online {
            cell.time.observe(point.seconds);
        } else {
            // Imported timings price execution; only this process's real
            // calls may consume its primer and establish hot observations.
            cell.prior = if cell.prior > 0. {
                crate::mtp::adaptive::ema(cell.prior, point.seconds)
            } else {
                point.seconds
            };
        }
        let local_setup = point.setup_s - point.shared_setup_s;
        if local_setup > 0. {
            // Host readback mode does not change the captured launch geometry.
            let cold_group = family(point.stage, point.block, false, point.work);
            let cold_key = (cold_group, point.work.rows);
            crate::mtp::adaptive::evict_oldest(&mut self.cold, cold_key, 2048, |c| c.last);
            let cold = self.cold.entry(cold_key).or_default();
            cold.last = tick;
            if point.replay {
                cold.capture.observe(local_setup);
            } else {
                cold.build.observe(local_setup);
            }
        }
    }
    pub fn cell_count(&self) -> usize {
        self.cells.len()
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
            .map(|&(replay, _)| replay)
            .unwrap_or(false);
        self.predict_mode(stage, block, full, work, replay)
    }
    pub fn profiled(&self, stage: Stage, block: usize, full: bool, work: Work) -> bool {
        let group = family(stage, block, full, work);
        self.cells
            .iter()
            .any(|(Key(f, rows, _, _), cell)| *f == group && *rows == work.rows && cell.prior > 0.)
    }
    /// Actual local observations; LRU eviction keeps the bounded surface able
    /// to learn new shapes even after filling 2048 entries.
    pub(super) fn local_evidence(
        &self,
        stage: Stage,
        block: usize,
        full: bool,
        work: Work,
    ) -> Option<(usize, usize)> {
        let group = family(stage, block, full, work);
        Some(
            self.cells
                .range(Key(group, work.rows, false, false)..=Key(group, work.rows, true, true))
                .fold((0, 0), |(hot, seen), (_, cell)| {
                    (hot + cell.time.samples(), seen + cell.time.observations())
                }),
        )
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
            max_uses: None,
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
        && crate::mtp::adaptive::valid_duration(point.seconds, point.setup_s)
        && !(point.replay && point.shared)
        && (0. ..=point.setup_s).contains(&point.shared_setup_s)
}

#[cfg(test)]
#[path = "cost_tests.rs"]
mod tests;
