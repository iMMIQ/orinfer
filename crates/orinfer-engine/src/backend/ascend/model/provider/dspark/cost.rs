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
    setup: f64,
}
#[derive(Default)]
pub(super) struct Model {
    cells: BTreeMap<Key, Cell>,
    modes: BTreeMap<(Family, usize), bool>,
    trace: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Prediction {
    pub seconds: f64,
    pub setup_s: f64,
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
            setup: 0.,
        });
        cell.time.observe(point.seconds);
        cell.setup = cell.setup.max(point.setup_s);
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
        // Select a context/prompt-work neighborhood before interpolating rows.
        let group = self
            .cells
            .keys()
            .filter(|Key(f, _, mode)| {
                f.0 == stage && f.1 == block && f.2 == full && f.3 == requested.3 && *mode == replay
            })
            .map(|Key(f, _, _)| *f)
            .min_by_key(|f| {
                (
                    f.4.ilog2().abs_diff(requested.4.ilog2()),
                    f.5.abs_diff(requested.5),
                )
            })?;
        let points = self
            .cells
            .iter()
            .filter(|(Key(f, _, mode), _)| *f == group && *mode == replay);
        let mut lower = None;
        let mut upper = None;
        for (Key(_, rows, _), cell) in points {
            if *rows <= work.rows {
                lower = Some((*rows, cell));
            }
            if *rows >= work.rows && upper.is_none() {
                upper = Some((*rows, cell));
            }
        }
        let value = |cell: &Cell| Prediction {
            seconds: cell.time.mean,
            setup_s: cell.setup,
        };
        match (lower, upper) {
            // Replay pays the captured tier; no invented savings within a tier.
            (_, Some((_, cell))) if replay => Some(value(cell)),
            (Some((lo, a)), Some((hi, b))) if lo < hi => {
                let fraction = (work.rows - lo) as f64 / (hi - lo) as f64;
                Some(Prediction {
                    seconds: a.time.mean + (b.time.mean - a.time.mean) * fraction,
                    setup_s: a.setup.max(b.setup),
                })
            }
            (Some((rows, cell)), _) => Some(Prediction {
                seconds: cell.time.mean * (work.rows as f64 / rows as f64).max(1.),
                setup_s: cell.setup,
            }),
            (_, Some((_, cell))) => Some(value(cell)),
            _ => None,
        }
    }
}
fn valid(point: &Point) -> bool {
    point.work.rows > 0
        && point.work.rows <= 65536
        && point.work.prefill_rows < point.work.rows
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
