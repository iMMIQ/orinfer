//! One reclamation order for owner admission and optional prefix snapshots.
use super::*;
impl memory::Pressure for Runtime {
    fn fits(&mut self, requirement: memory::Requirement) -> Result<bool> {
        self.team.has_budget(requirement)
    }
    fn evict_prefix(&mut self) -> Result<bool> {
        self.evict()
    }
    fn evict_program(&mut self) -> Result<bool> {
        self.team.reclaim_programs()
    }
}
