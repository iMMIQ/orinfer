//! Model semantics and execution composition, independent of precision selection.
use orinfer_model_sdk::{abi, architecture::BatchSegment, artifact::Result};

pub(crate) trait Adapter {
    fn prepare(
        &self,
        _program: &str,
        _tokens: &[u32],
        _history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        Ok(vec![])
    }
    fn prefetch_inputs(&self, _tokens: &[u32], _history: &[u32]) {}
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<abi::BatchPlan>;
    fn batch_operation(
        &self,
        segments: &[BatchSegment],
        operation: abi::BatchOperation,
        include_plan: bool,
    ) -> Result<abi::BatchPlan> {
        if operation != abi::BatchOperation::Ordinary {
            return Err("Model adapter does not provide this batch operation".into());
        }
        self.batch(segments, include_plan)
    }
    fn visual(
        &self,
        _tokens: &[u32],
        _images: &[abi::ImageGrid],
        _capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        Err("Model package has no visual layout adapter".into())
    }
}
