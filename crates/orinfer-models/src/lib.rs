//! Native model implementations built with the engine's common source workspace.
pub use orinfer_model_sdk::{artifact, execution, model, operators};
mod adapter;
mod flash_next;
mod nemotron_embed;
mod packet;
mod prefill;
mod qwen3_5;
mod qwen_reranker;
mod registry;
mod request;
mod sampling;
mod vision;

pub use adapter::Adapter;
use orinfer_model_sdk::catalog::Bindings;
pub use request::{CreateRequest, CreatedPlan};

/// Compose a registered model using the execution package's typed bindings.
pub fn create(
    mut request: CreateRequest,
    bindings: Bindings,
) -> artifact::Result<(Box<dyn Adapter>, CreatedPlan)> {
    request.operands = packet::Operands::from_kernels(&bindings.kernels);
    request.kernel_contracts = bindings
        .kernels
        .iter()
        .filter(|kernel| request.expert_contracts || bindings.plan_contracts.contains(&kernel.name))
        .cloned()
        .collect();
    request.metadata.dynamic_batch_kernels = bindings.dynamic;
    let build_bindings = request.build_bindings;
    let sampling_bindings = request
        .kernel_contracts
        .iter()
        .map(|k| k.name.clone())
        .collect();
    let (adapter, mut plan) = registry::create(request)?;
    sampling::register(&mut plan.metadata, &sampling_bindings)?;
    let metadata = &mut plan.metadata;
    let mut used = model::execution_kernels(&metadata.programs, metadata.row_execution.as_ref());
    for bank in &plan.expert_banks {
        used.extend(bank.consumers.keys().cloned());
    }
    metadata.kernels = bindings
        .kernels
        .into_iter()
        .filter(|kernel| build_bindings || used.contains(&kernel.name))
        .collect();
    metadata
        .dynamic_batch_kernels
        .retain(|name, _| build_bindings || used.contains(name));
    Ok((adapter, plan))
}
