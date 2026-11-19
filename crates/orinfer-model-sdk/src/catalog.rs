//! Compiled implementations are independent of tensor roles and model layers.
use crate::{
    artifact::{Argument, FileIdentity, Kernel, Result},
    operators::dynamic::{DynamicBatchKernel, RowArgument, RowExpression},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    Buffer,
    BufferSlice,
    I32,
    U32,
    I64,
    U64,
    F32,
}
impl ParameterKind {
    pub fn of(argument: &Argument) -> Self {
        match argument {
            Argument::Buffer { .. } => Self::Buffer,
            Argument::BufferSlice { .. } => Self::BufferSlice,
            Argument::I32 { .. } => Self::I32,
            Argument::U32 { .. } => Self::U32,
            Argument::I64 { .. } => Self::I64,
            Argument::U64 { .. } => Self::U64,
            Argument::F32 { .. } => Self::F32,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<crate::artifact::Launch>,
    pub module: usize,
    pub source: usize,
    pub host_abi: usize,
    pub symbol: String,
    #[serde(default)]
    pub block: [u32; 3],
    #[serde(default)]
    pub shared_memory_bytes: u32,
    #[serde(default)]
    pub cooperative: bool,
    pub parameters: Vec<ParameterKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub readonly_arguments: Vec<bool>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicDefinition {
    pub capacity: usize,
    pub grid: [RowExpression; 3],
    pub arguments: Vec<RowArgument>,
}
impl DynamicDefinition {
    pub fn bind(&self, name: &str) -> DynamicBatchKernel {
        DynamicBatchKernel {
            name: name.into(),
            capacity: self.capacity,
            grid: self.grid.clone(),
            arguments: self.arguments.clone(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub assets: Vec<FileIdentity>,
    pub definitions: Vec<Definition>,
    pub dynamic: Vec<DynamicDefinition>,
}
impl Catalog {
    pub fn validate(&self) -> Result<()> {
        if self.definitions.is_empty() {
            return Err("Empty kernel catalog".into());
        }
        for d in &self.definitions {
            if [d.module, d.source, d.host_abi]
                .iter()
                .any(|&i| i >= self.assets.len())
                || d.symbol.is_empty()
                || d.parameters.len() > 256
                || (!d.readonly_arguments.is_empty()
                    && d.readonly_arguments.len() != d.parameters.len())
            {
                return Err("Invalid compiled kernel definition".into());
            }
            crate::artifact::validate_geometry(
                d.launch.as_ref(),
                None,
                d.block,
                d.shared_memory_bytes,
                d.cooperative,
            )?;
        }
        Ok(())
    }
    pub fn validate_target(&self, target: &orinfer_device::Target) -> Result<()> {
        self.validate()?;
        target.validate()?;
        for definition in &self.definitions {
            let backend = definition.launch.as_ref().map_or(
                orinfer_device::Backend::Cuda,
                crate::artifact::Launch::backend,
            );
            if backend != target.backend {
                return Err("Kernel catalog backend differs from package target".into());
            }
        }
        Ok(())
    }
    pub fn bind(
        &self,
        definition: usize,
        name: String,
        grid: [u32; 3],
        args: Vec<Argument>,
    ) -> Result<Kernel> {
        let d = self
            .definitions
            .get(definition)
            .ok_or("Unknown compiled kernel definition")?;
        crate::artifact::validate_geometry(
            d.launch.as_ref(),
            Some(grid),
            d.block,
            d.shared_memory_bytes,
            d.cooperative,
        )?;
        if d.parameters.len() != args.len()
            || d.parameters
                .iter()
                .zip(&args)
                .any(|(&kind, a)| kind != ParameterKind::of(a))
        {
            return Err(format!(
                "{name}: native binding differs from compiled parameter ABI"
            ));
        }
        let asset = |i| {
            self.assets
                .get(i)
                .cloned()
                .ok_or_else(|| "Unknown kernel asset".to_owned())
        };
        Ok(Kernel {
            launch: d.launch.clone(),
            name,
            grid,
            args,
            module: asset(d.module)?,
            source: asset(d.source)?,
            host_abi: asset(d.host_abi)?,
            symbol: d.symbol.clone(),
            block: d.block,
            shared_memory_bytes: d.shared_memory_bytes,
            cooperative: d.cooperative,
            readonly_arguments: d.readonly_arguments.clone(),
        })
    }
}

/// Offline-factored tensor roles, parameters and layer-local launch choices.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BindingRegistry {
    pub roles: Vec<String>,
    pub parameters: Vec<Vec<RoleArgument>>,
    pub choices: Vec<Vec<BindingVariant>>,
    pub templates: Vec<BindingTemplate>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleArgument {
    Buffer(usize),
    BufferSlice(usize, usize),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(u32),
}
impl RoleArgument {
    fn bind(&self, roles: &[String]) -> Result<Argument> {
        let role = |i: usize| {
            roles
                .get(i)
                .cloned()
                .ok_or_else(|| "Unknown binding tensor role".to_owned())
        };
        Ok(match *self {
            Self::Buffer(i) => Argument::Buffer { name: role(i)? },
            Self::BufferSlice(i, offset) => Argument::BufferSlice {
                name: role(i)?,
                offset,
            },
            Self::I32(value) => Argument::I32 { value },
            Self::U32(value) => Argument::U32 { value },
            Self::I64(value) => Argument::I64 { value },
            Self::U64(value) => Argument::U64 { value },
            Self::F32(bits) => Argument::F32 {
                value: f32::from_bits(bits),
            },
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BindingVariant {
    pub suffix: String,
    pub definition: usize,
    pub grid: [u32; 3],
    pub arguments: Vec<usize>,
    pub dynamic: Option<usize>,
    pub plan_contract: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayerBinding {
    pub layer: Option<usize>,
    pub parameters: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BindingTemplate {
    pub prefix: String,
    pub suffix: String,
    pub choice: usize,
    pub layers: Vec<LayerBinding>,
}

pub struct Bindings {
    pub kernels: Vec<Kernel>,
    pub dynamic: std::collections::BTreeMap<String, DynamicBatchKernel>,
    pub plan_contracts: std::collections::BTreeSet<String>,
}

impl BindingRegistry {
    /// Resolve data references and verify the ABI at the published row extent.
    pub fn bind(&self, catalog: &Catalog) -> Result<Bindings> {
        let mut kernels = std::collections::BTreeMap::new();
        let mut dynamic = std::collections::BTreeMap::new();
        let mut plan_contracts = std::collections::BTreeSet::new();
        for template in &self.templates {
            let choices = self
                .choices
                .get(template.choice)
                .filter(|choices| !choices.is_empty())
                .ok_or("Missing binding specialization table")?;
            if template.layers.is_empty() {
                return Err("Empty binding template".into());
            }
            for layer in &template.layers {
                let parameters = self
                    .parameters
                    .get(layer.parameters)
                    .ok_or("Unknown binding layer parameters")?;
                let layer_name = layer.layer.map(|i| i.to_string()).unwrap_or_default();
                for choice in choices {
                    let name = format!(
                        "{}{layer_name}{}{}",
                        template.prefix, template.suffix, choice.suffix
                    );
                    let args = choice
                        .arguments
                        .iter()
                        .map(|&index| {
                            parameters
                                .get(index)
                                .ok_or("Unknown binding parameter slot")?
                                .bind(&self.roles)
                        })
                        .collect::<Result<_>>()?;
                    let kernel =
                        catalog.bind(choice.definition, name.clone(), choice.grid, args)?;
                    if let Some(index) = choice.dynamic {
                        let launch = catalog
                            .dynamic
                            .get(index)
                            .ok_or("Unknown dynamic launch definition")?
                            .bind(&name);
                        launch.validate_binding(&kernel)?;
                        dynamic.insert(name.clone(), launch);
                    }
                    if choice.plan_contract {
                        plan_contracts.insert(name.clone());
                    }
                    if name.is_empty() || kernels.insert(name, kernel).is_some() {
                        return Err("Empty or duplicate kernel binding name".into());
                    }
                }
            }
        }
        if kernels.is_empty() {
            return Err("Empty binding registry".into());
        }
        Ok(Bindings {
            kernels: kernels.into_values().collect(),
            dynamic,
            plan_contracts,
        })
    }
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn native_ascend_binding_retains_launch_and_checks_package_target() {
        let asset = FileIdentity {
            file: "kernel.bin".into(),
            sha256: "0".repeat(64),
            tensor: None,
        };
        let catalog = Catalog {
            assets: vec![asset],
            dynamic: vec![],
            definitions: vec![Definition {
                launch: Some(crate::artifact::Launch::Ascend {
                    block_dim: 24,
                    workspace_bytes: 8192,
                }),
                module: 0,
                source: 0,
                host_abi: 0,
                symbol: "projection".into(),
                block: [0; 3],
                shared_memory_bytes: 0,
                cooperative: false,
                parameters: vec![ParameterKind::U32],
                readonly_arguments: vec![],
            }],
        };
        let target = orinfer_device::Target::from_package("ascend910b").unwrap();
        catalog.validate_target(&target).unwrap();
        assert!(
            catalog
                .validate_target(&orinfer_device::Target::from_package("sm_87").unwrap())
                .is_err()
        );
        let kernel = catalog
            .bind(
                0,
                "projection".into(),
                [0; 3],
                vec![Argument::U32 { value: 16 }],
            )
            .unwrap();
        kernel.validate_launch(&target).unwrap();
        assert!(matches!(
            kernel.launch,
            Some(crate::artifact::Launch::Ascend {
                block_dim: 24,
                workspace_bytes: 8192
            })
        ));
        assert!(
            catalog
                .bind(
                    0,
                    "projection".into(),
                    [1; 3],
                    vec![Argument::U32 { value: 16 }]
                )
                .is_err()
        );
        assert!(
            catalog
                .bind(
                    0,
                    "projection".into(),
                    [0; 3],
                    vec![Argument::I32 { value: 16 }]
                )
                .is_err()
        );
    }

    #[test]
    fn data_bindings_preserve_layer_choices_offsets_and_scalar_bits() {
        use serde_json::{Value, json};
        let registry = json!({
            "roles": ["Weight0","Weight1","State0","State1"],
            "parameters": [
                [{"buffer":0},{"buffer_slice":[2,64]},{"i32":-7},{"f32":2147483648u32}],
                [{"buffer":1},{"buffer_slice":[3,128]},{"i32":-7},{"f32":2147483648u32}]
            ],
            "choices": [[
                {"suffix":"","definition":0,"grid":[1024,1,1],"arguments":[0,1,2,3],"dynamic":null,"plan_contract":true},
                {"suffix":"/rows/1-1","definition":1,"grid":[1,1,1],"arguments":[1,0,3],"dynamic":null,"plan_contract":false}
            ]],
            "templates": [{
                "prefix":"target/layer","suffix":"/projection","choice":0,
                "layers":[{"layer":0,"parameters":0},{"layer":1,"parameters":1}]
            }]
        });
        let catalog: Catalog = serde_json::from_value(json!({
            "assets":[{"file":"assets.safetensors","tensor":"fixture","sha256":"0".repeat(64)}],
            "definitions":[
                {"module":0,"source":0,"host_abi":0,"symbol":"projection","block":[32,1,1],
                 "parameters":["buffer","buffer_slice","i32","f32"]},
                {"module":0,"source":0,"host_abi":0,"symbol":"projection_one","block":[32,1,1],
                 "parameters":["buffer_slice","buffer","f32"]}
            ],
            "dynamic":[]
        }))
        .unwrap();
        let bind = |data: Value| {
            serde_json::from_value::<BindingRegistry>(data)
                .unwrap()
                .bind(&catalog)
        };
        let result = bind(registry.clone()).unwrap();
        assert_eq!(result.kernels.len(), 4);
        assert_eq!(result.plan_contracts.len(), 2);
        let binding = result
            .kernels
            .iter()
            .find(|k| k.name == "target/layer1/projection/rows/1-1")
            .unwrap();
        assert!(
            matches!(&binding.args[0], Argument::BufferSlice { name, offset: 128 } if name == "State1")
        );
        assert!(matches!(&binding.args[1], Argument::Buffer { name } if name == "Weight1"));
        assert!(
            matches!(&binding.args[2], Argument::F32 { value } if value.to_bits() == 0x80000000)
        );
        for (path, invalid) in [
            ("/templates/0/choice", json!(2)),
            ("/templates/0/layers/1/layer", json!(0)),
            ("/parameters/1/0/buffer", json!(4)),
            ("/choices/0/1/arguments/0", json!(9)),
            ("/choices/0/1/dynamic", json!(0)),
        ] {
            let mut data = registry.clone();
            *data.pointer_mut(path).unwrap() = invalid;
            assert!(bind(data).is_err(), "{path}");
        }
    }
}
