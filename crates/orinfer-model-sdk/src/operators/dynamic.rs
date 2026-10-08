//! Data-only launch contracts for symbolic row kernels; no online compiler.
use crate::artifact::{Argument, Kernel, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum RowExpression {
    Constant { value: u32 },
    Rows,
    Add { lhs: Box<Self>, rhs: Box<Self> },
    Subtract { lhs: Box<Self>, rhs: Box<Self> },
    Multiply { lhs: Box<Self>, rhs: Box<Self> },
    Divide { lhs: Box<Self>, rhs: Box<Self> },
    Remainder { lhs: Box<Self>, rhs: Box<Self> },
    Minimum { lhs: Box<Self>, rhs: Box<Self> },
}
impl RowExpression {
    fn evaluate(&self, rows: u32, depth: usize) -> Result<u32> {
        if depth > 32 {
            return Err("Row expression exceeds nesting limit".into());
        }
        let binary = |lhs: &Self, rhs: &Self| -> Result<(u32, u32)> {
            Ok((
                lhs.evaluate(rows, depth + 1)?,
                rhs.evaluate(rows, depth + 1)?,
            ))
        };
        let value = match self {
            Self::Constant { value } => Some(*value),
            Self::Rows => Some(rows),
            Self::Add { lhs, rhs } => {
                let (a, b) = binary(lhs, rhs)?;
                a.checked_add(b)
            }
            Self::Subtract { lhs, rhs } => {
                let (a, b) = binary(lhs, rhs)?;
                a.checked_sub(b)
            }
            Self::Multiply { lhs, rhs } => {
                let (a, b) = binary(lhs, rhs)?;
                a.checked_mul(b)
            }
            Self::Divide { lhs, rhs } => {
                let (a, b) = binary(lhs, rhs)?;
                a.checked_div(b)
            }
            Self::Remainder { lhs, rhs } => {
                let (a, b) = binary(lhs, rhs)?;
                a.checked_rem(b)
            }
            Self::Minimum { lhs, rhs } => {
                let (a, b) = binary(lhs, rhs)?;
                Some(a.min(b))
            }
        };
        value.ok_or_else(|| "Invalid row expression arithmetic".into())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RowArgument {
    pub index: usize,
    pub value: RowExpression,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicBatchKernel {
    pub name: String,
    #[serde(default = "default_capacity")]
    pub capacity: usize,
    pub grid: [RowExpression; 3],
    pub arguments: Vec<RowArgument>,
}
fn default_capacity() -> usize {
    128
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RowLaunch {
    pub grid: [u32; 3],
    pub arguments: Vec<(usize, i32)>,
}
impl DynamicBatchKernel {
    pub fn launch(&self, rows: usize) -> Result<RowLaunch> {
        if !(2..=128).contains(&self.capacity) || !(2..=self.capacity).contains(&rows) {
            return Err("Dynamic batch rows exceed the kernel capacity".into());
        }
        let mut grid = [0; 3];
        for (target, expression) in grid.iter_mut().zip(&self.grid) {
            *target = expression.evaluate(rows as u32, 0)?;
            if *target == 0 {
                return Err("Dynamic grid cannot be zero".into());
            }
        }
        let arguments = self
            .arguments
            .iter()
            .map(|a| {
                let value = a.value.evaluate(rows as u32, 0)?;
                Ok((
                    a.index,
                    i32::try_from(value).map_err(|_| "Dynamic argument exceeds i32")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RowLaunch { grid, arguments })
    }
    pub fn validate(&self, kernel: &Kernel) -> Result<()> {
        let mut indices = std::collections::BTreeSet::new();
        if self.name != kernel.name
            || self.arguments.iter().any(|a| {
                !indices.insert(a.index)
                    || !matches!(kernel.args.get(a.index), Some(Argument::I32 { .. }))
            })
        {
            return Err("Invalid dynamic batch kernel ABI".into());
        }
        let capacity = self.launch(self.capacity)?;
        if capacity.grid != kernel.grid || capacity.arguments.iter().any(|(index, value)|
            !matches!(kernel.args[*index], Argument::I32 { value: v } if v == *value)) {
            return Err("Dynamic batch capacity binding differs from host ABI".into());
        }
        for rows in 2..self.capacity {
            self.launch(rows)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dynamic_contract_cannot_replace_pointers_or_change_capacity_abi() {
        let identity = serde_json::json!({"file":"kernel.cubin", "sha256":"0".repeat(64)});
        let kernel: Kernel = serde_json::from_value(serde_json::json!({
            "name":"batch_m128/layer0/k1", "module":identity, "source":identity,
            "host_abi":identity, "symbol":"test", "grid":[128,1,1], "block":[128,1,1],
            "shared_memory_bytes":0, "cooperative":false,
            "args":[{"kind":"buffer","name":"Input"},{"kind":"i32","value":128}]
        }))
        .unwrap();
        let mut template = DynamicBatchKernel {
            name: kernel.name.clone(),
            capacity: 128,
            grid: [
                RowExpression::Rows,
                RowExpression::Constant { value: 1 },
                RowExpression::Constant { value: 1 },
            ],
            arguments: vec![RowArgument {
                index: 1,
                value: RowExpression::Rows,
            }],
        };
        template.validate(&kernel).unwrap();
        let launch = template.launch(33).unwrap();
        assert_eq!(launch.grid, [33, 1, 1]);
        assert_eq!(launch.arguments, [(1, 33)]);
        assert!(template.launch(1).is_err());
        assert!(template.launch(129).is_err());
        template.arguments[0].index = 0;
        assert!(template.validate(&kernel).is_err());
        template.arguments[0].index = 1;
        template.arguments.push(template.arguments[0].clone());
        assert!(template.validate(&kernel).is_err());
        template.arguments.pop();
        template.grid[0] = RowExpression::Constant { value: 64 };
        assert!(template.validate(&kernel).is_err());
    }
    #[test]
    fn expressions_preserve_partial_tiles_and_reject_bad_arithmetic() {
        let e: RowExpression = serde_json::from_value(serde_json::json!({"op":"divide",
            "lhs":{"op":"add","lhs":{"op":"rows"},"rhs":{"op":"constant","value":31}},
            "rhs":{"op":"constant","value":32}}))
        .unwrap();
        for (rows, expected) in [(3, 1), (5, 1), (31, 1), (33, 2), (127, 4)] {
            assert_eq!(e.evaluate(rows, 0).unwrap(), expected);
        }
        let bad = RowExpression::Divide {
            lhs: Box::new(RowExpression::Rows),
            rhs: Box::new(RowExpression::Constant { value: 0 }),
        };
        assert!(bad.evaluate(3, 0).is_err());
        let overflow = RowExpression::Multiply {
            lhs: Box::new(RowExpression::Rows),
            rhs: Box::new(RowExpression::Constant { value: u32::MAX }),
        };
        assert!(overflow.evaluate(3, 0).is_err());
    }
}
