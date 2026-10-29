//! Checked row arithmetic shared by A2 kernel, view and collective bindings.
use super::View;
use crate::artifact::Result;
use orinfer_model_sdk::operators::dynamic::RowExpression;

/// Keep ordinary one-row decode resident while bounding dynamic Graph instances.
pub(super) struct PacketLru {
    capacity: usize,
    order: std::collections::VecDeque<usize>,
}
impl PacketLru {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: std::collections::VecDeque::new(),
        }
    }
    pub(super) fn touch(&mut self, rows: usize) -> Option<usize> {
        if rows == 1 {
            return None;
        }
        if let Some(index) = self.order.iter().position(|r| *r == rows) {
            self.order.remove(index);
        }
        self.order.push_back(rows);
        (self.order.len() + 1 > self.capacity).then(|| self.order.pop_front().unwrap())
    }
}

pub(super) fn validate_replacement(
    source: &super::super::kernels::Spec,
    replacement: &super::super::kernels::Spec,
) -> Result<()> {
    let arithmetic = |spec: &super::super::kernels::Spec| {
        spec.parameters
            .iter()
            .filter(|(key, _)| {
                !matches!(
                    key.as_str(),
                    "tile_m" | "tile_n" | "block_k" | "pipeline_stages" | "_grid_limit"
                )
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let target = |spec: &super::super::kernels::Spec| {
        serde_json::to_value(&spec.target).map_err(|e| e.to_string())
    };
    if source.op != "dense.projection"
        || replacement.op != source.op
        || target(source)? != target(replacement)?
        || source.source_sha256 != replacement.source_sha256
        || source.compiler != replacement.compiler
        || arithmetic(source) != arithmetic(replacement)
        || source.operands.len() != replacement.operands.len()
        || source
            .operands
            .iter()
            .zip(&replacement.operands)
            .any(|(a, b)| {
                a.name != b.name
                    || a.dtype != b.dtype
                    || a.access != b.access
                    || a.access != super::super::kernels::Access::Scratch && a.shape != b.shape
            })
    {
        return Err("A2 local replacement changes arithmetic or physical operand contract".into());
    }
    Ok(())
}

pub(super) fn kernel_rows(
    actual: usize,
    capacity: usize,
    fixed: Option<usize>,
    expression: Option<&RowExpression>,
) -> Result<usize> {
    if actual == 0 || actual > capacity || fixed.is_some() && expression.is_some() {
        return Err("Invalid or ambiguous kernel row binding".into());
    }
    let bound = fixed.unwrap_or(expression.map_or(Ok(capacity), |e| e.resolve(capacity))?);
    let rows = fixed.unwrap_or(expression.map_or(Ok(actual), |e| e.resolve(actual))?);
    if bound == 0 || bound > capacity || rows == 0 || rows > bound {
        return Err("Kernel row expression exceeds its capacity".into());
    }
    Ok(rows)
}

pub(super) fn elements(
    count: usize,
    actual: usize,
    capacity: usize,
    fixed: bool,
    expression: Option<&RowExpression>,
) -> Result<usize> {
    if count == 0 || actual == 0 || actual > capacity || fixed && expression.is_some() {
        return Err("Invalid or ambiguous collective row binding".into());
    }
    if let Some(expression) = expression {
        let live = expression.resolve(actual)?;
        if expression.resolve(capacity)? != count || live == 0 || live > count {
            return Err("Collective expression differs from its capacity binding".into());
        }
        Ok(live)
    } else if fixed {
        Ok(count)
    } else {
        if !count.is_multiple_of(capacity) {
            return Err("Collective expression differs from packet capacity".into());
        }
        (count / capacity)
            .checked_mul(actual)
            .ok_or_else(|| "Collective extent overflow".into())
    }
}

pub(super) fn view_offset(view: &View, actual: usize) -> Result<usize> {
    match (&view.offset_expression, view.row_capacity) {
        (None, None) => Ok(view.offset),
        (Some(expression), Some(capacity)) if (2..=4096).contains(&capacity) => {
            if actual == 0 || actual > capacity || expression.resolve(capacity)? != view.offset {
                return Err("View expression differs from its capacity binding".into());
            }
            expression.resolve(actual)
        }
        _ => Err("Incomplete dynamic view contract".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph_churn_keeps_decode_and_recent_rows() {
        let mut cache = PacketLru::new(4);
        for row in [1, 17, 513, 1025, 17, 1] {
            assert_eq!(cache.touch(row), None);
        }
        assert_eq!(cache.touch(2048), Some(513));
        assert_eq!(cache.touch(513), Some(1025));
        assert_eq!(cache.touch(17), None);
    }
    fn expression(source: serde_json::Value) -> RowExpression {
        serde_json::from_value(source).unwrap()
    }
    fn partition() -> RowExpression {
        expression(serde_json::json!({"op":"divide", "lhs":{"op":"add",
            "lhs":{"op":"rows"},"rhs":{"op":"constant","value":7}},
            "rhs":{"op":"constant","value":8}}))
    }
    #[test]
    fn tp8_views_and_transport_cover_every_tail_without_capacity_padding() {
        let partition = partition();
        for actual in 1..=2048usize {
            let local = kernel_rows(actual, 2048, None, Some(&partition)).unwrap();
            assert_eq!(local, actual.div_ceil(8));
            assert!(local * 8 >= actual && local * 8 - actual < 8);
            for rank in 0..8 {
                let offset = RowExpression::Multiply {
                    lhs: Box::new(partition.clone()),
                    rhs: Box::new(RowExpression::Constant {
                        value: rank * 5120 * 2,
                    }),
                };
                let view = View {
                    parent: "global".into(),
                    offset: rank as usize * 256 * 5120 * 2,
                    row_capacity: Some(2048),
                    offset_expression: Some(offset),
                };
                assert_eq!(
                    view_offset(&view, actual).unwrap(),
                    rank as usize * local * 5120 * 2
                );
                assert!(view_offset(&view, actual).unwrap() + 256 * 5120 * 2 <= 2048 * 5120 * 2);
            }
        }
        let count = RowExpression::Multiply {
            lhs: Box::new(partition.clone()),
            rhs: Box::new(RowExpression::Constant { value: 5120 }),
        };
        assert_eq!(
            elements(256 * 5120, 513, 2048, false, Some(&count)).unwrap(),
            65 * 5120
        );
        assert!(elements(255 * 5120, 513, 2048, false, Some(&count)).is_err());
        assert!(elements(256 * 5120, 513, 2048, true, Some(&count)).is_err());
        assert!(kernel_rows(513, 2048, Some(256), Some(&partition)).is_err());
        assert!(kernel_rows(0, 2048, None, Some(&partition)).is_err());
    }
}
