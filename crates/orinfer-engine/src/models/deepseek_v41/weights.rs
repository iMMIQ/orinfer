//! Small tensor transforms from the original checkpoint; W4 banks are prepared offline.
use crate::{
    artifact::{Dtype, Result},
    weights::checkpoint::Checkpoint,
};
use half::bf16;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Weight {
    Matrix {
        name: String,
        dtype: Dtype,
        first_row: usize,
        rows: usize,
        first_column: usize,
        columns: usize,
        #[serde(default)]
        float32: bool,
    },
    Pad {
        name: String,
        elements: usize,
    },
    ChannelProduct {
        left: String,
        right: String,
    },
}

fn dtype(d: Dtype) -> Result<safetensors::Dtype> {
    Ok(match d {
        Dtype::I8 => safetensors::Dtype::I8,
        Dtype::Bf16 => safetensors::Dtype::BF16,
        Dtype::F32 => safetensors::Dtype::F32,
        _ => return Err("Unsupported original checkpoint matrix dtype".into()),
    })
}
pub(super) fn load(checkpoint: &mut Checkpoint, source: &Weight) -> Result<Vec<u8>> {
    match source {
        Weight::Matrix {
            name,
            dtype: kind,
            first_row,
            rows,
            first_column,
            columns,
            float32,
        } => {
            let tensor = checkpoint.read(name)?;
            let row_width = if tensor.shape.len() == 1 {
                1
            } else if tensor.shape.len() == 2 {
                tensor.shape[1]
            } else {
                return Err(format!("{name}: expected a matrix"));
            };
            let height = tensor.shape[0];
            if tensor.dtype != dtype(*kind)?
                || first_row.checked_add(*rows).is_none_or(|x| x > height)
                || first_column
                    .checked_add(*columns)
                    .is_none_or(|x| x > row_width)
            {
                return Err(format!("{name}: checkpoint matrix slice/dtype differs"));
            }
            let bytes = kind.bytes();
            let mut output = Vec::with_capacity(rows * columns * if *float32 { 4 } else { bytes });
            for row in *first_row..first_row + rows {
                let begin = (row * row_width + first_column) * bytes;
                let data = &tensor.data[begin..begin + columns * bytes];
                if *float32 && *kind == Dtype::Bf16 {
                    for value in data.as_chunks::<2>().0.iter() {
                        output.extend_from_slice(
                            &bf16::from_bits(u16::from_le_bytes(*value))
                                .to_f32()
                                .to_le_bytes(),
                        );
                    }
                } else if *float32 && *kind != Dtype::F32 {
                    return Err("Unsupported checkpoint conversion".into());
                } else {
                    output.extend_from_slice(data);
                }
            }
            Ok(output)
        }
        Weight::Pad { name, elements } => {
            let tensor = checkpoint.read(name)?;
            if tensor.dtype != safetensors::Dtype::F32 || tensor.data.len() > elements * 4 {
                return Err(format!("{name}: invalid FP32 padding"));
            }
            let mut output = vec![0; elements * 4];
            output[..tensor.data.len()].copy_from_slice(tensor.data);
            Ok(output)
        }
        Weight::ChannelProduct { left, right } => {
            let left = checkpoint.read(left)?;
            if left.dtype != safetensors::Dtype::BF16 {
                return Err("Engram channel weight must be BF16".into());
            }
            let values = left
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|x| bf16::from_bits(u16::from_le_bytes(*x)).to_f32())
                .collect::<Vec<_>>();
            let right = checkpoint.read(right)?;
            if right.dtype != safetensors::Dtype::BF16 || right.data.len() != values.len() * 2 {
                return Err("Engram channel weight extents differ".into());
            }
            Ok(values
                .iter()
                .zip(right.data.as_chunks::<2>().0.iter())
                .flat_map(|(a, b)| {
                    (a * bf16::from_bits(u16::from_le_bytes(*b)).to_f32()).to_le_bytes()
                })
                .collect())
        }
    }
}
