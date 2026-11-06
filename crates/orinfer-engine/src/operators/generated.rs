//! Exact derived installation assets, shared independently of the device backend.
//! A deterministic portable approximation plus an offline XOR correction stream
//! reproduces the original FP32 bytes across libm/CPU versions. Inference reads
//! ordinary installed files and never generates or hashes these assets on load.
use crate::artifact::{FileIdentity, Result, read_identity};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    Sin,
    Cos,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub output: FileIdentity,
    pub recipe: Recipe,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Recipe {
    RopeF32V1 {
        rows: usize,
        frequencies: Vec<u32>,
        component: ComponentKind,
        corrections: FileIdentity,
    },
}
#[derive(Serialize)]
pub struct EncodedRope {
    rows: usize,
    frequencies: Vec<u32>,
    cos_corrections: PathBuf,
    sin_corrections: PathBuf,
}

// These are codec arithmetic, not a replacement inference RoPE implementation.
// Keep each operation explicit: neither fused multiply-add nor host libm is used.
fn sin_cos(phase: f32) -> (f32, f32) {
    let x = f64::from(phase);
    let n = (x * std::f64::consts::FRAC_2_PI + 0.5) as u64;
    let r = (x - n as f64 * f64::from_bits(0x3ff9_21fb_5440_0000))
        - n as f64 * 6.077_100_506_506_192e-11;
    let z = r * r;
    let s = r + r
        * z
        * (-1.0 / 6.0
            + z * (1.0 / 120.0
                + z * (-1.0 / 5040.0
                    + z * (1.0 / 362880.0
                        + z * (-1.0 / 39916800.0
                            + z * (1.0 / 6227020800.0
                                + z * (-1.0 / 1307674368000.0 + z / 355687428096000.0)))))));
    let c = 1.0
        + z * (-1.0 / 2.0
            + z * (1.0 / 24.0
                + z * (-1.0 / 720.0
                    + z * (1.0 / 40320.0
                        + z * (-1.0 / 3628800.0
                            + z * (1.0 / 479001600.0
                                + z * (-1.0 / 87178291200.0 + z / 20922789888000.0)))))));
    let (s, c) = match n % 4 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    };
    (s as f32, c as f32)
}
fn dimensions(rows: usize, frequencies: &[u32]) -> Result<usize> {
    if !(2..=262144).contains(&rows)
        || frequencies.is_empty()
        || frequencies.len() > 256
        || frequencies.iter().any(|&f| {
            let x = f32::from_bits(f);
            !x.is_finite() || x <= 0.0 || x > 1.000001
        })
    {
        return Err("Invalid generated RoPE geometry/frequencies".into());
    }
    rows.checked_mul(frequencies.len())
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "Generated RoPE size overflow".into())
}
fn relative(path: &str) -> Result<&Path> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|p| !matches!(p, Component::Normal(_)))
    {
        return Err("Generated asset requires a relative normal path".into());
    }
    Ok(path)
}
impl Asset {
    pub(crate) fn validate(&self) -> Result<()> {
        relative(&self.output.file)?;
        if self.output.tensor.is_some() {
            return Err("Generated output must be a standalone file".into());
        }
        let Recipe::RopeF32V1 {
            rows,
            frequencies,
            corrections,
            ..
        } = &self.recipe;
        dimensions(*rows, frequencies)?;
        relative(&corrections.file)?;
        if corrections.tensor.is_some() || corrections.file == self.output.file {
            return Err("Invalid generated correction identity".into());
        }
        Ok(())
    }
    pub(crate) fn verify_recipe(&self, root: &Path) -> Result<()> {
        self.validate()?;
        let Recipe::RopeF32V1 { corrections, .. } = &self.recipe;
        read_identity(root, corrections)?;
        Ok(())
    }
}
fn generate(
    rows: usize,
    frequencies: &[u32],
    component: ComponentKind,
    mut patch: impl Read,
    mut output: impl Write,
) -> Result<()> {
    dimensions(rows, frequencies)?;
    let mut row = vec![0u8; frequencies.len() * 4];
    for position in 0..rows {
        patch
            .read_exact(&mut row)
            .map_err(|e| format!("Truncated RoPE correction: {e}"))?;
        for (bytes, &frequency) in row.as_chunks_mut::<4>().0.iter_mut().zip(frequencies) {
            let phase = position as f32 * f32::from_bits(frequency);
            let (s, c) = sin_cos(phase);
            let value = match component {
                ComponentKind::Sin => s,
                ComponentKind::Cos => c,
            };
            let bits = value.to_bits() ^ u32::from_le_bytes(*bytes);
            bytes.copy_from_slice(&bits.to_le_bytes());
        }
        output.write_all(&row).map_err(|e| e.to_string())?;
    }
    let mut extra = [0];
    if patch.read(&mut extra).map_err(|e| e.to_string())? != 0 {
        return Err("Trailing RoPE correction bytes".into());
    }
    Ok(())
}
/// Materialize only inside the installer's private, regular-file staging tree.
pub(crate) fn materialize(root: &Path, assets: &[Asset]) -> Result<()> {
    if assets.len() > 64 {
        return Err("Too many generated assets".into());
    }
    let mut outputs = BTreeSet::new();
    for asset in assets {
        asset.validate()?;
        if !outputs.insert(&asset.output.file) {
            return Err("Duplicate generated asset output".into());
        }
    }
    for asset in assets {
        let Recipe::RopeF32V1 {
            rows,
            frequencies,
            component,
            corrections,
        } = &asset.recipe;
        let compressed = read_identity(root, corrections)?;
        let patch = lzma_rust2::XzReader::new_mem_limit(compressed.as_slice(), true, 256 * 1024);
        let target = root.join(relative(&asset.output.file)?);
        let parent = target.parent().ok_or("Generated asset has no parent")?;
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        if !parent
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(root)
        {
            return Err("Generated asset leaves package directory".into());
        }
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(&target)
            .map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        generate(*rows, frequencies, *component, patch, &mut writer)?;
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Offline encoding writes raw XOR streams; Python publication compresses them.
/// Frequency inference is offline only; corrections preserve every original bit.
pub fn encode_rope(cos: &Path, sin: &Path, columns: usize, output: &Path) -> Result<EncodedRope> {
    let read = |path: &Path| -> Result<Vec<u32>> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let bytes = file.metadata().map_err(|e| e.to_string())?.len();
        if bytes > 256 * 1024 * 1024 || bytes % 4 != 0 {
            return Err("Invalid RoPE input size".into());
        }
        let mut input = BufReader::new(file);
        let mut values = Vec::with_capacity(bytes as usize / 4);
        for _ in 0..bytes / 4 {
            let mut b = [0; 4];
            input.read_exact(&mut b).map_err(|e| e.to_string())?;
            values.push(u32::from_le_bytes(b));
        }
        Ok(values)
    };
    let cos = read(cos)?;
    let sin = read(sin)?;
    if columns == 0 || columns > 256 || cos.len() != sin.len() || !cos.len().is_multiple_of(columns)
    {
        return Err("RoPE cosine/sine geometry differs".into());
    }
    let rows = cos.len() / columns;
    if !(128..=262144).contains(&rows) {
        return Err("RoPE encoding requires 128..262144 positions".into());
    }
    let points: BTreeSet<_> = (0..512)
        .map(|i| i * (rows - 1) / 511)
        .chain(0..32)
        .collect();
    let mut frequencies = Vec::new();
    for col in 0..columns {
        let s = f64::from(f32::from_bits(sin[columns + col]));
        let c = f64::from(f32::from_bits(cos[columns + col]));
        let guess = (s.atan2(c) as f32).to_bits();
        let mut best = None;
        for bits in guess.saturating_sub(4)..=guess.saturating_add(4) {
            let f = f32::from_bits(bits);
            if !f.is_finite() || f <= 0.0 || f > 1.000001 {
                continue;
            }
            let mut score = 0u64;
            for &position in &points {
                let (s, c) = sin_cos(position as f32 * f);
                score += u64::from(s.to_bits().abs_diff(sin[position * columns + col]));
                score += u64::from(c.to_bits().abs_diff(cos[position * columns + col]));
            }
            if best.is_none_or(|(cost, _)| score < cost) {
                best = Some((score, bits));
            }
        }
        let (cost, bits) = best.ok_or("No valid RoPE frequency")?;
        if cost > points.len() as u64 * 8 {
            return Err("Input is not a supported FP32 RoPE table".into());
        }
        frequencies.push(bits);
    }
    dimensions(rows, &frequencies)?;
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let cos_corrections = output.join("cos.xor");
    let sin_corrections = output.join("sin.xor");
    let mut cw = BufWriter::new(File::create(&cos_corrections).map_err(|e| e.to_string())?);
    let mut sw = BufWriter::new(File::create(&sin_corrections).map_err(|e| e.to_string())?);
    for position in 0..rows {
        for (col, &bits) in frequencies.iter().enumerate() {
            let (s, c) = sin_cos(position as f32 * f32::from_bits(bits));
            cw.write_all(&(c.to_bits() ^ cos[position * columns + col]).to_le_bytes())
                .map_err(|e| e.to_string())?;
            sw.write_all(&(s.to_bits() ^ sin[position * columns + col]).to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    cw.flush().map_err(|e| e.to_string())?;
    sw.flush().map_err(|e| e.to_string())?;
    Ok(EncodedRope {
        rows,
        frequencies,
        cos_corrections,
        sin_corrections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrections_restore_every_bit_and_reject_truncation_or_trailing_data() {
        let frequencies = [1.0f32.to_bits(), 0.00003125f32.to_bits()];
        let patch: Vec<u8> = (0u32..256).flat_map(|i| (i % 5).to_le_bytes()).collect();
        for component in [ComponentKind::Sin, ComponentKind::Cos] {
            let mut restored = Vec::new();
            generate(
                128,
                &frequencies,
                component,
                patch.as_slice(),
                &mut restored,
            )
            .unwrap();
            for (i, bytes) in restored.as_chunks::<4>().0.iter().enumerate() {
                let (s, c) = sin_cos((i / 2) as f32 * f32::from_bits(frequencies[i % 2]));
                let base = match component {
                    ComponentKind::Sin => s,
                    ComponentKind::Cos => c,
                };
                assert_eq!(u32::from_le_bytes(*bytes), base.to_bits() ^ (i as u32 % 5));
            }
            assert!(
                generate(
                    128,
                    &frequencies,
                    component,
                    &patch[..patch.len() - 1],
                    Vec::new()
                )
                .is_err()
            );
            let mut extra = patch.clone();
            extra.push(0);
            assert!(generate(128, &frequencies, component, extra.as_slice(), Vec::new()).is_err());
        }
        assert!(dimensions(usize::MAX, &frequencies).is_err());
        assert!(dimensions(128, &[f32::NAN.to_bits()]).is_err());
    }
    #[test]
    fn offline_encoder_roundtrips_independent_fp32_tables() {
        let directory = tempfile::tempdir().unwrap();
        let frequencies = [1.0f32, 0.00003125];
        let mut sine = Vec::new();
        let mut cosine = Vec::new();
        for position in 0..257 {
            for frequency in frequencies {
                let phase = position as f32 * frequency;
                sine.extend_from_slice(&phase.sin().to_le_bytes());
                cosine.extend_from_slice(&phase.cos().to_le_bytes());
            }
        }
        let cos = directory.path().join("cos.bin");
        let sin = directory.path().join("sin.bin");
        fs::write(&cos, &cosine).unwrap();
        fs::write(&sin, &sine).unwrap();
        let encoded = encode_rope(&cos, &sin, 2, &directory.path().join("encoded")).unwrap();
        for (component, path, reference) in [
            (ComponentKind::Sin, encoded.sin_corrections, sine),
            (ComponentKind::Cos, encoded.cos_corrections, cosine),
        ] {
            let patch = fs::read(path).unwrap();
            let mut output = Vec::new();
            generate(
                encoded.rows,
                &encoded.frequencies,
                component,
                patch.as_slice(),
                &mut output,
            )
            .unwrap();
            assert_eq!(output, reference);
        }
    }

    #[test]
    fn installation_checks_corrections_and_never_overwrites_existing_files() {
        use crate::artifact::sha256;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let mut encoder =
            lzma_rust2::XzWriter::new(Vec::new(), lzma_rust2::XzOptions::with_preset(1)).unwrap();
        encoder.write_all(&[0u8; 128 * 4]).unwrap();
        let patch = encoder.finish().unwrap();
        fs::write(root.join("patch.xz"), &patch).unwrap();
        let asset: Asset = serde_json::from_value(serde_json::json!({
            "output":{"file":"table.bin","sha256":"0".repeat(64)},
            "recipe":{"kind":"rope_f32_v1","rows":128,"frequencies":[1.0f32.to_bits()],
                "component":"sin","corrections":{"file":"patch.xz","sha256":sha256(&patch)}}
        }))
        .unwrap();
        materialize(root, std::slice::from_ref(&asset)).unwrap();
        let original = fs::read(root.join("table.bin")).unwrap();
        assert_eq!(original.len(), 128 * 4);
        assert!(materialize(root, std::slice::from_ref(&asset)).is_err());
        assert_eq!(fs::read(root.join("table.bin")).unwrap(), original);
        fs::remove_file(root.join("table.bin")).unwrap();
        fs::write(root.join("patch.xz"), b"corrupt").unwrap();
        assert!(
            materialize(root, std::slice::from_ref(&asset))
                .unwrap_err()
                .contains("sha256")
        );
        assert!(!root.join("table.bin").exists());
        for name in ["../escape", "/absolute", "./relative"] {
            assert!(relative(name).is_err(), "{name}");
        }
    }
}
