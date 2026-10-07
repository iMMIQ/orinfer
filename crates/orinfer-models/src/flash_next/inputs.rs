//! Bounded CPU E8P embedding/PLE lookup. History comes from the request owner;
//! immutable decoded rows are shared, never mutable n-gram state.
use crate::artifact::Result;
use half::f16;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fs::File,
    io::Read,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Book {
    table: Vec<i8>,
    signs: Vec<i8>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    file: String,
    sha256: String,
    offset: u64,
    first: usize,
    rows: usize,
    #[serde(default)]
    book: Option<usize>,
    #[serde(default)]
    scale_offset: Option<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Assets {
    version: u32,
    vocab: usize,
    eos: u32,
    multipliers: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    books: Vec<Book>,
    embedding: Vec<Part>,
    ple: Vec<Part>,
}
struct Table {
    values: Vec<[i8; 8]>,
    signs: Vec<i8>,
}
struct Source {
    file: File,
    offset: u64,
    first: usize,
    rows: usize,
    book: Option<usize>,
    scale_offset: Option<u64>,
    width: usize,
}
#[derive(Default)]
struct Cache {
    clock: u64,
    bytes: usize,
    values: HashMap<(usize, usize), (u64, Vec<u16>)>,
    order: BTreeMap<u64, (usize, usize)>,
}
impl Cache {
    fn get(&mut self, key: (usize, usize)) -> Option<Vec<u16>> {
        let (stamp, value) = self.values.get_mut(&key)?;
        self.order.remove(stamp);
        self.clock += 1;
        *stamp = self.clock;
        self.order.insert(*stamp, key);
        Some(value.clone())
    }
    fn put(&mut self, key: (usize, usize), value: Vec<u16>, budget: usize) {
        let size = value.len() * 2 + 512;
        if size > budget {
            return;
        }
        while self.bytes + size > budget {
            let (_, key) = self.order.pop_first().expect("nonempty bounded row cache");
            let (_, old) = self.values.remove(&key).expect("paired cache entry");
            self.bytes -= old.len() * 2 + 512;
        }
        self.clock += 1;
        self.bytes += size;
        self.values.insert(key, (self.clock, value));
        self.order.insert(self.clock, key);
    }
}
pub(super) struct Inputs {
    assets: Assets,
    tables: Vec<Table>,
    embedding: Vec<Source>,
    ple: Vec<Source>,
    embedding_cache: RefCell<Cache>,
    ple_cache: RefCell<Cache>,
}
fn path(root: &Path, name: &str) -> Result<PathBuf> {
    let relative = Path::new(name);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("Unsafe model input asset path".into());
    }
    let file = root
        .join(relative)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !file.starts_with(root) {
        return Err("Model input asset leaves model directory".into());
    }
    Ok(file)
}
fn digest(file: &mut File) -> Result<String> {
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hex::encode(hash.finalize()))
}
impl Inputs {
    pub(super) fn open(root: &str, identity: &crate::artifact::FileIdentity) -> Result<Self> {
        let root = Path::new(root).canonicalize().map_err(|e| e.to_string())?;
        let raw = std::fs::read(path(&root, &identity.file)?).map_err(|e| e.to_string())?;
        if hex::encode(Sha256::digest(&raw)) != identity.sha256 {
            return Err("CPU input metadata hash mismatch".into());
        }
        let assets: Assets = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if assets.version != 1
            || assets.vocab != 248320
            || assets.eos as usize >= assets.vocab
            || assets.multipliers.len() != 3
            || assets.sizes.len() != 16
            || assets.offsets.len() != 16
            || assets.multipliers.iter().any(|&m| {
                m.checked_mul((assets.vocab - 1) as u64)
                    .is_none_or(|x| x > i64::MAX as u64)
            })
        {
            return Err("Unsupported CPU embedding/PLE metadata".into());
        }
        let mut tables = vec![];
        for book in &assets.books {
            let width = book.signs.len();
            if book.table.len() != 2048
                || !matches!(width, 160 | 2560)
                || book.signs.iter().any(|s| !matches!(s, -1 | 1))
                || book
                    .table
                    .iter()
                    .any(|&t| t % 2 != 0 || t.unsigned_abs() > 126)
            {
                return Err("Invalid E8P book or rotation".into());
            }
            let bit = [0, 4, 1, 5, 2, 6, 3, 7];
            let values = (0..65536u32)
                .map(|code| {
                    let parity = (code & 255).count_ones() & 1;
                    let signs = (code & 255) ^ parity;
                    std::array::from_fn(|j| {
                        let base = book.table[((code >> 8) * 8) as usize + j] as i16;
                        (base * if (signs >> bit[j]) & 1 == 0 { 1 } else { -1 }
                            + if parity == 0 { 1 } else { -1 }) as i8
                    })
                })
                .collect();
            tables.push(Table {
                values,
                signs: book.signs.clone(),
            });
        }
        let load = |parts: &[Part], width: usize| -> Result<Vec<Source>> {
            let mut out = vec![];
            let mut first = 0usize;
            for part in parts {
                if part.first != first
                    || part.rows == 0
                    || match (part.book, part.scale_offset) {
                        (Some(book), None) => {
                            tables.get(book).is_none_or(|b| b.signs.len() != width)
                        }
                        (None, Some(_)) => width != 2560,
                        _ => true,
                    }
                {
                    return Err("Invalid or discontinuous CPU row slices".into());
                }
                let mut file = File::open(path(&root, &part.file)?).map_err(|e| e.to_string())?;
                if digest(&mut file)? != part.sha256 {
                    return Err("CPU row shard hash mismatch".into());
                }
                let bytes = (part.rows as u64)
                    .checked_mul(if part.book.is_some() {
                        (2 + width / 4) as u64
                    } else {
                        width as u64
                    })
                    .and_then(|n| n.checked_add(part.offset))
                    .ok_or("CPU row extent overflow")?;
                if bytes > file.metadata().map_err(|e| e.to_string())?.len() {
                    return Err("CPU rows exceed file".into());
                }
                if part.scale_offset.is_some_and(|offset| {
                    offset
                        .checked_add(part.rows as u64 * 2)
                        .is_none_or(|end| end > file.metadata().map(|m| m.len()).unwrap_or(0))
                }) {
                    return Err("CPU row scales exceed file".into());
                }
                out.push(Source {
                    file,
                    offset: part.offset,
                    first,
                    rows: part.rows,
                    book: part.book,
                    scale_offset: part.scale_offset,
                    width,
                });
                first = first
                    .checked_add(part.rows)
                    .ok_or("CPU row count overflow")?;
            }
            if out.is_empty() {
                return Err("Missing CPU row shards".into());
            }
            Ok(out)
        };
        let embedding = load(&assets.embedding, 2560)?;
        let ple = load(&assets.ple, 160)?;
        if embedding.last().map(|p| p.first + p.rows) != Some(assets.vocab) {
            return Err("Embedding vocabulary mismatch".into());
        }
        let total = ple
            .last()
            .map(|p| p.first + p.rows)
            .ok_or("Empty PLE table")? as u64;
        if assets
            .sizes
            .iter()
            .zip(&assets.offsets)
            .any(|(&s, &o)| s == 0 || o.checked_add(s).is_none_or(|end| end > total))
        {
            return Err("PLE head outside CPU row table".into());
        }
        Ok(Self {
            assets,
            tables,
            embedding,
            ple,
            embedding_cache: Default::default(),
            ple_cache: Default::default(),
        })
    }
    fn row(
        &self,
        sources: &[Source],
        cache: &RefCell<Cache>,
        row: usize,
        budget: usize,
    ) -> Result<Vec<u16>> {
        let shard = sources
            .partition_point(|s| s.first <= row)
            .checked_sub(1)
            .ok_or("CPU row outside table")?;
        let source = &sources[shard];
        let local = row - source.first;
        if local >= source.rows {
            return Err("CPU row outside table".into());
        }
        let key = (shard, local);
        if let Some(value) = cache.borrow_mut().get(key) {
            return Ok(value);
        }
        let width = source.width;
        let stride = if source.book.is_some() {
            2 + width / 4
        } else {
            width
        };
        let mut packet = vec![0u8; stride];
        source
            .file
            .read_exact_at(&mut packet, source.offset + (local * stride) as u64)
            .map_err(|e| e.to_string())?;
        let value = if let Some(book) = source.book {
            decode(&packet, &self.tables[book])?
        } else {
            let mut scale = [0u8; 2];
            source
                .file
                .read_exact_at(
                    &mut scale,
                    source.scale_offset.ok_or("Missing INT8 row scale")? + local as u64 * 2,
                )
                .map_err(|e| e.to_string())?;
            let scale = f16::from_bits(u16::from_le_bytes(scale)).to_f32();
            if !scale.is_finite() || scale <= 0.0 {
                return Err("Invalid INT8 row scale".into());
            }
            packet
                .iter()
                .map(|&v| f16::from_f32((v as i8) as f32 * scale).to_bits())
                .collect()
        };
        cache.borrow_mut().put(key, value.clone(), budget);
        Ok(value)
    }
    #[cfg(test)]
    pub(super) fn prepare(
        &self,
        tokens: &[u32],
        history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        self.prepare_for("target", tokens, history)
    }
    pub(super) fn prepare_for(
        &self,
        program: &str,
        tokens: &[u32],
        history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let draft = program.starts_with("mtp_warm_m") || program == "mtp_draft";
        if !draft && program != "target" {
            return Err("Unknown Flash input program".into());
        }
        if !matches!(tokens.len(), 1..=8 | 16 | 128 | 512)
            || tokens
                .iter()
                .chain(history)
                .any(|&t| t as usize >= self.assets.vocab)
        {
            return Err("Unsupported Flash input shape/token".into());
        }
        let mut previous: Vec<u32> = history
            .iter()
            .rev()
            .take_while(|&&t| t != self.assets.eos)
            .take(2)
            .copied()
            .collect();
        previous.reverse();
        let mut embedding = Vec::with_capacity(tokens.len() * 5120);
        let mut ple = Vec::with_capacity(embedding.capacity());
        for &token in tokens {
            for value in self.row(
                &self.embedding,
                &self.embedding_cache,
                token as usize,
                8 * 1024 * 1024,
            )? {
                embedding.extend(value.to_le_bytes());
            }
            if draft {
                continue;
            }
            let mut mixed = token as u64 * self.assets.multipliers[0];
            for shift in 1..=2 {
                let prior = previous
                    .len()
                    .checked_sub(shift)
                    .map(|i| previous[i])
                    .unwrap_or(self.assets.eos);
                mixed ^= prior as u64 * self.assets.multipliers[shift];
                for head in (shift - 1) * 8..shift * 8 {
                    let row =
                        (mixed % self.assets.sizes[head] + self.assets.offsets[head]) as usize;
                    for value in self.row(&self.ple, &self.ple_cache, row, 32 * 1024 * 1024)? {
                        ple.extend(value.to_le_bytes());
                    }
                }
            }
            if token == self.assets.eos {
                previous.clear();
            } else {
                previous.push(token);
                if previous.len() > 2 {
                    previous.remove(0);
                }
            }
        }
        if draft {
            return Ok(vec![(
                format!("DraftM{}_Embedding", tokens.len()),
                embedding,
            )]);
        }
        Ok(vec![
            (format!("M{}_Embedding", tokens.len()), embedding),
            (format!("M{}_Ple", tokens.len()), ple),
        ])
    }
}
fn decode(packet: &[u8], table: &Table) -> Result<Vec<u16>> {
    let scale = f16::from_bits(u16::from_le_bytes([packet[0], packet[1]])).to_f32();
    if !scale.is_finite() || scale <= 0. {
        return Err("Invalid E8P row scale".into());
    }
    let width = table.signs.len();
    let columns = width / 20;
    let mut rotated = vec![0f32; width];
    for (i, code) in packet[2..].as_chunks::<2>().0.iter().enumerate() {
        let values = &table.values[u16::from_le_bytes([code[0], code[1]]) as usize];
        for j in 0..8 {
            rotated[i * 8 + j] = values[j] as f32 * scale;
        }
    }
    let mut out = vec![0f32; width];
    let norm = (20f32).sqrt().recip();
    for i in 0..20 {
        for j in 0..20 {
            let sign = if j == 0 {
                1.
            } else if i == 0 {
                -1.
            } else {
                let delta = (j + 19 - i) % 19;
                if delta == 0 || (1..19).any(|k| k * k % 19 == delta) {
                    1.
                } else {
                    -1.
                }
            };
            let coefficient = sign * norm;
            for col in 0..columns {
                out[i * columns + col] += coefficient * rotated[j * columns + col];
            }
        }
    }
    for row in out.chunks_exact_mut(columns) {
        let mut half = 1;
        while half < columns {
            for block in row.chunks_exact_mut(half * 2) {
                for j in 0..half {
                    let a = block[j];
                    let b = block[j + half];
                    block[j] = a + b;
                    block[j + half] = a - b;
                }
            }
            half *= 2;
        }
    }
    let norm = (columns as f32).sqrt().recip();
    Ok(out
        .iter()
        .zip(&table.signs)
        .map(|(&x, &s)| f16::from_f32((x * norm) * s as f32).to_bits())
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Requires prepared Flash CPU row assets and frozen Python input rows"]
    fn reference_rows_match_with_eos_and_request_history() {
        let root = std::env::var("ORINFER_FLASH_MODEL").unwrap();
        let golden = std::env::var("ORINFER_FLASH_GOLDEN").unwrap();
        let raw = std::fs::read(Path::new(&root).join("cache/cpu/inputs.json")).unwrap();
        let inputs = Inputs::open(
            &root,
            &crate::artifact::FileIdentity {
                file: "cache/cpu/inputs.json".into(),
                sha256: hex::encode(Sha256::digest(&raw)),
            },
        )
        .unwrap();
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(golden).unwrap()).unwrap();
        let tokens: Vec<u32> = serde_json::from_value(golden["tokens"].clone()).unwrap();
        let mut history: Vec<u32> = serde_json::from_value(golden["history"].clone()).unwrap();
        let expected: Vec<Vec<u16>> = ["embedding", "ple"]
            .into_iter()
            .map(|k| serde_json::from_value(golden[k].clone()).unwrap())
            .collect();
        for (index, &token) in tokens.iter().enumerate() {
            let rows = inputs.prepare(&[token], &history).unwrap();
            for (kind, (_, raw)) in rows.iter().enumerate() {
                let actual: Vec<u16> = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u16::from_le_bytes([p[0], p[1]]))
                    .collect();
                let expected = &expected[kind][index * 2560..(index + 1) * 2560];
                let max = actual
                    .iter()
                    .zip(expected)
                    .map(|(&a, &b)| (f16::from_bits(a).to_f32() - f16::from_bits(b).to_f32()).abs())
                    .fold(0f32, f32::max);
                assert!(max <= 0.0005, "input {index}/{kind}: max FP16 error {max}");
            }
            history.push(token);
        }
        assert!(inputs.embedding_cache.borrow().bytes <= 8 * 1024 * 1024);
        assert!(inputs.ple_cache.borrow().bytes <= 32 * 1024 * 1024);
        assert!(inputs.prepare(&[248320], &[]).is_err());
    }
}
