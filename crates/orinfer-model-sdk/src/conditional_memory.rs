//! Deterministic n-gram addressing over offline-exported tokenizer metadata.
//! Hashing takes explicit committed history, so restored prefixes and rejected
//! speculative suffixes never depend on a mutable physical KV-page mirror.
use crate::artifact::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HashLayer {
    pub layer: usize,
    pub rows: u64,
    pub multipliers: [i64; 4],
    pub primes: Vec<i64>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HashLayout {
    pub schema_version: u32,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub token_map: Vec<u32>,
    pub pad_token: u32,
    pub barriers: Vec<u32>,
    pub heads: usize,
    pub layers: Vec<HashLayer>,
}
pub struct Hasher {
    layout: HashLayout,
    offsets: Vec<Vec<i64>>,
}
impl Hasher {
    pub fn new(layout: HashLayout) -> Result<Self> {
        if layout.schema_version != 1
            || layout.token_map.is_empty()
            || layout.pad_token as usize >= layout.token_map.len()
            || layout.heads == 0
            || layout.heads > 64
            || layout.layers.is_empty()
            || layout.layers.len() > 64
            || [&layout.config_sha256, &layout.tokenizer_sha256]
                .into_iter()
                .any(|h| !crate::artifact::is_sha256_hex(h))
            || layout
                .barriers
                .iter()
                .any(|&id| id as usize >= layout.token_map.len())
        {
            return Err("Invalid conditional-memory hash layout or source identity".into());
        }
        let mut layer_ids = std::collections::BTreeSet::new();
        let mut offsets = Vec::new();
        for layer in &layout.layers {
            if !layer_ids.insert(layer.layer)
                || layer.rows == 0
                || layer.rows > i64::MAX as u64
                || layer.primes.len() != layout.heads * 3
                || layer.primes.iter().any(|&p| p < 2)
                || layer.multipliers.iter().any(|&m| m <= 0 || m % 2 == 0)
            {
                return Err("Invalid conditional-memory layer geometry".into());
            }
            let mut offset = 0i64;
            let starts = layer
                .primes
                .iter()
                .map(|&p| {
                    let start = offset;
                    offset = offset
                        .checked_add(p)
                        .ok_or("Conditional-memory bucket overflow")?;
                    Ok(start)
                })
                .collect::<Result<Vec<_>>>()?;
            if offset as u64 != layer.rows {
                return Err("Conditional-memory bucket rows differ from table".into());
            }
            offsets.push(starts);
        }
        Ok(Self { layout, offsets })
    }
    pub fn layout(&self) -> &HashLayout {
        &self.layout
    }
    pub fn values_per_token(&self) -> usize {
        self.layout.layers.len() * self.layout.heads * 3
    }
    /// Flattened output is [token,layer,ngram-order,head]. Invalid media rows
    /// are zeroed and explicitly masked; the first media barrier pads older IDs.
    pub fn hash_into(
        &self,
        history: &[u32],
        tokens: &[u32],
        out: &mut Vec<i64>,
        valid: &mut Vec<bool>,
    ) -> Result<()> {
        for &token in tokens.iter().chain(history.iter().rev().take(3)) {
            if token as usize >= self.layout.token_map.len() {
                return Err("Conditional-memory token outside tokenizer".into());
            }
        }
        let width = self.values_per_token();
        let count = tokens
            .len()
            .checked_mul(width)
            .ok_or("Conditional-memory hash output overflow")?;
        out.clear();
        out.resize(count, 0);
        valid.clear();
        valid.resize(tokens.len(), false);
        let pad = i64::from(self.layout.token_map[self.layout.pad_token as usize]);
        for (row, &token) in tokens.iter().enumerate() {
            if self.layout.barriers.contains(&token) {
                continue;
            }
            valid[row] = true;
            let mut window = [pad; 4];
            for (shift, item) in window.iter_mut().enumerate() {
                let raw = if shift <= row {
                    Some(tokens[row - shift])
                } else {
                    history.len().checked_sub(shift - row).map(|i| history[i])
                };
                let Some(raw) = raw else {
                    break;
                };
                if self.layout.barriers.contains(&raw) {
                    break;
                }
                *item = i64::from(self.layout.token_map[raw as usize]);
            }
            for (l, layer) in self.layout.layers.iter().enumerate() {
                let mut rolling = window[0].wrapping_mul(layer.multipliers[0]);
                for (shift, &token) in window.iter().enumerate().skip(1) {
                    rolling ^= token.wrapping_mul(layer.multipliers[shift]);
                    let first = (shift - 1) * self.layout.heads;
                    for head in 0..self.layout.heads {
                        let bucket = first + head;
                        out[row * width + l * self.layout.heads * 3 + bucket] =
                            rolling.rem_euclid(layer.primes[bucket]) + self.offsets[l][bucket];
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hasher() -> Hasher {
        Hasher::new(HashLayout {
            schema_version: 1,
            config_sha256: "0".repeat(64),
            tokenizer_sha256: "1".repeat(64),
            token_map: (0..16).collect(),
            pad_token: 0,
            barriers: vec![14, 15],
            heads: 1,
            layers: vec![HashLayer {
                layer: 1,
                rows: 21,
                multipliers: [1, 3, 5, 7],
                primes: vec![5, 7, 9],
            }],
        })
        .unwrap()
    }
    fn hash(h: &Hasher, history: &[u32], tokens: &[u32]) -> (Vec<i64>, Vec<bool>) {
        let (mut out, mut valid) = (vec![], vec![]);
        h.hash_into(history, tokens, &mut out, &mut valid).unwrap();
        (out, valid)
    }
    #[test]
    fn chunked_prefill_and_speculative_rollback_use_explicit_history() {
        let h = hasher();
        let input = [1, 2, 3, 4, 5, 6];
        let whole = hash(&h, &[], &input);
        let first = hash(&h, &[], &input[..2]);
        let second = hash(&h, &input[..2], &input[2..]);
        assert_eq!([first.0, second.0].concat(), whole.0);
        hash(&h, &[1, 2], &[9, 10, 11]); // unaccepted branch has no hidden mutation
        assert_eq!(hash(&h, &[1, 2], &[3, 4]).0, whole.0[6..12]);
        assert_eq!(
            hash(&h, &[99, 1, 2, 3], &[4]).0,
            hash(&h, &[1, 2, 3], &[4]).0
        );
    }
    #[test]
    fn media_barriers_and_bucket_offsets_are_exact() {
        let h = hasher();
        let (values, valid) = hash(&h, &[], &[1, 2, 14, 3]);
        assert_eq!(valid, [true, true, false, true]);
        assert_eq!(&values[0..3], &[1, 6, 13]);
        assert_eq!(&values[3..6], &[1, 6, 13]); // 2 XOR 3*1 == 1
        assert_eq!(&values[6..9], &[0, 0, 0]);
        assert_eq!(&values[9..12], &hash(&h, &[], &[3]).0);
    }
    #[test]
    fn malformed_metadata_and_token_ids_fail_before_output_changes() {
        let h = hasher();
        let mut bad = h.layout.clone();
        bad.layers[0].rows += 1;
        assert!(Hasher::new(bad).is_err());
        let mut out = vec![42];
        let mut mask = vec![true];
        assert!(h.hash_into(&[], &[99], &mut out, &mut mask).is_err());
        assert_eq!(out, [42]);
        assert_eq!(mask, [true]);
    }
}
