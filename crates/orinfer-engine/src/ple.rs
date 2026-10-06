//! CPU n-gram row selection for Qwen4 per-layer embeddings.
//!
//! History is request-owned and cloneable for prefix snapshots. Hashes use the
//! model seed (1234 by default), independently of the sampling seed. Table
//! lookups stay on the CPU; GPU feature uploads are owned by the runtime.
use crate::artifact::Result;

pub mod table;

#[derive(Clone, Copy, Debug)]
pub struct NgramConfig {
    pub vocab_size: u32,
    pub eos_token_id: u32,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub vocab_base: u64,
    pub vocab_alignment: u64,
    pub layer_index: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct History {
    // Chronological tokens after the last EOS; missing context hashes as EOS.
    tokens: Vec<u32>,
}

#[derive(Debug)]
pub struct NgramHasher {
    config: NgramConfig,
    multipliers: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    table_rows: u64,
}

pub struct Prepared {
    /// Token-major, then head-major BF16 features, ready for a GPU upload.
    pub lookup: table::Lookup,
    /// Commit only after the corresponding GPU execution succeeds. Preparing
    /// a chunk, including failed I/O, never advances the caller's history.
    pub next_history: History,
}

const GAMMA: u64 = 0x9e3779b97f4a7c15;

fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(GAMMA);
    let value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) || value.is_multiple_of(3) {
        return value == 2 || value == 3;
    }
    let mut divisor = 5;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) || value.is_multiple_of(divisor + 2) {
            return false;
        }
        divisor += 6;
    }
    true
}

fn next_prime(value: u64) -> Result<u64> {
    let mut candidate = value.checked_add(1).ok_or("PLE prime overflow")?;
    while !prime(candidate) {
        candidate = candidate.checked_add(1).ok_or("PLE prime overflow")?;
    }
    Ok(candidate)
}

impl NgramHasher {
    pub fn new(config: NgramConfig) -> Result<Self> {
        if config.vocab_size == 0
            || config.eos_token_id >= config.vocab_size
            || !(2..=16).contains(&config.ngram_size)
            || !(1..=128).contains(&config.heads_per_ngram)
            || !(2..=u32::MAX as u64).contains(&config.vocab_base)
            || config.vocab_alignment == 0
            || config.layer_index > 1024
        {
            return Err("Invalid PLE n-gram configuration".into());
        }
        let bound = ((i64::MAX as u64 / u64::from(config.vocab_size)) / 2).max(1);
        let seed = config.seed.wrapping_add(10007 * config.layer_index as u64);
        let multipliers = (0..config.ngram_size)
            .map(|index| {
                let initial = seed.wrapping_add(GAMMA.wrapping_mul(index as u64 + 1));
                2 * (splitmix64(initial) % bound) + 1
            })
            .collect();
        let heads = (config.ngram_size - 1) * config.heads_per_ngram;
        let mut size = config.vocab_base - 1;
        for _ in 0..config.layer_index * heads {
            size = next_prime(size)?;
        }
        let mut sizes = Vec::with_capacity(heads);
        let mut offsets = Vec::with_capacity(heads);
        let mut rows = 0u64;
        for _ in 0..heads {
            size = next_prime(size)?;
            offsets.push(rows);
            sizes.push(size);
            rows = rows.checked_add(size).ok_or("PLE vocabulary overflow")?;
        }
        let table_rows = rows
            .div_ceil(config.vocab_alignment)
            .checked_mul(config.vocab_alignment)
            .ok_or("PLE vocabulary alignment overflow")?;
        Ok(Self {
            config,
            multipliers,
            sizes,
            offsets,
            table_rows,
        })
    }

    pub fn table_rows(&self) -> u64 {
        self.table_rows
    }

    pub fn heads(&self) -> usize {
        self.sizes.len()
    }

    pub fn prepare(
        &self,
        tokens: &[u32],
        history: &History,
        table: &mut table::Table,
    ) -> Result<Prepared> {
        if table.rows() != self.table_rows() {
            return Err("PLE table vocabulary does not match n-gram heads".into());
        }
        let length = tokens
            .len()
            .checked_mul(self.heads())
            .ok_or("PLE row batch overflow")?;
        let mut next_history = history.clone();
        let mut rows = Vec::with_capacity(length);
        for &token in tokens {
            rows.extend(self.append(token, &mut next_history)?);
        }
        Ok(Prepared {
            lookup: table.lookup(&rows)?,
            next_history,
        })
    }

    /// Hash a token and advance its request-owned history. On an invalid token,
    /// history remains unchanged. EOS is hashed with its preceding context and
    /// then resets the context for the next token.
    pub fn append(&self, token: u32, history: &mut History) -> Result<Vec<u64>> {
        if token >= self.config.vocab_size {
            return Err("PLE token outside vocabulary".into());
        }
        let mut mix = u64::from(token) * self.multipliers[0];
        let mut rows = Vec::with_capacity(self.heads());
        for shift in 1..self.config.ngram_size {
            let previous = history
                .tokens
                .len()
                .checked_sub(shift)
                .map(|index| history.tokens[index])
                .unwrap_or(self.config.eos_token_id);
            mix ^= u64::from(previous) * self.multipliers[shift];
            let start = (shift - 1) * self.config.heads_per_ngram;
            for head in start..start + self.config.heads_per_ngram {
                rows.push(mix % self.sizes[head] + self.offsets[head]);
            }
        }
        if token == self.config.eos_token_id {
            history.tokens.clear();
        } else {
            history.tokens.push(token);
            if history.tokens.len() >= self.config.ngram_size {
                history.tokens.remove(0);
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> NgramConfig {
        NgramConfig {
            vocab_size: 248320,
            eos_token_id: 248044,
            ngram_size: 3,
            heads_per_ngram: 8,
            vocab_base: 20000000,
            vocab_alignment: 128,
            layer_index: 0,
            seed: 1234,
        }
    }

    #[test]
    fn checkpoint_geometry_and_hashes_match_independent_reference() {
        let hasher = NgramHasher::new(config()).unwrap();
        // Expected values are generated independently from the model's integer
        // SplitMix64/XOR/modulo definition, including per-head prime moduli.
        assert_eq!(hasher.table_rows(), 320001536);
        assert_eq!(
            hasher.multipliers,
            vec![23703573157769, 20109073645365, 8052911324071]
        );
        let expected: [[u64; 16]; 6] = [
            [
                5727835, 21884476, 43702108, 66434789, 84094509, 104112171, 134886528, 157314943,
                162363667, 189466439, 200618315, 232121522, 258547276, 266199001, 289022207,
                305180972,
            ],
            [
                18462773, 34329996, 52266929, 69382332, 86913271, 106090998, 124858223, 143215740,
                178648117, 190447732, 216031532, 231341628, 251016382, 271050650, 284475351,
                312197724,
            ],
            [
                11085383, 38759060, 47007057, 71494768, 94214697, 109395741, 133049541, 146234704,
                167689211, 196321042, 211743113, 227563092, 242994021, 279483683, 285215424,
                315638419,
            ],
            [
                1401404, 25218075, 50865572, 78959896, 81215095, 102764567, 135836609, 141328471,
                165439174, 188692655, 205353654, 224025111, 258526110, 274253068, 286434575,
                309385298,
            ],
            [
                14294302, 26995765, 47759518, 73771333, 83513797, 107702784, 134868839, 146071007,
                177249532, 194113089, 215310392, 227549525, 256067950, 265546846, 288920109,
                307386659,
            ],
            [
                3745007, 25177969, 59639583, 68080282, 84924354, 118004746, 128374230, 156931756,
                166762062, 198605654, 206967647, 226398276, 251758918, 278018498, 293638175,
                313234297,
            ],
        ];
        let mut history = History::default();
        for (token, rows) in [100, 101, 248044, 102, 248319, 0].into_iter().zip(expected) {
            assert_eq!(hasher.append(token, &mut history).unwrap(), rows);
        }
    }

    #[test]
    fn chunking_restore_eos_and_request_isolation() {
        let hasher = NgramHasher::new(config()).unwrap();
        let mut history = History::default();
        hasher.append(10, &mut history).unwrap();
        hasher.append(20, &mut history).unwrap();
        let snapshot = history.clone();
        let expected = hasher.append(30, &mut history).unwrap();
        let mut restored = snapshot.clone();
        assert_eq!(hasher.append(30, &mut restored).unwrap(), expected);
        let mut other = History::default();
        hasher.append(40, &mut other).unwrap();
        assert_ne!(hasher.append(30, &mut other).unwrap(), expected);
        hasher.append(248044, &mut restored).unwrap();
        assert_eq!(restored, History::default());
        assert_eq!(
            hasher.append(31, &mut restored).unwrap(),
            hasher.append(31, &mut History::default()).unwrap()
        );
        let snapshot = restored.clone();
        assert!(hasher.append(248320, &mut restored).is_err());
        assert_eq!(restored, snapshot);
    }

    #[test]
    fn rejects_invalid_configuration() {
        let mut invalid = config();
        invalid.vocab_alignment = 0;
        assert!(NgramHasher::new(invalid).is_err());
        let mut invalid = config();
        invalid.eos_token_id = invalid.vocab_size;
        assert!(NgramHasher::new(invalid).is_err());
    }
}
