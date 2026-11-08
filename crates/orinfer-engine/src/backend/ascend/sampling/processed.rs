//! Packed vocabulary masks and compact TP probability summaries.
use super::*;
pub(crate) use crate::sampling::RowParameters as Parameters;

pub(super) const WIDTH: usize = 48;
pub(super) fn mask_stride(vocab: usize) -> usize {
    vocab.div_ceil(256) * 32
}
impl Parameters {
    fn pack(&self, origin: usize, vocab: usize, target: &mut [u8]) -> Result<[i32; 2]> {
        if self.top_logprobs.is_some_and(|n| n > 20) || target.len() != mask_stride(vocab) {
            return Err("Invalid compact sampling parameters".into());
        }
        if let Some(mask) = &self.mask {
            let shift = origin % 32;
            for word in 0..vocab.div_ceil(32) {
                let first = origin / 32 + word;
                let mut bits = mask.get(first).copied().unwrap_or(0) >> shift;
                if shift > 0 {
                    bits |= mask.get(first + 1).copied().unwrap_or(0) << (32 - shift);
                }
                let valid = (vocab - word * 32).min(32);
                if valid < 32 {
                    bits &= (1u32 << valid) - 1;
                }
                target[word * 4..word * 4 + 4].copy_from_slice(&bits.to_le_bytes());
            }
        }
        Ok([
            i32::from(self.mask.is_some()),
            self.top_logprobs.map_or(0, |n| n as i32 + 1),
        ])
    }
}
#[derive(Clone)]
pub(crate) struct Summary {
    pub selected: f32,
    pub maximum: f32,
    pub mass: f32,
    pub top: Vec<(u32, f32)>,
    pub requested: usize,
}
pub(crate) struct Sample {
    pub score: f32,
    pub token: u32,
    pub summary: Option<Summary>,
}
/// Combine shard log-sum-exp statistics without returning the vocabulary.
pub(crate) fn report(
    parts: &[&Summary],
    chosen: &Summary,
    token: u32,
) -> Result<crate::sampling::TokenLogprob> {
    if parts.is_empty()
        || !chosen.selected.is_finite()
        || chosen.requested > 20
        || parts.iter().any(|p| {
            p.requested != chosen.requested
                || !p.mass.is_finite()
                || p.mass < 0.0
                || (p.mass > 0.0 && !p.maximum.is_finite())
        })
    {
        return Err("Invalid sharded probability summary".into());
    }
    let maximum = parts
        .iter()
        .filter(|p| p.mass > 0.0)
        .map(|p| p.maximum as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let mass: f64 = parts
        .iter()
        .filter(|p| p.mass > 0.0)
        .map(|p| p.mass as f64 * (p.maximum as f64 - maximum).exp())
        .sum();
    if !maximum.is_finite() || !mass.is_finite() || mass <= 0.0 {
        return Err("Grammar permits no tokens".into());
    }
    let normalizer = maximum + mass.ln();
    let mut top = parts
        .iter()
        .flat_map(|p| p.top.iter().copied())
        .collect::<Vec<_>>();
    top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if top.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err("Duplicate token in sharded summary".into());
    }
    top.truncate(chosen.requested);
    Ok(crate::sampling::TokenLogprob {
        token,
        logprob: chosen.selected as f64 - normalizer,
        top: top
            .into_iter()
            .map(|(id, score)| (id, score as f64 - normalizer))
            .collect(),
    })
}

pub(super) struct Processed {
    masks: Buffer,
    options: Buffer,
    output: Buffer,
    program: Program,
    graph: Option<Graph>,
}
impl Processed {
    pub fn load(
        session: &Session,
        base: &Path,
        c: &Contract,
        logits: &Buffer,
        controls: &Buffer,
        active: &Buffer,
    ) -> Result<Self> {
        let local = Template::load(
            session,
            &resolve_file(
                base,
                c.processed_local
                    .as_ref()
                    .ok_or("Missing processed sampler")?,
            )?,
        )?;
        let merge = Template::load(
            session,
            &resolve_file(
                base,
                c.processed_merge
                    .as_ref()
                    .ok_or("Missing processed sampler merge")?,
            )?,
        )?;
        let masks = session.allocate(c.capacity * mask_stride(c.vocab))?;
        let options = session.allocate(c.capacity * 8)?;
        let partial = session.allocate(c.capacity * c.vocab.div_ceil(1024) * WIDTH * 4)?;
        let output = session.allocate(c.capacity * WIDTH * 4)?;
        let first = Kernel::bind(
            session,
            &local,
            &[
                logits.clone(),
                controls.clone(),
                active.clone(),
                masks.clone(),
                options.clone(),
                partial.clone(),
            ],
        )?;
        let second = Kernel::bind(
            session,
            &merge,
            &[partial, active.clone(), options.clone(), output.clone()],
        )?;
        Ok(Self {
            masks,
            options,
            output,
            program: Program::new(session, &[&first, &second])?,
            graph: None,
        })
    }
    pub fn sample(
        &mut self,
        session: &Session,
        c: &Contract,
        active: &Buffer,
        origin: u32,
        parameters: &[Parameters],
        capture: bool,
    ) -> Result<Vec<Sample>> {
        if capture && self.graph.is_none() {
            session.upload_inputs(&[(active.clone(), 0i32.to_le_bytes().to_vec())])?;
            self.graph = Some(self.program.capture(session)?);
        }
        let mut masks = vec![0; parameters.len() * mask_stride(c.vocab)];
        let options = parameters
            .iter()
            .zip(masks.chunks_mut(mask_stride(c.vocab)))
            .map(|(p, mask)| p.pack(origin as usize, c.vocab, mask))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut inputs = vec![
            (self.options.slice(0, options.len())?, options),
            (
                active.clone(),
                (parameters.len() as i32).to_le_bytes().to_vec(),
            ),
        ];
        if parameters.iter().any(|p| p.mask.is_some()) {
            inputs.push((self.masks.slice(0, masks.len())?, masks));
        }
        session.upload_inputs(&inputs)?;
        if capture {
            self.graph.as_ref().unwrap().enqueue(session)?;
        } else {
            self.program.enqueue(session)?;
        }
        let bytes = session.download_compute(&self.output, 0, parameters.len() * WIDTH * 4)?;
        bytes
            .as_chunks::<{ WIDTH * 4 }>()
            .0
            .iter()
            .zip(parameters)
            .map(|(data, p)| {
                let v = data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                let token = |x: f32| -> Result<u32> {
                    if !x.is_finite() || x.fract() != 0.0 || !(0.0..c.vocab as f32).contains(&x) {
                        return Err("Invalid compact sample token".into());
                    }
                    Ok(x as u32)
                };
                if !(v[0].is_finite() || v[0] == f32::NEG_INFINITY) {
                    return Err("Nonfinite processed sampling logits".into());
                }
                let summary = p
                    .top_logprobs
                    .map(|n| -> Result<Summary> {
                        let mut top = vec![];
                        for k in 0..n {
                            let score = v[5 + 2 * k];
                            if score.is_finite() {
                                top.push((origin + token(v[6 + 2 * k])?, score));
                            } else if score != f32::NEG_INFINITY {
                                return Err("Invalid compact top probabilities".into());
                            }
                        }
                        Ok(Summary {
                            selected: v[2],
                            maximum: v[3],
                            mass: v[4],
                            top,
                            requested: n,
                        })
                    })
                    .transpose()?;
                Ok(Sample {
                    score: v[0],
                    token: token(v[1])?,
                    summary,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_masks_slice_unaligned_shards_and_clear_tails() {
        let p = Parameters {
            mask: Some(vec![0x8000_0001, 0x8000_0003, 0xff].into()),
            top_logprobs: Some(0),
        };
        for (origin, vocab) in [(0, 33), (1, 35), (31, 37), (32, 33)] {
            let mut bytes = vec![0; mask_stride(vocab)];
            assert_eq!(p.pack(origin, vocab, &mut bytes).unwrap(), [1, 1]);
            for i in 0..bytes.len() * 8 {
                let actual = bytes[i / 8] & (1 << (i % 8)) != 0;
                let expected = i < vocab
                    && p.mask.as_ref().unwrap()[(origin + i) / 32] & (1 << ((origin + i) % 32))
                        != 0;
                assert_eq!(actual, expected, "origin={origin} bit={i}");
            }
        }
    }
    #[test]
    fn compact_reports_normalize_across_shards_and_include_nontop_draws() {
        let a = Summary {
            selected: -2.0,
            maximum: 1.0,
            mass: 1.0 + (-3.0f32).exp(),
            top: vec![(0, 1.0), (1, -2.0)],
            requested: 3,
        };
        let b = Summary {
            selected: 1.0,
            maximum: 1.0,
            mass: 1.0,
            top: vec![(33, 1.0)],
            requested: 3,
        };
        let z = (2.0f64 * 1.0f64.exp() + (-2.0f64).exp()).ln();
        let r = report(&[&a, &b], &a, 1).unwrap();
        assert!((r.logprob - (-2.0 - z)).abs() < 1e-7);
        assert_eq!(r.top.iter().map(|v| v.0).collect::<Vec<_>>(), [0, 33, 1]);
        let empty = Summary {
            mass: 0.,
            maximum: f32::NEG_INFINITY,
            selected: f32::NEG_INFINITY,
            top: vec![],
            requested: 3,
        };
        assert!(report(&[&empty], &a, 1).is_err());
    }
}
