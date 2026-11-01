//! Offline next-token scoring through the normal arbitrary-length prefill and
//! stateful decode programs. Frozen token histories never pass through sampling.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreRequests {
    pub seed: u64,
    pub cases: Vec<ScoreCase>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreCase {
    pub id: String,
    pub prompt_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
    #[serde(default)]
    pub images: Vec<crate::vision::ImageInput>,
    #[serde(default)]
    pub audios: Vec<crate::audio::AudioInput>,
    #[serde(default)]
    pub query_ids: Vec<Vec<u32>>,
}
#[derive(Serialize)]
pub struct TokenProbability {
    pub token_id: u32,
    pub logprob: f64,
}
#[derive(Serialize)]
pub struct Probe {
    pub case_id: String,
    pub execution_mode: &'static str,
    pub position: usize,
    pub seed: u64,
    pub context_sha256: String,
    pub reference_token_id: u32,
    pub reference_logprob: f64,
    pub top3: Vec<TokenProbability>,
    pub queried_logprobs: BTreeMap<u32, f64>,
}
#[derive(Serialize)]
pub struct ScoreReport {
    pub manifest_sha256: String,
    pub seed: u64,
    pub load_to_ready_s: f64,
    pub scoring_s: f64,
    pub probes: Vec<Probe>,
}

pub(super) fn scoring_context_hash(
    history: &[u32],
    images: &[crate::vision::ImageInput],
    audios: &[crate::audio::AudioInput],
) -> Result<String> {
    let mut context =
        serde_json::json!({"token_ids":history,"positions":(0..history.len()).collect::<Vec<_>>()});
    if !images.is_empty() {
        let identities: Vec<_> = images
            .iter()
            .map(|image| {
                let bytes: Vec<_> = image.pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
                let mut identity = serde_json::json!({"grid_height":image.grid_height,"grid_width":image.grid_width,
                    "pixels_sha256":crate::artifact::sha256(&bytes)});
                if !image.rgb.is_empty() { identity["rgb_sha256"] = crate::artifact::sha256(&image.rgb).into(); }
                identity
            })
            .collect();
        context["images"] = serde_json::json!(identities);
    }
    if !audios.is_empty() {
        context["audios"] = serde_json::json!(audios.iter().map(|audio| {
            let bytes: Vec<_> = audio.mel.iter().flat_map(|v| v.to_le_bytes()).collect();
            serde_json::json!({"frames":audio.frames,"mel_sha256":crate::artifact::sha256(&bytes)})
        }).collect::<Vec<_>>());
    }
    // Stable identity regardless of serde_json's feature-unified map backend.
    context.as_object_mut().unwrap().sort_keys();
    Ok(crate::artifact::sha256(
        &serde_json::to_vec(&context).map_err(|e| e.to_string())?,
    ))
}

fn probabilities(
    logits: &[f32],
    target: u32,
    query: &[u32],
) -> Result<(f64, Vec<TokenProbability>, BTreeMap<u32, f64>)> {
    if logits.len() < 3
        || logits.iter().any(|v| !v.is_finite())
        || std::iter::once(&target)
            .chain(query)
            .any(|&id| id as usize >= logits.len())
    {
        return Err("Invalid scoring logits or queried token".into());
    }
    let mut ids = [0, 1, 2];
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    for id in 3..logits.len() {
        if logits[id] > logits[ids[2]] {
            ids[2] = id;
            ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        }
    }
    let maximum = logits[ids[0]] as f64;
    let normalizer = maximum
        + logits
            .iter()
            .map(|&v| (v as f64 - maximum).exp())
            .sum::<f64>()
            .ln();
    let lp = |id: u32| logits[id as usize] as f64 - normalizer;
    let top3 = ids
        .into_iter()
        .map(|id| TokenProbability {
            token_id: id as u32,
            logprob: lp(id as u32),
        })
        .collect();
    let queried = std::iter::once(target)
        .chain(query.iter().copied())
        .map(|id| (id, lp(id)))
        .collect();
    Ok((lp(target), top3, queried))
}

impl ScoreRequests {
    pub(crate) fn validate(&self, vocab: usize, context: usize) -> Result<()> {
        let mut ids = BTreeSet::new();
        if self.cases.is_empty() {
            return Err("Scoring requires at least one case".into());
        }
        for case in &self.cases {
            if case.id.is_empty()
                || !ids.insert(&case.id)
                || case.prompt_ids.is_empty()
                || case.target_ids.is_empty()
                || case
                    .prompt_ids
                    .len()
                    .checked_add(case.target_ids.len())
                    .is_none_or(|n| n > context)
                || (!case.query_ids.is_empty() && case.query_ids.len() != case.target_ids.len())
                || case
                    .prompt_ids
                    .iter()
                    .chain(&case.target_ids)
                    .chain(case.query_ids.iter().flatten())
                    .any(|&id| id as usize >= vocab)
            {
                return Err(format!("{}: invalid scoring case/history/context", case.id));
            }
        }
        Ok(())
    }
}

impl ModelRuntime {
    pub(crate) fn score(mut self, requests: ScoreRequests) -> Result<ScoreReport> {
        requests.validate(self.manifest.vocab, self.manifest.max_context)?;
        // This model instance is consumed by scoring; no live request or MTP
        // branch can observe the diagnostic policy. All real kernels stay intact.
        self.manifest.mtp = None;
        self.prefix_cache.budget = 0;
        self.host_prefix_cache.budget = 0;
        let options = crate::sampling::Options {
            temperature: 0.,
            seed: requests.seed,
            repetition_penalty: 1.,
            ..Default::default()
        };
        let start = Instant::now();
        let mut probes = vec![];
        for case in requests.cases {
            // Keep the normal request arena leased while scoring. Whole-request
            // generation releases and resets it before returning to its caller.
            let mut request = self.start_request(
                requests::GenerationInput {
                    audios: case.audios.clone(),
                    input_tokens: case.prompt_ids.clone(),
                    images: case.images.clone(),
                    max_new_tokens: case.target_ids.len(),
                    sampling: options.clone(),
                    prefix_hints: vec![],
                },
                &|| false,
            )?;
            let schedule = crate::scheduler::Options {
                max_active: 1,
                ..Default::default()
            };
            while request.is_prefilling() {
                self.advance_requests(&mut [&mut request], &schedule)?;
            }
            let mut history = case.prompt_ids;
            for (position, &target) in case.target_ids.iter().enumerate() {
                if position != 0 {
                    self.upload_ids(&self.manifest.token, &[case.target_ids[position - 1]])?;
                    let last = history.len() - 1;
                    self.prepare_inputs(&history[last..], &history[..last])?;
                    self.upload_segment_controls(1)?;
                    self.launch_program("decode", ExecutionPhase::Decode)?;
                }
                if self.read_control(&self.manifest.position)? as usize != history.len() {
                    return Err(format!(
                        "{}: teacher-forced scoring position mismatch at {position}",
                        case.id
                    ));
                }
                let spec = self
                    .manifest
                    .buffers
                    .iter()
                    .find(|b| b.name == self.manifest.logits)
                    .ok_or("Scoring logits buffer missing")?;
                let logits = floats(
                    &self.execution.download_bytes(&spec.name, spec.bytes()?)?,
                    spec.dtype,
                );
                let query = case
                    .query_ids
                    .get(position)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let (reference_logprob, top3, queried_logprobs) =
                    probabilities(&logits, target, query)?;
                probes.push(Probe {
                    case_id: case.id.clone(),
                    execution_mode: "decode",
                    position,
                    seed: requests.seed,
                    context_sha256: scoring_context_hash(&history, &case.images, &case.audios)?,
                    reference_token_id: target,
                    reference_logprob,
                    top3,
                    queried_logprobs,
                });
                history.push(target);
            }
            self.finish_request(&mut request, false)?;
        }
        Ok(ScoreReport {
            manifest_sha256: self.stats.manifest_sha256,
            seed: requests.seed,
            load_to_ready_s: self.stats.load_to_ready_s,
            scoring_s: start.elapsed().as_secs_f64(),
            probes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_context_matches_python_and_preserves_image_order() {
        let image = crate::vision::ImageInput {
            rgb: Vec::new(),
            grid_height: 2,
            grid_width: 2,
            pixels: vec![0.; 6144],
        };
        let hash = scoring_context_hash(&[1, 2], std::slice::from_ref(&image), &[]).unwrap();
        assert_eq!(
            hash,
            "bf01648ff97c3d303d1e215026c2a49fa1b829b8ee66fb1edd201046f8ca8977"
        );
        let mut changed = image.clone();
        changed.pixels[0] = 1.;
        assert_ne!(
            hash,
            scoring_context_hash(&[1, 2], std::slice::from_ref(&changed), &[]).unwrap()
        );
        assert_ne!(
            scoring_context_hash(&[1, 2], &[image.clone(), changed.clone()], &[]).unwrap(),
            scoring_context_hash(&[1, 2], &[changed, image], &[]).unwrap()
        );
    }
    #[test]
    fn rgb_context_and_patch_bytes_match_python() {
        let image = crate::vision::ImageInput {
            grid_height: 2,
            grid_width: 4,
            rgb: (0..32 * 64 * 3).map(|n| (n * 37 % 256) as u8).collect(),
            ..Default::default()
        };
        assert_eq!(
            scoring_context_hash(&[1, 2], std::slice::from_ref(&image), &[]).unwrap(),
            "8b1eeb5f09290823371a0fec3be54f5ceaf1a7e7e1ef7f1e19dea0d9d77e2ce2"
        );
        let v: crate::vision::VisionSpec = serde_json::from_value(serde_json::json!({"hidden":8,
            "patch_size":16,"temporal_patch_size":2,"merge_size":2,"max_patches":64,"max_features":16,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":64,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12})).unwrap();
        v.feature_count(&image).unwrap();
        let raw: Vec<_> = image.normalized(&v).flat_map(f32::to_le_bytes).collect();
        assert_eq!(
            crate::artifact::sha256(&raw),
            "9d0d0fdf119bd16c9a3ebf528dd1be1153bfa805736b4dbfe176ef302e71eaa2"
        );
        let mut changed = image.clone();
        changed.rgb[0] ^= 255;
        assert_ne!(
            scoring_context_hash(&[1, 2], &[image.clone(), changed.clone()], &[]).unwrap(),
            scoring_context_hash(&[1, 2], &[changed, image], &[]).unwrap()
        );
    }
    #[test]
    fn audio_changes_scoring_identity() {
        let mut audio = crate::audio::AudioInput {
            frames: 50,
            mel: vec![0.; 12800],
        };
        let first = scoring_context_hash(&[1, 2], &[], std::slice::from_ref(&audio)).unwrap();
        audio.mel[0] = 1.;
        assert_ne!(first, scoring_context_hash(&[1, 2], &[], &[audio]).unwrap());
        assert_ne!(first, scoring_context_hash(&[1, 2], &[], &[]).unwrap());
    }
    #[test]
    fn full_vocabulary_normalization_queries_and_ties() {
        let (target, top, query) = probabilities(&[0., 2., 2., -1., 3.], 3, &[0, 4]).unwrap();
        assert_eq!(
            top.iter().map(|v| v.token_id).collect::<Vec<_>>(),
            [4, 1, 2]
        );
        let norm = (1. + 2_f64.exp() * 2. + (-1_f64).exp() + 3_f64.exp()).ln();
        assert!((target - (-1. - norm)).abs() < 1e-12);
        assert!((query[&0] + norm).abs() < 1e-12);
        assert_eq!(query[&3], target);
        assert!(probabilities(&[0., f32::NAN, 1.], 0, &[]).is_err());
        assert!(probabilities(&[0., 1., 2.], 3, &[]).is_err());
    }
    #[test]
    fn history_budget_query_positions_and_unique_cases_are_checked() {
        let mut requests = ScoreRequests {
            seed: 20261002,
            cases: vec![ScoreCase {
                id: "test".into(),
                prompt_ids: vec![1, 2],
                target_ids: vec![3, 4],
                images: vec![],
                audios: vec![],
                query_ids: vec![],
            }],
        };
        assert!(requests.validate(5, 4).is_ok());
        assert!(requests.validate(5, 3).is_err());
        assert!(requests.validate(4, 4).is_err());
        requests.cases[0].query_ids = vec![vec![1]];
        assert!(requests.validate(5, 4).is_err());
    }
}
