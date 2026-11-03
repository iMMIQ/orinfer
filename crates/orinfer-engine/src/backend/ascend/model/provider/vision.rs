//! Bounded image frames across the existing tensor-parallel workers.
use super::*;
use std::collections::VecDeque;
pub(super) struct Pending {
    pub start: usize,
    pub identity: [u8; 32],
    pub image: Option<crate::vision::ImageInput>,
}

pub(super) struct Wave {
    identities: Vec<[u8; 32]>,
    started: Instant,
    max_patches: usize,
}
/// Bind completed immutable features by content, rather than by a reusable
/// owner index. Keep every completed image, even when an earlier image is pending
/// and cross-wave cache retention is disabled or too small for the whole wave.
fn resolve(
    requests: &mut [&mut Request],
    cache: &mut crate::vision::FeatureCache<Vec<u8>>,
    completed: &BTreeMap<[u8; 32], Arc<Vec<u8>>>,
    installed: &mut BTreeSet<usize>,
) -> usize {
    let mut hits = 0;
    for (i, request) in requests.iter_mut().enumerate() {
        if request.is_finished() {
            continue;
        }
        let mut pending = VecDeque::new();
        while let Some(image) = request.pending_images.pop_front() {
            let data = if let Some(data) = completed.get(&image.identity) {
                Some(Arc::clone(data))
            } else {
                let data = cache.get(&image.identity);
                hits += usize::from(data.is_some());
                data
            };
            if let Some(data) = data {
                Arc::make_mut(&mut request.image_features).push(super::super::vision::Span {
                    start: image.start,
                    data,
                });
                installed.insert(i);
            } else {
                pending.push_back(image);
            }
        }
        request.pending_images = pending;
        if installed.contains(&i) {
            Arc::make_mut(&mut request.image_features).sort_unstable_by_key(|s| s.start);
        }
    }
    hits
}
impl Runtime {
    pub(super) fn pending_images(
        &self,
        input: &[u32],
        images: Vec<crate::vision::ImageInput>,
        offset: usize,
        media: &Media,
    ) -> Result<VecDeque<Pending>> {
        let mut pending = VecDeque::new();
        let Some(spec) = self.vision() else {
            return if images.is_empty() {
                Ok(pending)
            } else {
                Err("Model has no image adapter".into())
            };
        };
        for (image, (start, identity)) in images.into_iter().zip(&media.0) {
            let end = start
                .checked_add(spec.feature_count(&image)?)
                .ok_or("Image span overflow")?;
            if end > input.len() {
                return Err("Image span exceeds prompt".into());
            }
            if end > offset {
                pending.push_back(Pending {
                    start: *start,
                    identity: *identity,
                    image: Some(image),
                });
            }
        }
        Ok(pending)
    }
    pub(super) fn prepare_images(
        &mut self,
        requests: &mut [&mut Request],
        options: &scheduler::Options,
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let at = Instant::now();
        let mut installed = BTreeSet::new();
        let mut cache_hits = resolve(
            requests,
            &mut self.vision_cache,
            &BTreeMap::new(),
            &mut installed,
        );
        if self.image_wave.as_ref().is_some_and(|wave| {
            !requests.iter().any(|r| {
                !r.is_finished()
                    && r.pending_images
                        .iter()
                        .any(|p| wave.identities.contains(&p.identity))
            })
        }) {
            self.team.exchange(Command::CancelImages)?;
            self.image_wave = None;
        }
        let mut coalesced = 0;
        if self.image_wave.is_none() {
            let start = self.vision_cursor % requests.len();
            let depth = requests
                .iter()
                .map(|r| r.pending_images.len())
                .max()
                .unwrap_or(0);
            let mut images = Vec::new();
            let mut identities = Vec::new();
            'selection: for depth in 0..depth {
                for i in (start..requests.len()).chain(0..start) {
                    let request = &mut requests[i];
                    if request.is_finished() {
                        continue;
                    }
                    let Some(pending) = request.pending_images.get_mut(depth) else {
                        continue;
                    };
                    if identities.contains(&pending.identity) {
                        coalesced += 1;
                        continue;
                    }
                    let image = pending.image.take().ok_or("Unowned pending image frame")?;
                    images.push(image);
                    identities.push(pending.identity);
                    if images.len() == self.team.ranks() {
                        break 'selection;
                    }
                }
            }
            if !images.is_empty() {
                let max_patches = images
                    .iter()
                    .map(|i| i.grid_height * i.grid_width)
                    .max()
                    .unwrap();
                self.team.exchange(Command::BeginImages(Arc::new(images)))?;
                self.image_wave = Some(Wave {
                    identities,
                    started: at,
                    max_patches,
                });
                self.vision_cursor = (start + 1) % requests.len();
            }
        }
        let profiling = std::env::var_os("ORINFER_A2_VISION_PROFILE").is_some();
        let mut timings = profiling.then(Vec::new);
        let mut count = 0;
        let mut wave_wall_s = None;
        if let Some(wave) = &self.image_wave {
            let bounded = requests
                .iter()
                .any(|r| !r.prefilling && !r.is_finished() && r.ready_output.is_none());
            // Merge cheap segments in the same submission. Reserve most of the
            // output interval for admission, the language packet and its draft.
            let budget = if bounded {
                self.team.model.vision.as_ref().unwrap().advance_budget(
                    wave.max_patches as f64 * self.vision_seconds_per_patch,
                    (options.target_tpot_ms / 4000.).min(0.100),
                )
            } else {
                usize::MAX
            };
            let replies = self.team.exchange(Command::AdvanceImages(budget))?;
            let mut encoded = Vec::new();
            for (rank, reply) in replies.into_iter().enumerate() {
                let super::super::team::Reply::Image(image) = reply else {
                    return Err("Invalid image frame reply".into());
                };
                if rank >= wave.identities.len() && image.is_some() {
                    return Err("Unexpected unassigned image output".into());
                }
                if let Some(image) = image {
                    encoded.push((rank, image));
                }
            }
            if !encoded.is_empty() {
                if encoded.len() != wave.identities.len() {
                    return Err("TP ranks have different vision frame boundaries".into());
                }
                let wave = self.image_wave.take().unwrap();
                count = wave.identities.len();
                wave_wall_s = Some(wave.started.elapsed().as_secs_f64());
                let mut completed = BTreeMap::new();
                for (rank, image) in encoded {
                    let identity = wave.identities[rank];
                    if let Some(timings) = &mut timings {
                        timings.push(serde_json::json!({"rank":rank,"patches":image.patches,"span":image.span,"upload_s":image.upload_s,"compute_s":image.compute_s,"download_s":image.download_s,"graph_replay":image.graph_replay,"max_advance_s":image.max_advance_s}));
                    }
                    let per_patch = (image.upload_s + image.compute_s + image.download_s)
                        / image.patches as f64;
                    self.vision_seconds_per_patch =
                        self.vision_seconds_per_patch * 0.8 + per_patch * 0.2;
                    let data = Arc::new(image.data);
                    self.vision_cache
                        .insert(identity, Arc::clone(&data), data.capacity());
                    completed.insert(identity, data);
                }
                cache_hits += resolve(requests, &mut self.vision_cache, &completed, &mut installed);
            }
        }
        let changed = !installed.is_empty();
        for i in installed {
            let request = &requests[i];
            self.team.exchange(Command::InstallImages(
                request.slot,
                Arc::clone(&request.image_features),
            ))?;
        }
        if let Some(timings) = timings
            && (count > 0 || self.image_wave.is_some() || changed)
        {
            eprintln!(
                "A2 vision {}",
                serde_json::json!({"images":count,"inflight_images":self.image_wave.as_ref().map_or(0,|w|w.identities.len()),"cache_hits":cache_hits,"coalesced":coalesced,"cache_bytes":self.vision_cache.bytes(),"wall_s":at.elapsed().as_secs_f64(),"wave_wall_s":wave_wall_s,"encoders":timings})
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn owner(slot: usize, images: &[(usize, u8)]) -> Request {
        let output = Arc::new(Output {
            owner: 1,
            token: 0,
            value: 0.,
            logits: vec![],
        });
        let mut request = super::super::tests::request(output, Default::default());
        request.slot = slot;
        request.ready_output = None;
        request.input = vec![0; 64];
        request.pending_images = images
            .iter()
            .map(|&(start, key)| Pending {
                start,
                identity: [key; 32],
                image: None,
            })
            .collect();
        request
    }
    #[test]
    fn completed_frames_follow_content_across_owner_reuse_and_disabled_cache() {
        let mut cache = crate::vision::FeatureCache::new(0);
        let mut first = owner(7, &[(10, 1), (30, 2)]);
        let mut second = owner(2, &[(5, 2)]);
        let features = Arc::new(vec![2u8; 5120 * 2]);
        let completed = BTreeMap::from([([2; 32], Arc::clone(&features))]);
        let mut installed = BTreeSet::new();
        resolve(
            &mut [&mut second, &mut first],
            &mut cache,
            &completed,
            &mut installed,
        );
        assert_eq!(installed, BTreeSet::from([0, 1]));
        assert!(second.pending_images.is_empty());
        assert_eq!(first.pending_images.len(), 1);
        assert_eq!(first.image_features[0].start, 30);
        assert!(Arc::ptr_eq(
            &first.image_features[0].data,
            &second.image_features[0].data
        ));
        // A later completion may arrive after cancellation and slot reuse.
        let mut replacement = owner(7, &[(1, 3)]);
        let completed = BTreeMap::from([([1; 32], Arc::new(vec![1; 5120 * 2]))]);
        installed.clear();
        resolve(
            &mut [&mut replacement, &mut first],
            &mut cache,
            &completed,
            &mut installed,
        );
        assert_eq!(installed, BTreeSet::from([1]));
        assert!(replacement.image_features.is_empty());
        assert_eq!(replacement.pending_images.len(), 1);
        assert_eq!(
            first
                .image_features
                .iter()
                .map(|s| s.start)
                .collect::<Vec<_>>(),
            [10, 30]
        );
        assert_eq!(cache.bytes(), 0);
    }
}
