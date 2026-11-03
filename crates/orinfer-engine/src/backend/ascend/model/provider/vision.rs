//! Bounded image admission across the existing tensor-parallel workers.
use super::*;
use std::collections::VecDeque;
pub(super) struct Pending {
    pub start: usize,
    pub identity: [u8; 32],
    pub image: crate::vision::ImageInput,
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
                    image,
                });
            }
        }
        Ok(pending)
    }
    pub(super) fn prepare_images(&mut self, requests: &mut [&mut Request]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let mut selected: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut identities: std::collections::BTreeMap<[u8; 32], usize> = Default::default();
        let mut images = Vec::new();
        let mut installed = std::collections::BTreeSet::new();
        let mut cache_hits = 0;
        let mut coalesced = 0;
        let at = Instant::now();
        let start = self.vision_cursor % requests.len();
        loop {
            let mut progressed = false;
            for i in (start..requests.len()).chain(0..start) {
                let request = &mut requests[i];
                if request.is_finished() {
                    continue;
                }
                let Some(pending) = request.pending_images.front() else {
                    continue;
                };
                if let Some(data) = self.vision_cache.get(&pending.identity) {
                    Arc::make_mut(&mut request.image_features).push(super::super::vision::Span {
                        start: pending.start,
                        data,
                    });
                    request.pending_images.pop_front();
                    installed.insert(i);
                    cache_hits += 1;
                } else if let Some(&rank) = identities.get(&pending.identity) {
                    let pending = request.pending_images.pop_front().unwrap();
                    selected[rank].push((i, pending.start));
                    coalesced += 1;
                } else if images.len() < self.team.ranks() {
                    let pending = request.pending_images.pop_front().unwrap();
                    identities.insert(pending.identity, images.len());
                    selected.push(vec![(i, pending.start)]);
                    images.push(pending.image);
                } else {
                    continue;
                }
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        if images.is_empty() && installed.is_empty() {
            return Ok(());
        }
        self.vision_cursor = (start + 1) % requests.len();
        let count = images.len();
        let profiling = std::env::var_os("ORINFER_A2_VISION_PROFILE").is_some();
        let mut timings = profiling.then(Vec::new);
        let replies = if images.is_empty() {
            Vec::new()
        } else {
            self.team
                .exchange(Command::EncodeImages(Arc::new(images)))?
        };
        for (rank, reply) in replies.into_iter().enumerate() {
            let super::super::team::Reply::Image(encoded) = reply else {
                return Err("Invalid image encoder reply".into());
            };
            if rank >= count {
                if encoded.is_some() {
                    return Err("Unexpected unassigned image output".into());
                }
                continue;
            }
            let encoded = encoded.ok_or("Missing assigned image output")?;
            if let Some(timings) = &mut timings {
                timings.push(serde_json::json!({"rank":rank,"owners":selected[rank].iter().map(|(i,_)|requests[*i].slot).collect::<Vec<_>>(),"patches":encoded.patches,"span":encoded.span,"upload_s":encoded.upload_s,"compute_s":encoded.compute_s,"download_s":encoded.download_s,"graph_replay":encoded.graph_replay}));
            }
            let data = Arc::new(encoded.data);
            let identity = identities
                .iter()
                .find(|(_, index)| **index == rank)
                .unwrap()
                .0;
            self.vision_cache
                .insert(*identity, Arc::clone(&data), data.capacity());
            for &(i, start) in &selected[rank] {
                Arc::make_mut(&mut requests[i].image_features).push(super::super::vision::Span {
                    start,
                    data: Arc::clone(&data),
                });
                installed.insert(i);
            }
            let per_patch = (encoded.upload_s + encoded.compute_s + encoded.download_s)
                / encoded.patches as f64;
            self.vision_seconds_per_patch = self.vision_seconds_per_patch * 0.8 + per_patch * 0.2;
        }
        for i in installed {
            let request = &requests[i];
            self.team.exchange(Command::InstallImages(
                request.slot,
                Arc::clone(&request.image_features),
            ))?;
        }
        if let Some(timings) = timings {
            eprintln!(
                "A2 vision {}",
                serde_json::json!({"images":count,"cache_hits":cache_hits,"coalesced":coalesced,"cache_bytes":self.vision_cache.bytes(),"wall_s":at.elapsed().as_secs_f64(),"encoders":timings})
            );
        }
        Ok(())
    }
}
