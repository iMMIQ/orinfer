//! Bounded host-cache transfers. Pageable cache bytes never back an active DMA.
use super::*;

impl Session {
    /// Call after synchronizing producers. Regions may leave holes in the host bank.
    /// Each completed slab is copied before reuse; failed DMA retains native backing.
    pub(crate) fn download_regions(
        &self,
        regions: &[(Buffer, usize, usize, usize)],
        output: &mut [u8],
    ) -> Result<()> {
        for (buffer, source, destination, bytes) in regions {
            self.require(buffer)?;
            buffer.range(*source, *bytes)?;
            MemoryRange {
                offset: *destination,
                bytes: *bytes,
            }
            .within(output.len())?;
        }
        let capacity = (4 << 20).min(self.context.limits.pinned_bytes);
        self.pinned_cache
            .borrow_mut()
            .retain(|h| h.bytes <= capacity || Rc::strong_count(h) != 1);
        let mut index = 0;
        let mut offset = 0;
        while index < regions.len() {
            let host = self.input_pinned(capacity)?;
            let mut pieces = vec![];
            let mut written = 0;
            while index < regions.len() && written < capacity {
                let (buffer, source, destination, bytes) = &regions[index];
                let count = (bytes - offset).min(capacity - written);
                if count > 0 {
                    pieces.push((
                        buffer.clone(),
                        source + offset,
                        destination + offset,
                        written,
                        count,
                    ));
                }
                written += count;
                offset += count;
                if offset == *bytes {
                    index += 1;
                    offset = 0;
                }
            }
            if pieces.is_empty() {
                continue;
            }
            let ticket = self.submit(
                self.context.transfer,
                Retained {
                    _buffers: pieces.iter().map(|(b, ..)| b.clone()).collect(),
                    _host: Some(Rc::clone(&host)),
                    _extra: None,
                },
                |stream| {
                    for (buffer, source, _, start, count) in &pieces {
                        let pointer = buffer.range(*source, *count)?;
                        check(
                            // SAFETY: Validated extents; completion retains both DMA buffers. D2H=2.
                            unsafe {
                                (self.context.api.memcpy)(
                                    host.pointer.cast::<u8>().add(*start).cast(),
                                    *count,
                                    pointer,
                                    *count,
                                    2,
                                    stream,
                                )
                            },
                            "ACL host-cache download",
                        )?;
                    }
                    Ok(())
                },
            )?;
            ticket.wait()?;
            for (_, _, destination, start, count) in pieces {
                // SAFETY: DMA completed; both host ranges span count bytes.
                unsafe {
                    ptr::copy_nonoverlapping(
                        host.pointer.cast::<u8>().add(start),
                        output.as_mut_ptr().add(destination),
                        count,
                    );
                }
            }
        }
        self.reap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ascend::tests::Fixture;
    #[test]
    fn scattered_host_cache_transfers_split_and_validate_before_dma() {
        let mut fixture = Fixture::new();
        let session = fixture.session_with_limits(Limits {
            device_bytes: 128,
            pinned_bytes: 8,
            timeout_ms: 100,
        });
        let a = session.allocate(29).unwrap();
        let b = session.allocate(29).unwrap();
        let expected = (0..29).collect::<Vec<u8>>();
        session.fill(&a, 99).unwrap();
        session
            .upload_regions(&[
                (a.clone(), 3, &expected[..17]),
                (a.clone(), 22, &expected[17..24]),
            ])
            .unwrap()
            .wait()
            .unwrap();
        session.synchronize().unwrap();
        let mut host = vec![0; 35];
        session
            .download_regions(&[(a.clone(), 3, 2, 17), (a.clone(), 22, 25, 7)], &mut host)
            .unwrap();
        assert_eq!(&host[2..19], &expected[..17]);
        assert_eq!(&host[25..32], &expected[17..24]);
        assert_eq!(&host[19..25], &[0; 6]);
        assert!(session.pinned_bytes() <= 8);
        session.fill(&b, 0).unwrap();
        session
            .upload_regions(&[(b.clone(), 3, &host[2..19]), (b.clone(), 22, &host[25..32])])
            .unwrap()
            .wait()
            .unwrap();
        session.synchronize().unwrap();
        let mut actual = vec![0; 35];
        session
            .download_regions(
                &[(b.clone(), 3, 2, 17), (b.clone(), 22, 25, 7)],
                &mut actual,
            )
            .unwrap();
        assert_eq!(host, actual);
        assert!(
            session
                .download_regions(&[(a.clone(), 0, 0, 4), (a.clone(), 28, 10, 2)], &mut host)
                .is_err()
        );
        assert_eq!(host, actual, "bad tail must not start an earlier valid DMA");
        drop((a, b, session));
        fixture.assert_clean();
    }
}
