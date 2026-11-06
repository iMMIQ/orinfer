//! Release large, exclusively owned row payloads without a long resident munmap.
//! Allocation and inference keep the same Vec layout; only destruction changes.

pub(super) fn discard(data: &mut [u8]) {
    #[cfg(target_os = "linux")]
    if data.len() >= 128 << 20 {
        // Advice is best effort. Vec still owns the complete allocation and
        // performs normal deallocation even when the kernel rejects advice.
        let _ = discard_chunks(data, 128 << 20);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = data;
}

#[cfg(target_os = "linux")]
fn discard_chunks(data: &mut [u8], chunk_bytes: usize) -> bool {
    // SAFETY: sysconf takes a defined selector and owns no caller memory.
    let Ok(page) = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }) else {
        return false;
    };
    if !page.is_power_of_two() {
        return false;
    }
    let first = data.as_ptr().align_offset(page);
    if first >= data.len() {
        return true;
    }
    let end = first + (data.len() - first) / page * page;
    let chunk = (chunk_bytes / page).max(1) * page;
    for offset in (first..end).step_by(chunk) {
        let bytes = chunk.min(end - offset);
        // SAFETY: Only full pages wholly inside this exclusive payload are
        // discarded. Allocation headers, neighboring objects and partial tail
        // pages are untouched. The bytes are dead, have no DMA readers, and
        // remain mapped until their owning Vec immediately deallocates them.
        if unsafe {
            libc::madvise(
                data.as_mut_ptr().add(offset).cast(),
                bytes,
                libc::MADV_DONTNEED,
            )
        } != 0
        {
            return false;
        }
    }
    true
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn bounded_discard_keeps_allocator_edges_and_neighboring_bytes() {
        // SAFETY: sysconf takes a defined selector and owns no caller memory.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let mut allocation = vec![0x5a; page * 9 + 37];
        let begin = 23;
        let end = allocation.len() - 51;
        let first_page = begin + allocation[begin..end].as_ptr().align_offset(page);
        let last_page = first_page + (end - first_page) / page * page;
        assert!(last_page - first_page >= page * 6);
        assert!(discard_chunks(&mut allocation[begin..end], page * 2));
        assert!(allocation[..first_page].iter().all(|&b| b == 0x5a));
        assert!(allocation[first_page..last_page].iter().all(|&b| b == 0));
        assert!(allocation[last_page..].iter().all(|&b| b == 0x5a));
        // A slice smaller than a page has no whole pages to discard.
        let small_end = first_page.min(page / 2);
        assert!(discard_chunks(&mut allocation[1..small_end], page));
        assert!(allocation[1..small_end].iter().all(|&b| b == 0x5a));
    }
}
