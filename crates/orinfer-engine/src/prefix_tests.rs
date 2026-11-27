use super::*;
#[test]
fn image_identity_preserves_text_prefix_and_separates_changed_or_reordered_images() {
    let vision: crate::vision::VisionSpec = serde_json::from_value(serde_json::json!({
        "layout":"deepseek_v41","max_image_tokens":1024,"hidden":5120,
        "patch_size":14,"temporal_patch_size":1,"merge_size":3,"max_patches":9216,
        "max_features":4096,"pixels":"p","grid":"g","length":"n","output":"o",
        "features":"f","feature_index":"i","mrope_positions":"r",
        "plans":[{"patches":9216,"program":"vision"}],"image_token_id":10,
        "vision_start_token_id":10,"vision_end_token_id":10
    }))
    .unwrap();
    let red = crate::vision::ImageInput {
        grid_height: 2,
        grid_width: 4,
        rgb: vec![127; 2 * 4 * 14 * 14 * 3],
        ..Default::default()
    };
    let mut blue = red.clone();
    blue.rgb[0] = 128;
    let images = [red.clone(), blue.clone()];
    let tokens = vision.expand(&[1, 10, 2, 10, 3], &images, 32).unwrap();
    let original = Media::images(&tokens, &images, Some(&vision)).unwrap();
    let same = Media::images(&tokens, &images, Some(&vision)).unwrap();
    let changed = Media::images(&tokens, &[red.clone(), red], Some(&vision)).unwrap();
    let reversed = Media::images(&tokens, &[blue, images[0].clone()], Some(&vision)).unwrap();
    assert_eq!(original, same);
    assert_eq!(original.common_tokens(&same, tokens.len()), tokens.len());
    assert_eq!(original.common_tokens(&changed, tokens.len()), 7);
    assert_eq!(original.common_tokens(&reversed, tokens.len()), 1);
    let mut invalid = tokens;
    invalid[2] = 4;
    assert!(Media::images(&invalid, &images, Some(&vision)).is_err());
}
#[test]
fn decoded_insertion_preserves_its_prompt_but_admission_can_reclaim_it() {
    let mut cache = Cache::new(100);
    let prompt = cache.insert(entry(&[1, 2], &[(1, 10)], true));
    cache.insert(entry(&[1, 2, 3, 4, 5], &[(2, 10)], false));
    assert!(cache.evict_one_except(Some(prompt)).is_some());
    assert_eq!(cache.find(&[1, 2], &Media::default()), Some(prompt));
    assert!(cache.evict_one_except(Some(prompt)).is_none());
    assert_eq!(cache.bytes, 10);
    assert!(cache.evict_one().is_some());
    assert_eq!(cache.bytes, 0);
}
#[derive(Clone)]
struct Blocks(Vec<(u64, usize)>);
impl Resident for Blocks {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        out.extend(self.0.iter().copied());
    }
}
fn entry(tokens: &[u32], blocks: &[(u64, usize)], valid: bool) -> Entry<Blocks> {
    Entry::new(
        tokens.to_vec(),
        Media::default(),
        Blocks(blocks.to_vec()),
        blocks.iter().map(|b| b.1).sum(),
        tokens.len() - 1,
        valid,
    )
}
#[test]
fn capture_prefers_unproven_continuations_even_when_they_are_recent() {
    let mut cache = Cache::new(100);
    let prompt = cache.insert(entry(&[1], &[(1, 50)], true));
    let tail = cache.insert(entry(&[2, 3, 4, 5], &[(2, 10)], false));
    assert_eq!(cache.victim_except(None), Some(prompt));
    assert_eq!(cache.victim_for_capture(|_| false), Some(tail));
    cache.touch(tail);
    assert_eq!(cache.victim_for_capture(|_| false), Some(prompt));
    assert_eq!(cache.victim_for_capture(|id| id == prompt), Some(tail));
    cache.mark_recent(prompt).unwrap();
    assert_eq!(cache.entries[&prompt].hits, 0);
}

#[test]
fn selects_cheapest_compatible_endpoint_and_preserves_raw_overlap() {
    let mut c = Cache::new(100);
    let a = c.insert(entry(&[1, 2], &[(1, 10)], true));
    let b = c.insert(entry(&[1, 2, 3], &[(2, 10)], true));
    let query = [1, 2, 3, 4];
    assert_eq!(
        c.match_prefix_by(&query, &Media::default(), |e| {
            Some(if e.tokens.len() == 2 { 1. } else { 2. })
        }),
        Match {
            checkpoint: Some(a),
            matched_tokens: 3
        }
    );
    assert_eq!(
        c.match_prefix_by(&query, &Media::default(), |_| None),
        Match {
            checkpoint: None,
            matched_tokens: 3
        }
    );
    assert_eq!(c.find(&query, &Media::default()), Some(b));
}
#[test]
fn splits_branches_and_never_manufactures_an_intermediate_state() {
    let mut c = Cache::new(1000);
    let a = c.insert(entry(&[1, 2, 3, 4], &[(1, 10)], true));
    assert_eq!(
        c.match_prefix(&[1, 2, 3, 9], &Media::default()),
        Match {
            checkpoint: None,
            matched_tokens: 3
        }
    );
    let b = c.insert(entry(&[1, 2], &[(2, 10)], true));
    let d = c.insert(entry(&[1, 2, 3, 9], &[(3, 10)], true));
    assert_eq!(c.find(&[1, 2, 3, 4, 5], &Media::default()), Some(a));
    assert_eq!(c.find(&[1, 2, 3, 8], &Media::default()), Some(b));
    c.remove(b);
    assert_eq!(c.find(&[1, 2, 3, 8], &Media::default()), None);
    assert_eq!(c.find(&[1, 2, 3, 9], &Media::default()), Some(d));
    c.remove(a);
    c.remove(d);
    assert_eq!(c.bytes, 0);
    assert!(c.root.children.is_empty());
}
#[test]
fn image_changes_only_invalidate_the_affected_suffix() {
    let mut c = Cache::new(1000);
    let plain = c.insert(entry(&[1, 2], &[(1, 10)], true));
    let mut a = entry(&[1, 2, 3, 4, 5], &[(2, 10)], true);
    a.media = Media(vec![(2, [1; 32]), (4, [2; 32])]);
    c.insert(a);
    let changed = Media(vec![(2, [1; 32]), (4, [3; 32])]);
    assert_eq!(
        c.match_prefix(&[1, 2, 3, 4, 5, 6], &changed),
        Match {
            checkpoint: Some(plain),
            matched_tokens: 4
        }
    );
    assert_eq!(
        c.match_prefix(&[1, 2, 3, 4, 5, 6], &Media(vec![(2, [9; 32])]))
            .matched_tokens,
        2
    );
}
#[test]
fn byte_budget_counts_shared_blocks_and_has_no_eight_entry_cap() {
    let mut c = Cache::new(1000);
    for i in 1..=20 {
        c.insert(entry(&vec![1; i], &[(1, 100), (i as u64 + 1, 10)], true));
    }
    assert_eq!(c.entries.len(), 20);
    assert_eq!(c.bytes, 300);
    for _ in 0..19 {
        c.evict_one().unwrap();
    }
    assert_eq!(c.bytes, 110);
    c.evict_one();
    assert_eq!(c.bytes, 0);
}
#[test]
fn output_state_requires_an_extension_when_its_head_is_stale() {
    let mut c = Cache::new(100);
    let p = c.insert(entry(&[1, 2], &[(1, 10)], true));
    let g = c.insert(entry(&[1, 2, 3], &[(2, 10)], false));
    assert_eq!(c.find(&[1, 2, 3], &Media::default()), Some(p));
    assert_eq!(c.find(&[1, 2, 3, 4], &Media::default()), Some(g));
    let mut upgraded = entry(&[1, 2, 3], &[(3, 10)], true);
    upgraded.media = Media::default();
    c.insert(upgraded);
    assert_ne!(c.find(&[1, 2, 3], &Media::default()), Some(p));
}
#[test]
fn radix_matches_a_linear_oracle_after_arbitrary_splits_and_evictions() {
    let mut c = Cache::new(1_000_000);
    let mut random = 20261002u64;
    for step in 0..400 {
        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
        let len = 1 + ((random >> 32) as usize % 12);
        let tokens: Vec<_> = (0..len)
            .map(|i| ((random >> (i * 3 % 48)) % 5) as u32)
            .collect();
        let mut e = entry(&tokens, &[(step + 1, 10)], step % 3 != 0);
        e.media = Media(vec![(2, [(step % 3) as u8; 32])]);
        c.insert(e);
        if step % 4 == 0 {
            c.evict_one();
        }
        let query: Vec<_> = tokens.iter().copied().chain([7]).collect();
        let media = Media(vec![(2, [(step % 2) as u8; 32])]);
        let found = c.match_prefix(&query, &media);
        let mut expected_match = 0;
        let mut expected_checkpoint = 0;
        for e in c.entries.values() {
            let raw = e
                .tokens
                .iter()
                .zip(&query)
                .take_while(|(a, b)| a == b)
                .count();
            let common = e.media.common_tokens(&media, raw);
            expected_match = expected_match.max(common);
            if common == e.tokens.len() && (common < query.len() || e.logits_valid) {
                expected_checkpoint = expected_checkpoint.max(common);
            }
        }
        assert_eq!(
            found.matched_tokens, expected_match,
            "step {step}, query {query:?}, media {media:?}"
        );
        assert_eq!(
            found
                .checkpoint
                .map(|id| c.entries[&id].tokens.len())
                .unwrap_or(0),
            expected_checkpoint
        );
    }
}
#[test]
fn exact_matches_returns_only_same_tokens_and_media() {
    let mut c = Cache::new(1_000_000);
    let m1 = Media(vec![(1, [1u8; 32])]);
    let m2 = Media(vec![(1, [2u8; 32])]);
    let mut a = entry(&[10, 20, 30], &[(1, 10)], false);
    a.media = m1.clone();
    let id_a = c.insert(a);
    let mut b = entry(&[10, 20, 30], &[(2, 10)], true);
    b.media = m1.clone();
    let id_b = c.insert(b);
    let mut diff_media = entry(&[10, 20, 30], &[(3, 10)], true);
    diff_media.media = m2.clone();
    let id_m2 = c.insert(diff_media);
    let mut shorter = entry(&[10, 20], &[(4, 10)], true);
    shorter.media = m1.clone();
    c.insert(shorter);
    let mut longer = entry(&[10, 20, 30, 40], &[(5, 10)], true);
    longer.media = m1.clone();
    c.insert(longer);

    let mut got = c.exact_matches(&[10, 20, 30], &m1);
    got.sort_unstable();
    let mut expected = vec![id_a, id_b];
    expected.sort_unstable();
    assert_eq!(got, expected);
    assert_eq!(c.exact_matches(&[10, 20, 30], &m2), vec![id_m2]);
    assert!(c.exact_matches(&[10, 20, 99], &m1).is_empty());
    assert!(c.exact_matches(&[10], &m1).is_empty());
}
#[test]
fn incremental_accounting_matches_full_recomputation_at_scale() {
    let mut c = Cache::new(100_000_000);
    for i in 1..=600u64 {
        let shared = (i % 17) + 1;
        let tokens = vec![(i % 7) as u32, (i / 7) as u32, (i % 13) as u32];
        c.insert(entry(
            &tokens,
            &[(shared, 1000), (1000 + i, 50)],
            i % 2 == 0,
        ));
        if i % 5 == 0 {
            c.evict_one().unwrap();
        }
    }
    let mut oracle = BTreeMap::new();
    let mut logical = 0usize;
    for e in c.entries.values() {
        e.snapshot.allocations(&mut oracle);
        logical += e.bytes;
    }
    assert_eq!(c.bytes, oracle.values().sum::<usize>());
    assert_eq!(c.resident_with(None), c.bytes);
    let extra = Blocks(vec![(1, 1000), (999_999, 777)]);
    assert_eq!(c.resident_with(Some(&extra)), c.bytes + 777);
    let mut stats = Statistics::default();
    c.update_statistics(&mut stats);
    assert_eq!(stats.logical_bytes, logical);
    assert_eq!(stats.resident_bytes, c.bytes);
    assert_eq!(stats.shared_bytes, logical - c.bytes);
}
