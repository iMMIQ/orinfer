use super::*;
fn observe(c: &mut Costs, w: Work, path: Path, seconds: f64) {
    for _ in 0..4 {
        c.observe(w, path, seconds, 0.);
    }
}
#[test]
fn incompatible_decode_context_does_not_hide_a_valid_mixed_cost() {
    let mut c = Costs::new(4000.);
    let wanted = Work {
        tokens: 1008,
        context: 32768,
        decode_rows: 16,
        decode_context: 262144,
        domain: 256,
        ..Default::default()
    };
    observe(
        &mut c,
        Work {
            tokens: 512,
            decode_context: 32768,
            ..wanted
        },
        Path::Direct,
        0.14,
    );
    observe(
        &mut c,
        Work {
            tokens: 1024,
            decode_rows: 64,
            ..wanted
        },
        Path::Direct,
        0.17,
    );
    let p = c.estimate_path(wanted, Path::Direct);
    assert!(p.samples > 0);
    assert!((p.upper_s - 0.166).abs() < 1e-9);
}
#[test]
fn decode_only_geometry_does_not_hide_a_valid_mixed_cost() {
    let mut c = Costs::new(4000.);
    let wanted = Work {
        tokens: 1008,
        context: 32768,
        decode_rows: 16,
        decode_context: 262144,
        domain: 256,
        ..Default::default()
    };
    observe(
        &mut c,
        Work {
            tokens: 0,
            ..wanted
        },
        Path::Direct,
        0.09,
    );
    observe(
        &mut c,
        Work {
            tokens: 1024,
            decode_rows: 64,
            ..wanted
        },
        Path::Direct,
        0.17,
    );
    let p = c.estimate_path(wanted, Path::Direct);
    assert!(p.samples > 0);
    assert!((p.upper_s - 0.166).abs() < 1e-9);
}
#[test]
fn learns_token_context_and_execution_cost_without_mixing_modes_or_final_windows() {
    let mut c = Costs::new(4000.);
    for context in [8192, 32768] {
        for tokens in [512, 2048] {
            let w = Work {
                tokens,
                context,
                domain: 128,
                ..Default::default()
            };
            observe(
                &mut c,
                w,
                Path::Direct,
                0.04 + tokens as f64 * 0.00005 * (context as f64 / 8192.),
            );
            observe(
                &mut c,
                w,
                Path::Replay,
                0.02 + tokens as f64 * 0.00005 * (context as f64 / 8192.),
            );
        }
    }
    let w = Work {
        tokens: 1024,
        context: 16384,
        domain: 128,
        ..Default::default()
    };
    assert!((c.estimate_path(w, Path::Replay).seconds - 0.1224).abs() < 1e-8);
    assert!((c.estimate_path(w, Path::Direct).seconds - 0.1424).abs() < 1e-8);
    observe(
        &mut c,
        Work {
            final_rows: 128,
            ..w
        },
        Path::Replay,
        1.,
    );
    assert!((c.estimate_path(w, Path::Replay).seconds - 0.1224).abs() < 1e-8);
    assert!(
        c.estimate_path(
            Work {
                final_rows: 128,
                ..w
            },
            Path::Replay
        )
        .seconds
            > 0.9
    );
}
#[test]
fn final_decoder_window_is_not_multiplied_when_prompt_size_changes() {
    let mut c = Costs::new(4000.);
    let w = Work {
        context: 8192,
        domain: 128,
        ..Default::default()
    };
    for tokens in [512, 2048] {
        observe(
            &mut c,
            Work { tokens, ..w },
            Path::Direct,
            0.02 + tokens as f64 * 0.00005,
        );
    }
    let final_prompt = Work {
        tokens: 128,
        final_rows: 128,
        full: true,
        ..w
    };
    observe(&mut c, final_prompt, Path::Direct, 0.1);
    let p = c.estimate(Work {
        tokens: 1024,
        ..final_prompt
    });
    assert!((p.seconds - 0.1448).abs() < 1e-9);
    let mixed = Work {
        decode_rows: 8,
        ..final_prompt
    };
    observe(&mut c, mixed, Path::Direct, 0.15);
    let p = c.estimate(Work {
        tokens: 1024,
        ..mixed
    });
    assert!((p.seconds - 0.1948).abs() < 1e-9);
}
#[test]
fn cold_direct_primer_does_not_block_hot_mode_transfer_to_larger_windows() {
    let mut c = Costs::new(4000.);
    let w = Work {
        domain: 128,
        ..Default::default()
    };
    for tokens in [512, 2048] {
        observe(
            &mut c,
            Work { tokens, ..w },
            Path::Replay,
            tokens as f64 / 10240.,
        );
    }
    let mixed = Work {
        tokens: 512,
        decode_rows: 8,
        ..w
    };
    c.observe(mixed, Path::Direct, 0.3, 0.2);
    observe(&mut c, mixed, Path::Replay, 0.1);
    let p = c.estimate(Work {
        tokens: 1024,
        ..mixed
    });
    assert_eq!(p.path, Path::Direct);
    assert_eq!(p.samples, 0);
    assert!((p.seconds - 0.15).abs() < 1e-9);
    assert!(p.upper_s < 0.2);
}
#[test]
fn adjacent_row_timing_noise_cannot_create_unbounded_extrapolation() {
    let mut c = Costs::new(4000.);
    let w = Work {
        domain: 128,
        ..Default::default()
    };
    observe(&mut c, Work { tokens: 1024, ..w }, Path::Direct, 0.12);
    observe(&mut c, Work { tokens: 1025, ..w }, Path::Direct, 0.14);
    let p = c.estimate(Work { tokens: 2048, ..w });
    assert!((0.24..0.29).contains(&p.seconds));
    assert!(p.upper_s < 0.3);
}
#[test]
fn sparse_mixed_observations_preserve_decoder_intercept() {
    let mut c = Costs::new(4000.);
    let w = Work {
        context: 8192,
        domain: 128,
        ..Default::default()
    };
    for tokens in [512, 2048] {
        observe(
            &mut c,
            Work { tokens, ..w },
            Path::Direct,
            0.02 + tokens as f64 * 0.00005,
        );
    }
    observe(
        &mut c,
        Work {
            tokens: 128,
            decode_rows: 8,
            ..w
        },
        Path::Direct,
        0.125,
    );
    let p = c.estimate(Work {
        tokens: 1024,
        decode_rows: 8,
        ..w
    });
    assert!((p.seconds - 0.1698).abs() < 1e-8);
    assert!(p.upper_s < 0.2);
}
#[test]
fn preparation_and_first_observation_do_not_poison_hot_cost_or_claim_replay_residency() {
    let mut c = Costs::new(4000.);
    let w = Work {
        tokens: 1024,
        ..Default::default()
    };
    c.observe(w, Path::Direct, 2., 1.);
    observe(&mut c, w, Path::Direct, 0.1);
    assert!((c.estimate(w).seconds - 0.1).abs() < 1e-9);
    assert_eq!(c.estimate(w).path, Path::Direct);
    c.observe(w, Path::Replay, 0.08, 0.4);
    observe(&mut c, w, Path::Replay, 0.06);
    assert!((c.estimate(w).seconds - 0.06).abs() < 1e-9);
    assert_eq!(c.estimate(Work { tokens: 1025, ..w }).path, Path::Direct);
}
#[test]
fn cache_restore_paths_and_observation_memory_are_bounded() {
    let mut c = Costs::new(4000.);
    c.observe_restore(false, 100000, 0.01);
    c.observe_restore(true, 100000, 0.2);
    assert_eq!(c.restore(false, 100001), 0.01);
    assert_eq!(c.restore(true, 100001), 0.2);
    assert_eq!(c.restore(false, 0), 0.);
    for tokens in 1..=1000 {
        c.observe(
            Work {
                tokens,
                ..Default::default()
            },
            Path::Direct,
            0.1,
            0.,
        );
    }
    assert_eq!(c.curves.cells.len(), 256);
    assert!(c.curves.modes.len() <= 256);
    c.observe(
        Work {
            tokens: 1,
            ..Default::default()
        },
        Path::Direct,
        f64::NAN,
        0.,
    );
    assert_eq!(c.curves.cells.len(), 256);
}

mod profile_tests {
    use super::super::*;

    #[test]
    fn stage_buckets_reuse_observations_across_layers_and_dynamic_lengths() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(600, 4096, 0.2);
        for chunk in 513..=640 {
            for layer in 0..64 {
                let seconds = if chunk == 513 && layer == 0 {
                    0.008
                } else {
                    0.002
                };
                costs.observe_stage(chunk, 4096, false, seconds);
            }
        }
        costs.observe_stage(600, 4096, true, 0.07);
        assert_eq!(costs.stages.len(), 2);
        assert_eq!(costs.timings.cells.len(), 1);
        assert_eq!(costs.stage_upper(900, 4096, 64, false), 0.008);
        assert_eq!(costs.stage_upper(900, 4096, 64, true), 0.07);
    }

    #[test]
    fn cold_stage_uses_the_fitted_dynamic_cost_and_a_millisecond_floor() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(4, 4, 0.005);
        costs.observe_at(2048, 2048, 0.21);
        let fitted = 0.005 + (0.21 - 0.005) * 1024. / 2048.;
        assert!((costs.chunk_cost(1024) - fitted).abs() < 1e-12);
        assert!((costs.stage_upper(1024, 4096, 64, false) - fitted * 1.2 / 64.).abs() < 1e-12);
        assert_eq!(costs.stage_upper(4, 4096, 64, false), 0.001);
    }

    #[test]
    fn near_end_boundaries_are_admitted_before_a_short_pass_is_measured() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(2048, 2048, 0.4);
        let checkpoints = BTreeSet::new();
        for tail in 2..=8 {
            assert!(costs.admit(2048 - tail, 2048, 100 << 20, 0, &checkpoints));
        }
        assert!(!costs.admit(2039, 2048, 100 << 20, 0, &checkpoints));
        costs.observe_restore(100 << 20, 100.);
        assert!(!costs.admit(2044, 2048, 100 << 20, 0, &checkpoints));
    }
    #[test]
    fn one_token_tail_uses_the_split_cost_and_confirmed_reuse() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(2048, 2048, 0.8);
        let checkpoints = BTreeSet::new();
        let bytes = 100 << 20;
        for suffix in [256, 312, 768] {
            let start = 2048 - suffix;
            assert!(costs.score(2047, 2048, bytes).is_some());
            assert!(!costs.admit(2047, 2048, bytes, start, &checkpoints));
            for tail in 2..=8 {
                assert!(costs.admit(2048 - tail, 2048, bytes, start, &checkpoints));
            }
        }
        assert!(costs.admit_with_reuse(2047, 2048, bytes, 2048 - 768, &checkpoints, true));
        // A cold prompt with a large baseline must still reject an unconfirmed
        // one-token tail whose extra weight pass exceeds 50 ms.
        assert!(!costs.admit(4095, 4096, bytes, 0, &checkpoints));
        assert!(costs.admit_with_reuse(4095, 4096, bytes, 0, &checkpoints, true));
        costs.observe_at(1, 2048, 0.01);
        assert!(costs.admit(2047, 2048, bytes, 2048 - 312, &checkpoints));
    }
    #[test]
    fn first_short_pass_replaces_the_cold_estimate() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(4, 2048, f64::NAN);
        assert_eq!(costs.weight_pass_s(), 0.1);
        costs.observe_at(4, 2048, 0.005);
        assert_eq!(costs.weight_pass_s(), 0.005);
        costs.observe_at(8, 2048, 0.01);
        assert!((costs.weight_pass_s() - 0.006).abs() < 1e-12);
    }
    #[test]
    fn dynamic_restore_prices_the_real_tail_instead_of_observed_partitions() {
        let mut costs = Profiles::new([4096]).with_dynamic_capacity(4096);
        costs.observe_at(1203, 4508, 1.3);
        costs.observe_at(21, 4508, 0.4);
        let ancestor = costs.score(3303, 4508, 150 << 20).unwrap();
        let parent = costs.score(3970, 4508, 180 << 20).unwrap();
        assert!(parent < ancestor);
        for tail in [1, 537, 538, 1203, 4096] {
            assert!(costs.remaining(0, tail) < costs.remaining(0, tail + 1));
        }
        let two_passes = costs.remaining(0, 4097);
        let split = costs.remaining(0, 4096) + costs.remaining(0, 1);
        assert!((two_passes - split).abs() < 1e-12);
    }
    #[test]
    fn confirmed_reuse_pays_for_a_checkpoint_but_not_an_unprofitable_split() {
        let mut costs = Profiles::new((2..=16).chain([128, 512, 2048, 4096]));
        for n in 1..16 {
            costs.observe(n, 0.1);
        }
        for (n, seconds) in [
            (16, 0.075),
            (128, 0.21),
            (512, 0.6),
            (2048, 1.7),
            (4096, 3.),
        ] {
            costs.observe(n, seconds);
        }
        let checkpoints = BTreeSet::from([8192]);
        assert!(!costs.admit(3584, 8223, 160 << 20, 0, &checkpoints));
        assert!(costs.admit_with_reuse(3584, 8223, 160 << 20, 0, &checkpoints, true));
        assert!(!costs.admit_with_reuse(2403, 8223, 160 << 20, 0, &checkpoints, true));
        costs.observe_restore(160 << 20, 100.);
        assert!(!costs.admit_with_reuse(3584, 8223, 160 << 20, 0, &checkpoints, true));
    }
    #[test]
    fn rejects_small_prefixes_that_destroy_large_chunks() {
        let costs = Profiles::new([2, 4, 8, 512, 2048]);
        let checkpoints = BTreeSet::new();
        for end in [512, 2048, 8192, 32768] {
            for prefix in [1, 26] {
                assert!(costs.score(prefix, end, 160 << 20).is_none());
                assert!(!costs.admit(prefix, end, 160 << 20, 0, &checkpoints));
            }
            assert!(costs.score(end, end, 2 << 30).is_some());
        }
        assert!(costs.admit(512, 520, 180 << 20, 0, &checkpoints));
        assert!(costs.score(527, 536, 180 << 20).is_some());
        assert!(costs.admit(8192, 32768, 450 << 20, 0, &checkpoints));
        assert!(costs.admit(26, 46, 160 << 20, 0, &checkpoints));
        assert!(costs.admit(62, 71, 160 << 20, 0, &checkpoints));
        assert!(!costs.admit(62, 71, 160 << 20, 47, &checkpoints));
        assert!(costs.admit(44, 200, 160 << 20, 0, &checkpoints));
        assert!(costs.admit(82, 200, 160 << 20, 0, &checkpoints));
        assert!(!costs.admit(82, 200, 160 << 20, 0, &BTreeSet::from([44])));
    }
    #[test]
    fn measurements_change_selection_and_account_for_restore() {
        let mut costs = Profiles::new([8, 512]);
        assert!(costs.score(8, 512, 160 << 20).is_none());
        costs.observe(512, 5.);
        costs.observe(8, 0.02);
        assert!(costs.score(8, 512, 160 << 20).is_some());
        assert!(costs.admit(8, 512, 160 << 20, 0, &BTreeSet::new()));
        costs.observe_restore(160 << 20, 100.);
        assert!(costs.score(8, 512, 160 << 20).is_none());
    }
    #[test]
    fn mixed_chunks_follow_measured_cost_and_never_pad_a_tail() {
        let mut costs = Profiles::new([16, 128, 512, 2048, 4096]);
        assert_eq!(costs.bounded_chunk(8192, 200.), 128);
        costs.observe(16, 0.04);
        costs.observe(128, 0.25);
        costs.observe(512, 0.6);
        assert_eq!(costs.bounded_chunk(8192, 200.), 16);
        assert_eq!(costs.bounded_chunk(8192, 700.), 512);
        assert_eq!(costs.bounded_chunk(15, 200.), 1);
        assert_eq!(costs.bounded_chunk(8192, 1.), 1);
        assert_eq!(costs.bounded_chunk(8192, f64::INFINITY), 4096);
    }
    #[test]
    fn context_buckets_do_not_reuse_a_short_attention_timing_for_a_long_prompt() {
        let mut costs = Profiles::new([128, 512]);
        costs.observe_at(128, 1024, 0.1);
        costs.observe_at(128, 8192, 0.5);
        assert_eq!(costs.bounded_chunk_at(8192, 896, 200.), 128);
        assert_eq!(costs.bounded_chunk_at(8192, 8064, 200.), 1);
        costs.observe_at(512, 8192, f64::NAN);
        assert!(costs.chunk_upper(512, 8192) > 0.);
    }
    #[test]
    fn soft_budget_does_not_lock_prefill_into_one_token_weight_passes() {
        let mut costs = Profiles::new([16, 128]);
        costs.observe_at(16, 16, 0.4);
        costs.observe_at(128, 128, 0.8);
        assert_eq!(costs.bounded_chunk_at(512, 0, 200.), 16);
        // No padding across a tail, and no large tile forced beyond a budget.
        assert_eq!(costs.bounded_chunk_at(15, 0, 1.), 1);
        assert_eq!(costs.bounded_chunk_at(512, 0, 1000.), 128);
        costs.observe_at(16, 16, 5.);
        assert_eq!(costs.bounded_chunk_at(512, 0, 1.), 1);
    }
    #[test]
    fn exact_tail_profiles_do_not_turn_soft_budget_fallback_into_tiny_passes() {
        let mut costs = Profiles::new((2..=16).chain([128]));
        for n in 2..=16 {
            costs.observe_at(n, n, 0.4);
        }
        assert_eq!(costs.bounded_chunk_at(512, 0, 1.), 16);
        assert_eq!(costs.bounded_chunk_at(15, 0, 1.), 15);
        assert_eq!(costs.bounded_chunk_at(1, 0, 1.), 1);
    }
}
