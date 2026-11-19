//! Real-weight continuous vision lengths, tail guards, Graph replay and multi-image isolation.
use super::*;
use std::path::Path;
#[derive(serde::Deserialize)]
struct VisionFixture {
    baseline: PathBuf,
    model: PathBuf,
    output: PathBuf,
}
#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn validate_flash_dynamic_vision() {
    let f: VisionFixture =
        crate::model::read(Path::new(&std::env::var("ORINFER_VISION_FIXTURE").unwrap())).unwrap();
    let grids = [
        (2, 2),
        (10, 14),
        (16, 16),
        (18, 30),
        (32, 32),
        (48, 64),
        (64, 128),
    ];
    let mut reference = vec![];
    for (kind, root) in [(0, &f.baseline), (1, &f.model)] {
        let mut model = ModelRuntime::load_with_options(
            root,
            LoadOptions {
                cuda_graph: crate::execution::CudaGraphMode::Full,
                ..Default::default()
            },
        )
        .unwrap();
        model.execution.prepare_reference_sequence().unwrap();
        let v = model.manifest.vision.as_ref().unwrap();
        for (index, &(height, width)) in grids.iter().enumerate() {
            let pixels = (0..height * width * v.patch_values().unwrap())
                .map(|n| ((n * 37 % 199) as f32 - 99.) / 100.)
                .collect();
            let image = crate::vision::ImageInput {
                rgb: Vec::new(),
                grid_height: height,
                grid_width: width,
                pixels,
            };
            let tokens = std::iter::once(v.vision_start_token_id)
                .chain(std::iter::repeat_n(v.image_token_id, height * width / 4))
                .chain([v.vision_end_token_id])
                .collect::<Vec<_>>();
            let n = height * width / 4 * v.hidden * 2;
            for replay in 0..2 {
                // Changed pixels must survive cached graph reuse.
                let mut changed = image.clone();
                if replay == 1 {
                    for value in &mut changed.pixels {
                        *value = -*value;
                    }
                }
                model
                    .prepare_visual(&tokens, &[changed], &|| false)
                    .unwrap();
                let actual = model.execution.download_bytes(&v.features, n).unwrap();
                if kind == 0 {
                    reference.push(actual);
                } else {
                    assert_eq!(
                        actual,
                        reference[index * 2 + replay],
                        "grid={height}x{width}, replay={replay}"
                    );
                }
            }
        }
        // Different images use the same workspace and retain their feature order.
        let images: Vec<_> = [(10, 14), (18, 30)]
            .into_iter()
            .enumerate()
            .map(|(i, (h, w))| crate::vision::ImageInput {
                rgb: Vec::new(),
                grid_height: h,
                grid_width: w,
                pixels: (0..h * w * v.patch_values().unwrap())
                    .map(|n| ((n * 37 % 199) as f32 - 99.) / 100. * if i == 0 { 1. } else { -1. })
                    .collect(),
            })
            .collect();
        let tokens = images
            .iter()
            .flat_map(|image| {
                std::iter::once(v.vision_start_token_id)
                    .chain(std::iter::repeat_n(
                        v.image_token_id,
                        image.grid_height * image.grid_width / 4,
                    ))
                    .chain([v.vision_end_token_id])
            })
            .collect::<Vec<_>>();
        model.prepare_visual(&tokens, &images, &|| false).unwrap();
        let n = (140 + 540) / 4 * v.hidden * 2;
        let actual = model.execution.download_bytes(&v.features, n).unwrap();
        if kind == 0 {
            reference.push(actual);
        } else {
            assert_eq!(actual, *reference.last().unwrap());
        }
    }
    std::fs::write(&f.output,serde_json::to_vec_pretty(&serde_json::json!({"complete":true,"image_lengths":[4,140,256,540,1024,3072,8192],"replays":2,"multi_image":true,"features_exact":true})).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn validate_visual_capacity_and_compact_inputs() {
    let f: VisionFixture =
        crate::model::read(Path::new(&std::env::var("ORINFER_VISION_FIXTURE").unwrap())).unwrap();
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Full,
            ..Default::default()
        },
    )
    .unwrap();
    let v = model.manifest.vision.clone().unwrap();
    // A larger logical limit must not allocate a full feature bank at startup.
    assert_eq!(model.execution.sizes[&v.features], v.hidden * 2);
    assert!(v.max_features > 32770);
    let high = 32768;
    let mut limits = BTreeMap::from([
        (v.features.clone(), (high + 2) * v.hidden * 2),
        (v.feature_index.clone(), 16 * 4),
        (v.mrope_positions.clone(), 16 * 12),
    ]);
    if let Some(name) = model
        .manifest
        .mtp
        .as_ref()
        .and_then(|m| m.feature_index.as_ref())
    {
        limits.insert(name.clone(), 16 * 4);
    }
    let slot = model.execution.lease_sequence(&limits).unwrap();
    assert_eq!(
        model.execution.sizes[&v.features],
        (high + 2) * v.hidden * 2
    );
    let overlays: Vec<_> = model
        .manifest
        .kernels
        .iter()
        .filter(|k| {
            k.args.get(1).is_some_and(
                |a| matches!(a, crate::artifact::Argument::Buffer { name } if name == &v.features),
            )
        })
        .cloned()
        .collect();
    assert!(
        overlays.len() >= 2,
        "Include target and MTP visual boundaries"
    );
    let mut checked = Vec::new();
    for kernel in overlays {
        let buffer = |i| match &kernel.args[i] {
            crate::artifact::Argument::Buffer { name } => name.clone(),
            _ => panic!("visual boundary pointer"),
        };
        let (output, index, position) = (buffer(0), buffer(2), buffer(3));
        let launch = model
            .manifest
            .dynamic_batch_kernels
            .get(&kernel.name)
            .map(|k| k.launch(2).unwrap());
        let rows = if launch.is_some() {
            2
        } else {
            kernel.grid[0] as usize
        };
        assert!((1..=16).contains(&rows));
        let plan = [crate::execution::Invocation {
            operation: crate::model::Operation::Kernel {
                name: kernel.name.clone(),
            },
            sequence: Some(slot),
            launch,
            views: BTreeMap::new(),
        }];
        // Each operation has one registered plan per membership/row shape.
        // Keep target and draft captures in their actual runtime roles.
        let operation = if kernel.name.contains("/mtp/") || kernel.name.contains("/warm/") {
            orinfer_model_sdk::packet::BatchOperation::Warm
        } else if kernel.name.contains("mtp_verify/") {
            orinfer_model_sdk::packet::BatchOperation::Verification
        } else {
            orinfer_model_sdk::packet::BatchOperation::Ordinary
        };
        for replay in 0..3 {
            let values = [0.25 + replay as f32 * 0.125, -0.5];
            let bytes: Vec<_> = values
                .into_iter()
                .flat_map(|x| {
                    std::iter::repeat_n(half::f16::from_f32(x).to_bits().to_le_bytes(), v.hidden)
                })
                .flatten()
                .collect();
            model
                .execution
                .upload_bytes_at(&v.features, high * v.hidden * 2, &bytes)
                .unwrap();
            let mut indices = [high as u32; 16];
            indices[1] += 1;
            model.upload_ids(&index, &indices).unwrap();
            model.upload_ids(&position, &[0]).unwrap();
            model
                .execution
                .execute_batch_operation(operation, vec![(slot, rows)], &plan, false, true)
                .unwrap();
            let n = rows.min(2) * v.hidden * 2;
            let actual = model.execution.download_bytes(&output, n).unwrap();
            assert!(
                actual == bytes[..n],
                "{} replay={replay}: expected {:?}, got {:?}",
                kernel.name,
                &bytes[..16],
                &actual[..16]
            );
        }
        checked.push(kernel.name);
    }
    let graph_hits = model.execution.batch_statistics().graph_hits;
    assert!(
        graph_hits > 0,
        "Changed features must execute through Graph replay"
    );
    model
        .execution
        .release_sequence(slot, &model.manifest.reset_buffers)
        .unwrap();
    model.execution.prepare_reference_sequence().unwrap();
    // Compact RGB and legacy FP32 input must encode identically, including a
    // changed image on the cached visual graph and reversed image order.
    let mut images: Vec<_> = (0..2)
        .map(|owner| crate::vision::ImageInput {
            grid_height: 16,
            grid_width: 16,
            rgb: (0..256 * 256 * 3)
                .map(|n| ((n * 37 + owner * 91) % 256) as u8)
                .collect(),
            ..Default::default()
        })
        .collect();
    for replay in 0..2 {
        if replay == 1 {
            images.reverse();
            images[0].rgb[0] ^= 255;
        }
        let tokens: Vec<_> = images
            .iter()
            .flat_map(|image| {
                std::iter::once(v.vision_start_token_id)
                    .chain(std::iter::repeat_n(
                        v.image_token_id,
                        v.feature_count(image).unwrap(),
                    ))
                    .chain([v.vision_end_token_id])
            })
            .collect();
        let expanded: Vec<_> = images
            .iter()
            .map(|image| crate::vision::ImageInput {
                grid_height: image.grid_height,
                grid_width: image.grid_width,
                pixels: image.normalized(&v).collect(),
                ..Default::default()
            })
            .collect();
        model.prepare_visual(&tokens, &expanded, &|| false).unwrap();
        let expected = model
            .execution
            .download_bytes(&v.features, 128 * v.hidden * 2)
            .unwrap();
        model.prepare_visual(&tokens, &images, &|| false).unwrap();
        assert_eq!(
            model
                .execution
                .download_bytes(&v.features, expected.len())
                .unwrap(),
            expected
        );
    }
    std::fs::write(
        f.output,
        serde_json::to_vec_pretty(&serde_json::json!({"complete":true,
        "feature_limit":v.max_features,"high_feature_indices":[high,high+1],"graph_calls_per_boundary":3,"boundaries":checked,"graph_hits":graph_hits,
        "bootstrap_features":1,"request_features":high+2,"compact_matches_float":true,
        "changed_images":true,"reversed_images":true}))
        .unwrap(),
    )
    .unwrap();
}
