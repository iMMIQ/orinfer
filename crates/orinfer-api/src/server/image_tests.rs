use super::*;
#[test]
fn actual_image_sizes_and_optional_shrinking_obey_independent_limits() {
    let spec: VisionSpec = serde_json::from_value(json!({"hidden":8,"patch_size":16,
        "temporal_patch_size":2,"merge_size":2,"max_patches":1024,"max_features":512,
        "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
        "mrope_positions":"r","plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
        "vision_start_token_id":11,"vision_end_token_id":12}))
    .unwrap();
    let part = |size| {
        let image =
            image::RgbImage::from_fn(size, size, |x, y| image::Rgb([x as u8, y as u8, 191]));
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}", STANDARD.encode(bytes.get_ref()))}})
    };
    let run = |parts, options: &ImageProcessing| {
        messages_with_options(
            vec![json!({"role":"user","content":parts})],
            Some(&spec),
            options,
            &mut super::super::preparation::Context::unbounded(),
        )
    };
    let small = part(256);
    let large = part(512);
    let parts = vec![small.clone(), small.clone(), large.clone(), small.clone()];
    let (_, images) = run(parts, &ImageProcessing::default()).unwrap();
    assert_eq!(
        images
            .iter()
            .map(|i| (i.grid_height, i.grid_width))
            .collect::<Vec<_>>(),
        [(16, 16), (16, 16), (32, 32), (16, 16)]
    );
    let parts = vec![small, large.clone(), large];
    assert!(
        run(parts.clone(), &ImageProcessing::default())
            .unwrap_err()
            .contains("576")
    );
    let (_, images) = run(
        parts.clone(),
        &ImageProcessing {
            max_tokens: Some(400),
            auto_resize: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!((images[0].grid_height, images[0].grid_width), (16, 16));
    assert!(images[1].grid_height > 20);
    assert!(
        images
            .iter()
            .map(|i| spec.feature_count(i).unwrap())
            .sum::<usize>()
            <= 400
    );
    let (_, images) = run(
        parts.clone(),
        &ImageProcessing {
            max_pixels: Some(MIN_PIXELS),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        images
            .iter()
            .all(|i| i.grid_height == 16 && i.grid_width == 16)
    );
    assert!(
        run(
            parts,
            &ImageProcessing {
                max_images: Some(2),
                ..Default::default()
            }
        )
        .unwrap_err()
        .contains("max_images")
    );
}
#[test]
#[ignore = "Needs Transformers-generated image references"]
fn checkpoint_image_processor_reference() {
    let root = std::path::PathBuf::from(std::env::var("ORINFER_IMAGE_REFERENCE").unwrap());
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
    let spec: VisionSpec = serde_json::from_value(reference["vision"].clone()).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let input = preprocess(
            &std::fs::read(root.join(case["image"].as_str().unwrap())).unwrap(),
            &spec,
            "auto",
            &mut crate::server::preparation::Context::unbounded(),
        )
        .unwrap();
        assert_eq!(
            input.grid_height,
            case["grid"][1].as_u64().unwrap() as usize
        );
        assert_eq!(input.grid_width, case["grid"][2].as_u64().unwrap() as usize);
        let bytes = std::fs::read(root.join(case["pixels"].as_str().unwrap())).unwrap();
        let expected: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let actual: Vec<_> = input.normalized(&spec).collect();
        assert_eq!(actual.len(), expected.len());
        let (mut square, mut baseline, mut maximum) = (0.0f64, 0.0f64, 0.0f32);
        for (&a, &b) in actual.iter().zip(&expected) {
            square += (a - b) as f64 * (a - b) as f64;
            baseline += b as f64 * b as f64;
            maximum = maximum.max((a - b).abs());
        }
        let relative = (square / baseline).sqrt();
        eprintln!(
            "{} resize relative L2 {relative:.6}, max {maximum:.6}",
            case["image"]
        );
        assert!(relative < 0.02 && maximum < 0.05);
    }
}
#[test]
#[ignore = "Real 12-image RGB preprocessing and memory accounting"]
fn many_images_share_the_configured_preprocessing_pool() {
    let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
        "temporal_patch_size":2,"merge_size":2,"max_patches":32768,"max_features":32768,
        "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
        "mrope_positions":"r","plans":[{"patches":32768,"program":"v"}],"image_token_id":10,
        "vision_start_token_id":11,"vision_end_token_id":12}))
    .unwrap();
    let image = image::RgbImage::from_pixel(2048, 1024, image::Rgb([32, 128, 224]));
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let url = format!(
        "data:image/png;base64,{}",
        STANDARD.encode(encoded.get_ref())
    );
    let request: super::super::ChatRequest = serde_json::from_value(json!({"model":"test","messages":[
        {"role":"user","content":(0..12).map(|_|json!({"type":"image_url","image_url":{"url":url}})).collect::<Vec<_>>()}]})).unwrap();
    let estimate =
        crate::server::preparation::request_memory(&request, 262144, Some(&spec), 2048, 8192);
    let mut context = estimate.context(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
        false,
    )));
    let (_, images) = messages(request.messages, Some(&spec), &mut context).unwrap();
    assert_eq!(images.len(), 12);
    assert_eq!(
        images.iter().map(ImageInput::host_bytes).sum::<usize>(),
        72 << 20
    );
    assert_eq!(
        images
            .iter()
            .map(|i| spec.feature_count(i).unwrap())
            .sum::<usize>(),
        24576
    );
    assert!(
        images
            .iter()
            .all(|i| i.pixels.is_empty() && !i.rgb.is_empty())
    );
    context.checkpoint(0).unwrap();
}
#[test]
#[ignore = "Real five-image request near the 32768-feature capacity; needs host RAM"]
fn automatic_large_images_fit_real_model_capacity() {
    let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
        "temporal_patch_size":2,"merge_size":2,"max_patches":32768,"max_features":32768,
        "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
        "mrope_positions":"r","plans":[{"patches":32768,"program":"v"}],"image_token_id":10,
        "vision_start_token_id":11,"vision_end_token_id":12}))
    .unwrap();
    let image = image::RgbImage::from_pixel(4096, 2048, image::Rgb([32, 128, 224]));
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let url = format!(
        "data:image/png;base64,{}",
        STANDARD.encode(encoded.get_ref())
    );
    let request: super::super::ChatRequest = serde_json::from_value(json!({"model":"test","messages":[
        {"role":"user","content":(0..5).map(|_|json!({"type":"image_url","image_url":{"url":url}})).collect::<Vec<_>>()}]})).unwrap();
    assert!(5 * (4096 / 32) * (2048 / 32) > spec.max_features);
    let estimate =
        crate::server::preparation::request_memory(&request, 262144, Some(&spec), 2048, 8192);
    let mut context = estimate.context(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
        false,
    )));
    let (_, images) = messages_with_options(
        request.messages,
        Some(&spec),
        &ImageProcessing {
            auto_resize: true,
            ..Default::default()
        },
        &mut context,
    )
    .unwrap();
    assert_eq!(images.len(), 5);
    let features: usize = images.iter().map(|i| spec.feature_count(i).unwrap()).sum();
    assert!(features <= spec.max_features);
    assert!(features > 30000);
    context.checkpoint(0).unwrap();
}
#[test]
fn automatic_images_share_features_and_preserve_input_order() {
    let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
        "temporal_patch_size":2,"merge_size":2,"max_patches":1024,"max_features":256,
        "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
        "mrope_positions":"r","plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
        "vision_start_token_id":11,"vision_end_token_id":12}))
    .unwrap();
    let encode = |size, color| {
        let image = image::RgbImage::from_pixel(size, size, image::Rgb(color));
        let mut output = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut output, image::ImageFormat::Png)
            .unwrap();
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(output.get_ref())
        )
    };
    let red = encode(512, [255, 0, 0]);
    let blue = encode(512, [0, 0, 255]);
    let part = |url: &str, detail: &str| json!({"type":"image_url","image_url":{"url":url,"detail":detail}});
    let run = |parts: Vec<Value>| {
        messages_with_options(
            vec![json!({"role":"user","content":parts})],
            Some(&spec),
            &ImageProcessing {
                auto_resize: true,
                ..Default::default()
            },
            &mut crate::server::preparation::Context::unbounded(),
        )
    };
    let (rendered, images) = run(vec![part(&red, "auto"), part(&blue, "auto")]).unwrap();
    assert_eq!(images.len(), 2);
    assert_eq!(rendered[0]["content"], format!("{MARKER}{MARKER}"));
    assert!(
        images
            .iter()
            .map(|i| spec.feature_count(i).unwrap())
            .sum::<usize>()
            <= 256
    );
    assert_eq!(images[0].normalized(&spec).next().unwrap(), 1.0);
    assert_eq!(images[1].normalized(&spec).next().unwrap(), -1.0);
    let (_, reverse) = run(vec![part(&blue, "auto"), part(&red, "auto")]).unwrap();
    assert_eq!(images[0].rgb, reverse[1].rgb);
    assert_eq!(images[1].rgb, reverse[0].rgb);
    let (_, single) = run(vec![part(&red, "auto")]).unwrap();
    assert_eq!(spec.feature_count(&single[0]).unwrap(), 256);
    let error = run(vec![part(&red, "high"), part(&blue, "high")]).unwrap_err();
    assert!(error.contains("feature capacity") && error.contains("512"));
    let small_blue = encode(256, [0, 0, 255]);
    let (_, mixed) = run(vec![part(&red, "auto"), part(&small_blue, "high")]).unwrap();
    assert_eq!((mixed[1].grid_height, mixed[1].grid_width), (16, 16));
    assert_eq!(mixed[0].normalized(&spec).next().unwrap(), 1.0);
    assert_eq!(mixed[1].normalized(&spec).next().unwrap(), -1.0);
    assert!(
        mixed
            .iter()
            .map(|i| spec.feature_count(i).unwrap())
            .sum::<usize>()
            <= 256
    );
    let error = run((0..5)
        .map(|_| part("https://invalid.invalid/never-fetch", "auto"))
        .collect())
    .unwrap_err();
    assert!(error.contains("minimum image detail"));
}
#[test]
fn patch_order_temporal_duplication_and_image_parts() {
    let spec: VisionSpec = serde_json::from_value(json!({
        "hidden":5120,"patch_size":16,"temporal_patch_size":2,"merge_size":2,
        "max_patches":1024,"max_features":256,"pixels":"p","grid":"g","length":"l",
        "output":"o","features":"f","feature_index":"i","mrope_positions":"r",
        "plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
        "vision_start_token_id":11,"vision_end_token_id":12
    }))
    .unwrap();
    let rgb = image::RgbImage::from_fn(256, 256, |x, y| {
        image::Rgb([
            if x / 16 % 2 == 0 { 255 } else { 0 },
            if y / 16 % 2 == 0 { 255 } else { 0 },
            0,
        ])
    });
    let mut data = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(rgb)
        .write_to(&mut data, image::ImageFormat::Png)
        .unwrap();
    let pixels = preprocess(
        data.get_ref(),
        &spec,
        "auto",
        &mut crate::server::preparation::Context::unbounded(),
    )
    .unwrap();
    assert_eq!((pixels.grid_height, pixels.grid_width), (16, 16));
    let normalized: Vec<_> = pixels.normalized(&spec).collect();
    for (row, (red, green)) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
        .into_iter()
        .enumerate()
    {
        let p = &normalized[row * 1536..(row + 1) * 1536];
        assert!(p[..512].iter().all(|&v| v == red));
        assert!(p[512..1024].iter().all(|&v| v == green));
        assert!(p[1024..].iter().all(|&v| v == -1.0));
    }
    let url = format!("data:image/png;base64,{}", STANDARD.encode(data.get_ref()));
    let input = vec![
        json!({"role":"user","content":[{"type":"text","text":"A"},{"type":"image_url","image_url":{"url":url}},{"type":"text","text":"B"},{"type":"image_url","image_url":{"url":url}}]}),
    ];
    let (out, images) = messages(
        input.clone(),
        Some(&spec),
        &mut crate::server::preparation::Context::unbounded(),
    )
    .unwrap();
    assert_eq!(images.len(), 2);
    assert_eq!(out[0]["content"], format!("A{MARKER}B{MARKER}"));
    assert!(
        messages(
            input,
            None,
            &mut crate::server::preparation::Context::unbounded()
        )
        .is_err()
    );
    assert!(
        preprocess(
            data.get_ref(),
            &spec,
            "invalid",
            &mut crate::server::preparation::Context::unbounded()
        )
        .is_err()
    );
}
#[test]
fn smart_resize_rounding_and_aspect() {
    assert_eq!(resize_shape(256, 256, 32, 1_048_576).unwrap(), (256, 256));
    assert_eq!(resize_shape(80, 112, 32, 1_048_576).unwrap(), (224, 320));
    assert_eq!(
        resize_shape(2000, 1000, 32, 1_048_576).unwrap(),
        (1440, 704)
    );
    assert_eq!(resize_shape(500, 100, 32, 65_536).unwrap(), (576, 128));
    assert!(resize_shape(1, 300, 32, 1_048_576).is_err());
    assert!(
        load(
            "file:///tmp/image.png",
            &crate::server::preparation::Context::unbounded()
        )
        .is_err()
    );
    assert!(
        load(
            "data:video/mp4;base64,AA==",
            &crate::server::preparation::Context::unbounded()
        )
        .is_err()
    );
}
