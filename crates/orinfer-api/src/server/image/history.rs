//! Multi-turn images keep their geometry and identity as history grows.
use super::*;
use std::sync::Arc;

fn spec() -> VisionSpec {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/vision/deepseek-v41.json"
    ))
    .unwrap();
    let mut spec: VisionSpec = serde_json::from_value(fixture["spec"].clone()).unwrap();
    spec.max_features = 262144;
    spec
}
fn part(value: u8) -> Value {
    let image = image::RgbImage::from_pixel(256, 256, image::Rgb([value, 27, 91]));
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    json!({"type":"image_url","uuid":"drawing-1","image_url":{"url":format!("data:image/png;base64,{}",STANDARD.encode(bytes.get_ref()))}})
}
fn context(cache: &Arc<cache::Cache>) -> super::super::preparation::Context {
    let mut context = super::super::preparation::Context::unbounded();
    context.images = Some(Arc::clone(cache));
    context
}
#[test]
fn long_history_uuid_references_and_fixed_per_image_budgets() {
    let cache = Arc::new(cache::Cache::new(32 << 20));
    let spec = spec();
    let parts = vec![part(18); 50];
    let run = |parts, options: &ImageProcessing| {
        messages_with_options(
            vec![json!({"role":"user","content":parts})],
            Some(&spec),
            options,
            &mut context(&cache),
        )
    };
    let (_, first) = run(parts.clone(), &ImageProcessing::default()).unwrap();
    let total: usize = first.iter().map(|i| spec.feature_count(i).unwrap()).sum();
    assert!(total > 8192);
    assert_eq!(cache.statistics()["misses"], 1);
    assert_eq!(cache.statistics()["hits"], 49);
    let reference = json!({"type":"image_url","uuid":"drawing-1","image_url":null});
    let (_, next) = run(vec![reference.clone(); 51], &ImageProcessing::default()).unwrap();
    assert_eq!(next.len(), 51);
    for (a, b) in first.iter().zip(&next) {
        assert_eq!((a.grid_height, a.grid_width), (b.grid_height, b.grid_width));
        assert_eq!(a.rgb, b.rgb);
    }
    assert_eq!(cache.statistics()["reference_hits"], 51);
    // A total budget never silently changes historical image resolution.
    assert!(
        run(
            parts,
            &ImageProcessing {
                max_tokens: Some(8192),
                auto_resize: true,
                ..Default::default()
            }
        )
        .unwrap_err()
        .contains("feature capacity")
    );
    let (_, smaller) = run(
        vec![reference; 51],
        &ImageProcessing {
            max_tokens_per_image: Some(64),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(smaller.iter().all(|i| spec.feature_count(i).unwrap() <= 64));
    assert!(
        run(vec![part(19)], &ImageProcessing::default())
            .unwrap_err()
            .contains("different contents")
    );
}
#[test]
fn duplicate_preprocessing_coalesces_and_a_cache_hit_still_checks_memory() {
    let cache = Arc::new(cache::Cache::new(32 << 20));
    let source = part(17);
    let barrier = Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        for _ in 0..2 {
            let cache = Arc::clone(&cache);
            let source = source.clone();
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                messages(
                    vec![json!({"role":"user","content":[source]})],
                    Some(&spec()),
                    &mut context(&cache),
                )
                .unwrap();
            });
        }
    });
    assert_eq!(cache.statistics()["misses"], 1);
    assert_eq!(cache.statistics()["hits"], 1);
    // A retained image is charged to each active request, independently of LRU.
    let data = cache.source("drawing-1").unwrap();
    let mut limited = super::super::preparation::Context::new(
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        1 << 20,
    );
    limited.images = Some(Arc::clone(&cache));
    assert!(
        preprocess(&data, &spec(), "auto", &mut limited)
            .unwrap_err()
            .contains("memory budget")
    );
    assert!(cache.source("missing").unwrap_err().contains("resend"));
}
