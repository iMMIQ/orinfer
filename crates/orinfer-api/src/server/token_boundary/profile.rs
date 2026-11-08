//! CPU-only comparison of tokenizer dispatch on real long plain regions.
use super::*;

#[test]
#[ignore = "Requires ORINFER_TOKENIZER_PROFILE with a tokenizer and long-text corpus"]
fn compare_single_region_dispatch() {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };
    let path = std::path::PathBuf::from(std::env::var_os("ORINFER_TOKENIZER_PROFILE").unwrap());
    let spec: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(spec["tokenizer"].as_str().unwrap()).unwrap();
    let texts: Vec<String> = serde_json::from_value(spec["texts"].clone()).unwrap();
    let oracle = Boundary::new(&tokenizer).unwrap();
    let expected = texts
        .iter()
        .map(|s| {
            oracle
                .plain
                .encode_fast(s.as_str(), false)
                .unwrap()
                .get_ids()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let mut reports = vec![];
    for direct in [false, true, true, false] {
        let boundary = Boundary::new(&tokenizer).unwrap();
        let next = AtomicUsize::new(0);
        let started = Instant::now();
        let timings = std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    let next = &next;
                    let texts = &texts;
                    let expected = &expected;
                    let plain = &boundary.plain;
                    scope.spawn(move || {
                        let mut timings = vec![];
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= texts.len() {
                                break;
                            }
                            let at = Instant::now();
                            let encoding = if direct {
                                plain.encode_fast(texts[i].as_str(), false).unwrap()
                            } else {
                                plain
                                    .encode_batch_fast(vec![texts[i].as_str()], false)
                                    .unwrap()
                                    .pop()
                                    .unwrap()
                            };
                            timings.push(at.elapsed().as_secs_f64());
                            assert_eq!(encoding.get_ids(), &expected[i], "region {i}");
                        }
                        timings
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let report = serde_json::json!({"direct":direct,"workers":8,"regions":texts.len(),
            "wall_s":started.elapsed().as_secs_f64(),"summed_request_s":timings.iter().sum::<f64>(),
            "tokens":expected.iter().map(Vec::len).sum::<usize>(),"ids_exact":true});
        eprintln!("TOKENIZER_DISPATCH {report}");
        reports.push(report);
    }
    std::fs::write(
        spec["output"].as_str().unwrap(),
        serde_json::to_vec_pretty(&reports).unwrap(),
    )
    .unwrap();
}
