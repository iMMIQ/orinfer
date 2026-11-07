use super::*;
use std::{fs, io::Write, path::PathBuf, sync::atomic::AtomicUsize, time::Duration};

static SERIAL: AtomicUsize = AtomicUsize::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "orinfer-batch-test-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn input(&self, text: &str) -> Input {
        fs::write(self.0.join("input.jsonl"), text).unwrap();
        Input::open(&self.0.join("input.jsonl")).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn response(id: &str, reason: &str) -> Value {
    json!({"custom_id":id,"response":{"status_code":200,"body":{
        "choices":[{"finish_reason":reason,"message":{"role":"assistant","content":"译文"}}]}},"error":null})
}

// Other tests spawn workers concurrently. A forked child can briefly retain a
// CLOEXEC descriptor until exec, after its owner in this process has closed it.
// Only post-close resume assertions wait; live-owner exclusion stays immediate.
fn reopen_after_close(path: &std::path::Path, identity: &Value, input: &Input) -> Journal {
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    loop {
        match Journal::open(path, true, identity, input) {
            Ok(journal) => return journal,
            Err(error)
                if error == "Batch output is locked by another process"
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(error) => panic!("Reopen closed journal: {error}"),
        }
    }
}

#[test]
fn standard_and_direct_rows_preserve_physical_line_ids_and_errors() {
    let d = Directory::new();
    let mut input = d.input("\n{\"custom_id\":\"one\",\"method\":\"POST\",\"url\":\"/v1/chat/completions\",\"body\":{\"messages\":[]}}\n{\"messages\":[]}\nbroken\n{\"custom_id\":\"bad\",\"method\":\"GET\",\"url\":\"/v1/models\",\"body\":{}}\n");
    assert_eq!(input.ids.len(), 4);
    let first = input.next().unwrap().unwrap();
    assert_eq!(first.id, "one");
    assert_eq!(first.line, 2);
    assert_eq!(first.body.unwrap(), json!({"messages":[]}));
    assert_eq!(input.next().unwrap().unwrap().id, "line-3");
    assert!(input.next().unwrap().unwrap().body.is_err());
    let bad = input.next().unwrap().unwrap();
    assert_eq!(bad.id, "bad");
    assert!(bad.body.is_err());
    assert!(input.next().unwrap().is_none());
    fs::write(
        d.0.join("input.jsonl"),
        "{\"custom_id\":\"same\"}\n{\"custom_id\":\"same\"}\n",
    )
    .unwrap();
    assert!(
        Input::open(&d.0.join("input.jsonl"))
            .err()
            .unwrap()
            .contains("Duplicate")
    );
}

#[test]
fn resume_retries_errors_retains_truncation_and_repairs_only_uncommitted_tail() {
    let d = Directory::new();
    let input = d.input("{}\n{}\n{}\n");
    let path = d.0.join("output.jsonl");
    let identity = json!({"input":input.sha256,"model":"v1"});
    {
        let mut journal = Journal::open(&path, false, &identity, &input).unwrap();
        journal.append(&response("line-1", "stop")).unwrap();
        journal
            .append(&failure("line-2", "request_failed", "bad".into()))
            .unwrap();
        journal.append(&response("line-3", "length")).unwrap();
        journal.sync().unwrap();
    }
    let committed = fs::read(&path).unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{\"custom_id\":")
        .unwrap();
    {
        let mut journal = reopen_after_close(&path, &identity, &input);
        assert!(journal.done("line-1"));
        assert!(!journal.done("line-2"));
        assert!(journal.done("line-3"));
        assert_eq!(fs::read(&path).unwrap(), committed);
        // A second writer cannot repair or append to the live journal.
        assert!(Journal::open(&path, true, &identity, &input).is_err());
        journal.append(&response("line-2", "stop")).unwrap();
    }
    let journal = reopen_after_close(&path, &identity, &input);
    assert!(journal.done("line-2"));
    assert_eq!(journal.states.len(), 3);
}

#[test]
fn identity_mismatch_corruption_and_alias_never_modify_existing_files() {
    let d = Directory::new();
    let input = d.input("{}\n");
    let path = d.0.join("output.jsonl");
    let identity = json!({"model":"v1"});
    let mut journal = Journal::open(&path, false, &identity, &input).unwrap();
    journal.append(&response("line-1", "stop")).unwrap();
    drop(journal);
    let before = fs::read(&path).unwrap();
    assert!(Journal::open(&path, true, &json!({"model":"v2"}), &input).is_err());
    assert!(Journal::open(&path, false, &identity, &input).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"broken\n")
        .unwrap();
    let corrupt = fs::read(&path).unwrap();
    assert!(Journal::open(&path, true, &identity, &input).is_err());
    assert_eq!(fs::read(&path).unwrap(), corrupt);
    let alias = d.0.join("alias.jsonl");
    fs::hard_link(d.0.join("input.jsonl"), &alias).unwrap();
    assert!(Journal::open(&alias, true, &identity, &input).is_err());
    assert_eq!(fs::read(d.0.join("input.jsonl")).unwrap(), b"{}\n");
}

#[test]
fn offline_defaults_preserve_explicit_chat_controls_and_reject_streaming() {
    let defaults =
        json!({"temperature":0,"seed":20261002,"max_completion_tokens":256,"stream":false});
    let body = json!({"messages":[{"role":"user","content":"翻译"}],"max_tokens":64});
    let r = request(body, "translation", defaults.as_object().unwrap()).unwrap();
    assert_eq!(r.model, "translation");
    assert_eq!(r.max_tokens, Some(64));
    assert_eq!(r.max_completion_tokens, None);
    assert_eq!(r.seed, Some(20261002));
    for body in [
        json!({"messages":[],"stream":true}),
        json!({"messages":[],"stream_options":{}}),
        json!({"messages":[],"unknown":true}),
    ] {
        assert!(request(body, "translation", defaults.as_object().unwrap()).is_err());
    }
}

#[test]
fn input_mutation_is_rejected() {
    let d = Directory::new();
    let mut input = d.input("{}\n");
    fs::write(d.0.join("input.jsonl"), "{}\n{}\n").unwrap();
    assert!(input.next().is_err());
}

#[test]
fn global_order_preserves_physical_ids_skips_blank_lines_and_can_reset() {
    let d = Directory::new();
    let mut input = d.input("\n{}\n{\"custom_id\":\"named\"}\n\n{}\n");
    input
        .reorder(vec![2, 0, 1], vec![None, Some(32), None])
        .unwrap();
    assert_eq!(input.next().unwrap().unwrap().id, "line-5");
    assert_eq!(input.next().unwrap().unwrap().line, 2);
    let named = input.next().unwrap().unwrap();
    assert_eq!(named.id, "named");
    assert_eq!(named.prefix_hint, Some(32));
    assert!(input.next().unwrap().is_none());
    assert!(input.reorder(vec![0, 0, 1], vec![]).is_err());
    assert!(input.reorder(vec![0, 1, 3], vec![]).is_err());
    input.reorder(vec![], vec![]).unwrap();
    assert_eq!(input.next().unwrap().unwrap().id, "line-2");
}

#[tokio::test]
async fn queued_results_are_drained_before_resume() {
    let d = Directory::new();
    let input = d.input("{}\n{}\n{}\n");
    let path = d.0.join("queued.jsonl");
    let identity = json!({"model":"test"});
    let mut journal = Journal::open(&path, false, &identity, &input).unwrap();
    let writer = files::Writer::start(&journal, 1).unwrap();
    for i in 1..=3 {
        writer
            .append(
                &mut journal,
                Completed {
                    value: response(&format!("line-{i}"), "stop"),
                    bytes: 0,
                    _memory: None,
                },
            )
            .await
            .unwrap();
    }
    writer.finish().await.unwrap();
    drop(journal);
    let journal = reopen_after_close(&path, &identity, &input);
    assert_eq!(journal.states.len(), 3);
}
