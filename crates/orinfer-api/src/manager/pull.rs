//! Native HF/ModelScope snapshots; downloads never quantize or compile online.
use super::{ModelRecord, Result, WorkerOptions, locked_file, registry};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path},
    time::Duration,
};

#[derive(Deserialize)]
pub(super) struct Pull {
    pub name: String,
    pub source: String,
    pub revision: Option<String>,
    pub package: Option<String>,
    #[serde(default)]
    pub options: Option<WorkerOptions>,
}
fn safe_path(name: &str) -> Result<()> {
    if name.is_empty()
        || name.contains('\\')
        || name.as_bytes().contains(&0)
        || !Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err("Unsafe model repository file path".into());
    }
    Ok(())
}
fn request(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    let request = client.get(url);
    if let Some(token) = token {
        request.bearer_auth(token)
    } else {
        request
    }
}
fn download(
    client: &reqwest::blocking::Client,
    url: &str,
    target: &Path,
    size: Option<u64>,
    hash: Option<&str>,
    token: Option<&str>,
) -> Result<()> {
    if target.is_file()
        && size.is_none_or(|n| target.metadata().is_ok_and(|m| m.len() == n))
        && hash.is_none_or(|h| digest(target).is_ok_and(|actual| actual == h))
    {
        return Ok(());
    }
    fs::create_dir_all(target.parent().ok_or("Missing download directory")?)
        .map_err(|e| e.to_string())?;
    let partial = target.with_file_name(format!(
        "{}.part",
        target
            .file_name()
            .ok_or("Missing download name")?
            .to_string_lossy()
    ));
    if let Some(expected_size) = size
        && let Ok(metadata) = partial.metadata()
        && metadata.len() >= expected_size
    {
        if metadata.len() == expected_size
            && hash.is_none_or(|h| digest(&partial).is_ok_and(|actual| actual == h))
        {
            fs::rename(&partial, target).map_err(|e| e.to_string())?;
            return Ok(());
        }
        fs::remove_file(&partial).map_err(|e| e.to_string())?;
    }
    let mut last_error = String::new();
    for attempt in 0..4 {
        let result = (|| {
            let offset = partial.metadata().map(|m| m.len()).unwrap_or(0);
            let mut builder = request(client, url, token);
            if offset != 0 {
                builder = builder.header("Range", format!("bytes={offset}-"));
            }
            let mut response = builder
                .send()
                .map_err(|e| e.to_string())?
                .error_for_status()
                .map_err(|e| e.to_string())?;
            let resume = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
            if resume
                && response
                    .headers()
                    .get("content-range")
                    .and_then(|h| h.to_str().ok())
                    .is_none_or(|h| !h.starts_with(&format!("bytes {offset}-")))
            {
                return Err("Download returned an unexpected byte range".into());
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(resume)
                .truncate(!resume)
                .open(&partial)
                .map_err(|e| e.to_string())?;
            std::io::copy(&mut response, &mut file).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            if size.is_some_and(|n| file.metadata().map(|m| m.len() != n).unwrap_or(true)) {
                return Err("Incomplete model download".into());
            }
            if let Some(expected) = hash
                && digest(&partial)? != expected
            {
                fs::remove_file(&partial).map_err(|e| e.to_string())?;
                return Err("Model file hash mismatch".into());
            }
            fs::rename(&partial, target).map_err(|e| e.to_string())?;
            Ok(())
        })();
        match result {
            Ok(()) => return Ok(()),
            Err(error) => last_error = error,
        }
        if attempt < 3 {
            std::thread::sleep(Duration::from_secs(1 << attempt));
        }
    }
    Err(format!("Download {url}: {last_error}"))
}
fn digest(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub(super) fn pull(data_dir: &Path, input: Pull) -> Result<ModelRecord> {
    registry::validate_name(&input.name)?;
    let (provider, repo) = input
        .source
        .split_once("://")
        .ok_or("Use hf://owner/model or ms://owner/model")?;
    if !matches!(provider, "hf" | "ms")
        || repo.split('/').count() != 2
        || !repo
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"/_-.".contains(&b))
    {
        return Err("Invalid model source".into());
    }
    let revision =
        input
            .revision
            .as_deref()
            .unwrap_or(if provider == "hf" { "main" } else { "master" });
    if revision.is_empty()
        || !revision
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b))
    {
        return Err("Invalid model revision".into());
    }
    let client = reqwest::blocking::Client::builder()
        .user_agent("orinfer")
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let token = if provider == "hf" {
        std::env::var("HF_TOKEN").ok()
    } else {
        None
    };
    let mut files = Vec::new();
    let version;
    if provider == "hf" {
        let metadata: Value = request(
            &client,
            &format!("https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"),
            token.as_deref(),
        )
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())?;
        version = metadata["sha"]
            .as_str()
            .ok_or("Missing HF revision identity")?
            .to_owned();
        if !version.bytes().all(|b| b.is_ascii_hexdigit()) || version.len() != 40 {
            return Err("Invalid HF commit identity".into());
        }
        for file in metadata["siblings"]
            .as_array()
            .ok_or("Missing HF model file list")?
        {
            let name = file["rfilename"]
                .as_str()
                .ok_or("Missing model file name")?;
            safe_path(name)?;
            files.push((
                name.to_owned(),
                format!("https://huggingface.co/{repo}/resolve/{version}/{name}"),
                file["size"].as_u64(),
                file["lfs"]["sha256"].as_str().map(str::to_owned),
            ));
        }
    } else {
        let url = reqwest::Url::parse_with_params(
            &format!("https://modelscope.cn/api/v1/models/{repo}/repo/files"),
            &[("Revision", revision), ("Recursive", "true")],
        )
        .map_err(|e| e.to_string())?;
        let metadata: Value = client
            .get(url)
            .send()
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .json()
            .map_err(|e| e.to_string())?;
        version = orinfer_engine::artifact::sha256(
            &serde_json::to_vec(&metadata["Data"]["Files"]).map_err(|e| e.to_string())?,
        );
        for file in metadata["Data"]["Files"]
            .as_array()
            .ok_or("Missing ModelScope model file list")?
        {
            if file["Type"] == "tree" {
                continue;
            }
            let name = file["Path"].as_str().ok_or("Missing model file name")?;
            safe_path(name)?;
            let url = reqwest::Url::parse_with_params(
                &format!("https://modelscope.cn/api/v1/models/{repo}/repo"),
                &[("Revision", revision), ("FilePath", name)],
            )
            .map_err(|e| e.to_string())?;
            files.push((
                name.to_owned(),
                url.to_string(),
                file["Size"].as_u64(),
                file["Sha256"].as_str().map(str::to_owned),
            ));
        }
    }
    if !files.iter().any(|f| f.0 == "cache/model.json") {
        return Err("Repository must contain an orinfer native model (cache/model.json)".into());
    }
    let key = orinfer_engine::artifact::sha256(format!("{}@{version}", input.source).as_bytes());
    let directory = data_dir.join("snapshots").join(key);
    fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let _lock = locked_file(&directory.join("download.lock"))?;
    // Download the descriptor first, keeping retries and publication independent of loading.
    files.sort_by_key(|f| (f.0 != "cache/model.json", f.0.clone()));
    for (name, url, size, hash) in files {
        eprintln!("PULL {} {name}", input.source);
        download(
            &client,
            &url,
            &directory.join(name),
            size,
            hash.as_deref(),
            token.as_deref(),
        )?;
    }
    let descriptor: Value = serde_json::from_slice(
        &fs::read(directory.join("cache/model.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let expected = descriptor["execution_package"]
        .as_str()
        .ok_or("Missing execution package digest")?;
    if !orinfer_engine::artifact::is_sha256_hex(expected) {
        return Err("Invalid execution package digest".into());
    }
    if let Some(package) = input.package {
        let archive = if package.starts_with("https://") {
            let path = directory.join("execution.tar.xz");
            download(&client, &package, &path, None, None, None)?;
            path
        } else {
            Path::new(&package)
                .canonicalize()
                .map_err(|e| e.to_string())?
        };
        if !orinfer_engine::operators::default_cache()
            .join(expected)
            .is_dir()
        {
            let installed = orinfer_engine::operators::install::install(
                &archive,
                &orinfer_engine::operators::default_cache(),
            )?;
            if installed.digest != expected {
                return Err("Downloaded execution package does not match the model".into());
            }
        }
    }
    // Validate content once on installation; ordinary model starts retain structural checks.
    orinfer_engine::model::validate_model(&directory)?;
    let record = registry::inspect(&input.name, &directory, input.options.unwrap_or_default())?;
    let mut source = fs::File::create(directory.join("source.json")).map_err(|e| e.to_string())?;
    source
        .write_all(
            serde_json::to_string_pretty(&json!({"source":input.source,"revision":version}))
                .unwrap()
                .as_bytes(),
        )
        .map_err(|e| e.to_string())?;
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repository_files_cannot_write_outside_snapshot() {
        for name in ["cache/model.json", "model-00001.safetensors"] {
            safe_path(name).unwrap();
        }
        for name in ["/x", "../x", "cache/../../x", "a\\b", ""] {
            assert!(safe_path(name).is_err());
        }
    }

    #[test]
    fn downloads_resume_ranges_restart_if_ignored_and_publish_complete_parts() {
        use std::{
            io::{BufRead, BufReader},
            net::TcpListener,
        };
        let root =
            std::env::temp_dir().join(format!("orinfer-download-test-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let payload = b"native-model-shard";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            for (resume, range_requested) in
                [(true, true), (false, true), (false, false), (false, false)]
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut headers = String::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    headers.push_str(&line);
                }
                assert_eq!(
                    headers.to_ascii_lowercase().contains("range: bytes=4-"),
                    range_requested
                );
                let data = if resume { &payload[4..] } else { &payload[..] };
                let range = if resume {
                    format!(
                        "Content-Range: bytes 4-{}/{}\r\n",
                        payload.len() - 1,
                        payload.len()
                    )
                } else {
                    String::new()
                };
                write!(
                    stream,
                    "HTTP/1.1 {}\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n",
                    if resume {
                        "206 Partial Content"
                    } else {
                        "200 OK"
                    },
                    data.len(),
                    range
                )
                .unwrap();
                stream.write_all(data).unwrap();
            }
        });
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let hash = orinfer_engine::artifact::sha256(payload);
        for name in ["resume", "restart", "complete", "corrupt", "oversized"] {
            let initial = match name {
                "complete" => payload.to_vec(),
                "corrupt" => vec![b'x'; payload.len()],
                "oversized" => vec![b'x'; payload.len() + 1],
                _ => payload[..4].to_vec(),
            };
            fs::write(root.join(format!("{name}.part")), initial).unwrap();
            download(
                &client,
                &url,
                &root.join(name),
                Some(payload.len() as u64),
                Some(&hash),
                None,
            )
            .unwrap();
            assert_eq!(fs::read(root.join(name)).unwrap(), payload);
        }
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
