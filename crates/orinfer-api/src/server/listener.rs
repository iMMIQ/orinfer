//! Bind before GPU loading; unlink only this server's own Unix socket.
use std::{
    io,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::PathBuf,
};

pub(super) enum Listener {
    Tcp(tokio::net::TcpListener),
    Unix(tokio::net::UnixListener),
}
pub(super) struct SocketPath {
    path: PathBuf,
    device: u64,
    inode: u64,
    // Pin the filesystem inode so unlink/rebind cannot recycle its identity.
    _inode: std::fs::File,
}
impl Drop for SocketPath {
    fn drop(&mut self) {
        if let Ok(metadata) = std::fs::symlink_metadata(&self.path)
            && metadata.dev() == self.device
            && metadata.ino() == self.inode
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
pub(super) async fn bind(address: &str) -> io::Result<(Listener, Option<SocketPath>)> {
    if let Some(path) = address.strip_prefix("unix:") {
        let listener = tokio::net::UnixListener::bind(path)?;
        let inode = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_PATH | libc::O_NOFOLLOW)
            .open(path)?;
        let metadata = inode.metadata()?;
        Ok((
            Listener::Unix(listener),
            Some(SocketPath {
                path: path.into(),
                device: metadata.dev(),
                inode: metadata.ino(),
                _inode: inode,
            }),
        ))
    } else {
        Ok((
            Listener::Tcp(tokio::net::TcpListener::bind(address).await?),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn path() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "orinfer-listener-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[tokio::test]
    async fn occupied_tcp_fails_before_missing_model_assets_are_loaded() {
        let occupied = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let error = super::super::serve(crate::ServerConfig {
            model_dir: std::env::temp_dir(),
            listen: occupied.local_addr().unwrap().to_string(),
            gpu_lock: path(),
            ..Default::default()
        })
        .await
        .unwrap_err();
        assert!(error.starts_with("Listener bind failed:"), "{error}");
    }
    #[tokio::test]
    async fn unix_socket_is_removed_on_startup_failure_and_normal_shutdown() {
        let path = path();
        let (listener, owner) = bind(&format!("unix:{}", path.display())).await.unwrap();
        assert!(path.exists());
        drop(listener);
        drop(owner);
        assert!(!path.exists());
        let (listener, owner) = bind(&format!("unix:{}", path.display())).await.unwrap();
        drop(listener);
        drop(owner);
        assert!(!path.exists());
    }
    #[tokio::test]
    async fn a_replaced_or_occupied_unix_socket_is_never_unlinked() {
        let path = path();
        let (old, owner) = bind(&format!("unix:{}", path.display())).await.unwrap();
        std::fs::remove_file(&path).unwrap();
        let (replacement, replacement_owner) =
            bind(&format!("unix:{}", path.display())).await.unwrap();
        drop(old);
        drop(owner);
        assert!(path.exists());
        assert_eq!(
            bind(&format!("unix:{}", path.display()))
                .await
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::AddrInUse
        );
        assert!(path.exists());
        drop(replacement);
        drop(replacement_owner);
        assert!(!path.exists());
    }
}
