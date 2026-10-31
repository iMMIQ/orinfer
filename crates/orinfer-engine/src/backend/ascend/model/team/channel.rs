//! Fail-closed, ordered replies with an owner-thread wakeup and one deadline.
use super::{Command, Reply};
use crate::{artifact::Result, error::EngineError};
use std::{
    cell::RefCell,
    sync::mpsc,
    thread::{self, Thread},
    time::{Duration, Instant},
};

type Response = std::result::Result<Reply, EngineError>;
pub(super) fn command_timeout(raw: Option<&str>) -> Result<Duration> {
    let millis = raw
        .unwrap_or("30000")
        .parse::<u64>()
        .ok()
        .filter(|n| (1..=1_800_000).contains(n))
        .ok_or("ORINFER_A2_COMMAND_TIMEOUT_MS must be within 1..1800000")?;
    Ok(Duration::from_millis(millis))
}
pub(super) struct Peer {
    pub send: mpsc::Sender<Command>,
    pub receive: mpsc::Receiver<Response>,
}
pub(super) struct Replies {
    send: Option<mpsc::Sender<Response>>,
    owner: Thread,
}
impl Replies {
    pub fn send(&self, value: Response) -> std::result::Result<(), mpsc::SendError<Response>> {
        let result = self.send.as_ref().unwrap().send(value);
        self.owner.unpark();
        result
    }
}
impl Drop for Replies {
    fn drop(&mut self) {
        // A panic/disconnect must also wake the owner waiting on another rank.
        // Disconnect before waking: otherwise the owner can observe Empty and
        // park again before the sender field is automatically dropped.
        drop(self.send.take());
        self.owner.unpark();
    }
}
pub(super) fn reply_channel() -> (Replies, mpsc::Receiver<Response>) {
    let (send, receive) = mpsc::channel();
    (
        Replies {
            send: Some(send),
            owner: thread::current(),
        },
        receive,
    )
}
pub(super) struct Transport {
    peers: Vec<Peer>,
    failure: RefCell<Option<EngineError>>,
    timeout: Duration,
}
impl Transport {
    pub fn new(timeout: Duration) -> Self {
        Self {
            peers: vec![],
            failure: RefCell::new(None),
            timeout,
        }
    }
    pub fn push(&mut self, peer: Peer) {
        self.peers.push(peer);
    }
    pub fn iter(&self) -> std::slice::Iter<'_, Peer> {
        self.peers.iter()
    }
    fn fail(&self, error: EngineError) -> String {
        let mut failure = self.failure.borrow_mut();
        crate::error::forward(failure.get_or_insert(error).clone())
    }
    fn broken(&self, message: String) -> String {
        self.fail(EngineError::take(message, true))
    }
    pub fn send(&self, command: Command) -> Result<()> {
        if let Some(error) = self.failure.borrow().as_ref() {
            return Err(crate::error::forward(error.clone()));
        }
        for (rank, peer) in self.peers.iter().enumerate() {
            if peer.send.send(command.clone()).is_err() {
                return Err(self.broken(format!("A2 rank {rank} command channel disconnected")));
            }
        }
        Ok(())
    }
    pub fn exchange(&self, command: Command) -> Result<Vec<Reply>> {
        self.send(command)?;
        self.receive(self.timeout)
    }
    pub fn collect(&self) -> Result<Vec<Reply>> {
        self.receive(self.timeout)
    }
    pub fn receive(&self, timeout: Duration) -> Result<Vec<Reply>> {
        if let Some(error) = self.failure.borrow().as_ref() {
            return Err(crate::error::forward(error.clone()));
        }
        let deadline = Instant::now() + timeout;
        let mut replies: Vec<_> = (0..self.peers.len()).map(|_| None).collect();
        let mut remaining = self.peers.len();
        while remaining != 0 {
            for (rank, (peer, slot)) in self.peers.iter().zip(&mut replies).enumerate() {
                if slot.is_some() {
                    continue;
                }
                match peer.receive.try_recv() {
                    Ok(Ok(reply)) => {
                        *slot = Some(reply);
                        remaining -= 1;
                    }
                    Ok(Err(error)) => return Err(self.fail(error)),
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(
                            self.broken(format!("A2 rank {rank} reply channel disconnected"))
                        );
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
            if remaining == 0 {
                break;
            }
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return Err(self.broken(format!(
                    "A2 worker reply deadline exceeded ({remaining} ranks pending)"
                )));
            };
            // A notification arriving between the scan and park is retained.
            thread::park_timeout(left);
        }
        Ok(replies.into_iter().map(Option::unwrap).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn command_deadline_is_bounded_and_configurable() {
        assert_eq!(command_timeout(None).unwrap(), Duration::from_secs(30));
        assert_eq!(
            command_timeout(Some("60000")).unwrap(),
            Duration::from_secs(60)
        );
        for bad in ["0", "-1", "1800001", "x"] {
            assert!(command_timeout(Some(bad)).is_err());
        }
    }

    #[test]
    fn later_rank_failure_wakes_owner_preserves_device_code_and_retires_team() {
        for code in [507015, 207001] {
            thread::scope(|scope| {
                let mut transport = Transport::new(Duration::from_secs(2));
                let (release, gate) = mpsc::channel();
                let (send, commands) = mpsc::channel();
                let (reply, receive) = reply_channel();
                transport.push(Peer { send, receive });
                let delayed = scope.spawn(move || {
                    assert!(matches!(commands.recv().unwrap(), Command::Reclaim));
                    gate.recv_timeout(Duration::from_secs(5)).unwrap();
                    reply.send(Ok(Reply::Capacity(true))).unwrap();
                    assert!(matches!(
                        commands.try_recv(),
                        Err(mpsc::TryRecvError::Empty)
                    ));
                });
                let (send, commands) = mpsc::channel();
                let (reply, receive) = reply_channel();
                transport.push(Peer { send, receive });
                scope.spawn(move || {
                    assert!(matches!(commands.recv().unwrap(), Command::Reclaim));
                    crate::error::record_ascend(code);
                    reply
                        .send(Err(EngineError::take("rank failure".into(), true)))
                        .unwrap();
                });
                let message = transport.exchange(Command::Reclaim).err().unwrap();
                let error = EngineError::take(message, false);
                assert_eq!(error.ascend_code, Some(code));
                assert!(error.is_fatal());
                assert!(transport.exchange(Command::Reset(vec![0])).is_err());
                assert!(EngineError::take("retired".into(), false).is_fatal());
                release.send(()).unwrap();
                delayed.join().unwrap();
            });
        }
    }

    #[test]
    fn expired_reply_deadline_discards_late_replies_and_never_resends() {
        let mut transport = Transport::new(Duration::from_millis(2));
        let (send, commands) = mpsc::channel();
        let (reply, receive) = reply_channel();
        transport.push(Peer { send, receive });
        let error = transport.exchange(Command::Reclaim).err().unwrap();
        assert_eq!(EngineError::take(error, false).kind, ErrorKind::Invariant);
        reply.send(Ok(Reply::Capacity(true))).unwrap();
        assert!(transport.exchange(Command::Reclaim).is_err());
        assert!(EngineError::take("retired".into(), false).is_fatal());
        assert!(matches!(commands.try_recv().unwrap(), Command::Reclaim));
        assert!(matches!(
            commands.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn panic_on_a_later_rank_is_fatal_even_while_an_earlier_rank_is_pending() {
        thread::scope(|scope| {
            let mut transport = Transport::new(Duration::from_secs(2));
            let (send, _commands) = mpsc::channel();
            let (_reply, receive) = reply_channel();
            transport.push(Peer { send, receive });
            let (send, commands) = mpsc::channel();
            let (reply, receive) = reply_channel();
            transport.push(Peer { send, receive });
            let worker = scope.spawn(move || {
                let _reply = reply;
                commands.recv().unwrap();
                panic!("rank panic");
            });
            let error = transport.exchange(Command::Reclaim).err().unwrap();
            assert!(error.contains("rank 1"));
            assert!(EngineError::take(error, false).is_fatal());
            assert!(worker.join().is_err());
        });
    }
}
