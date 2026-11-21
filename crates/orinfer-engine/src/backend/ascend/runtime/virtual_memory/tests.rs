use crate::backend::ascend::tests::Fixture;

#[test]
fn demand_growth_retains_addresses_contents_and_completion_ownership() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let grain = 2 << 20;
    let buffer = session.reserve(grain * 8 - 19).unwrap();
    let address = buffer.pointer();
    assert_eq!(session.allocated_bytes(), 0);
    assert_eq!(buffer.growth_bytes(17).unwrap(), grain);
    session.grow(&buffer, 17).unwrap();
    assert_eq!(session.allocated_bytes(), grain);
    session
        .upload(&buffer, 0, &[37; 17])
        .unwrap()
        .wait()
        .unwrap();
    session.grow(&buffer, grain + 3).unwrap();
    assert_eq!(session.allocated_bytes(), grain * 2);
    assert_eq!(address, buffer.pointer());
    assert_eq!(session.download(&buffer, 0, 17).unwrap(), vec![37; 17]);
    assert_eq!(session.download(&buffer, grain, 3).unwrap(), vec![0; 3]);
    assert!(session.grow(&buffer, buffer.bytes() + 1).is_err());
    session.release_pages(&buffer).unwrap();
    assert_eq!(session.allocated_bytes(), 0);
    session.grow(&buffer, 19).unwrap();
    assert_eq!(session.download(&buffer, 0, 19).unwrap(), vec![0; 19]);
    let ticket = session.upload(&buffer, 0, &[91; 19]).unwrap();
    drop(buffer);
    assert_eq!(session.allocated_bytes(), grain);
    ticket.wait().unwrap();
    drop(ticket);
    assert_eq!(session.allocated_bytes(), 0);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn demand_growth_failure_preserves_mapped_history_and_budget() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let grain = 2 << 20;
    let buffer = session.reserve(grain * 4).unwrap();
    session.grow(&buffer, 1).unwrap();
    session.upload(&buffer, 0, &[73]).unwrap().wait().unwrap();
    fixture.failure(100);
    assert!(session.grow(&buffer, grain + 1).is_err());
    assert_eq!(session.allocated_bytes(), grain);
    fixture.failure(0);
    assert_eq!(session.download(&buffer, 0, 1).unwrap(), vec![73]);
    session.grow(&buffer, grain + 1).unwrap();
    drop(buffer);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn growth_batch_fences_once_and_keeps_history_and_disjoint_owners() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let grain = 2 << 20;
    let first = session.reserve(grain * 3).unwrap();
    let second = session.reserve(grain * 3).unwrap();
    session.grow(&first, 1).unwrap();
    let pending = session.upload(&first, 0, &[73]).unwrap();
    let syncs = fixture.counter(12);
    session
        .grow_many(&[(first.clone(), grain + 3), (second.clone(), 17)])
        .unwrap();
    assert_eq!(fixture.counter(12) - syncs, 3);
    assert_eq!(session.allocated_bytes(), 3 * grain);
    assert_eq!(session.download(&first, 0, 1).unwrap(), [73]);
    assert_eq!(session.download(&first, grain, 3).unwrap(), [0; 3]);
    assert_eq!(session.download(&second, 0, 17).unwrap(), [0; 17]);
    session.upload(&second, 0, &[91]).unwrap().wait().unwrap();
    assert_eq!(session.download(&first, 0, 1).unwrap(), [73]);
    let syncs = fixture.counter(12);
    session.grow_many(&[(first.clone(), grain)]).unwrap();
    assert_eq!(fixture.counter(12), syncs, "warm growth needs no fence");
    drop((first, second, pending, session));
    fixture.assert_clean();
}

#[test]
fn logical_page_identity_follows_caller_order_across_native_address_orders() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let grain = 2 << 20;
    let mut buffers = (0..4)
        .map(|_| session.reserve(grain * 2).unwrap())
        .collect::<Vec<_>>();
    buffers.sort_by_key(|b| std::cmp::Reverse(std::rc::Rc::as_ptr(&b.allocation) as usize));
    let mut requests = buffers.iter().cloned().map(|b| (b, 17)).collect::<Vec<_>>();
    requests.push((buffers[0].slice(grain, 3).unwrap(), 3));
    session.grow_many(&requests).unwrap();
    let ids = buffers
        .iter()
        .map(|b| b.kv_preview(17).unwrap()[0].0.id)
        .collect::<Vec<_>>();
    assert!(
        ids.windows(2).all(|ids| ids[1] == ids[0] + 2),
        "TP logical allocation order must be independent of native addresses: {ids:?}"
    );
    assert_eq!(session.allocated_bytes(), 5 * grain);
    drop((requests, buffers, session));
    fixture.assert_clean();
}
#[test]
fn growth_batch_deduplicates_views_and_validates_the_whole_budget() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let grain = 2 << 20;
    let first = session.reserve(grain * 3).unwrap();
    let view = first.slice(grain, 17).unwrap();
    session
        .grow_many(&[
            (first.clone(), 1),
            (view.clone(), 17),
            (first.clone(), grain),
        ])
        .unwrap();
    assert_eq!(session.allocated_bytes(), 2 * grain);
    let huge = session.reserve(256 << 20).unwrap();
    assert!(session.grow_many(&[(huge.clone(), 256 << 20)]).is_err());
    assert_eq!(session.allocated_bytes(), 2 * grain);
    let other = fixture.session();
    let foreign = other.reserve(grain).unwrap();
    assert!(
        session
            .grow_many(&[(huge.clone(), 1), (foreign.clone(), 1)])
            .is_err()
    );
    assert!(
        session
            .grow_many(&[(huge.clone(), 1), (first.clone(), 4 * grain)])
            .is_err()
    );
    assert_eq!(session.allocated_bytes(), 2 * grain);
    drop((first, view, huge, foreign, other, session));
    fixture.assert_clean();
}

#[test]
fn growth_budget_reuses_backing_released_by_the_prior_fence() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.reserve(256 << 20).unwrap();
    let temporary = session.allocate(2 << 20).unwrap();
    let pending = session.upload(&temporary, 0, &[17]).unwrap();
    drop(temporary);
    assert_eq!(session.allocated_bytes(), 2 << 20);
    session.grow_many(&[(buffer.clone(), 256 << 20)]).unwrap();
    assert_eq!(session.allocated_bytes(), 256 << 20);
    assert_eq!(session.download(&buffer, 0, 1).unwrap(), [0]);
    drop((pending, buffer, session));
    fixture.assert_clean();
}
