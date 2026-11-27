use super::*;
#[test]
fn shared_mapping_cleanup_never_releases_snapshot_owned_handle() {
    let allocation = Rc::new(super::super::snapshot::Allocation {
        id: crate::backend::kv::allocation_id().unwrap(),
        used: 8192,
        native: None,
        owner: orinfer_model_sdk::device::DeviceOwner::new(
            orinfer_model_sdk::device::Backend::Cuda,
            0,
        ),
        address: 0,
        bytes: 8192,
        handle: Some(7),
    });
    let mut slabs = vec![Slab {
        id: kv::allocation_id().unwrap(),
        offset: 0,
        bytes: 8192,
        handle: 7,
        mapped: true,
        shared: Some(Rc::clone(&allocation)),
    }];
    assert_eq!(Rc::strong_count(&allocation), 2);
    release_owned(
        &mut slabs,
        |_| Ok(()),
        |_| panic!("snapshot registry owns the handle"),
    )
    .unwrap();
    assert_eq!(Rc::strong_count(&allocation), 1);
    assert!(!overlaps(8192, 9000, 0, 8192));
    assert!(overlaps(8191, 8192, 0, 8192));
    assert!(!overlaps(0, 0, 0, 8192));
}
#[test]
#[ignore = "Requires exclusive GPU experiment lock"]
fn vmm_shared_prefix_cow_and_graph_replay() {
    let mut s = Session::new(Driver::load().unwrap());
    // SAFETY: Test exclusively owns a context reference and its CUDA resources.
    unsafe {
        check(sys::cuInit(0), "init").unwrap();
        check(sys::cuDeviceGet(&mut s.device, 0), "device").unwrap();
        check(sys::cuCtxGetCurrent(&mut s.previous), "previous").unwrap();
        let mut context = std::ptr::null_mut();
        check(
            sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
            "retain",
        )
        .unwrap();
        s.retained = true;
        check(sys::cuCtxSetCurrent(context), "current").unwrap();
        check(sys::cuStreamCreate(&mut s.stream, 1), "stream").unwrap();
    }
    let mut source = Reservation::reserve(&s, 4 << 20, 1).unwrap();
    let bytes = source.granularity * 2;
    source.grow_bytes(&s, bytes).unwrap();
    let source_address = source.address;
    let allocation = Rc::new(super::super::snapshot::Allocation {
        id: crate::backend::kv::allocation_id().unwrap(),
        used: 8192,
        native: None,
        owner: s.owner,
        address: source.address,
        bytes,
        handle: Some(source.allocation_handle().unwrap()),
    });
    s.prefix_buffers.get_mut().insert("source".into(), source);
    let mut left = Reservation::reserve(&s, bytes * 2, 1).unwrap();
    let mut right = Reservation::reserve(&s, bytes * 2, 1).unwrap();
    let left_address = left.address;
    let right_address = right.address;
    // Capture a writer before replacing physical backing at the same VA.
    left.grow_bytes(&s, bytes).unwrap();
    // SAFETY: Captured memset targets a live owned mapping; all source writes finish first.
    unsafe {
        check(
            sys::cuMemsetD8Async(source_address, 37, bytes, s.stream),
            "source initialization",
        )
        .unwrap();
        check(sys::cuStreamSynchronize(s.stream), "init sync").unwrap();
        check(
            sys::cuStreamBeginCapture_v2(
                s.stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            ),
            "capture",
        )
        .unwrap();
        s.capturing = true;
        check(
            sys::cuMemsetD8Async(left_address, 91, 4, s.stream),
            "record write",
        )
        .unwrap();
        check(
            sys::cuStreamEndCapture(s.stream, &mut s.graph),
            "capture end",
        )
        .unwrap();
        s.capturing = false;
        check(
            sys::cuGraphInstantiateWithFlags(&mut s.exec, s.graph, 0),
            "instantiate",
        )
        .unwrap();
    }
    left.release_slabs(&s.driver).unwrap();
    left.map_shared(&s, 0, Rc::clone(&allocation)).unwrap();
    right.map_shared(&s, 0, Rc::clone(&allocation)).unwrap();
    assert_eq!(left.physical_bytes() + right.physical_bytes(), 0);
    assert_eq!(left.writable_bytes(bytes, 4).unwrap(), 0);
    assert_eq!(left.writable_bytes(0, 4).unwrap(), bytes);
    left.make_writable(&s, 0, 4).unwrap();
    assert_eq!(left.address, left_address);
    assert_eq!(left.physical_bytes(), bytes);
    assert_eq!(right.physical_bytes(), 0);
    // SAFETY: The graph uses unchanged VAs; synchronized downloads cover four bytes.
    unsafe {
        check(sys::cuGraphLaunch(s.exec, s.stream), "COW replay").unwrap();
        check(sys::cuStreamSynchronize(s.stream), "replay sync").unwrap();
        for (address, expected) in [
            (left_address, 91u8),
            (right_address, 37),
            (source_address, 37),
        ] {
            let mut values = [0u8; 4];
            check(
                sys::cuMemcpyDtoH_v2(values.as_mut_ptr().cast(), address, values.len()),
                "verify sharing isolation",
            )
            .unwrap();
            assert_eq!(values, [expected; 4]);
        }
    }
    right.grow_bytes(&s, bytes + 1).unwrap();
    assert_eq!(right.physical_bytes(), right.granularity);
    s.virtual_buffers.get_mut().insert("left".into(), left);
    s.virtual_buffers.get_mut().insert("right".into(), right);
    s.cleanup().unwrap();
}
#[test]
fn cleanup_retries_preserve_mapping_and_handle_ownership() {
    let mut slabs = vec![Slab {
        id: kv::allocation_id().unwrap(),
        offset: 0,
        bytes: 4096,
        handle: 7,
        mapped: true,
        shared: None,
    }];
    let mut releases = 0;
    assert!(
        release_owned(
            &mut slabs,
            |_| Err("unmap fault".into()),
            |_| {
                releases += 1;
                Ok(())
            }
        )
        .is_err()
    );
    assert_eq!(releases, 0);
    assert!(slabs[0].mapped);
    assert!(release_owned(&mut slabs, |_| Ok(()), |_| Err("release fault".into())).is_err());
    assert!(!slabs[0].mapped);
    assert_eq!(slabs[0].handle, 7);
    release_owned(
        &mut slabs,
        |_| panic!("must not unmap twice"),
        |h| {
            assert_eq!(h, 7);
            Ok(())
        },
    )
    .unwrap();
    assert!(slabs.is_empty());
}
#[test]
#[ignore = "Requires exclusive GPU experiment lock"]
fn vmm_graph_growth_reset_and_last_token() {
    use super::super::Session;
    let mut s = Session::new(Driver::load().unwrap());
    // SAFETY: Test owns a primary-context reference, stream and graph handles.
    unsafe {
        check(sys::cuInit(0), "init").unwrap();
        check(sys::cuDeviceGet(&mut s.device, 0), "device").unwrap();
        check(sys::cuCtxGetCurrent(&mut s.previous), "previous").unwrap();
        let mut context: sys::CUcontext = std::ptr::null_mut();
        check(
            sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
            "retain",
        )
        .unwrap();
        s.retained = true;
        check(sys::cuCtxSetCurrent(context), "current").unwrap();
        check(sys::cuStreamCreate(&mut s.stream, 1), "stream").unwrap();
    }
    let mut r = Reservation::reserve(&s, 262144 * 2048, 2048).unwrap();
    let address = r.address;
    r.grow(&s, 1).unwrap();
    assert_eq!(r.mapped, r.granularity);
    // SAFETY: Capture a write into a mapped buffer; subsequent mappings keep
    // the graph address stable. Every map/unmap follows stream completion.
    unsafe {
        check(sys::cuStreamSynchronize(s.stream), "init sync").unwrap();
        check(
            sys::cuStreamBeginCapture_v2(
                s.stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            ),
            "capture",
        )
        .unwrap();
        s.capturing = true;
        check(
            sys::cuMemsetD8Async(address, 37, 4, s.stream),
            "record write",
        )
        .unwrap();
        check(
            sys::cuStreamEndCapture(s.stream, &mut s.graph),
            "capture end",
        )
        .unwrap();
        s.capturing = false;
        check(
            sys::cuGraphInstantiateWithFlags(&mut s.exec, s.graph, 0),
            "instantiate",
        )
        .unwrap();
    }
    for tokens in [1024, 1025, 8193, 262144] {
        r.grow(&s, tokens).unwrap();
        assert_eq!(address, r.address);
        assert_eq!(r.mapped, r.extent(tokens).unwrap());
        let last = address + (tokens * 2048 - 4) as u64;
        let mut value = [0u8; 4];
        // SAFETY: First and last writes are within mapped extents.
        unsafe {
            check(
                sys::cuMemsetD8Async(last, 71, 4, s.stream),
                "last token write",
            )
            .unwrap();
            check(sys::cuGraphLaunch(s.exec, s.stream), "graph replay").unwrap();
            check(sys::cuStreamSynchronize(s.stream), "sync").unwrap();
            check(
                sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), last, 4),
                "last token read",
            )
            .unwrap();
            assert_eq!(value, [71; 4]);
            check(
                sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), address, 4),
                "graph read",
            )
            .unwrap();
            assert_eq!(value, [37; 4]);
        }
    }
    r.release_slabs(&s.driver).unwrap();
    assert_eq!(r.mapped, 0);
    r.grow(&s, 1).unwrap();
    let mut value = [1u8; 4];
    // SAFETY: Fresh physical storage is zeroed, graph still uses the same VA.
    unsafe {
        check(sys::cuStreamSynchronize(s.stream), "reset sync").unwrap();
        check(
            sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), address, 4),
            "reset read",
        )
        .unwrap();
        assert_eq!(value, [0; 4]);
        check(sys::cuGraphLaunch(s.exec, s.stream), "reset replay").unwrap();
        check(sys::cuStreamSynchronize(s.stream), "reset replay sync").unwrap();
        check(
            sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), address, 4),
            "replay read",
        )
        .unwrap();
        assert_eq!(value, [37; 4]);
    }
    s.virtual_buffers.borrow_mut().insert("test".into(), r);
    s.cleanup().unwrap();
}
#[test]
fn allocation_abi_and_checked_extents() {
    assert_eq!(std::mem::size_of::<AllocationProp>(), 32);
    assert_eq!(std::mem::size_of::<AccessDesc>(), 12);
    assert_eq!(round_up(2048, 2 << 20).unwrap(), 2 << 20);
    assert!(round_up(usize::MAX, 2 << 20).is_err());
    assert!(round_up(1, 3).is_err());
    let r = Reservation {
        address: 0,
        stride: 2048,
        region: kv::Reservation::new(4 << 20, 2 << 20),
    };
    assert_eq!(r.extent(1024).unwrap(), 2 << 20);
    assert_eq!(r.extent(1025).unwrap(), 4 << 20);
    assert!(r.extent(2049).is_err());
    assert_eq!(r.required_extent(2049).unwrap(), 6 << 20);
    let scale = Reservation {
        address: 0,
        stride: 32,
        region: kv::Reservation::new(8 << 20, 2 << 20),
    };
    assert_eq!(scale.extent(65536).unwrap(), 2 << 20);
    assert_eq!(scale.extent(65537).unwrap(), 4 << 20);
    assert_eq!(scale.extent(262144).unwrap(), 8 << 20);
    assert!(scale.extent(usize::MAX).is_err());
}
