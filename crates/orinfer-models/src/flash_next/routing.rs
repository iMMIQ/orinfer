//! Single-token routing metadata fusion belongs to the Flash execution recipe.
use super::contracts::{
    Storage, connect, distinct, fuse, kernel_at, kernel_buffer_arg as buffer, retire,
};
use crate::{
    artifact::{Dtype, Kernel, Result},
    model::Manifest,
};

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel], exhaustive: bool) -> Result<()> {
    let contracts = super::contracts::kernels(kernels);
    let storage = exhaustive.then(|| Storage::new(&manifest.buffers));
    let anchors = kernels
        .iter()
        .filter(|k| k.symbol == "single_route_metadata_kernel_kernel");
    fuse(
        &mut manifest.programs,
        anchors,
        |program, operations, index, route, skip| {
            if program != "decode" {
                return Err("Single-token routing requires decode program".into());
            }
            let prefix = kernel_at(&contracts, operations, index + 1)?;
            let tiles = kernel_at(&contracts, operations, index + 2)?;
            if route.args.len() != 8
                || prefix.args.len() != 4
                || tiles.args.len() != 4
                || prefix.symbol != "main_kernel"
                || tiles.symbol != "main_kernel"
            {
                return Err("Unexpected single-token routing ABI".into());
            }
            for (old, connections) in [
                (prefix, [(0, 0), (1, 2), (2, 4), (3, 6)]),
                (tiles, [(0, 0), (1, 5), (2, 6), (3, 7)]),
            ] {
                connect(old, route, &connections)?;
            }
            distinct((0..8).map(|i| buffer(route, i)))?;
            if let Some(storage) = &storage {
                for (i, elements) in [512, 10, 512, 10, 1, 11, 512, 11].into_iter().enumerate() {
                    storage.check(buffer(route, i)?, Dtype::I32, false, |b| {
                        Ok(b.shape == [elements])
                    })?;
                }
            }
            retire(skip, [index + 1, index + 2])
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::Argument;

    fn fixture() -> (Manifest, Vec<Kernel>) {
        let names = [
            "Counts",
            "IDs",
            "Offsets",
            "Relative",
            "TileCount",
            "TileExpert",
            "TileOffsets",
            "TileRow",
        ];
        let sizes = [512, 10, 512, 10, 1, 11, 512, 11];
        let buffers: Vec<_> = names
            .iter()
            .zip(sizes)
            .map(|(name, n)| {
                serde_json::json!({
                    "name":name,"dtype":"i32","shape":[n],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write"
                })
            })
            .collect();
        let manifest = serde_json::from_value(serde_json::json!({
            "schema_version":1,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":buffers,
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test","programs":{"decode":
            (0..3).map(|i|serde_json::json!({"kind":"kernel","name":format!("decode/k{i}")})).collect::<Vec<_>>()}
        })).unwrap();
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        let args = [
            names.to_vec(),
            vec!["Counts", "Offsets", "TileCount", "TileOffsets"],
            vec!["Counts", "TileExpert", "TileOffsets", "TileRow"],
        ];
        let kernels = args.into_iter().enumerate().map(|(i,args)| serde_json::from_value(
            serde_json::json!({"name":format!("decode/k{i}"),"symbol":if i==0 {"single_route_metadata_kernel_kernel"} else {"main_kernel"},
                "module":identity,"source":identity,"host_abi":identity,"grid":[1,1,1],
                "block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
                "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()})
        ).unwrap()).collect();
        (manifest, kernels)
    }
    #[test]
    fn removes_only_the_two_routing_metadata_consumers() {
        let (mut m, k) = fixture();
        build(&mut m, &k, true).unwrap();
        assert_eq!(m.programs["decode"].len(), 1);
    }
    #[test]
    fn rejects_wrong_binding_and_aliased_input() {
        let (mut m, mut k) = fixture();
        k[2].args[2] = Argument::Buffer {
            name: "Relative".into(),
        };
        assert!(
            build(&mut m, &k, true)
                .unwrap_err()
                .contains("producer/consumer")
        );
        let (mut m, mut k) = fixture();
        k[0].args[1] = Argument::Buffer {
            name: "Relative".into(),
        };
        assert!(
            build(&mut m, &k, true)
                .unwrap_err()
                .contains("Aliased fusion buffer")
        );
    }
    #[test]
    fn rejects_batched_program_and_missing_contract() {
        let (mut m, k) = fixture();
        let ops = m.programs.remove("decode").unwrap();
        m.programs.insert("verify_m4".into(), ops);
        assert!(
            build(&mut m, &k, true)
                .unwrap_err()
                .contains("decode program")
        );
        let (mut m, mut k) = fixture();
        k.pop();
        assert!(build(&mut m, &k, true).unwrap_err().contains("contract"));
    }
}
