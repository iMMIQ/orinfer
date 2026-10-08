"""Offline exact-row Flash decode fallback; optimized fixed profiles stay intact.

Shared kernels have symbolic token rows, including routing and dispatch. Expert
arenas keep bounded capacity but Counts/TileCount exclude inactive assignments.
Private GDN/QSA reuse the published dynamic address-table kernels. No compiler
or shape discovery is needed while serving.
"""

import argparse
import copy
import shutil
import subprocess
from pathlib import Path

from tools.model.publication import (
    atomic_model,
    clone_model,
    commit_package,
    file_hash,
    load_model,
    source_path,
    link_or_copy,
)
from tools.model.upgrade_dynamic_batch import contract, expression
from tools.operators.abi import evaluate, parse_host, validate_parameter_count


def factory(label, capacity):
    from kernels.model import flash_next as f, hyperconnection as h, moe, ple
    from kernels.model.int8_projection import int8_projection
    from kernels.model.int8_swiglu import int8_swiglu
    from kernels.model.rotation_a8 import rotate_activation_a8
    from kernels.model.greedy import greedy_partials, greedy_merge

    opts = dict(dynamic_rows=True)
    parts = label.split("-")
    if label.startswith("initialize-"):
        return f.hc_initialize.__wrapped__(capacity, **opts)
    if label.startswith("hc-norm-"):
        return h.hc_norm.__wrapped__(capacity, 2560, 4, dtype="float16", **opts)
    if label.startswith("hc-down-partial-"):
        return h.hc_down_partial.__wrapped__(capacity, 320, 10240, int(parts[-1]), **opts)
    if label.startswith("hc-down-finish-"):
        return h.hc_down_finish.__wrapped__(capacity, 320, int(parts[-1]), **opts)
    if label.startswith("hc-up-mix-"):
        return h.hc_up_mix.__wrapped__(capacity, 2560, 320, **opts)
    if label.startswith("hc-injection-"):
        return h.hc_injection.__wrapped__(capacity, int(parts[-2]), int(parts[-1]), **opts)
    if label.startswith("hc-silu-"):
        return h.hc_silu.__wrapped__(capacity, 320, 4, "float16", **opts)
    if label.startswith("hc-mix-"):
        return h.hc_mix.__wrapped__(capacity, 2560, 4, "float16", **opts)
    if label.startswith("hc-combine-"):
        return h.hc_combine.__wrapped__(capacity, 2560, 4, "float16", **opts)
    if parts[0] == "hc":
        _, _, n, k, bm, bn, gm, silu = parts
        return h.hc_projection.__wrapped__(
            capacity,
            int(n),
            int(k),
            block_m=int(bm),
            dtype="float16",
            block_n=int(bn),
            group_m=int(gm),
            silu=silu == "True",
            **opts,
        )
    if label.startswith("dense-a8-"):
        _, _, _, n, k, dtype, bm, bn, bk, gm = parts
        return int8_projection.__wrapped__(
            capacity, int(n), int(k), dtype, int(bm), int(bn), int(bk), int(gm), **opts
        )
    if parts[0] == "dense" and parts[1] != "quant":
        _, _, n, k, wd, od = parts
        return f.dense_projection.__wrapped__(capacity, int(n), int(k), wd, od, **opts)
    if label.startswith("gdn-sigmoid-"):
        return f.gdn_sigmoid_norm.__wrapped__(capacity, **opts)
    if label.startswith("router-"):
        return moe.router_topk.__wrapped__(capacity, 512, 10, **opts)
    if label.startswith("histogram-"):
        return moe.expert_histogram.__wrapped__(capacity, 512, 10, **opts)
    if label.startswith("rotate-a8-"):
        ffn = parts[2] == "ffn"
        return rotate_activation_a8.__wrapped__(
            capacity * (10 if ffn else 1), 640 if ffn else 2560, swiglu=ffn, **opts
        )
    if label.startswith("dispatch-"):
        return moe.expert_dispatch.__wrapped__(capacity, 2560, 512, 10, scale_group=2560, **opts)
    if label.startswith("shared-w8-swiglu-"):
        return int8_swiglu.__wrapped__(capacity, 640, 2560, **opts)
    if label.startswith("shared-swiglu-"):
        return f.swiglu.__wrapped__(capacity, 640, **opts)
    if label.startswith("combine-"):
        return moe.moe_combine.__wrapped__(capacity, 2560, capacity * 10, 10, **opts)
    if label.startswith("ple-gate-"):
        return ple.ple_gate.__wrapped__(capacity, 2560, 4, **opts)
    if label.startswith("ple-add-"):
        return f.residual_add.__wrapped__(capacity, 10240, **opts)
    if label.startswith("greedy-partials-"):
        return greedy_partials.__wrapped__(int(parts[-1]), capacity, **opts)
    if label.startswith("greedy-merge-"):
        return greedy_merge.__wrapped__(int(parts[-1]), capacity, **opts)
    # Already-symbolic row kernels, or capacity-bounded expert tile programs.
    if label == "gdn-gates" or label.startswith(
        ("dense-quant-", "offsets-", "tiles-", "expert-gu-", "expert-down-")
    ):
        return None
    raise ValueError(f"Unsupported Flash shared kernel: {label}")


def upgrade(model, destination, output):
    import tilelang
    from kernels.model.rows import explicit_rows
    from tools.model.package import model_library
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    meta = data["metadata"]
    if (
        data["architecture"] != "flash_next"
        or not meta.get("batch_layout", {}).get("state_columns")
        or package.get("dynamic_batch_kernels")
    ):
        raise ValueError("Expected parallel Flash batch package without dynamic contracts")
    operator = clone_model(model, destination, origin)
    compiled, exports, contracts, additions = {}, {}, [], []
    for old in package["kernels"]:
        if old["name"].split("/")[0] not in {
            f"{prefix}_m{n}"
            for prefix in ("flash_batch", "flash_private")
            for n in (4, 8, 16, 32, 64, 128)
        }:
            continue
        capacity = int(old["name"].split("/")[0].rsplit("m", 1)[1])
        old_host = source_path(origin, old["host_abi"]["file"]).read_text()
        (host,) = parse_host(old_host)
        if old["name"].startswith("flash_private_"):
            contracts.append(contract(old, host, capacity))
            continue
        if "/private/" in old["name"]:
            continue
        label = Path(old["module"]["file"]).parent.name
        multiplier = 10 if label.startswith("rotate-a8-ffn-") else 1
        k = copy.deepcopy(old)
        k["name"] = old["name"].replace("flash_batch_", "flash_dynamic_", 1)
        key = f"dynamic{capacity}-{label}"
        if key not in compiled:
            fn = factory(label, capacity)
            if fn is not None:
                fn = explicit_rows(fn)
                kernel = tilelang.compile(
                    fn,
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                )
                export_kernel(kernel, output / "aot" / key)
                assets = {}
                for field, filename in [
                    ("module", "kernel.cubin"),
                    ("source", "kernel.cu"),
                    ("host_abi", "host.txt"),
                ]:
                    path = operator / "flash-dynamic/kernels" / key / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    link_or_copy(output / "aot" / key / filename, path)
                    assets[field] = dict(
                        file=str(path.relative_to(operator)), sha256=file_hash(path)
                    )
                exports[key] = assets
                compiled[key] = parse_host((output / "aot" / key / "host.txt").read_text())[0]
                validate_parameter_count(
                    (output / "aot" / key / "kernel.cu").read_text(), compiled[key]
                )
            else:
                compiled[key] = None
        generated = compiled[key]
        if generated is not None:
            bound = {
                a["value"]: value
                for a, value in zip(host["ordered_arguments"], old["args"])
                if a["ctype"] == "ctypes.c_void_p"
            }
            dims = dict(
                m=capacity * multiplier,
                M=capacity * multiplier,
                rows=capacity * multiplier,
                batch=capacity * multiplier,
            )
            k.update(exports[key])
            k["symbol"] = generated["symbol"]
            k["args"] = []
            for a in generated["ordered_arguments"]:
                if a["ctype"] == "ctypes.c_void_p":
                    k["args"].append(copy.deepcopy(bound[a["value"]]))
                elif a["ctype"] == "ctypes.c_int32":
                    k["args"].append(dict(kind="i32", value=evaluate(a["value"], dims)))
                else:
                    raise ValueError("Unexpected symbolic Flash ABI")
            launch = generated["launch_expressions"]
            k["grid"] = [evaluate(launch["gridDim" + axis], dims) for axis in "XYZ"]
            k["block"] = [evaluate(launch["blockDim" + axis], dims) for axis in "XYZ"]
            k["shared_memory_bytes"] = evaluate(launch["sharedMemBytes"], dims)
            host = generated
        validate_parameter_count(source_path(operator, k["source"]["file"]).read_text(), host)
        launch_contract = contract(k, host, capacity, multiplier)
        if label.startswith(("expert-gu-", "expert-down-")):
            launch_contract["grid"][0] = expression("(m * 10 + 15) // 16 + min(512, m * 10)")
        contracts.append(launch_contract)
        additions.append(k)
        print("DYNAMIC", k["name"], label, flush=True)
    package["kernels"].extend(additions)
    package["dynamic_batch_kernels"] = contracts
    path = operator / "lib/model.so"
    path.unlink()
    shutil.copyfile(model_library(), path)
    package["execution"]["library"]["sha256"] = file_hash(path)
    commit_package(destination, operator, data, package)
    subprocess.run(
        [str(Path("target/release/orinfer").resolve()), "plan-model", str(destination)],
        check=True,
        stdout=(output / "registered-plan.json").open("w"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    a = p.parse_args()
    if a.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(a.compile_cache, a.output / "cache/0.1.15")
    with atomic_model(a.model_output) as staging:
        upgrade(a.model.resolve(), staging, a.output)


if __name__ == "__main__":
    main()
