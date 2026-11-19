"""Rebuild the measured Flash Next expert projections for small decode and verification packets."""

import re
from functools import partial

from orinfer_build.tuning.expert_tuning import tile_capacity


def publish(package, manifest, compiler):
    from kernels.cuda.model.e8p_exact import e8p_exact_grouped

    text = package["config_signature"]["text"]
    experts, top = text["num_experts"], text["num_experts_per_tok"]
    geometry = {(1280, 2560), (2560, 640)}
    launches = manifest["dynamic_batch_kernels"]
    replacements, pairs = {}, {}
    ranges = {"4-4": 4, "8-8": 8, "9-15": 1024}
    for old in package["kernels"]:
        name = old["name"]
        match = re.fullmatch(r"(flash_decode/layer\d+/post)/k\d+/rows/(4-4|8-8|9-15)", name)
        if match is None or old["symbol"] != "exact_e8p_int4_kernel":
            continue
        section, extent = match.groups()
        rows = ranges[extent]
        dynamic = extent == "9-15"
        pointers = compiler.pointers(old)
        bank = compiler.buffers[pointers["P"]["name"]]
        count, blocks, outputs, words = bank["shape"]
        inputs = blocks * 64
        tiles = tile_capacity(rows, experts, top, 8)
        launch = launches.get(name)
        if (
            bank["dtype"] != "u16"
            or count != experts
            or words != 8
            or (outputs, inputs) not in geometry
            or old["grid"] != [(outputs + 127) // 128, tiles, 1]
            or old["block"] != [128, 1, 1]
            or launch is None
            or launch["capacity"] != 1024
            or (not dynamic and launch["arguments"])
            or (
                not dynamic
                and launch["grid"] != [dict(op="constant", value=n) for n in old["grid"]]
            )
        ):
            raise ValueError(f"Unexpected Flash decode expert contract: {name}")
        seen = pairs.setdefault((section, extent), set())
        if (outputs, inputs) in seen:
            raise ValueError(f"Duplicate Flash decode expert geometry: {name}")
        seen.add((outputs, inputs))

        new, rebuilt_launch = compiler.emit(
            old,
            name,
            pointers,
            rows,
            build=partial(
                e8p_exact_grouped.__wrapped__,
                rows * top,
                experts,
                tiles,
                outputs,
                inputs,
                block_m=8,
                stages=4,
                warp_private=True,
                dynamic_rows=dynamic,
            ),
            key=f"expert-decode-warp-{rows}-{experts}-{top}-{outputs}-{inputs}",
            symbolic=dynamic,
            multiplier=top if dynamic else 1,
        )
        if (
            new["grid"] != old["grid"]
            or new["block"] != old["block"]
            or new["args"] != old["args"]
            or (dynamic and rebuilt_launch != launch)
        ):
            raise ValueError(f"Changed Flash decode expert routing geometry: {name}")
        replacements[name] = new

    sections = [
        name for name in manifest["programs"] if re.fullmatch(r"flash_decode/layer\d+/post", name)
    ]
    expected = {(section, rows) for section in sections for rows in ranges}
    if not expected or pairs.keys() != expected or any(pair != geometry for pair in pairs.values()):
        raise ValueError("Missing paired Flash decode expert projections")
    package["kernels"] = [replacements.get(k["name"], k) for k in package["kernels"]]
    return sorted(replacements)
