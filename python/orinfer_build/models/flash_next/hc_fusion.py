"""Replace decode HC combine bindings with the fused combine-norm kernel.

The fused TileLang kernel is bit-identical to the separate hc_combine +
hc_norm chain (same fragment statements, same stream-dtype rounding). The
Rust adapter removes the now-redundant norm operation when it registers the
fused symbol; this tool only constructs checked operand bindings.
"""

from orinfer_build.common.files import write_json

H = 2560
STREAMS = 4
FUSED_KEY = f"hc-combine-norm-m1-h{H}-s{STREAMS}"


def _role_set(compiler, kernel):
    return set(compiler.pointers(kernel))


def decode_pairs(manifest, package, compiler):
    """Adjacent (combine, norm) kernel names inside the single-row decode program."""
    contracts = {k["name"]: k for k in package["kernels"]}
    pairs = []
    operations = manifest["programs"].get("decode", [])
    for first, second in zip(operations, operations[1:]):
        if first["kind"] != "kernel" or second["kind"] != "kernel":
            continue
        combine = contracts.get(first["name"])
        norm = contracts.get(second["name"])
        if (
            combine is not None
            and norm is not None
            and _role_set(compiler, combine) == {"Block", "Inject", "Output", "Residual"}
            and len(combine["args"]) == 4
            and _role_set(compiler, norm) == {"Normed", "Residual", "Weight"}
            and len(norm["args"]) == 3
        ):
            pairs.append((first["name"], second["name"]))
    return pairs


def publish(package, manifest, compiler, output):
    from kernels.cuda.model import hyperconnection

    output.mkdir(parents=True, exist_ok=True)
    pairs = decode_pairs(manifest, package, compiler)
    if not pairs:
        raise ValueError("No decode HC combine/norm pairs found to fuse")
    replacements = {}
    contracts = {k["name"]: k for k in package["kernels"]}
    for combine_name, norm_name in pairs:
        combine = contracts[combine_name]
        norm = contracts[norm_name]
        c = compiler.pointers(combine)
        n = compiler.pointers(norm)
        bindings = {
            "Block": c["Block"],
            "Residual": c["Residual"],
            "Inject": c["Inject"],
            "Weight": n["Weight"],
            "Output": c["Output"],
            "Normed": n["Normed"],
        }
        item, _ = compiler.emit(
            combine,
            combine_name,
            bindings,
            1,
            build=lambda: hyperconnection.hc_combine_norm.__wrapped__(
                1, H, streams=STREAMS, dtype="float16"
            ),
            key=FUSED_KEY,
            symbolic=False,
        )
        replacements[combine_name] = item
    write = output / "hc-combine-norm.json"
    write_json(
        write,
        dict(
            fused=[dict(combine=a, norm=b) for a, b in pairs],
            symbol=replacements[pairs[0][0]]["symbol"],
        ),
    )
    print(f"HC combine-norm fused {len(pairs)} decode pairs ->", write, flush=True)
    return replacements
