"""Typed integer construction contracts derived from operands and the host ABI.

The formal builder consumes a published model's architectural binding contract.
It never recovers computation parameters from generated CUDA or from filenames.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True, order=True)
class Projection:
    kind: str
    rows: int
    outputs: int
    inputs: int
    dtype: str
    dynamic: bool

    def spec(self):
        return asdict(self)


def select_projections(groups, *, kinds=None, shapes=None):
    """Restrict a rebuild by logical geometry, preserving every other binding."""
    selected = {
        spec: items
        for spec, items in groups.items()
        if (not kinds or spec.kind in kinds)
        and (not shapes or [spec.outputs, spec.inputs] in shapes)
    }
    if not selected:
        raise ValueError("No projection contracts match the requested scope")
    return selected


def bounded_measurements(manifest):
    """Measure each declared dynamic selection at its endpoints and centre."""
    result = {}
    for variants in manifest.get("row_execution", {}).get("variants", {}).values():
        for variant in variants:
            first, last = variant["min_rows"], variant["max_rows"]
            if first == last:
                continue
            samples = {first, last, math.isqrt(first * last)}
            for name in variant["kernels"]:
                result.setdefault(name, set()).update(samples)
    return {name: sorted(rows) for name, rows in result.items()}


def fixed_extents(manifest):
    result = {}
    for variants in manifest.get("row_execution", {}).get("variants", {}).values():
        for variant in variants:
            if variant["min_rows"] != variant["max_rows"]:
                continue
            for name in variant["kernels"]:
                result.setdefault(name, set()).add(variant["min_rows"])
    return result


def prefill_names(manifest):
    programs = {p["prefill_program"] for p in manifest["prefill_plans"]}
    programs.update(s for p in manifest["prefill_plans"] for s in p.get("stages", []))
    return {
        op["name"]
        for name in programs
        for op in manifest["programs"][name]
        if op["kind"] == "kernel"
    }


def projection(kernel, host, pointers, buffers, dynamic, fixed):
    roles = set(pointers)
    if roles == {"A", "W", "S", "AS", "Out"}:
        kind, weight, output = "q4", "W", "Out"
        names = dict(W=pointers["W"], WS=pointers["S"])
    elif roles == {"A", "Weight", "WeightScale", "TokenScale", "Output"}:
        kind, weight, output = "w8", "Weight", "Output"
        names = dict(W=pointers["Weight"], WS=pointers["WeightScale"])
    elif roles == {"A", "Gate", "Up", "GateScale", "UpScale", "TokenScale", "Output"}:
        kind, weight, output = "swiglu", "Gate", "Output"
        names = {r: pointers[r] for r in ("Gate", "Up", "GateScale", "UpScale")}
    else:
        return None
    w = buffers[pointers[weight]["name"]]
    expected = "u16" if kind == "q4" else "i8"
    if (
        len(w["shape"]) != 2
        or w["dtype"] != expected
        or any(v.get("offset", 0) for v in names.values())
    ):
        raise ValueError("Unsupported integer weight geometry/view")
    n, k = w["shape"]
    k *= 4 if kind == "q4" else 1
    if k % 128 or min(n, k) < 1:
        raise ValueError("Invalid integer matrix geometry")
    scalar = {
        a["value"]: b["value"]
        for a, b in zip(host["ordered_arguments"], kernel["args"])
        if a["ctype"].endswith("c_int32") and a["value"] in ("m", "rows")
    }
    symbolic = bool(scalar)
    extents = fixed.get(kernel["name"], set())
    if symbolic:
        rows = dynamic.get(kernel["name"], {}).get("capacity", next(iter(scalar.values())))
        if any(v != rows for v in scalar.values()):
            raise ValueError("Unexpected integer row multiplier")
    elif extents:
        if len(extents) != 1:
            raise ValueError("Static projection used for multiple live row counts")
        rows = next(iter(extents))
    else:
        # Only the single-token reference/head contracts lack a dynamic scalar
        # and an AOT row selection. Validate the bound activation scale extent.
        scale = pointers["AS" if kind == "q4" else "TokenScale"]
        rows = 1
        if buffers[scale["name"]]["dtype"] != "f16":
            raise ValueError("Expected FP16 token-scale reference contract")
    output_dtype = buffers[pointers[output]["name"]]["dtype"]
    if output_dtype not in ("f16", "f32") or kind == "swiglu" and output_dtype != "f16":
        raise ValueError("Unsupported integer output precision")
    for role, operand in names.items():
        b = buffers[operand["name"]]
        if not b.get("data"):
            raise ValueError("Projection weights must be immutable")
        count = math.prod(b["shape"])
        if role.endswith("Scale") or role == "WS":
            if b["dtype"] != "f16" or count != n:
                raise ValueError("Invalid output-row scale contract")
        elif b["dtype"] != expected or b["shape"] != w["shape"]:
            raise ValueError("Fused projection operands differ in geometry")
    return Projection(
        kind, rows, n, k, {"f16": "float16", "f32": "float32"}[output_dtype], symbolic
    ), names


def expert_matrices(package, compiler):
    """Real K64 E8P geometries; route/up/down publication stays coupled."""
    matrices = {}
    for kernel in package["kernels"]:
        p = compiler.pointers(kernel)
        if not {"Low", "High", "P", "Book", "WS", "Counts", "Out"} <= p.keys():
            continue
        weight = compiler.buffers[p["P"]["name"]]
        experts, blocks, outputs, lanes = weight["shape"]
        if lanes != 8 or weight["dtype"] != "u16" or not weight.get("data"):
            raise ValueError("Unsupported compressed expert construction contract")
        key = (experts, outputs, blocks * 64)
        matrices.setdefault(key, {role: p[role] for role in ("P", "Book", "WS")})
    return matrices
