"""Find a closed shared-KV boundary without changing producer arithmetic."""

from copy import deepcopy


def source_boundary(packet, contract, describe):
    body = packet["body"]
    start = next(i for i, s in enumerate(body) if s.get("name") == "layer19") + 1
    end = next(i for i, s in enumerate(body) if s.get("name") == "layer20")
    writes = []
    for i in range(start, end):
        step = body[i]
        if step["op"] != "kernel" or "owner_work" not in step:
            continue
        spec = describe(step)
        columns = contract["pointers"][step["buffers"][1]]
        if any(
            a["write"] and contract["columns"][columns[a["column"]]]["buffer"].startswith("cache.")
            for a in spec["parameters"]["_owner_access"]
        ):
            writes.append(i)
    if not writes:
        raise ValueError("Producer layer has no persistent KV writes")
    boundary = max(writes) + 1
    depth = 0
    for step in body[start:boundary]:
        depth += int(step["op"] == "fork") - int(step["op"] == "join")
    if depth != 1 or body[start]["op"] != "fork":
        raise ValueError("Expected only the outer mHC branch at the shared-KV boundary")
    # mHC coefficients are independent of the main projection branch. Join
    # them before selection, then omit that same join from the attention tail.
    level, join = depth, None
    for i in range(boundary, end):
        level += int(body[i]["op"] == "fork") - int(body[i]["op"] == "join")
        if level == 0:
            join = i
            break
    if join is None:
        raise ValueError("Producer attention has no matching outer join")
    # Q-B and its RoPE output have no producer-side consumers. Delay both
    # until after selection; Q-A remains paired with the existing KV branch.
    query = []
    for i in range(start, boundary):
        step = body[i]
        if step["op"] != "kernel" or "owner_work" in step:
            continue
        spec = describe(step)
        outputs = [
            n
            for n, operand in zip(step["buffers"], spec["operands"], strict=True)
            if operand["access"] in ("write", "read_write")
        ]
        if any(n.endswith((".q.b", ".q.rope")) for n in outputs):
            query.append(i)
    if len(query) != 2:
        raise ValueError("Expected the query projection and RoPE pair before the KV boundary")
    delayed = {
        name
        for i in query
        for name, port in zip(body[i]["buffers"], describe(body[i])["operands"], strict=True)
        if port["access"] in ("write", "read_write")
    }
    for i in range(start, boundary):
        step = body[i]
        if i in query:
            continue
        if step["op"] == "kernel":
            reads = {
                name
                for name, port in zip(step["buffers"], describe(step)["operands"], strict=True)
                if port["access"] in ("read", "read_write")
            }
        else:
            reads = {step.get("source")}
        if delayed & reads:
            raise ValueError("A delayed query output is still required by the KV producer")
    prefix = deepcopy([s for i, s in enumerate(body[:boundary]) if i not in query])
    prefix += [
        dict(op="join"),
        dict(op="snapshot", name="source.kv.20", buffer=packet["coordinates"]["slots"]),
    ]
    tail = deepcopy(
        [body[i] for i in query] + [s for i, s in enumerate(body[boundary:], boundary) if i != join]
    )
    return prefix, tail


def live_rows(early):
    """Row tensors crossing the boundary; sequence-local tensors are explicit."""
    if not early:
        return [
            ("residual.a", 20480, "bf16", True),
            ("collapsed", 5120, "bf16", True),
            ("selected", 512, "i32", False),
        ]
    return [
        ("residual.a", 20480, "bf16", True),
        ("collapsed", 5120, "bf16", True),
        ("pre.b", 8, "f32", True),
        ("hc.post", 8, "f32", True),
        ("hc.combination", 16, "f32", True),
        ("norm.attn", 5120, "bf16", False),
        ("q.norm.quant", 1280, "i8", False),
        ("q.norm.quant.scale", 1, "f32", False),
    ]
