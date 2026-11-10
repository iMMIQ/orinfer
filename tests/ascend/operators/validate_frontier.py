"""Validate device continuation across acceptance, remapping and changed Graph inputs."""

import argparse
import ctypes
import json
from pathlib import Path

from orinfer_build.common.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(7)
    model = json.loads(args.plan.read_text())
    c = model["device_inputs"]
    owners, cap = model["batch"]["owners"], c["capacity"]
    outputs = model["batch"]["output_capacity"]
    libraries = []
    checks = []

    def bind(name, tensors):
        path = args.plan.parent / name
        spec = json.loads(path.read_text())
        lib = ctypes.CDLL(str((path.parent / spec["library"]["file"]).resolve()))
        libraries.append(lib)
        entry = lib.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(tensors))(*[x.data_ptr() for x in tensors])

        def call(rows):
            if entry(pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_):
                raise RuntimeError(f"Device continuation launch failed: {spec['op']}")

        return call

    def integer(*shape):
        return torch.zeros(shape, dtype=torch.int32, device="npu")

    state, proposals = integer(owners, 4), integer(owners, 16)
    mapping, ids, positions = integer(cap, 8), integer(cap), integer(cap)
    prep = c["preparations"][f"{cap}:0:1"]
    token_call = bind(prep["tokens"], [mapping, state, proposals, ids, positions])
    host_state = torch.zeros((owners, 4), dtype=torch.int32)
    host_state[:, 0] = torch.arange(owners) + 1234
    host_state[:, 1] = torch.arange(owners) * 777 + 100257
    host_state[:, 3] = 3
    host_proposals = torch.arange(owners * 16, dtype=torch.int32).reshape(owners, 16) + 3000
    state.copy_(host_state)
    proposals.copy_(host_proposals)
    for rows in [1, 17, 32, 513, cap]:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            token_call(rows)
        for phase in range(3):
            m = torch.zeros((rows, 8), dtype=torch.int32)
            m[:, 0] = (torch.arange(rows) * 7 + phase) % owners
            m[:, 1] = torch.arange(rows) % 16
            m[:, 2] = 16
            m[:, 3] = torch.arange(rows) % 4
            m[:, 4] = 2049
            m[:, 5] = 987
            m[:, 6] = 3 + (phase == 2)  # recycled owners cannot reuse old state
            mapping[:rows].copy_(m)
            ids.fill_(-777)
            graph.replay()
            torch.npu.synchronize()
            expected, p = m[:, 5].clone(), m[:, 4] + m[:, 1]
            for i in range(rows):
                owner, offset, _, mode, _, _, epoch, _ = m[i].tolist()
                if mode and epoch == host_state[owner, 3]:
                    expected[i] = (
                        host_proposals[owner, offset - 1]
                        if mode == 2 and offset
                        else 128799
                        if mode == 3 and offset
                        else host_state[owner, 0]
                    )
                    p[i] = host_state[owner, 1] + offset
            assert torch.equal(ids[:rows].cpu(), expected)
            assert torch.equal(positions[:rows].cpu(), p)
            assert bool((ids[rows:].cpu() == -777).all())
        checks.append(dict(op="tokens", rows=rows, changed_graph=True))

    # Pack rank-major samples with the actual row count, including ties across
    # shards. The output must agree with the host's global token tie rule.
    width = 5120
    gathered = torch.empty((8 * outputs * 2,), device="npu")
    selected = torch.zeros((outputs, 2), device="npu")
    merge = bind(c["merge"], [gathered, selected])
    for rows in [1, 17, outputs]:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            merge(rows)
        for phase in range(2):
            pairs = torch.randn(8, rows, 2)
            pairs[:, :, 1] = torch.randint(0, 16160, (8, rows)).float()
            pairs[7, ::2, 0] = 123
            pairs[3, ::2, 0] = 123
            pairs[:, :, 1] = (pairs[:, :, 1] + phase) % 16160
            gathered[: 8 * rows * 2].copy_(pairs.flatten())
            graph.replay()
            torch.npu.synchronize()
            global_ids = pairs[:, :, 1].long() + torch.arange(8)[:, None] * 16160
            maxima = pairs[:, :, 0].max(dim=0).values
            expected_ids = (
                torch.where(pairs[:, :, 0] == maxima[None], global_ids, 1 << 30).min(dim=0).values
            )
            actual = selected[:rows].cpu()
            assert torch.equal(actual[:, 0], maxima)
            assert torch.equal(actual[:, 1].long(), expected_ids)
        checks.append(dict(op="merge", rows=rows, changed_graph=True))

    owner_map = integer(owners, 8)
    history = integer(owners, 4)
    host_history = torch.tensor([[121, 122, 123, 9]] * owners, dtype=torch.int32)
    history.copy_(host_history)
    cache = torch.zeros((owners, width), dtype=torch.bfloat16, device="npu")
    local_table = torch.zeros((16160, width), dtype=torch.bfloat16, device="npu")
    compact = torch.zeros_like(cache)
    reduced = torch.zeros_like(cache)
    commit = bind(
        c["commit"],
        [owner_map, selected, positions, proposals, state, ids, history, local_table, compact],
    )
    scatter = bind(c["scatter_embedding"], [owner_map, reduced, cache])
    # Commit and next-input construction execute in one Graph, without a host
    # token/length round trip. Acceptance varies from first rejection to full.
    count = min(owners, 17)
    om = torch.zeros((count, 8), dtype=torch.int32)
    next_map = torch.zeros((count, 8), dtype=torch.int32)
    at = 0
    x = torch.zeros(outputs, 2)
    p = torch.zeros(cap, dtype=torch.int32)
    expected_state = host_state.clone()
    expected_history = host_history.clone()
    input_ids = torch.zeros(cap, dtype=torch.int32)
    for lane in range(count):
        owner = count - 1 - lane
        depth = [0, 1, 5, 16][lane % 4]
        accepted = [0, 0, 3, 16][lane % 4]
        om[lane] = torch.tensor([owner, at, depth + 1, 0, at, at, 9, 0])
        p[at : at + depth + 1] = 262000 + lane + torch.arange(depth + 1)
        input_ids[at] = 1111 + owner
        input_ids[at + 1 : at + depth + 1] = host_proposals[owner, :depth]
        x[at : at + depth, 1] = host_proposals[owner, :depth].float()
        x[at + accepted, 1] = 7000 + lane + lane % 8 * 16160
        next_map[lane] = torch.tensor([owner, 0, 1, 1, 0, 0, 9, 0])
        expected_state[owner] = torch.tensor(
            [7000 + lane + lane % 8 * 16160, p[at] + accepted + 1, accepted, 9]
        )
        committed_history = (
            host_history[owner, :3].flip(0).tolist() + input_ids[at : at + accepted + 1].tolist()
        )
        expected_history[owner, :3] = torch.tensor(committed_history[-3:][::-1])
        at += depth + 1
    owner_map[:count].copy_(om)
    mapping[:count].copy_(next_map)
    selected.copy_(x)
    positions.copy_(p)
    ids.copy_(input_ids)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        commit(count)
        token_call(count)
    expected_cache = torch.zeros_like(cache, device="cpu")
    for rank in range(8):
        om[:, 3] = rank
        owner_map[:count].copy_(om)
        positions.copy_(p)
        ids.copy_(input_ids)
        history.copy_(host_history)
        state.copy_(host_state)
        for lane in range(count):
            value = (
                ((7000 + lane + torch.arange(width) * 3 + rank * 7) % 257 - 128).float() / 8
            ).to(torch.bfloat16)
            local_table[7000 + lane].copy_(value)
            if lane % 8 == rank:
                expected_cache[count - 1 - lane] = value
        graph.replay()
        torch.npu.synchronize()
        local = compact[:count].cpu()
        for lane in range(count):
            expected = (
                expected_cache[count - 1 - lane]
                if lane % 8 == rank
                else torch.zeros(width, dtype=torch.bfloat16)
            )
            assert torch.equal(local[lane], expected)
        reduced[:count].add_(compact[:count])
    assert torch.equal(state.cpu(), expected_state)
    assert torch.equal(history.cpu(), expected_history)
    assert torch.equal(ids[:count].cpu(), expected_state[next_map[:, 0].long(), 0])
    scatter(count)
    torch.npu.synchronize()
    assert torch.equal(cache.cpu(), expected_cache)
    checks.append(
        dict(
            op="commit_to_next_input",
            owners=count,
            varied_acceptance=True,
            sparse_rank_reduction=True,
        )
    )

    draft_spec = json.loads((args.plan.parent / c["draft_embedding"]).read_text())
    draft_cap = draft_spec["operands"][0]["shape"][0]
    draft_map = integer(draft_cap, 8)
    mask = torch.arange(width, dtype=torch.float32).to(torch.bfloat16).npu()
    draft_out = torch.empty((draft_cap, width), dtype=torch.bfloat16, device="npu")
    cached = bind(c["draft_embedding"], [draft_map, cache, mask, draft_out])
    for rows in [1, 17, min(draft_cap, count * 17)]:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            cached(rows)
        for phase in range(2):
            m = torch.zeros((rows, 8), dtype=torch.int32)
            m[:, 0] = (torch.arange(rows) + phase) % count
            m[:, 1] = torch.arange(rows) % 17
            draft_map[:rows].copy_(m)
            draft_out.fill_(-777)
            graph.replay()
            torch.npu.synchronize()
            expected = mask.cpu()[None, :].repeat(rows, 1)
            anchors = m[:, 1] == 0
            expected[anchors] = cache.cpu()[m[anchors, 0].long()]
            assert torch.equal(draft_out[:rows].cpu(), expected)
            assert bool((draft_out[rows:].cpu() == -777).all())
        checks.append(dict(op="draft_embedding", rows=rows, changed_graph=True))

    # RoPE and recurrent metadata must use device positions at context edges.
    context = model["capacity"]
    rope_cpu = (
        (
            torch.arange(context).remainder(61).reshape(1, context, 1)
            + torch.arange(4).reshape(4, 1, 1) * 64
        )
        .expand(4, context, 64)
        .bfloat16()
        .contiguous()
    )
    rope = rope_cpu.npu()
    trig = [torch.empty(cap, 64, dtype=torch.bfloat16, device="npu") for _ in range(6)]
    slots, c2_slots, swa, source_swa = (
        integer(cap),
        integer(cap),
        integer(cap, 128),
        integer(cap, 128),
    )
    ring, mix = integer(cap, 8), torch.empty(cap, 8, device="npu")
    coords = bind(
        prep["coordinates"],
        [mapping, positions, rope, *trig, slots, c2_slots, swa, source_swa, ring, mix],
    )
    rows = 17
    m = torch.zeros(rows, 8, dtype=torch.int32)
    m[:, 1] = torch.arange(rows) % 4
    m[:, 2] = 4
    mapping[:rows].copy_(m)
    coord_graph = torch.npu.NPUGraph()
    with torch.npu.graph(coord_graph):
        coords(rows)
    for phase in range(2):
        p = torch.tensor(
            ([0, 127, 128, 2048, 2049, 100257, context - 1] * 3)[:rows], dtype=torch.int32
        )
        if phase:
            p = p.flip(0)
        positions[:rows].copy_(p)
        coord_graph.replay()
        torch.npu.synchronize()
        assert torch.equal(slots[:rows].cpu(), p)
        assert torch.equal(c2_slots[:rows].cpu(), torch.where(p % 2 == 1, p // 2, -1))
        for plane, actual in enumerate(trig):
            index = p if plane < 4 else p // 2 * 2
            table = plane if plane < 4 else plane - 2
            assert torch.equal(actual[:rows].cpu(), rope_cpu[table, index.long()])
        for i, position in enumerate(p.tolist()):
            expected = torch.full((128,), -1, dtype=torch.int32)
            past = torch.arange(max(0, position - 127), position + 1, dtype=torch.int32)
            expected[: len(past)] = past
            assert torch.equal(swa[i].cpu(), expected)
            assert torch.equal(source_swa[i].cpu(), expected)
        expected_ring = torch.zeros(rows, 8, dtype=torch.int32)
        expected_ring[:, 0] = p
        expected_ring[:, 1] = expected_ring[:, 3] = expected_ring[:, 5] = 1
        expected_ring[:, 2] = torch.where(m[:, 1] == 0, -1, m[:, 1] - 1)
        expected_ring[:, 4] = p - m[:, 1] + m[:, 2] - 1
        assert torch.equal(ring[:rows].cpu(), expected_ring)
        expected_mix = torch.zeros(rows, 8)
        expected_mix[:, 0] = 1
        assert torch.equal(mix[:rows].cpu(), expected_mix)
    checks.append(dict(op="coordinates", rows=rows, context_edges=True, changed_graph=True))

    layout = json.loads((args.plan.parent / model["engram"]["file"]).read_text())
    token_map = torch.tensor(layout["token_map"], dtype=torch.int32, device="npu")
    # Positive odd multipliers deliberately exercise signed 64-bit wrapping.
    multipliers_cpu = torch.tensor([l["multipliers"] for l in layout["layers"]], dtype=torch.int64)
    multipliers_cpu[0, 0] = 0x7000000000000001
    multipliers = multipliers_cpu.npu()
    bucket_cpu = []
    for layer in layout["layers"]:
        offset, buckets = 0, []
        for prime in layer["primes"]:
            buckets.append([prime, offset])
            offset += prime
        bucket_cpu.append(buckets)
    buckets = torch.tensor(bucket_cpu, dtype=torch.int64, device="npu")
    hashes = torch.empty(cap, c["hash_width"], dtype=torch.int64, device="npu")
    valid = integer(cap)
    hash_call = bind(
        c["hashes"], [mapping, ids, history, token_map, multipliers, buckets, hashes, valid]
    )
    histories = host_history.clone()
    histories[0, :3] = torch.tensor([-1, -1, -1])
    histories[1, 0] = layout["barriers"][0]
    history.copy_(histories)
    for rows in [1, 17, 513, cap]:
        m = torch.zeros(rows, 8, dtype=torch.int32)
        m[:, 0] = (torch.arange(rows) // 17) % owners
        m[:, 1] = torch.arange(rows) % 17
        mapping[:rows].copy_(m)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            hash_call(rows)
        for phase in range(2):
            raw = torch.randint(0, 129280, (rows,), dtype=torch.int32)
            raw[::11] = layout["barriers"][phase]
            ids[:rows].copy_(raw)
            graph.replay()
            torch.npu.synchronize()
            expected = torch.zeros(rows, c["hash_width"], dtype=torch.int64)
            expected_valid = torch.ones(rows, dtype=torch.int32)
            for row in range(rows):
                if raw[row] in layout["barriers"]:
                    expected_valid[row] = 0
                    continue
                window = [layout["token_map"][layout["pad_token"]]] * 4
                for shift in range(4):
                    index = row - shift
                    raw_token = (
                        int(raw[index])
                        if shift <= m[row, 1]
                        else int(histories[m[row, 0], shift - m[row, 1] - 1])
                    )
                    if raw_token < 0 or raw_token in layout["barriers"]:
                        break
                    window[shift] = layout["token_map"][raw_token]
                for layer_index, layer in enumerate(layout["layers"]):
                    rolling = (window[0] * int(multipliers_cpu[layer_index, 0])) & ((1 << 64) - 1)
                    for order in range(1, 4):
                        rolling ^= (window[order] * int(multipliers_cpu[layer_index, order])) & (
                            (1 << 64) - 1
                        )
                        signed = rolling - (1 << 64) if rolling >= 1 << 63 else rolling
                        for head in range(layout["heads"]):
                            bucket = (order - 1) * layout["heads"] + head
                            prime, offset = bucket_cpu[layer_index][bucket]
                            expected[row, layer_index * 24 + bucket] = signed % prime + offset
            assert torch.equal(valid[:rows].cpu(), expected_valid)
            actual = hashes[:rows].cpu()
            assert torch.equal(actual, expected), (rows, phase, (actual != expected).nonzero()[:5])
        checks.append(
            dict(
                op="hashes",
                rows=rows,
                signed_wrapping=True,
                media_barriers=True,
                changed_graph=True,
            )
        )

    draft_indices = integer(owners * 16)
    save = bind(c["save_draft"], [owner_map, draft_indices, proposals])
    order = torch.arange(owners - 1, -1, -1)
    om = torch.zeros(owners, 8, dtype=torch.int32)
    om[:, 0] = order
    om[:, 2] = 16
    owner_map.copy_(om)
    packed = torch.arange(owners * 16, dtype=torch.int32) + 100
    draft_indices.copy_(packed)
    save(owners)
    torch.npu.synchronize()
    assert torch.equal(proposals.cpu()[order], packed.reshape(owners, 16))
    checks.append(dict(op="save_draft", reversed_owners=True))

    for packet in model["ranks"][0]["packets"]:
        if packet.get("role") != "draft":
            continue
        names = packet["coordinates"]
        block = names["draft_block"]
        draft_prep = c["preparations"][f"{packet['rows']}:{block}:0"]
        rows = 17 * block
        m = torch.zeros(rows, 8, dtype=torch.int32)
        m[:, 1] = torch.arange(rows) % block
        m[:, 2] = block
        mapping[:rows].copy_(m)
        coordinate = bind(
            draft_prep["coordinates"],
            [mapping, positions, rope, *trig, slots, c2_slots, swa, source_swa, ring, mix],
        )
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            coordinate(rows)
        for phase in range(2):
            base = 100257 + (torch.arange(rows) // block) * 17 + phase
            positions[:rows].copy_(base + m[:, 1])
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(c2_slots[:rows].cpu(), m[:, 1])
            expected = (base[:, None] - 128 + torch.arange(128)[None]) % names.get(
                "draft_swa_storage_rows", 128
            )
            assert torch.equal(swa[:rows].cpu(), expected.int())
        checks.append(dict(op="draft_coordinates", block=block, changed_graph=True))

    for key, name in c["embeddings"].items():
        capacity, width = map(int, key.split(":"))
        table = torch.empty(129280, width, dtype=torch.bfloat16, device="npu")
        out = torch.empty(capacity, width, dtype=torch.bfloat16, device="npu")
        embed = bind(name, [table, ids, out])
        chosen = torch.tensor([0, 129279, 128799, 2049, 1], dtype=torch.int32)
        values = torch.randn(len(chosen), width).bfloat16()
        for i, token in enumerate(chosen.tolist()):
            table[token].copy_(values[i])
        rows = min(capacity, 17)
        ids[:rows].copy_(chosen[torch.arange(rows) % len(chosen)])
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            embed(rows)
        for phase in range(2):
            index = (torch.arange(rows) + phase) % len(chosen)
            ids[:rows].copy_(chosen[index])
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(out[:rows].cpu(), values[index])
        checks.append(dict(op="embedding", capacity=capacity, width=width))
        del table, out
    atomic_json(args.output, dict(checks=checks, passed=True))
    print(json.dumps(dict(passed=True, checks=len(checks))))


if __name__ == "__main__":
    main()
