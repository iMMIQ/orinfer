"""Canonicalize dynamic launch arithmetic and adjacent measured selections."""

import copy


def divide_rows(value, divisor):
    if isinstance(value, dict):
        if value == {"op": "rows"}:
            # Valid images have multiples of four patches; ceil keeps generic
            # validation-only short rows legal without changing image launches.
            return dict(
                op="divide",
                lhs=dict(op="add", lhs=value, rhs=dict(op="constant", value=divisor - 1)),
                rhs=dict(op="constant", value=divisor),
            )
        return {k: divide_rows(v, divisor) for k, v in value.items()}
    if isinstance(value, list):
        return [divide_rows(v, divisor) for v in value]
    return value


def coalesce_adjacent_selections(package):
    """Merge identical launch contracts even in a multi-implementation portfolio.

    Exact AOT implementations remain separate when any argument, grid or
    symbolic expression differs. Coupled expert route tiles are never inferred.
    """
    from collections import defaultdict

    launches = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    slots = defaultdict(list)
    for k in package["kernels"]:
        if "/rows/" in k["name"] and k["name"] in launches:
            slot, extent = k["name"].split("/rows/")
            first, last = map(int, extent.split("-"))
            slots[slot].append((first, last, k))
    removed, added, dynamic = set(), [], []

    def identity(k):
        return [{a: b for a, b in item.items() if a != "name"} for item in (k, launches[k["name"]])]

    for slot, items in slots.items():
        spans = []
        for first, last, k in sorted(items, key=lambda x: x[0]):
            if spans and first <= spans[-1][1]:
                raise ValueError("Overlapping row selections")
            if spans and first == spans[-1][1] + 1 and identity(k) == identity(spans[-1][2][0]):
                spans[-1][1] = last
                spans[-1][2].append(k)
            else:
                spans.append([first, last, [k]])
        for first, last, ks in spans:
            if len(ks) == 1:
                continue
            name = f"{slot}/rows/{first}-{last}"
            item, launch = copy.deepcopy(ks[0]), copy.deepcopy(launches[ks[0]["name"]])
            item["name"] = launch["name"] = name
            removed.update(k["name"] for k in ks)
            added.append(item)
            dynamic.append(launch)
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + dynamic
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in added}
    )
    return len(removed) - len(added)
