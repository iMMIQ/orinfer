"""Factor invocation slots into layer parameters and shared specialization tables.

Only compiler-owned binding names are interpreted here. Tensor roles, argument
order, scalar bits, offsets, grids and dynamic contracts are never inferred.
"""

import copy
import re


def key(value):
    from orinfer_build.contracts.catalog import canonical

    return canonical(value)


def factor(bindings):
    slots = {}
    for binding in bindings:
        base = re.split(r"/(?:rows|skip|choice)/", binding["name"], maxsplit=1)[0]
        slots.setdefault(base, []).append(binding)
    parameters, parameter_ids, choices, choice_ids, templates = [], {}, [], {}, {}

    def intern(value, values, ids):
        identity = key(value)
        if identity not in ids:
            ids[identity] = len(values)
            values.append(value)
        return ids[identity]

    for base, items in slots.items():
        values, value_ids, variants = [], {}, []
        for binding in sorted(items, key=lambda b: b["name"][len(base) :]):
            variant = {k: copy.deepcopy(v) for k, v in binding.items() if k not in ("name", "args")}
            variant["suffix"] = binding["name"][len(base) :]
            variant["arguments"] = [intern(a, values, value_ids) for a in binding["args"]]
            variants.append(variant)
        choice = intern(variants, choices, choice_ids)
        match = re.search(r"(?:^|/)layer(0|[1-9][0-9]*)(?=/|$)", base)
        prefix, suffix, layer = base, "", None
        if match:
            prefix, suffix = base[: match.start(1)], base[match.end(1) :]
            layer = int(match[1])
        identity = (prefix, suffix, layer is not None, choice)
        template = templates.setdefault(
            identity, dict(prefix=prefix, suffix=suffix, choice=choice, layers=[])
        )
        template["layers"].append(
            dict(layer=layer, parameters=intern(values, parameters, parameter_ids))
        )
    result = dict(parameters=parameters, choices=choices, templates=list(templates.values()))
    # Prove the generated representation itself, not just the earlier cubin catalog.
    before = {b["name"]: b for b in bindings}
    after = expand(result)
    if len(before) != len(bindings) or len(after) != len(before):
        raise ValueError("Duplicate binding in native templates")
    if {b["name"]: b for b in after} != before:
        raise ValueError("Layer template failed exact binding roundtrip")
    return result


def expand(registry):
    bindings = []
    for template in registry["templates"]:
        for instance in template["layers"]:
            layer = "" if instance["layer"] is None else str(instance["layer"])
            base = template["prefix"] + layer + template["suffix"]
            parameters = registry["parameters"][instance["parameters"]]
            for choice in registry["choices"][template["choice"]]:
                binding = {
                    k: copy.deepcopy(v)
                    for k, v in choice.items()
                    if k not in ("suffix", "arguments")
                }
                binding["name"] = base + choice["suffix"]
                binding["args"] = [copy.deepcopy(parameters[i]) for i in choice["arguments"]]
                bindings.append(binding)
    return bindings
