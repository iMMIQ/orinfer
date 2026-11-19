"""Factor invocation slots into layer parameters and shared specialization tables.

Only compiler-owned binding names are interpreted here. Tensor roles, argument
order, scalar bits, offsets, grids and dynamic contracts are never inferred.
"""

import copy
import re
import struct


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
    roles, role_ids = [], {}
    tagged = [
        [encode_argument(a, lambda name: intern(name, roles, role_ids)) for a in group]
        for group in parameters
    ]
    result = dict(
        roles=roles, parameters=tagged, choices=choices, templates=list(templates.values())
    )
    # Prove the generated representation itself, not just the earlier cubin catalog.
    before = {b["name"]: b for b in bindings}
    after = expand(result)
    if len(before) != len(bindings) or len(after) != len(before):
        raise ValueError("Duplicate binding in native templates")

    def encoded(bindings):
        return {
            name: dict(
                binding, args=[encode_argument(a, lambda name: name) for a in binding["args"]]
            )
            for name, binding in bindings.items()
        }

    if encoded({b["name"]: b for b in after}) != encoded(before):
        raise ValueError("Layer template failed exact binding roundtrip")
    return result


def encode_argument(arg, role):
    kind = arg["kind"]
    if kind == "buffer":
        value = role(arg["name"])
    elif kind == "buffer_slice":
        value = [role(arg["name"]), arg["offset"]]
    elif kind == "f32":
        value = struct.unpack("<I", struct.pack("<f", arg["value"]))[0]
    elif kind in ("i32", "u32", "i64", "u64"):
        value = arg["value"]
    else:
        raise ValueError("Unknown binding argument type")
    return {kind: value}


def lookup(items, index):
    if type(index) is not int or not 0 <= index < len(items):
        raise ValueError("Invalid binding table index")
    return items[index]


def decode_argument(arg, roles):
    ((kind, value),) = arg.items()
    if kind == "buffer":
        return dict(kind=kind, name=lookup(roles, value))
    if kind == "buffer_slice":
        role, offset = value
        if type(offset) is not int or not 0 <= offset < 2**64:
            raise ValueError("Invalid binding slice offset")
        return dict(kind=kind, name=lookup(roles, role), offset=offset)
    if kind == "f32":
        value = struct.unpack("<f", struct.pack("<I", value))[0]
    elif kind in ("i32", "u32", "i64", "u64"):
        signed = kind.startswith("i")
        bits = int(kind[1:]) - signed
        if type(value) is not int or not (-(2**bits) if signed else 0) <= value < 2**bits:
            raise ValueError("Invalid binding scalar")
    else:
        raise ValueError("Unknown binding argument type")
    return dict(kind=kind, value=value)


def expand(registry):
    bindings = []
    if set(registry) != {"roles", "parameters", "choices", "templates"}:
        raise ValueError("Invalid binding registry fields")
    parameters = [
        [decode_argument(a, registry["roles"]) for a in group] for group in registry["parameters"]
    ]
    seen = set()
    for template in registry["templates"]:
        choices = lookup(registry["choices"], template["choice"])
        if not choices or not template["layers"]:
            raise ValueError("Empty binding template")
        for instance in template["layers"]:
            layer = "" if instance["layer"] is None else str(instance["layer"])
            base = template["prefix"] + layer + template["suffix"]
            arguments = lookup(parameters, instance["parameters"])
            for choice in choices:
                binding = {
                    k: copy.deepcopy(v)
                    for k, v in choice.items()
                    if k not in ("suffix", "arguments")
                }
                binding["name"] = base + choice["suffix"]
                if not binding["name"] or binding["name"] in seen:
                    raise ValueError("Empty or duplicate binding name")
                seen.add(binding["name"])
                binding["args"] = [copy.deepcopy(lookup(arguments, i)) for i in choice["arguments"]]
                bindings.append(binding)
    if not bindings:
        raise ValueError("Empty binding registry")
    return bindings
