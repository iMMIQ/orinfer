"""Structural interning for final JSON execution plans; no inference arithmetic."""

import argparse
import json
from pathlib import Path

ENCODING = "shared-json-v1"


def encode(value):
    nodes, identities = [], {}

    def intern(value):
        if isinstance(value, dict):
            node = {key: intern(child) for key, child in value.items()}
            identity = ("object", tuple(node.items()))
        elif isinstance(value, list):
            node = [intern(child) for child in value]
            identity = ("array", tuple(node))
        else:
            node = value
            identity = (type(value).__name__, value)
        if identity not in identities:
            identities[identity] = len(nodes)
            nodes.append(node)
        return identities[identity]

    root = intern(value)
    return dict(encoding=ENCODING, nodes=nodes, root=root)


def decode(document):
    if not isinstance(document, dict) or document.get("encoding") != ENCODING:
        return document
    if set(document) != {"encoding", "nodes", "root"}:
        raise ValueError("Invalid shared JSON document")
    nodes, root = document["nodes"], document["root"]
    if not isinstance(nodes, list) or not 0 < len(nodes) <= 2000000:
        raise ValueError("Invalid shared JSON node count")
    if type(root) is not int or not 0 <= root < len(nodes):
        raise ValueError("Invalid shared JSON root")
    sizes, depths = [], []
    for index, node in enumerate(nodes):
        refs = (
            list(node.values())
            if isinstance(node, dict)
            else node
            if isinstance(node, list)
            else []
        )
        if any(type(ref) is not int or not 0 <= ref < index for ref in refs):
            raise ValueError("Shared JSON references must precede their parents")
        size = sum(sizes[ref] + 1 for ref in refs) + (
            sum(6 * len(key.encode()) + 4 for key in node) if isinstance(node, dict) else 2
        )
        if not isinstance(node, (dict, list)):
            size = len(json.dumps(node, ensure_ascii=False).encode())
        depth = 1 + max((depths[ref] for ref in refs), default=0)
        if size > 128 * 1024 * 1024 or depth > 128:
            raise ValueError("Shared JSON expansion exceeds its bounds")
        sizes.append(size)
        depths.append(depth)

    def expand(index):
        node = nodes[index]
        if isinstance(node, dict):
            return {key: expand(ref) for key, ref in node.items()}
        if isinstance(node, list):
            return [expand(ref) for ref in node]
        return node

    return expand(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expand", action="store_true")
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    result = decode(value) if args.expand else encode(value)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
