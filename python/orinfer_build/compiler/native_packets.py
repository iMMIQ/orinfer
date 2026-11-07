"""Inspect native C ABI packet plans on the CPU during offline publication."""

import ctypes as C
import json
from pathlib import Path


class Bytes(C.Structure):
    _fields_ = [("data", C.c_void_p), ("len", C.c_size_t)]

    def text(self):
        return C.string_at(self.data, self.len).decode() if self.len else ""


class Segment(C.Structure):
    _fields_ = [("slot", C.c_size_t), ("tokens", C.c_size_t)]


class View(C.Structure):
    _fields_ = [("name", Bytes), ("buffer", Bytes), ("offset", C.c_size_t)]


class Argument(C.Structure):
    _fields_ = [("index", C.c_size_t), ("value", C.c_int32)]


class Command(C.Structure):
    _fields_ = [
        ("kind", C.c_uint32),
        ("name", Bytes),
        ("destination", Bytes),
        ("bytes", C.c_size_t),
        ("sequence", C.c_size_t),
        ("has_launch", C.c_uint32),
        ("grid", C.c_uint32 * 3),
        ("arguments", C.POINTER(Argument)),
        ("argument_count", C.c_size_t),
        ("views", C.POINTER(View)),
        ("view_count", C.c_size_t),
    ]


class Binding(C.Structure):
    _fields_ = [("present", C.c_uint32), ("sequence", C.c_size_t), ("buffer", Bytes)]


class Output(C.Structure):
    _fields_ = [
        ("owner", C.c_void_p),
        ("commands", C.POINTER(Command)),
        ("command_count", C.c_size_t),
        ("bindings", C.POINTER(Binding)),
        ("binding_count", C.c_size_t),
    ]


class Api(C.Structure):
    _fields_ = [
        ("abi", C.c_uint32),
        ("size", C.c_size_t),
        ("describe", C.c_void_p),
        ("create", C.c_void_p),
        ("destroy", C.c_void_p),
        ("batch", C.c_void_p),
        ("free_bytes", C.c_void_p),
        ("free_batch", C.c_void_p),
    ]


class Packets:
    def __init__(self, model, *, indirect=True):
        from orinfer_build.packaging.publication import load_model

        model = Path(model).resolve(strict=True)
        data, origin, expanded = load_model(model)
        self.data = data
        package = json.loads((origin / "package.json").read_text())
        self.library = C.CDLL(str(origin / package["execution"]["library"]["file"]))
        entry = self.library.orinfer_model_v1
        entry.argtypes = [C.c_uint32]
        entry.restype = C.POINTER(Api)
        api = entry(1).contents
        if api.abi != 1 or api.size < C.sizeof(Api):
            raise ValueError("Incompatible native packet ABI")
        self.free_bytes = C.CFUNCTYPE(None, Bytes)(api.free_bytes)
        self.free_batch = C.CFUNCTYPE(None, Output)(api.free_batch)
        self.destroy = C.CFUNCTYPE(None, C.c_void_p)(api.destroy)
        self.batch = C.CFUNCTYPE(
            C.c_int32,
            C.c_void_p,
            C.POINTER(Segment),
            C.c_size_t,
            C.c_uint32,
            C.POINTER(Output),
            C.POINTER(Bytes),
        )(api.batch)
        create = C.CFUNCTYPE(
            C.c_int32, C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p), C.POINTER(Bytes)
        )(api.create)
        metadata = data["metadata"]
        if not indirect:
            metadata.pop("packet_table", None)
            if metadata.get("mtp"):
                metadata["mtp"].pop("packet_table", None)
                metadata["mtp"]["batch_programs"] = [
                    name
                    for name in metadata["mtp"].get("batch_programs", [])
                    if name != "mtp_packet_indirect"
                ]
        for name in ["batch_profiles", "prefill_batch_profiles", "greedy_sampling", "batch_gdn"]:
            if name in package:
                metadata[name] = package[name]
        request = dict(
            catalog=package["catalog"],
            model_root=str(model),
            verify_weights=False,
            config=json.loads((model / "config.json").read_text()),
            architecture=data["architecture"],
            compute_policy=data["compute_policy"],
            expected_signature=package["config_signature"],
            metadata=metadata,
            prefill_profiles=package["prefill_profiles"],
            build_bindings=True,
        )
        raw = json.dumps(request).encode()
        self.handle = C.c_void_p()
        output = Bytes()
        status = create(raw, len(raw), C.byref(self.handle), C.byref(output))
        try:
            text = output.text()
            if status:
                raise ValueError(text)
            self.manifest = json.loads(text)["metadata"]
        finally:
            self.free_bytes(output)
        self.expanded = expanded

    def close(self):
        if self.handle:
            self.destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def plan(self, segments, operation):
        segments = (Segment * len(segments))(*(Segment(*s) for s in segments))
        output, error = Output(), Bytes()
        status = self.batch(
            self.handle, segments, len(segments), operation | 1, C.byref(output), C.byref(error)
        )
        try:
            if status:
                raise ValueError(error.text())
            commands = []
            for item in output.commands[: output.command_count]:
                command = dict(
                    kind=item.kind,
                    name=item.name.text(),
                    destination=item.destination.text(),
                    bytes=item.bytes,
                    sequence=None if item.sequence == C.c_size_t(-1).value else item.sequence,
                    views={
                        v.name.text(): dict(buffer=v.buffer.text(), offset=v.offset)
                        for v in item.views[: item.view_count]
                    },
                )
                if item.has_launch:
                    command["launch"] = dict(
                        grid=list(item.grid),
                        arguments=[
                            (a.index, a.value) for a in item.arguments[: item.argument_count]
                        ],
                    )
                commands.append(command)
            bindings = [
                (b.sequence, b.buffer.text()) if b.present else None
                for b in output.bindings[: output.binding_count]
            ]
            return commands, bindings
        finally:
            self.free_bytes(error)
            self.free_batch(output)


def profiles(manifest):
    """Representative role boundaries, lengths and reversed request ownership."""
    yield 0, [(2, 1), (7, 1)]
    for tokens in [3, 17, 32, 128, 512]:
        yield 0, [(2, tokens), (7, 1)]
        yield 0, [(2, tokens), (7, tokens)]
    if not manifest.get("mtp"):
        return
    for tokens in [1, 2, 3, 8]:
        for op in [2, 4, 6, 8, 10]:
            yield op, [(2, tokens), (7, tokens)]
    for tokens in [2, 3, 8]:
        yield 12, [(2, tokens), (7, 1)]
        yield 12 | 256, [(2, tokens), (7, 2)]
