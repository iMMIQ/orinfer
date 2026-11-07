"""Build one bounded dynamic ViT/aligner program from shared A2 operator factories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.models.deepseek_v41.vision import certify_values, tensor_shapes


def program(
    config,
    describe,
    *,
    patches=9216,
    features=1024,
    attention_m=128,
    attention_k=256,
    workspace_slots=64,
    gemm_n=256,
):
    v = config["vision_config"]
    hidden, intermediate = v["hidden_size"], v["intermediate_size"]
    output = config["text_config"]["hidden_size"]
    heads = v["num_attention_heads"]
    if hidden != 1024 or hidden // heads != 64 or v["downsample_ratio"] != 3:
        raise ValueError("Require the published DeepSeek ViT geometry")
    stride = (3 * v["patch_size"] ** 2 + 127) // 128 * 128
    buffers, body, segment_ends = {}, [], []

    def buffer(name, shape, dtype, **extra):
        size = 2 if dtype in ("BF16", "F16") else 4
        for dim in shape:
            size *= dim
        item = dict(name=name, bytes=size, **extra)
        if name in buffers and buffers[name] != item:
            raise ValueError("Conflicting vision buffer: " + name)
        buffers[name] = item
        return name

    def weight(name, shape):
        return buffer(
            name, shape, "BF16", prepared_tensor=dict(name=name, dtype="BF16", shape=shape)
        )

    for name, shape in tensor_shapes(config).items():
        if name == "vision.patch_embed.proj.weight":
            shape = [hidden, stride]
        weight(name, shape)
    buffer("vision.input", [patches, stride], "BF16")
    buffer("vision.phase", [patches, 64], "F32")
    buffer("vision.grid", [8], "I32")
    for name in ["h0", "h1", "norm", "attention"]:
        buffer("vision." + name, [patches, hidden], "BF16")
    buffer("vision.gated", [patches, intermediate], "BF16")
    buffer("vision.qkv_storage", [3, patches, hidden], "BF16")
    buffer(
        "vision.qkv", [2, patches, hidden], "BF16", view=dict(parent="vision.qkv_storage", offset=0)
    )
    buffer(
        "vision.values",
        [patches, hidden],
        "F16",
        view=dict(parent="vision.qkv_storage", offset=2 * patches * hidden * 2),
    )
    projection_width = max(2 * intermediate, 3 * hidden, output)
    buffer("vision.projection", [patches, projection_width], "F32")
    for width in {hidden, 3 * hidden, 2 * intermediate}:
        buffer(
            f"vision.projection.{width}",
            [patches, width],
            "F32",
            view=dict(parent="vision.projection", offset=0),
        )
    buffer(
        "vision.skip.gated",
        [patches, intermediate],
        "BF16",
        view=dict(parent="vision.qkv_storage", offset=0),
    )
    buffer(
        "vision.skip.merged", [features, output], "BF16", view=dict(parent="vision.h1", offset=0)
    )
    buffer(
        "vision.projection.merged",
        [features, output],
        "F32",
        view=dict(parent="vision.projection", offset=0),
    )
    buffer("vision.zero_bias", [projection_width], "BF16")
    for width in {hidden, 2 * intermediate, output}:
        buffer(
            f"vision.zero_bias.{width}",
            [width],
            "BF16",
            view=dict(parent="vision.zero_bias", offset=0),
        )
    buffer("vision.merged", [features, 9 * hidden], "BF16")
    buffer("vision.output", [features, output], "BF16")
    blocks = workspace_slots
    for name, shape, dtype in [
        ("vision.scores", [blocks * 2, attention_m, attention_k], "F32"),
        ("vision.probabilities", [blocks, attention_m, attention_k], "F16"),
        ("vision.products", [workspace_slots, attention_m, 64], "F32"),
    ]:
        buffer(name, shape, dtype)

    def op(module, name, arguments, ports, rows):
        capacity = patches if rows == "patches" else features
        specs = [dict(name=p[0], dtype=p[2], shape=p[3], access=p[4]) for p in ports]
        manifest = describe(module, name, dict(rows=capacity, **arguments), specs, capacity)
        body.append(dict(manifest=manifest, buffers=[p[1] for p in ports], rows=rows))

    def matmul(source, name, n, k, rows="patches"):
        capacity = patches if rows == "patches" else features
        destination = f"vision.projection.{n}" if rows == "patches" else "vision.projection.merged"
        op(
            "floating",
            "partials",
            dict(
                output=n,
                reduction=k,
                input_parts=1,
                weight_parts=1,
                splits=1,
                tile_m=128,
                tile_n=gemm_n,
                block_k=128,
                pipeline_stages=3,
            ),
            [
                ("activation", source, "bf16", [1, capacity, k], "read"),
                ("weight", name, "bf16", [1, n, k], "read"),
                ("out", destination, "f32", [capacity, n], "write"),
            ],
            rows,
        )
        return destination

    def epilogue(
        source,
        bias,
        skip,
        destination,
        width,
        *,
        activation="linear",
        residual=False,
        rows="patches",
    ):
        capacity = patches if rows == "patches" else features
        columns = width * (2 if activation == "silu_mul" else 1)
        op(
            "vision",
            "epilogue",
            dict(
                width=width,
                activation=activation,
                biased=bias is not None,
                residual=residual,
                tile=width,
            ),
            [
                ("source", source, "f32", [capacity, columns], "read"),
                ("bias", bias or f"vision.zero_bias.{columns}", "bf16", [columns], "read"),
                ("skip", skip, "bf16", [capacity, width], "read"),
                ("output", destination, "bf16", [capacity, width], "write"),
            ],
            rows,
        )

    def norm(source, name):
        op(
            "vector",
            "rms_norm",
            dict(width=hidden, eps=1e-6),
            [
                ("x", source, "bf16", [patches, hidden], "read"),
                ("weight", name, "bf16", [hidden], "read"),
                ("out", "vision.norm", "bf16", [patches, hidden], "write"),
            ],
            "patches",
        )

    x = matmul("vision.input", "vision.patch_embed.proj.weight", hidden, stride)
    epilogue(x, "vision.patch_embed.proj.bias", "vision.h1", "vision.h0", hidden)
    for layer in range(v["num_hidden_layers"]):
        prefix = f"vision.blocks.{layer}"
        norm("vision.h0", prefix + ".norm1.weight")
        x = matmul("vision.norm", prefix + ".attn.wqkv.weight", 3 * hidden, hidden)
        op(
            "vision",
            "qkv_rope",
            dict(
                capacity=patches,
                heads=heads,
                dimension=64,
                padding_multiple=attention_k,
            ),
            [
                ("source", x, "f32", [patches, 3 * hidden], "read"),
                ("bias", prefix + ".attn.wqkv.bias", "bf16", [3 * hidden], "read"),
                ("phase", "vision.phase", "f32", [patches, 64], "read"),
                ("qkv", "vision.qkv", "bf16", [2, patches, hidden], "write"),
                ("values", "vision.values", "f16", [patches, hidden], "write"),
            ],
            "patches",
        )
        op(
            "vision",
            "attention",
            dict(
                capacity=patches,
                heads=heads,
                dimension=64,
                tile_m=attention_m,
                tile_k=attention_k,
                workspace_slots=workspace_slots,
            ),
            [
                ("qkv", "vision.qkv", "bf16", [2, patches, hidden], "read"),
                ("values", "vision.values", "f16", [patches, hidden], "read"),
                ("output", "vision.attention", "bf16", [patches, hidden], "write"),
                (
                    "scores",
                    "vision.scores",
                    "f32",
                    [blocks * 2, attention_m, attention_k],
                    "scratch",
                ),
                (
                    "probabilities",
                    "vision.probabilities",
                    "f16",
                    [blocks, attention_m, attention_k],
                    "scratch",
                ),
                (
                    "products",
                    "vision.products",
                    "f32",
                    [workspace_slots, attention_m, 64],
                    "scratch",
                ),
            ],
            "patches",
        )
        x = matmul("vision.attention", prefix + ".attn.wo.weight", hidden, hidden)
        epilogue(x, prefix + ".attn.wo.bias", "vision.h0", "vision.h1", hidden, residual=True)
        norm("vision.h1", prefix + ".norm2.weight")
        x = matmul("vision.norm", prefix + ".mlp.w1.weight", 2 * intermediate, hidden)
        epilogue(x, None, "vision.skip.gated", "vision.gated", intermediate, activation="silu_mul")
        x = matmul("vision.gated", prefix + ".mlp.w2.weight", hidden, intermediate)
        epilogue(x, None, "vision.h1", "vision.h0", hidden, residual=True)
        if (layer + 1) % 4 == 0:
            segment_ends.append(len(body))

    norm("vision.h0", "vision.norm.weight")
    op(
        "vision",
        "merge",
        dict(capacity=patches, width=hidden, ratio=3, channels=512),
        [
            ("source", "vision.norm", "bf16", [patches, hidden], "read"),
            ("grid", "vision.grid", "i32", [8], "read"),
            ("output", "vision.merged", "bf16", [features, 9 * hidden], "write"),
        ],
        "features",
    )
    x = matmul("vision.merged", "aligner.w1.weight", output, 9 * hidden, "features")
    epilogue(
        x,
        "aligner.w1.bias",
        "vision.skip.merged",
        "vision.output",
        output,
        activation="gelu",
        rows="features",
    )
    x = matmul("vision.output", "aligner.w2.weight", output, output, "features")
    epilogue(x, "aligner.w2.bias", "vision.skip.merged", "vision.output", output, rows="features")
    spec = dict(
        layout="deepseek_v41",
        dtype="bf16",
        pixel_stride=stride,
        min_pixels=v["min_pixels"],
        max_image_tokens=v["max_image_tokens"],
        hidden=output,
        patch_size=v["patch_size"],
        temporal_patch_size=1,
        merge_size=3,
        max_patches=patches,
        max_features=8192,
        pixels="vision.input",
        grid="vision.grid",
        length="vision.length",
        output="vision.output",
        features="vision.features",
        feature_index="vision.feature_index",
        mrope_positions="vision.positions",
        plans=[dict(patches=patches, min_patches=1, program="vision")],
        image_token_id=config["image_token_id"],
        vision_start_token_id=config["image_token_id"],
        vision_end_token_id=config["image_token_id"],
    )
    if segment_ends and segment_ends[-1] < len(body):
        segment_ends[-1] = len(body)
    else:
        segment_ends.append(len(body))
    return dict(
        schema_version=1,
        spec=spec,
        rope_theta=v["rope_theta"],
        patch_columns=stride,
        features_capacity=features,
        buffers=list(buffers.values()),
        body=body,
        segment_ends=segment_ends,
    )


def build(
    source,
    output,
    workers,
    gemm_n=256,
    attention_m=128,
    attention_k=256,
    weights=None,
    workspace_slots=64,
):
    from orinfer_build.models.deepseek_v41.build import compile_task

    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    if weights is None:
        raise ValueError("Vision build requires prepared --weights to certify FP16 V")
    value_range = certify_values(weights / "vision.safetensors", config)
    atomic_json(output / "value-range.json", value_range)
    tasks = {}

    def describe(module, name, arguments, specs, capacity):
        identity = json.dumps([module, name, arguments, specs, capacity], sort_keys=True)
        if identity not in tasks:
            tasks[identity] = dict(
                module="kernels.ascend." + module,
                name=name,
                arguments=arguments,
                specs=specs,
                row_capacity=capacity,
                dynamic_arguments=dict(rows=1),
                placeholder=f"@vision{len(tasks)}",
            )
        return tasks[identity]["placeholder"]

    result = program(
        config,
        describe,
        gemm_n=gemm_n,
        attention_m=attention_m,
        attention_k=attention_k,
        workspace_slots=workspace_slots,
    )
    # Deduplicate by operator contract, not by image size, layer or concurrent owner.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(
            pool.map(compile_task, [(task, str(output)) for task in tasks.values()])
        )
    for operation in result["body"]:
        operation["manifest"] = replacements[operation["manifest"]]
    atomic_json(output / "vision-program.json", result)
    atomic_json(
        output / "build-result.json",
        dict(
            operators=len(tasks),
            layers=config["vision_config"]["num_hidden_layers"],
            steps=len(result["body"]),
            schema_version=1,
        ),
    )
    return output / "vision-program.json"


def rebuild(source, output, operators, workers):
    from orinfer_build.models.deepseek_v41.build import compile_task, kernel_rebuild_task
    from orinfer_build.models.deepseek_v41.vision_attach import hardlinks

    source, output = source.resolve(), output.resolve()
    if output.exists() or not operators:
        raise ValueError("Rebuild requires a new output and explicit operator names")
    hardlinks(source.parent, output)
    program = json.loads(source.read_text())
    tasks = {}
    for operation in program["body"]:
        path = operation["manifest"]
        spec = json.loads((source.parent / path).read_text())
        if spec["op"] in operators:
            tasks.setdefault(path, kernel_rebuild_task(spec, path))
    if not tasks:
        raise ValueError("No selected operators in the vision program")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(
            pool.map(compile_task, [(task, str(output)) for task in tasks.values()])
        )
    for operation in program["body"]:
        operation["manifest"] = replacements.get(operation["manifest"], operation["manifest"])
    atomic_json(output / "vision-program.json", program)
    atomic_json(output / "rebuild-result.json", dict(operators=operators, compiled=len(tasks)))
    return output / "vision-program.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--program", type=Path)
    parser.add_argument("--operators", nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--gemm-n", type=int, choices=[128, 256], default=256)
    parser.add_argument("--attention-m", type=int, choices=[32, 64, 128, 256], default=128)
    parser.add_argument("--attention-k", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--workspace-slots", type=int, default=64)
    args = parser.parse_args()
    print(
        rebuild(args.program, args.output, args.operators, args.compile_workers)
        if args.program
        else build(
            args.source,
            args.output,
            args.compile_workers,
            args.gemm_n,
            args.attention_m,
            args.attention_k,
            args.weights,
            args.workspace_slots,
        )
    )


if __name__ == "__main__":
    main()
