"""Select one floating MMA schedule per audio geometry across dynamic extents."""

from orinfer_build.compiler.compile import primitive
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.contracts.row_selection import aggregate_winner
from kernels.cuda.audio import encoder as audio
from kernels.cuda.vision import encoder as common


def select(tuner, *, convolution=None, outputs=480, inputs=None, activation="none"):
    import torch

    configs = [
        dict(bm=m, bn=n, bk=k, stages=2)
        for m, n, k in [
            (32, 64, 32),
            (64, 64, 32),
            (64, 128, 32),
            (128, 64, 32),
            (128, 128, 32),
            (64, 64, 64),
        ]
    ]
    if not convolution:
        configs += [dict(bm=m, bn=n, bk=64, stages=2) for m, n in [(64, 128), (128, 128)]]
    spec = dict(
        kind="audio_convolution" if convolution else "audio_linear",
        geometry=convolution,
        outputs=outputs,
        inputs=inputs,
        activation=activation,
        objective="relative_geomean" if convolution else "total_latency",
    )

    def build(config):
        if convolution:
            return primitive(audio.convolution, *convolution, outputs=outputs, **config)
        return primitive(common.linear, outputs, inputs, activation, **config)

    def fixtures(rows):
        if convolution:
            channels, height, width = convolution
            x = (
                torch.randn((rows, height, width, channels), device="cuda", dtype=torch.float16)
                * 0.1
            )
            w = torch.randn((outputs, 3, 3, channels), device="cuda", dtype=torch.float16) * 0.01
            shape = (rows, (height + 1) // 2, (width + 1) // 2, outputs)
        else:
            x = torch.randn((rows, inputs), device="cuda", dtype=torch.float16) * 0.1
            w = torch.randn((outputs, inputs), device="cuda", dtype=torch.float16) * 0.01
            shape = (rows, outputs)
        bias = torch.randn((outputs,), device="cuda", dtype=torch.float16) * 0.01

        def fixture(kernel, config):
            storage = torch.full((shape[0] + 1, *shape[1:]), 91, device="cuda", dtype=torch.float16)
            y = storage[:rows]

            def run():
                launch(kernel, x, w, bias, y)

            def check():
                if convolution:
                    reference = torch.nn.functional.conv2d(
                        x.permute(0, 3, 1, 2), w.permute(0, 3, 1, 2), bias, stride=2, padding=1
                    ).permute(0, 2, 3, 1)
                    reference = torch.nn.functional.gelu(reference)
                else:
                    reference = (x.float() @ w.float().T + bias.float()).half()
                    if activation == "gelu":
                        reference = torch.nn.functional.gelu(reference)
                torch.testing.assert_close(y, reference, atol=0.002, rtol=0)
                assert bool(torch.all(storage[rows:] == 91))

            return run, check, x.neg_

        yield fixture

    result = tuner.select(
        spec,
        configs,
        build,
        fixtures,
        measure_rows=[1, 4, 16, 32] if convolution else [1, 13, 55, 104, 196, 416, 1587, 4096],
    )
    # Publish one schedule for all row counts, selected from measured workloads.
    winner = (
        aggregate_winner(result["measurements"])
        if convolution
        else min(
            result["measurements"][0]["times"],
            key=lambda name: sum(row["times"][name] for row in result["measurements"]),
        )
    )
    return build(configs[int(winner)])
