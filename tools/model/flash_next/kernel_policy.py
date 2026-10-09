"""Validated static tiles shared by initial AOT builds and package upgrades."""


def expert_tile_config(rows):
    if type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError("Unsupported expert row count")
    block_m = 64 if rows >= 1024 else 32 if rows >= 256 else 16
    shortbook = 2 <= rows <= 8 or rows >= 256
    return dict(
        block_m=block_m,
        block_n=128 if shortbook else 64,
        shortbook=shortbook,
        num_stages=2 if 2 <= rows <= 4 else 1 if shortbook and block_m <= 32 else 2,
        byte_permute=2 <= rows <= 8 or block_m == 64,
    )


def router_tile(rows, outputs, inputs, experts, hidden):
    return 32 if rows <= 8 and outputs == experts and inputs == hidden else 64
