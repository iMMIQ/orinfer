"""Exact SM87 state primitives; no numeric cast or scheduler implementation.

GDN canonical [layer,request,Vhead,K,V] FP32 is viewed as int32 words;
conv [layer,request,history,channel] FP16 as paired int32 words; int64
positions as two int32 words. All outputs are caller-owned and disjoint.
"""

import tilelang.language as T
from tools.operators.common import orin_jit

GDN_LAYERS = 48
GDN_WORDS = 48 * 128 * 128
CONV_WORDS = 3 * 10240 // 2


def _dimension(value, name):
    if type(value) is not int or not 0 < value <= 2147483647:
        raise ValueError(f"{name} must be a positive int32 dimension")


def validate_request_map(source, destination, source_pool, destination_pool):
    """CPU scheduler boundary: readonly source duplication is legal; writes unique."""
    _dimension(source_pool, "source_pool")
    _dimension(destination_pool, "destination_pool")
    if not isinstance(source, (tuple, list)) or not source:
        raise ValueError("source must be a nonempty CPU list/tuple")
    if not isinstance(destination, (tuple, list)) or len(source) != len(destination):
        raise ValueError("source/destination map lengths must match")
    if any(type(i) is not int or not 0 <= i < source_pool for i in source):
        raise ValueError("source request out of bounds")
    if any(type(i) is not int or not 0 <= i < destination_pool for i in destination):
        raise ValueError("destination request out of bounds")
    if len(set(destination)) != len(destination):
        raise ValueError("duplicate destination request would race")
    return tuple(source), tuple(destination)


def validate_buffer_ranges(buffers):
    """Require disjoint contiguous allocation spans: [(name,pointer,nbytes),...].

    Includes metadata and both readonly/write tensors. The sole allowed page
    sharing is repeated entries inside a readonly pageTable, not pointer alias.
    Rust must compute checked byte lengths and call this equivalent before raw
    launch; stable graph pointers do not waive revalidating changed metadata.
    """
    spans = []
    for name, pointer, nbytes in buffers:
        if type(pointer) is not int or pointer <= 0 or type(nbytes) is not int or nbytes <= 0:
            raise ValueError("invalid allocation span")
        if pointer + nbytes > 2**64:
            raise ValueError("allocation span overflows uint64")
        spans.append((name, pointer, pointer + nbytes))
    for i, (name, begin, end) in enumerate(spans):
        for other, obegin, oend in spans[i + 1 :]:
            if begin < oend and obegin < end:
                raise ValueError(f"alias rejected: {name}/{other}")


def validate_paged_metadata(page_table, lengths, num_pages, physical_tokens, block_size=128):
    """Validate only pages that will be read; unused table entries may be -1.

    Shared prefix pages (including duplicate readonly page IDs) are legal.
    This gather never mutates pages and performs no page ownership/COW action.
    """
    _dimension(num_pages, "num_pages")
    _dimension(physical_tokens, "physical_tokens")
    _dimension(block_size, "block_size")
    if not isinstance(lengths, (tuple, list)) or not lengths:
        raise ValueError("lengths must be a nonempty CPU list/tuple")
    if not isinstance(page_table, (tuple, list)) or len(page_table) != len(lengths):
        raise ValueError("pageTable batch mismatch")
    width = len(page_table[0]) if isinstance(page_table[0], (tuple, list)) else 0
    if width <= 0:
        raise ValueError("pageTable must have positive width")
    for row, length in zip(page_table, lengths):
        if not isinstance(row, (tuple, list)) or len(row) != width:
            raise ValueError("pageTable must be rectangular")
        if type(length) is not int or not 0 <= length <= physical_tokens:
            raise ValueError("sequence length outside physical output")
        count = (length + block_size - 1) // block_size
        if count > width:
            raise ValueError("pageTable is too short")
        if any(type(p) is not int or not -2147483648 <= p <= 2147483647 for p in row):
            raise ValueError("pageTable values must be int32")
        if any(not 0 <= p < num_pages for p in row[:count]):
            raise ValueError("valid token page out of bounds")
    return tuple(tuple(r) for r in page_table), tuple(lengths)


@orin_jit
def request_word_copy(
    layers: int = 48, words: int = GDN_WORDS, block: int = 4096, threads: int = 256
):
    """(Source,SI,DI,Dest); int32 [L,Sp,W]/[L,Dp,W], int32 SI/DI[N].

    Source may repeat to clone branches. Dest indices unique. No aliases.
    All batches/pool sizes dynamic; copy/checkpoint/restore share this API.
    """
    assert layers > 0 and words > 0 and block % threads == 0
    sp, dp, count = T.dynamic("source_pool"), T.dynamic("destination_pool"), T.dynamic("count")

    @T.prim_func
    def kernel(
        Source: T.Tensor((layers, sp, words), T.int32),
        SI: T.Tensor((count,), T.int32),
        DI: T.Tensor((count,), T.int32),
        Dest: T.Tensor((layers, dp, words), T.int32),
    ):
        with T.Kernel(T.ceildiv(words, block), count, layers, threads=threads) as (bx, r, layer):
            for j in T.Parallel(block):
                w = bx * block + j
                if w < words:
                    Dest[layer, DI[r], w] = Source[layer, SI[r], w]

    return kernel


@orin_jit
def request_word_zero(
    layers: int = 48, words: int = GDN_WORDS, block: int = 4096, threads: int = 256
):
    """(DI,Dest); initialize selected unique requests with exact positive-zero bits."""
    assert layers > 0 and words > 0 and block % threads == 0
    dp, count = T.dynamic("destination_pool"), T.dynamic("count")

    @T.prim_func
    def kernel(DI: T.Tensor((count,), T.int32), Dest: T.Tensor((layers, dp, words), T.int32)):
        with T.Kernel(T.ceildiv(words, block), count, layers, threads=threads) as (bx, r, layer):
            for j in T.Parallel(block):
                w = bx * block + j
                if w < words:
                    Dest[layer, DI[r], w] = 0

    return kernel


@orin_jit
def paged_kv_gather(block_size: int = 128, block: int = 2048, threads: int = 256):
    """(Kpages,Vpages,PageTable,SeqLengths,Kout,Vout), int32 paired FP16 bits.

    Public FP16 layouts pages[P,BS,4,256], out[B,Tphysical,4,256]; raw word
    layouts pages[P,BS,512], out[B,Tphysical,512]. Pad is exact zero and its
    pages are never loaded. All dimensions except BS/head/D are dynamic.
    """
    assert block_size > 0 and block % threads == 0
    pages, batch = T.dynamic("pages"), T.dynamic("batch")
    width, tokens = T.dynamic("table_width"), T.dynamic("tokens")

    @T.prim_func
    def kernel(
        Kpages: T.Tensor((pages, block_size, 512), T.int32),
        Vpages: T.Tensor((pages, block_size, 512), T.int32),
        PageTable: T.Tensor((batch, width), T.int32),
        SeqLengths: T.Tensor((batch,), T.int32),
        Kout: T.Tensor((batch, tokens, 512), T.int32),
        Vout: T.Tensor((batch, tokens, 512), T.int32),
    ):
        with T.Kernel(T.ceildiv(tokens * 512, block), batch, threads=threads) as (bx, b):
            for j in T.Parallel(block):
                index = bx * block + j
                if index < tokens * 512:
                    t, w = index // 512, index % 512
                    if t < SeqLengths[b]:
                        page = PageTable[b, t // block_size]
                        Kout[b, t, w] = Kpages[page, t % block_size, w]
                        Vout[b, t, w] = Vpages[page, t % block_size, w]
                    else:
                        Kout[b, t, w] = 0
                        Vout[b, t, w] = 0

    return kernel
