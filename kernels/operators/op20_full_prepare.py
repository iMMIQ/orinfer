"""SM87 full-attention preparation; production math is TileLang.

Tensor API: X[M,14336] is 24 interleaved (Q256,gate256) then K1024,V1024.
Norm/cache arithmetic matches the frozen native FP16 path. Gate is raw FP16.
Checked execution is reset_metadata -> validate_metadata -> full_prepare on the
same explicit stream. Status is consumed before Rust commits positions/lengths.
"""
import tilelang.language as T
from tools.operators.common import orin_jit


def validate_host_metadata(req_ids, positions, page_table, num_pages,
                           block_size=128, max_position=262144):
    """CPU scheduler check. Returns physical slots; rejects duplicate owners.

    Call on every metadata update. Shared prefix pages must be made writable by
    the allocator before calling; uniqueness here covers this launch's writers.
    """
    if any(type(v) is not int or v <= 0 for v in
           (num_pages, block_size, max_position)) or num_pages*block_size > 2147483647:
        raise ValueError('invalid physical/cache dimensions')
    if not req_ids or len(req_ids) != len(positions) or not page_table:
        raise ValueError('empty/mismatched metadata')
    width = len(page_table[0])
    if not width or any(len(row) != width for row in page_table):
        raise ValueError('nonrectangular page table')
    seen, slots = set(), []
    for req, pos in zip(req_ids, positions):
        if type(req) is not int or not 0 <= req < len(page_table):
            raise ValueError('request id out of bounds')
        if type(pos) is not int or not 0 <= pos < min(max_position, width*block_size):
            raise ValueError('absolute position out of bounds')
        page = page_table[req][pos//block_size]
        if type(page) is not int or not 0 <= page < num_pages:
            raise ValueError('physical page out of bounds')
        slot = page*block_size + pos%block_size
        if slot in seen:
            raise ValueError('duplicate physical KV writer')
        seen.add(slot); slots.append(slot)
    return tuple(slots)


@orin_jit
def reset_metadata(num_pages: int, block_size: int = 128):
    slots = num_pages*block_size
    @T.prim_func
    def kernel(Owner: T.Tensor((slots,), T.int32), Status: T.Tensor((1,), T.int32)):
        with T.Kernel(T.ceildiv(slots, 256), threads=256) as b:
            for j in T.Parallel(256):
                if b*256+j < slots:
                    Owner[b*256+j] = 0
                if b == 0 and j == 0:
                    Status[0] = 0
    return kernel


@orin_jit
def validate_metadata(batch: int, max_pages: int, num_pages: int,
                      block_size: int = 128, max_position: int = 262144):
    rows = T.dynamic('rows')
    @T.prim_func
    def kernel(Req: T.Tensor((rows,), T.int32), Pos: T.Tensor((rows,), T.int32),
               Pages: T.Tensor((batch, max_pages), T.int32),
               Owner: T.Tensor((num_pages*block_size,), T.int32),
               Status: T.Tensor((1,), T.int32)):
        with T.Kernel(T.ceildiv(rows, 128), threads=128) as b:
            for j in T.Parallel(128):
                row = b*128+j
                if row < rows:
                    req = Req[row]; pos = Pos[row]
                    if req < 0 or req >= batch or pos < 0 or pos >= max_position or pos >= max_pages*block_size:
                        T.atomic_max(Status[0], 1)
                    else:
                        page = Pages[req, pos//block_size]
                        if page < 0 or page >= num_pages:
                            T.atomic_max(Status[0], 2)
                        else:
                            slot = page*block_size+pos%block_size
                            previous = T.atomic_add(Owner[slot], 1, return_prev=True)
                            if previous != 0:
                                T.atomic_max(Status[0], 3)
    return kernel


@orin_jit
def full_prepare(batch: int, max_pages: int, num_pages: int,
                 block_size: int = 128, max_position: int = 262144):
    """Build dynamic M: (X,WQ,WK,Cache,Req,Pos,Pages,Status,Q,Gate,K,V).

    Cache[max_position,64] FP16 = cos32,sin32, theta1e7, Neox pairs
    (j,j+32); text three MRoPE axes are identical. Output Q/Gate[M,24,256]
    and independent K/V[num_pages,block_size,4,256]. Status[0]==0 required.
    No seqLens argument: Rust commits only after same-stream status completion.
    """
    rows = T.dynamic('rows')
    @T.prim_func
    def kernel(X: T.Tensor((rows, 14336), T.float16),
               WQ: T.Tensor((256,), T.float16), WK: T.Tensor((256,), T.float16),
               Cache: T.Tensor((max_position, 64), T.float16),
               Req: T.Tensor((rows,), T.int32), Pos: T.Tensor((rows,), T.int32),
               Pages: T.Tensor((batch, max_pages), T.int32),
               Status: T.Tensor((1,), T.int32),
               Q: T.Tensor((rows, 24, 256), T.float16),
               Gate: T.Tensor((rows, 24, 256), T.float16),
               K: T.Tensor((num_pages, block_size, 4, 256), T.float16),
               V: T.Tensor((num_pages, block_size, 4, 256), T.float16)):
        with T.Kernel(rows*28, threads=128) as work:
            row = work//28
            head = work%28
            values = T.alloc_fragment((256,), T.float32)
            square = T.alloc_fragment((256,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            norm = T.alloc_shared((256,), T.float16)
            if Status[0] == 0:
                for j in T.Parallel(256):
                    if head < 24:
                        values[j] = T.cast(X[row, head*512+j], T.float32)
                        Gate[row, head, j] = X[row, head*512+256+j]
                    else:
                        values[j] = T.cast(X[row, 12288+(head-24)*256+j], T.float32)
                    square[j] = values[j]*values[j]
                T.reduce_sum(square, total, dim=0)
                for j in T.Parallel(256):
                    if head < 24:
                        norm[j] = (values[j]*T.rsqrt(total[0]/256+1e-6))*(1+T.cast(WQ[j], T.float32))
                    else:
                        norm[j] = (values[j]*T.rsqrt(total[0]/256+1e-6))*(1+T.cast(WK[j], T.float32))
                T.sync_threads()
                for j in T.Parallel(256):
                    values[j] = T.cast(norm[j], T.float32)
                    if j < 64:
                        idx = j%32
                        partner = T.if_then_else(j < 32, j+32, j-32)
                        # Native PyTorch FP16 mul, mul, add/sub each round.
                        a = T.cast(T.cast(norm[j], T.float32)*T.cast(Cache[Pos[row], idx], T.float32), T.float16)
                        b = T.cast(T.cast(norm[partner], T.float32)*T.cast(Cache[Pos[row], idx+32], T.float32), T.float16)
                        if j < 32:
                            values[j] = T.cast(a, T.float32)-T.cast(b, T.float32)
                        else:
                            values[j] = T.cast(a, T.float32)+T.cast(b, T.float32)
                    if head < 24:
                        Q[row, head, j] = values[j]
                    else:
                        page = Pages[Req[row], Pos[row]//block_size]
                        K[page, Pos[row]%block_size, head-24, j] = values[j]
                        V[page, Pos[row]%block_size, head-24, j] = X[row, 13312+(head-24)*256+j]
    return kernel


@orin_jit
def bandwidth_copy():
    """Independent streaming FP16 copy for measured bandwidth budget evidence."""
    rows = T.dynamic('rows')
    @T.prim_func
    def kernel(X: T.Tensor((rows, 14336), T.float16),
               Y: T.Tensor((rows, 14336), T.float16)):
        with T.Kernel(T.ceildiv(rows*14336, 4096), threads=256) as block:
            for j in T.Parallel(4096):
                flat = block*4096+j
                if flat < rows*14336:
                    Y[flat//14336, flat%14336] = X[flat//14336, flat%14336]
    return kernel
