"""Optional exact Q8 KV cache and explicit FP16 gather baseline, SM87.

Finite token-major FP16 K/V only. Metadata validation is scheduler policy;
shared immutable pages require caller COW before write. This is NOT inline
paged quantized attention: gather allocates and writes a complete FP16 copy.
"""
import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit


def validate_write_metadata(pages, requests, positions, num_pages,
                            readonly_pages=(), block_size=128):
    if type(num_pages) is not int or num_pages < 1 or block_size != 128:
        raise ValueError('invalid cache dimensions')
    if not pages or not pages[0] or any(len(r) != len(pages[0]) for r in pages):
        raise ValueError('nonrectangular or empty page table')
    if len(requests) != len(positions) or not requests:
        raise ValueError('empty or mismatched writes')
    protected = set(readonly_pages)
    if any(type(p) is not int or not 0 <= p < num_pages for p in protected):
        raise ValueError('invalid readonly page')
    owners = {}
    for r, row in enumerate(pages):
        for p in row:
            if type(p) is int and 0 <= p < num_pages:
                owners.setdefault(p, set()).add(r)
    written = set()
    for r, pos in zip(requests, positions):
        if type(r) is not int or not 0 <= r < len(pages):
            raise ValueError('request out of bounds')
        if type(pos) is not int or not 0 <= pos < len(pages[0])*block_size:
            raise ValueError('position out of bounds')
        page = pages[r][pos//block_size]
        if type(page) is not int or not 0 <= page < num_pages:
            raise ValueError('physical page out of bounds')
        if page in protected or len(owners.get(page, ())) > 1:
            raise ValueError('readonly/shared page write requires caller COW')
        slot = (page, pos % block_size)
        if slot in written:
            raise ValueError('duplicate physical writer')
        written.add(slot)
    return True


def validate_disjoint_buffers(*buffers):
    """Reject overlapping contiguous tensor spans before capture/launch."""
    spans = []
    for tensor in buffers:
        if not tensor.is_contiguous():
            raise ValueError('noncontiguous buffer')
        lo = tensor.data_ptr()
        hi = lo + tensor.numel()*tensor.element_size()
        if any(lo < b and a < hi for a, b in spans):
            raise ValueError('aliased buffers')
        spans.append((lo, hi))
    return True


def validate_finite_inputs(*inputs):
    """Host-side preflight, outside capture; finite inputs are mandatory.

    Tensor isfinite reduction is validation only, never quantization math.
    Call again whenever input contents change; this check synchronizes.
    """
    if any(not bool(x.isfinite().all()) for x in inputs):
        raise ValueError('nonfinite KV input')
    return True


@orin_jit
def kv_quantize_pack(max_pages: int, num_pages: int, block_size: int = 128):
    """K,V[N,4,256] half; Req,Pos[N]; Pages[B,MP]; QK,QV int8;
    SK,SV[P,128,4] FP32. Unique writer required. No workspace.
    scale=amax/127 FP32 (zero=1); code=clamp(rint_rne(x/scale),-127,127).
    """
    assert max_pages > 0 and num_pages > 0 and block_size == 128
    rows, batch = T.dynamic('rows'), T.dynamic('batch')
    @T.prim_func
    def kernel(K:T.Tensor((rows,4,256),T.float16),
               V:T.Tensor((rows,4,256),T.float16),
               Req:T.Tensor((rows,),T.int32), Pos:T.Tensor((rows,),T.int32),
               Pages:T.Tensor((batch,max_pages),T.int32),
               QK:T.Tensor((num_pages,block_size,4,256),T.int8),
               QV:T.Tensor((num_pages,block_size,4,256),T.int8),
               SK:T.Tensor((num_pages,block_size,4),T.float32),
               SV:T.Tensor((num_pages,block_size,4),T.float32)):
        with T.Kernel(rows,8,threads=256) as (row,h):
            values=T.alloc_fragment((256,),T.float32)
            absolute=T.alloc_fragment((256,),T.float32)
            maximum=T.alloc_fragment((1,),T.float32)
            scale=T.alloc_fragment((1,),T.float32)
            # Replicate reduction scalars explicitly so a thread0 store cannot
            # push the entire 256-element reduction to a serial fragment.
            T.annotate_layout({
                values:tilelang.Fragment((256,),forward_thread_fn=lambda d:d,
                    forward_index_fn=lambda d:0),
                absolute:tilelang.Fragment((256,),forward_thread_fn=lambda d:d,
                    forward_index_fn=lambda d:0),
                maximum:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,
                    replicate=256),
                scale:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,
                    replicate=256)})
            for d in T.Parallel(256):
                if h < 4: values[d]=T.cast(K[row,h,d],T.float32)
                else: values[d]=T.cast(V[row,h-4,d],T.float32)
                absolute[d]=T.abs(values[d])
            T.reduce_max(absolute,maximum,dim=0)
            scale[0]=1.0
            if maximum[0] > 0:
                scale[0]=T.call_extern('float32','__fdiv_rn',maximum[0],127.0)
            r=Req[row];pos=Pos[row]
            if r >= 0 and r < batch and pos >= 0 and pos < max_pages*block_size:
                page=Pages[r,pos//block_size]
                if page >= 0 and page < num_pages:
                    if T.get_thread_binding() == 0:
                        if h < 4: SK[page,pos%block_size,h]=scale[0]
                        else: SV[page,pos%block_size,h-4]=scale[0]
                    for d in T.Parallel(256):
                        ratio=T.call_extern('float32','__fdiv_rn',values[d],scale[0])
                        code=T.cast(T.max(-127.0,T.min(127.0,T.round(ratio))),T.int8)
                        if h < 4: QK[page,pos%block_size,h,d]=code
                        else: QV[page,pos%block_size,h-4,d]=code
    return kernel


@orin_jit
def kv_page_gather(max_pages: int, num_pages: int, block_size: int = 128):
    """QK,QV,SK,SV,Pages,Lengths,Ktmp,Vtmp; dynamic B.

    Temporary [B,MP*128,4,256] FP16, zero beyond length. Only valid cache
    tokens read, including scales; unused pages may be invalid/NaN poisoned.
    Dequantization explicitly rounds half(float(code)*FP32scale).
    """
    assert max_pages > 0 and num_pages > 0 and block_size == 128
    batch=T.dynamic('batch')
    @T.prim_func
    def kernel(QK:T.Tensor((num_pages,block_size,4,256),T.int8),
               QV:T.Tensor((num_pages,block_size,4,256),T.int8),
               SK:T.Tensor((num_pages,block_size,4),T.float32),
               SV:T.Tensor((num_pages,block_size,4),T.float32),
               Pages:T.Tensor((batch,max_pages),T.int32),
               Lengths:T.Tensor((batch,),T.int32),
               Ktmp:T.Tensor((batch,max_pages*block_size,4,256),T.float16),
               Vtmp:T.Tensor((batch,max_pages*block_size,4,256),T.float16)):
        with T.Kernel(max_pages*block_size,4,batch,threads=256) as (pos,h,b):
            for d in T.Parallel(256):
                Ktmp[b,pos,h,d]=0.0;Vtmp[b,pos,h,d]=0.0
                if pos < Lengths[b]:
                    page=Pages[b,pos//block_size]
                    if page >= 0 and page < num_pages:
                        Ktmp[b,pos,h,d]=T.cast(QK[page,pos%block_size,h,d],T.float32)*SK[page,pos%block_size,h]
                        Vtmp[b,pos,h,d]=T.cast(QV[page,pos%block_size,h,d],T.float32)*SV[page,pos%block_size,h]
    return kernel
