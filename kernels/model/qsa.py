"""Original QSA semantics on SM87: group-4 index, exact radix top-k, sparse GQA.

The index stores the mean of raw projected keys, normalized with zero-centered
RMS gamma and rotated at the FIRST token of each complete group. Scores are
sum_h relu(q_h @ k) / sqrt(128). Selection is exact, with lower block IDs winning
score ties; selected blocks plus the incomplete causal tail feed attention.
All arithmetic is TileLang; allocations and request positions belong to callers.
"""
import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit


def geometry(m, capacity):
    if type(m) is not int or type(capacity) is not int or not 1 <= m <= capacity <= 262144:
        raise ValueError('QSA requires 1 <= rows <= capacity <= 262144')
    return (capacity + 3) // 4


@orin_jit
def index_query(m: int, capacity: int = 0):
    mrope=capacity>0
    @T.macro
    def prepare(QK, Weight, Position, Query, Coordinates):
        with T.Kernel(m,4,threads=128) as (row,head):
            x=T.alloc_fragment((128,),T.float32)
            square=T.alloc_fragment((128,),T.float32)
            total=T.alloc_fragment((1,),T.float32)
            normalized=T.alloc_shared((128,),T.float32)
            for d in T.Parallel(128):
                x[d]=T.cast(QK[row,head,d],T.float32)
                square[d]=x[d]*x[d]
            T.reduce_sum(square,total,dim=0)
            for d in T.Parallel(128):
                normalized[d]=x[d]*T.rsqrt(total[0]/128+1e-6)*Weight[d]
            for d in T.Parallel(128):
                pair=T.if_then_else(d<32,d,T.if_then_else(d<64,d-32,0))
                coordinate=T.alloc_var(T.int32)
                coordinate=Position[0]+row
                if mrope:
                    axis=T.if_then_else(pair%3==1 and pair<33,1,T.if_then_else(pair%3==2 and pair<30,2,0))
                    coordinate=Coordinates[Position[0]+row,axis]
                angle=T.cast(coordinate,T.float32)*T.pow(1e7,-2.0*pair/64)
                if d<32:Query[row,head,d]=normalized[d]*T.cos(angle)-normalized[d+32]*T.sin(angle)
                elif d<64:Query[row,head,d]=normalized[d]*T.cos(angle)+normalized[d-32]*T.sin(angle)
                else:Query[row,head,d]=normalized[d]
    if mrope:
        @T.prim_func
        def main(QK: T.Tensor((m,5,128),T.float16), Weight: T.Tensor((128,),T.float32),
                 Position: T.Tensor((1,),T.int32), Query: T.Tensor((m,4,128),T.float16),
                 Coordinates:T.Tensor((capacity,3),T.int32)):
            prepare(QK, Weight, Position, Query, Coordinates)
    else:
        @T.prim_func
        def main(QK: T.Tensor((m,5,128),T.float16), Weight: T.Tensor((128,),T.float32),
                 Position: T.Tensor((1,),T.int32), Query: T.Tensor((m,4,128),T.float16)):
            prepare(QK, Weight, Position, Query, Position)
    return main


@orin_jit
def index_compress(m: int, capacity: int, mrope: bool = False):
    blocks=geometry(m,capacity)
    @T.macro
    def prepare(QK, Pending, Weight, Position, Cache, Coordinates):
        with T.Kernel(T.ceildiv(m+3,4),threads=128) as group:
            block=Position[0]//4+group
            if (block+1)*4<=Position[0]+m:
                x=T.alloc_fragment((128,),T.float32)
                square=T.alloc_fragment((128,),T.float32)
                total=T.alloc_fragment((1,),T.float32)
                normalized=T.alloc_shared((128,),T.float32)
                T.clear(x)
                for offset in T.serial(4):
                    token=block*4+offset
                    for d in T.Parallel(128):
                        if token<Position[0]:x[d]+=T.cast(Pending[token%4,d],T.float32)
                        else:x[d]+=T.cast(QK[token-Position[0],4,d],T.float32)
                for d in T.Parallel(128):
                    x[d]=T.cast(T.cast(x[d]*.25,T.float16),T.float32)
                    square[d]=x[d]*x[d]
                T.reduce_sum(square,total,dim=0)
                for d in T.Parallel(128):
                    normalized[d]=x[d]*T.rsqrt(total[0]/128+1e-6)*Weight[d]
                for d in T.Parallel(128):
                    pair=T.if_then_else(d<32,d,T.if_then_else(d<64,d-32,0))
                    coordinate=T.alloc_var(T.int32)
                    coordinate=block*4
                    if mrope:
                        axis=T.if_then_else(pair%3==1 and pair<33,1,T.if_then_else(pair%3==2 and pair<30,2,0))
                        coordinate=Coordinates[block*4,axis]
                    angle=T.cast(coordinate,T.float32)*T.pow(1e7,-2.0*pair/64)
                    if d<32:Cache[block,d]=normalized[d]*T.cos(angle)-normalized[d+32]*T.sin(angle)
                    elif d<64:Cache[block,d]=normalized[d]*T.cos(angle)+normalized[d-32]*T.sin(angle)
                    else:Cache[block,d]=normalized[d]
    if mrope:
        @T.prim_func
        def main(QK:T.Tensor((m,5,128),T.float16), Pending:T.Tensor((4,128),T.float16),
                 Weight:T.Tensor((128,),T.float32), Position:T.Tensor((1,),T.int32),
                 Cache:T.Tensor((blocks,128),T.float16),
                 Coordinates:T.Tensor((capacity,3),T.int32)):
            prepare(QK, Pending, Weight, Position, Cache, Coordinates)
    else:
        @T.prim_func
        def main(QK:T.Tensor((m,5,128),T.float16), Pending:T.Tensor((4,128),T.float16),
                 Weight:T.Tensor((128,),T.float32), Position:T.Tensor((1,),T.int32),
                 Cache:T.Tensor((blocks,128),T.float16)):
            prepare(QK, Pending, Weight, Position, Cache, Position)
    return main


@orin_jit
def index_pending(m: int):
    """Run after compression. One owner per ring slot, including arbitrary chunks."""
    @T.prim_func
    def main(QK:T.Tensor((m,5,128),T.float16), Position:T.Tensor((1,),T.int32),
             Pending:T.Tensor((4,128),T.float16)):
        with T.Kernel(4,threads=128) as slot:
            last=Position[0]+m-1
            token=last-(last-slot)%4
            if token>=Position[0]:
                for d in T.Parallel(128):Pending[slot,d]=QK[token-Position[0],4,d]
    return main


@orin_jit
def index_scores(m: int, capacity: int):
    blocks=geometry(m,capacity)
    stripes=min((blocks+63)//64,8 if m>=16 else 64)
    @T.prim_func
    def main(Query:T.Tensor((m,4,128),T.float16), Cache:T.Tensor((blocks,128),T.float16),
             Position:T.Tensor((1,),T.int32), Scores:T.Tensor((blocks,m),T.float32)):
        with T.Kernel(T.ceildiv(m,4),stripes,threads=128) as (qr,stripe):
            q=T.alloc_shared((16,128),T.float16)
            k=T.alloc_shared((64,128),T.float16)
            acc=T.alloc_fragment((16,64),T.float32)
            reduced=T.alloc_fragment((4,64),T.float32)
            view=T.reshape(acc,(4,4,64))
            # Stable graph/workspace addresses, but only live causal tiles run.
            for turn in T.serial(T.ceildiv(T.max(0,T.ceildiv((Position[0]+T.min(m,qr*4+4))//4,64)-stripe),stripes)):
                kb=stripe+turn*stripes
                T.clear(reduced)
                for i,d in T.Parallel(16,128):
                    q[i,d]=0
                    if qr*4+i//4<m:q[i,d]=Query[qr*4+i//4,i%4,d]
                for i,d in T.Parallel(64,128):
                    k[i,d]=0
                    if kb*64+i<(Position[0]+m)//4:k[i,d]=Cache[kb*64+i,d]
                T.gemm(q,k,acc,transpose_B=True,clear_accum=True)
                for i,j in T.Parallel(16,64):acc[i,j]=T.max(acc[i,j],0.0)
                T.reduce_sum(view,reduced,dim=1)
                for i,j in T.Parallel(4,64):
                    if qr*4+i<m and kb*64+j<(Position[0]+qr*4+i+1)//4:
                        Scores[kb*64+j,qr*4+i]=reduced[i,j]*128**-.5
    return main


@orin_jit
def radix_histogram(m: int, capacity: int, shift: int, tile: int = 1024):
    blocks=geometry(m,capacity);segments=(blocks+tile-1)//tile
    stripes=min(segments,2)
    if shift not in (24,16,8,0):raise ValueError('Invalid radix byte')
    @T.prim_func
    def main(Scores:T.Tensor((blocks,m),T.float32), Prefix:T.Tensor((m,),T.int32),
             Position:T.Tensor((1,),T.int32), Hist:T.Tensor((m,segments,256),T.int32)):
        with T.Kernel(m,stripes,threads=128) as (row,stripe):
            hist=T.alloc_shared((256,),T.int32)
            for turn in T.serial(T.ceildiv(T.max(0,T.ceildiv((Position[0]+row+1)//4,tile)-stripe),stripes)):
                segment=stripe+turn*stripes
                T.clear(hist)
                for i in T.Parallel(tile):
                    block=segment*tile+i
                    if block<(Position[0]+row+1)//4:
                        key=T.reinterpret(T.int32,Scores[block,row])
                        if shift==24 or key>>(shift+8)==Prefix[row]>>(shift+8):
                            T.atomic_add(hist[(key>>shift)&255],1)
                T.copy(hist,Hist[row,segment,:])
    return main


@orin_jit
def radix_choose(m: int, capacity: int, shift: int, tile: int = 1024):
    blocks=geometry(m,capacity);segments=(blocks+tile-1)//tile
    @T.prim_func
    def main(Hist:T.Tensor((m,segments,256),T.int32), Prefix:T.Tensor((m,),T.int32),
             Remaining:T.Tensor((m,),T.int32), Position:T.Tensor((1,),T.int32)):
        with T.Kernel(m,threads=128) as row:
            total=T.alloc_fragment((256,),T.int32)
            shared=T.alloc_shared((256,),T.int32)
            T.clear(total)
            for s in T.serial(T.ceildiv((Position[0]+row+1)//4,tile)):
                for b in T.Parallel(256):total[b]+=Hist[row,s,b]
            T.copy(total,shared)
            if T.get_thread_binding()==0:
                remain=T.alloc_var(T.int32)
                prefix=T.alloc_var(T.int32)
                found=T.alloc_var(T.int32)
                remain=T.if_then_else(shift==24,T.min(512,(Position[0]+row+1)//4),Remaining[row])
                prefix=T.if_then_else(shift==24,0,Prefix[row])
                found=0
                for b in T.serial(256):
                    bucket=255-b
                    if found==0:
                        if remain>shared[bucket]:remain-=shared[bucket]
                        else:
                            prefix=prefix | (bucket<<shift)
                            found=1
                Prefix[row]=prefix
                Remaining[row]=remain
    return main


@orin_jit
def selection_counts(m: int, capacity: int, tile: int = 1024):
    blocks=geometry(m,capacity);segments=(blocks+tile-1)//tile
    stripes=min(segments,2)
    @T.prim_func
    def main(Scores:T.Tensor((blocks,m),T.float32), Prefix:T.Tensor((m,),T.int32),
             Position:T.Tensor((1,),T.int32), Counts:T.Tensor((m,2,segments),T.int32)):
        with T.Kernel(m,stripes,threads=128) as (row,stripe):
            greater=T.alloc_fragment((tile,),T.int32)
            equal=T.alloc_fragment((tile,),T.int32)
            g=T.alloc_fragment((1,),T.int32);e=T.alloc_fragment((1,),T.int32)
            for turn in T.serial(T.ceildiv(T.max(0,T.ceildiv((Position[0]+row+1)//4,tile)-stripe),stripes)):
                segment=stripe+turn*stripes
                for i in T.Parallel(tile):
                    greater[i]=0;equal[i]=0
                    block=segment*tile+i
                    if block<(Position[0]+row+1)//4:
                        key=T.reinterpret(T.int32,Scores[block,row])
                        greater[i]=T.cast(key>Prefix[row],T.int32)
                        equal[i]=T.cast(key==Prefix[row],T.int32)
                T.reduce_sum(greater,g,dim=0);T.reduce_sum(equal,e,dim=0)
                if T.get_thread_binding()==0:
                    Counts[row,0,segment]=g[0];Counts[row,1,segment]=e[0]
    return main


@orin_jit
def selection_offsets(m: int, capacity: int, tile: int = 1024):
    blocks=geometry(m,capacity);segments=(blocks+tile-1)//tile
    padded=1<<(segments-1).bit_length()
    @T.prim_func
    def main(Counts:T.Tensor((m,2,segments),T.int32), Offsets:T.Tensor((m,2,segments),T.int32),
             Greater:T.Tensor((m,),T.int32), Position:T.Tensor((1,),T.int32),
             Selected:T.Tensor((m,2051),T.int32)):
        with T.Kernel(m,threads=128) as row:
            g=T.alloc_fragment((padded,),T.int32);e=T.alloc_fragment((padded,),T.int32)
            for s in T.Parallel(padded):
                g[s]=0;e[s]=0
                if s<segments and s*tile<(Position[0]+row+1)//4:
                    g[s]=Counts[row,0,s];e[s]=Counts[row,1,s]
            T.cumsum(g,dim=0);T.cumsum(e,dim=0)
            for s in T.Parallel(segments):
                if s*tile<(Position[0]+row+1)//4:
                    Offsets[row,0,s]=g[s]-Counts[row,0,s]
                    Offsets[row,1,s]=e[s]-Counts[row,1,s]
            for s in T.Parallel(padded):
                if s==padded-1:Greater[row]=g[s]
            for i in T.Parallel(2051):
                Selected[row,i]=-1
                full=(Position[0]+row+1)//4
                tail=(Position[0]+row+1)%4
                if i>=T.min(full,512)*4 and i<T.min(full,512)*4+tail:
                    Selected[row,i]=full*4+i-T.min(full,512)*4
    return main


@orin_jit
def selection_scatter(m: int, capacity: int, tile: int = 1024):
    blocks=geometry(m,capacity);segments=(blocks+tile-1)//tile
    stripes=min(segments,2)
    @T.prim_func
    def main(Scores:T.Tensor((blocks,m),T.float32), Prefix:T.Tensor((m,),T.int32),
             Offsets:T.Tensor((m,2,segments),T.int32), Greater:T.Tensor((m,),T.int32),
             Position:T.Tensor((1,),T.int32), Selected:T.Tensor((m,2051),T.int32)):
        with T.Kernel(m,stripes,threads=128) as (row,stripe):
            g=T.alloc_fragment((tile,),T.int32);e=T.alloc_fragment((tile,),T.int32)
            gp=T.alloc_fragment((tile,),T.int32);ep=T.alloc_fragment((tile,),T.int32)
            for turn in T.serial(T.ceildiv(T.max(0,T.ceildiv((Position[0]+row+1)//4,tile)-stripe),stripes)):
                segment=stripe+turn*stripes
                for i in T.Parallel(tile):
                    g[i]=0;e[i]=0
                    block=segment*tile+i
                    if block<(Position[0]+row+1)//4:
                        key=T.reinterpret(T.int32,Scores[block,row])
                        g[i]=T.cast(key>Prefix[row],T.int32);e[i]=T.cast(key==Prefix[row],T.int32)
                    gp[i]=g[i];ep[i]=e[i]
                T.cumsum(gp,dim=0);T.cumsum(ep,dim=0)
                for i in T.Parallel(tile):
                    rank=T.alloc_var(T.int32)
                    rank=-1
                    if g[i]!=0:rank=Offsets[row,0,segment]+gp[i]-1
                    elif e[i]!=0:rank=Greater[row]+Offsets[row,1,segment]+ep[i]-1
                    if rank>=0 and rank<T.min(512,(Position[0]+row+1)//4):
                        for offset in T.unroll(4):Selected[row,rank*4+offset]=(segment*tile+i)*4+offset
    return main


@orin_jit
def sparse_attention(m: int, capacity: int, splits: int = 8):
    geometry(m,capacity)
    if splits not in (1,2,4,8):raise ValueError('Invalid QSA splits')
    @T.prim_func
    def main(Query:T.Tensor((m,24,256),T.float16), K:T.Tensor((capacity,2,256),T.int8),
             V:T.Tensor((capacity,2,256),T.int8),
             KS:T.Tensor((capacity,2,4),T.float16), VS:T.Tensor((capacity,2,4),T.float16), Selected:T.Tensor((m,2051),T.int32),
             Position:T.Tensor((1,),T.int32), Max:T.Tensor((m,24,splits),T.float32),
             Den:T.Tensor((m,24,splits),T.float32), Out:T.Tensor((m,24,splits,256),T.float32)):
        with T.Kernel(m,24,splits,threads=128) as (row,head,split):
            products=T.alloc_fragment((16,256),T.float32)
            scores=T.alloc_fragment((16,),T.float32);probs=T.alloc_fragment((16,),T.float32)
            acc=T.alloc_fragment((256,),T.float32);update=T.alloc_fragment((256,),T.float32)
            max_=T.alloc_fragment((1,),T.float32);den=T.alloc_fragment((1,),T.float32)
            tile_max=T.alloc_fragment((1,),T.float32);tile_sum=T.alloc_fragment((1,),T.float32)
            new_max=T.alloc_fragment((1,),T.float32);alpha=T.alloc_fragment((1,),T.float32)
            T.annotate_layout({
                products:tilelang.Fragment((16,256),
                    forward_thread_fn=lambda i,d:(i%4)*32+d%32,
                    forward_index_fn=lambda i,d:(i//4)*8+d//32),
                scores:tilelang.Fragment((16,),forward_thread_fn=lambda i,rep:(i%4)*32+rep,
                    forward_index_fn=lambda i:i//4,replicate=32),
                probs:tilelang.Fragment((16,),forward_thread_fn=lambda i,rep:(i%4)*32+rep,
                    forward_index_fn=lambda i:i//4,replicate=32),
                update:tilelang.Fragment((256,),forward_thread_fn=lambda d,rep:rep*32+d%32,
                    forward_index_fn=lambda d:d//32,replicate=4),
                acc:tilelang.Fragment((256,),forward_thread_fn=lambda d:d%128,
                    forward_index_fn=lambda d:d//128),
                max_:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128),
                den:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128),
                tile_max:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128),
                tile_sum:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128),
                new_max:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128),
                alpha:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128)})
            T.clear(acc);max_[0]=-3.402823466e38;den[0]=0
            length=T.min((Position[0]+row+1)//4,512)*4+(Position[0]+row+1)%4
            count=T.ceildiv(length,splits)
            start=split*count;end=T.min(start+count,length)
            for tile in T.serial(T.ceildiv(T.max(0,end-start),16)):
                for i,d in T.Parallel(16,256):
                    slot=start+tile*16+i
                    products[i,d]=0
                    if slot<end:
                        token=Selected[row,slot]
                        if token>=0 and token<=Position[0]+row:
                            products[i,d]=T.cast(Query[row,head,d],T.float32)*(T.cast(K[token,head//12,d],T.float32)*T.cast(KS[token,head//12,d//64],T.float32))
                T.reduce_sum(products,scores,dim=1)
                for i in T.Parallel(16):
                    slot=start+tile*16+i
                    scores[i]*=.0625
                    if slot>=end:scores[i]=-3.402823466e38
                T.reduce_max(scores,tile_max,dim=0)
                new_max[0]=T.max(max_[0],tile_max[0])
                alpha[0]=T.exp(max_[0]-new_max[0])
                for i in T.Parallel(16):
                    probs[i]=0
                    if start+tile*16+i<end:probs[i]=T.exp(scores[i]-new_max[0])
                T.reduce_sum(probs,tile_sum,dim=0)
                for i,d in T.Parallel(16,256):
                    slot=start+tile*16+i
                    products[i,d]=0
                    if slot<end:
                        token=Selected[row,slot]
                        if token>=0 and token<=Position[0]+row:products[i,d]=probs[i]*(T.cast(V[token,head//12,d],T.float32)*T.cast(VS[token,head//12,d//64],T.float32))
                T.reduce_sum(products,update,dim=0)
                for d in T.Parallel(256):acc[d]=acc[d]*alpha[0]+update[d]
                den[0]=den[0]*alpha[0]+tile_sum[0];max_[0]=new_max[0]
            T.copy(acc,Out[row,head,split,:])
            if T.get_thread_binding()==0:Max[row,head,split]=max_[0];Den[row,head,split]=den[0]
    return main


@orin_jit
def sparse_merge(m: int, splits: int = 8):
    @T.prim_func
    def main(Max:T.Tensor((m,24,splits),T.float32), Den:T.Tensor((m,24,splits),T.float32),
             Out:T.Tensor((m,24,splits,256),T.float32), Gate:T.Tensor((m,24,256),T.float16),
             Output:T.Tensor((m,24,256),T.float16)):
        with T.Kernel(m,24,threads=128) as (row,head):
            maximum=T.alloc_local((1,),T.float32);den=T.alloc_local((1,),T.float32)
            acc=T.alloc_fragment((256,),T.float32)
            maximum[0]=-3.402823466e38;den[0]=0;T.clear(acc)
            for s in T.serial(splits):maximum[0]=T.max(maximum[0],Max[row,head,s])
            for s in T.serial(splits):
                scale=T.exp(Max[row,head,s]-maximum[0]);den[0]+=Den[row,head,s]*scale
                for d in T.Parallel(256):acc[d]+=Out[row,head,s,d]*scale
            for d in T.Parallel(256):Output[row,head,d]=acc[d]/T.max(den[0],1e-30)/(1+T.exp(-T.cast(Gate[row,head,d],T.float32)))
    return main


@orin_jit
def kv_store(m: int, capacity: int):
    """Group-64 symmetric INT8 K/V; quantize the exact prepared FP16 values."""
    geometry(m,capacity)
    @T.prim_func
    def main(Key:T.Tensor((m,2,256),T.float16), Value:T.Tensor((m,2,256),T.float16),
             Position:T.Tensor((1,),T.int32), K:T.Tensor((capacity,2,256),T.int8),
             V:T.Tensor((capacity,2,256),T.int8), KS:T.Tensor((capacity,2,4),T.float16),
             VS:T.Tensor((capacity,2,4),T.float16)):
        with T.Kernel(m,2,threads=128) as (row,head):
            ka=T.alloc_fragment((4,64),T.float32);va=T.alloc_fragment((4,64),T.float32)
            km=T.alloc_fragment((4,),T.float32);vm=T.alloc_fragment((4,),T.float32)
            ks=T.alloc_shared((4,),T.float16);vs=T.alloc_shared((4,),T.float16)
            for g,d in T.Parallel(4,64):
                ka[g,d]=T.abs(T.cast(Key[row,head,g*64+d],T.float32))
                va[g,d]=T.abs(T.cast(Value[row,head,g*64+d],T.float32))
            T.reduce_max(ka,km,dim=1);T.reduce_max(va,vm,dim=1)
            for g in T.Parallel(4):
                ks[g]=T.max(km[g]/127.0,2**-24);vs[g]=T.max(vm[g]/127.0,2**-24)
            if Position[0]>=0 and Position[0]+row<capacity:
                for g in T.Parallel(4):
                    KS[Position[0]+row,head,g]=ks[g];VS[Position[0]+row,head,g]=vs[g]
                for d in T.Parallel(256):
                    kr=T.call_extern('float32','__fdiv_rn',T.cast(Key[row,head,d],T.float32),T.cast(ks[d//64],T.float32))
                    vr=T.call_extern('float32','__fdiv_rn',T.cast(Value[row,head,d],T.float32),T.cast(vs[d//64],T.float32))
                    K[Position[0]+row,head,d]=T.cast(T.max(-127.,T.min(127.,T.round(kr))),T.int8)
                    V[Position[0]+row,head,d]=T.cast(T.max(-127.,T.min(127.,T.round(vr))),T.int8)
    return main
