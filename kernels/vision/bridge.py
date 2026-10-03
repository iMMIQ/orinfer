"""Vision feature injection and interleaved multimodal RoPE for the text plan."""
import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def embedding_features(vocab: int, hidden: int, context: int, features: int,
                       group: int = 128):
    rows=T.dynamic('rows'); padded=((hidden+group-1)//group)*group; pairs=(hidden+1)//2
    @T.prim_func
    def kernel(P:T.Tensor((vocab,padded//2),T.uint8),S:T.Tensor((vocab,padded//group),T.float16),
               Z:T.Tensor((vocab,padded//group),T.int8),I:T.Tensor((rows,),T.int32),
               Step:T.Tensor((1,),T.int32),Index:T.Tensor((context,),T.int32),
               Features:T.Tensor((features,hidden),T.float16),Y:T.Tensor((rows,hidden),T.float16)):
        with T.Kernel(T.ceildiv(rows*pairs,512),threads=128) as b:
            for j in T.Parallel(512):
                flat=b*512+j
                if flat<rows*pairs:
                    row=flat//pairs;col=(flat%pairs)*2;idx=Index[Step[0]+row]
                    if idx>=0:
                        Y[row,col]=Features[idx,col]
                        if col+1<hidden:Y[row,col+1]=Features[idx,col+1]
                    else:
                        token=I[row];packed=T.cast(P[token,col//2],T.int32)
                        zero=T.cast(Z[token,col//group],T.float32);scale=T.cast(S[token,col//group],T.float32)
                        Y[row,col]=(T.cast(packed&15,T.float32)-zero)*scale
                        if col+1<hidden:Y[row,col+1]=(T.cast(packed>>4,T.float32)-zero)*scale
    return kernel


@orin_jit
def full_prepare_mrope(max_pages: int, context: int, section: tuple[int,int,int],
                       block_size: int = 128, max_position: int = 8704):
    rows=T.dynamic('rows')
    @T.prim_func
    def kernel(X:T.Tensor((rows,14336),T.float16),WQ:T.Tensor((256,),T.float16),WK:T.Tensor((256,),T.float16),
               Cache:T.Tensor((max_position,64),T.float16),Req:T.Tensor((rows,),T.int32),
               Pos:T.Tensor((rows,),T.int32),Pages:T.Tensor((1,max_pages),T.int32),Status:T.Tensor((1,),T.int32),
               MRope:T.Tensor((context,3),T.int32),Q:T.Tensor((rows,24,256),T.float16),
               Gate:T.Tensor((rows,24,256),T.float16),K:T.Tensor((max_pages,block_size,4,256),T.float16),
               V:T.Tensor((max_pages,block_size,4,256),T.float16)):
        with T.Kernel(rows*28,threads=128) as work:
            row=work//28;head=work%28
            values=T.alloc_fragment((256,),T.float32)
            square=T.alloc_fragment((256,),T.float32)
            total=T.alloc_fragment((1,),T.float32)
            norm=T.alloc_shared((256,),T.float16)
            if Status[0]==0:
                for j in T.Parallel(256):
                    if head<24:
                        values[j]=T.cast(X[row,head*512+j],T.float32)
                        Gate[row,head,j]=X[row,head*512+256+j]
                    else:values[j]=T.cast(X[row,12288+(head-24)*256+j],T.float32)
                    square[j]=values[j]*values[j]
                T.reduce_sum(square,total,dim=0)
                for j in T.Parallel(256):
                    weight=T.if_then_else(head<24,WQ[j],WK[j])
                    norm[j]=(values[j]*T.rsqrt(total[0]/256+1e-6))*(1+T.cast(weight,T.float32))
                T.sync_threads()
                for j in T.Parallel(256):
                    values[j]=T.cast(norm[j],T.float32)
                    if j<64:
                        idx=j%32;partner=T.if_then_else(j<32,j+32,j-32)
                        axis=T.if_then_else(idx%3==1 and idx<section[1]*3,1,
                             T.if_then_else(idx%3==2 and idx<section[2]*3,2,0))
                        position=MRope[Pos[row],axis]
                        a=T.cast(T.cast(norm[j],T.float32)*T.cast(Cache[position,idx],T.float32),T.float16)
                        b=T.cast(T.cast(norm[partner],T.float32)*T.cast(Cache[position,idx+32],T.float32),T.float16)
                        values[j]=T.if_then_else(j<32,T.cast(a,T.float32)-T.cast(b,T.float32),T.cast(a,T.float32)+T.cast(b,T.float32))
                    if head<24:Q[row,head,j]=values[j]
                    else:
                        page=Pages[Req[row],Pos[row]//block_size]
                        K[page,Pos[row]%block_size,head-24,j]=values[j]
                        V[page,Pos[row]%block_size,head-24,j]=X[row,13312+(head-24)*256+j]
    return kernel
