"""Independent Torch reference math; never used by production kernels."""
import math

import torch
import torch.nn.functional as F


def norm(x, weight, *, zero_centered=True, eps=1e-6):
    gamma = weight.float()+(1.0 if zero_centered else 0.0)
    result = x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*gamma
    return result.to(x.dtype)


def read_streams(x, weight, *, inject=True):
    streams = x.shape[1]
    normalized = norm(x,weight['norm'].reshape(x.shape[1:]))
    low = F.silu(F.linear(normalized.flatten(1),weight['down'])/streams)
    up = F.linear(low,weight['up']).reshape_as(x)
    mixed = (normalized*torch.sigmoid(up)).mean(1)
    gate = 2*torch.sigmoid(F.linear(normalized.flatten(1),weight['inject'])/streams) if inject else None
    return mixed,gate


def write_streams(x, block, gate):
    return x+block[:,None,:]*gate[:,:,None]


def causal_conv(x, weight, *, dilation=1, round_before_silu=False):
    """Independent depthwise convolution with zero request-owned initial history."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0] or dilation < 1:
        raise ValueError('Invalid reference causal convolution')
    padding = (weight.shape[1]-1)*dilation
    value = F.conv1d(F.pad(x.float().T[None],(padding,0)),weight.float()[:,None],
                     groups=x.shape[1],dilation=dilation)[0].T
    if round_before_silu:value = value.to(x.dtype)
    return F.silu(value).to(x.dtype)


def ple(x, features, weight, *, dilation=3):
    key = F.linear(features,weight['key']).reshape_as(x)
    value = F.linear(features,weight['value'])
    key = norm(key,weight['norm_key'].reshape(x.shape[1:]))
    query = norm(x,weight['norm_query'].reshape(x.shape[1:]))
    gate = (key*query).sum(-1,keepdim=True)/math.sqrt(x.shape[-1])
    gate = gate.abs().clamp_min(1e-6).sqrt()*gate.sign()
    gated = torch.sigmoid(gate)*value[:,None,:]
    normalized = norm(gated,weight['norm_conv'].reshape(x.shape[1:]))
    convolved = causal_conv(normalized.flatten(1),weight['conv'],dilation=dilation,round_before_silu=True)
    return x+(gated.flatten(1)+convolved).reshape_as(x)


def delta_rule(query, key, value, decay, beta, *, state=None, normalize=True):
    """Sequential FP32 delta recurrence, including grouped query/key heads."""
    if query.shape != key.shape or query.ndim != 3 or value.ndim != 3:
        raise ValueError('Invalid delta input geometry')
    m,hk,dk = key.shape;mv,hv,dv = value.shape
    if m != mv or hv%hk or decay.shape != (m,hv) or beta.shape != (m,hv):
        raise ValueError('Invalid grouped delta geometry')
    q,k,v = query.float(),key.float(),value.float()
    if normalize:
        q = q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
        k = k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    q = q.repeat_interleave(hv//hk,1)/math.sqrt(dk)
    k = k.repeat_interleave(hv//hk,1)
    current = torch.zeros((hv,dk,dv),dtype=torch.float32,device=query.device) if state is None else state.float().clone()
    if current.shape != (hv,dk,dv):raise ValueError('Invalid initial delta state')
    output = []
    for i in range(m):
        current = current*decay[i].float().exp()[:,None,None]
        prediction = torch.einsum('hkv,hk->hv',current,k[i])
        update = beta[i].float()[:,None]*(v[i]-prediction)
        current = current+k[i][:,:,None]*update[:,None,:]
        output.append(torch.einsum('hkv,hk->hv',current,q[i]))
    return torch.stack(output).to(query.dtype),current


def neox(x, rotary, theta=1e7):
    """Original HF half-width text RoPE; channels after rotary are unchanged."""
    if rotary%2 or rotary < 2 or rotary > x.shape[-1]:raise ValueError('Invalid rotary width')
    positions = torch.arange(x.shape[0],device=x.device,dtype=torch.float32)
    frequencies = theta**(-torch.arange(0,rotary,2,device=x.device,dtype=torch.float32)/rotary)
    angle = positions[:,None,None]*frequencies[None,None,:]
    first,second = x[...,:rotary].float().chunk(2,-1)
    result = x.clone()
    result[...,:rotary] = torch.cat((first*angle.cos()-second*angle.sin(),
                                   second*angle.cos()+first*angle.sin()),-1).to(x.dtype)
    return result


def attention(qgate, key, value, qweight, kweight, *, rotary, theta=1e7):
    """Exact causal short QSA; each invocation is an independent request."""
    m,heads,_,dim = qgate.shape;kh = key.shape[1]
    q = neox(norm(qgate[:,:,0],qweight),rotary,theta).float()
    k = neox(norm(key,kweight),rotary,theta).repeat_interleave(heads//kh,1).float()
    v = value.repeat_interleave(heads//kh,1).float()
    scores = torch.einsum('mhd,shd->hms',q,k)/math.sqrt(dim)
    scores.masked_fill_(torch.ones((m,m),device=q.device,dtype=torch.bool).triu(1)[None],-torch.inf)
    output = torch.einsum('hms,shd->mhd',scores.softmax(-1),v).to(qgate.dtype)
    return output*torch.sigmoid(qgate[:,:,1])
