"""FP32 mathematical Gated Delta Rule references, not production kernels.

Q/K are L2-normalized. q_scale defaults to1 for already-scaled Q, or pass
1/sqrt(DK) for unscaled normalized Q. Never apply the model scale twice.
Logical nonchunk layout: [B,H,T,D]. V/g/beta have HV heads, Q/K HK heads.
Chunk layout: [B,H,C,BT,D], gates [B,HV,C,BT], state [B,HV,DK,DV].
Reference expands Q/K for readability only; production maps hv//(HV/HK).
Intermediate formulas define shared op11..16 semantics; model rounding is
separately locked and compared. All gates are natural logarithmic decays.
"""
import torch


def expand_heads(tensor, heads):
    assert heads % tensor.shape[1] == 0
    return tensor.repeat_interleave(heads // tensor.shape[1], dim=1).float()


def recurrent(q, k, v, g, beta, state, q_scale=1.0):
    q, k = expand_heads(q, v.shape[1]), expand_heads(k, v.shape[1])
    q = q * q_scale
    state = state.float().clone()
    output = torch.empty_like(v, dtype=torch.float32)
    for token in range(v.shape[2]):
        qt, kt, vt = q[:, :, token], k[:, :, token], v[:, :, token].float()
        decayed = state * g[:, :, token].float().exp()[..., None, None]
        predicted = torch.einsum("bhk,bhkv->bhv", kt, decayed)
        delta = beta[:, :, token].float()[..., None] * (vt - predicted)
        state = decayed + kt[..., :, None] * delta[..., None, :]
        output[:, :, token] = torch.einsum("bhk,bhkv->bhv", qt, state)
    return output, state


def chunk_matrices(q, k, cumulative_g, beta, q_scale=1.0):
    """L = I + strict_lower(beta_i * <k_i,k_j> * exp(G_i-G_j))."""
    heads, bt = beta.shape[1], beta.shape[-1]
    q, k = expand_heads(q, heads), expand_heads(k, heads)
    q = q * q_scale
    gc = cumulative_g.float()
    diff = gc[..., :, None] - gc[..., None, :]
    # Upper-triangle positive exponents must never overflow before masking.
    decay = diff.clamp(max=0).exp()
    kk = (k @ k.transpose(-1, -2)) * beta.float()[..., :, None] * decay
    lower = torch.tril(kk, diagonal=-1)
    eye = torch.eye(bt, dtype=torch.float32, device=k.device)
    causal_qk = torch.tril((q @ k.transpose(-1, -2)) * decay)
    return lower + eye, causal_qk


def triangle_transform(system):
    bt = system.shape[-1]
    eye = torch.eye(bt, dtype=torch.float32, device=system.device)
    identity = eye.expand_as(system)
    return torch.linalg.solve_triangular(system.float(), identity, upper=False,
                                         unitriangular=True)


def wy(transform, k, v, cumulative_g, beta):
    k = expand_heads(k, v.shape[1])
    bk = beta.float()[..., None] * k * cumulative_g.float().exp()[..., None]
    bv = beta.float()[..., None] * v.float()
    return transform.float() @ bk, transform.float() @ bv


def chunk_scan(k, cumulative_g, w, u, initial_state):
    """Return entering states, residuals R=U-W*S_enter, and final state."""
    k = expand_heads(k, u.shape[1])
    state = initial_state.float().clone()
    states, residuals = [], []
    for chunk in range(u.shape[2]):
        states.append(state.clone())
        r = u[:, :, chunk].float() - w[:, :, chunk].float() @ state
        gc = cumulative_g[:, :, chunk].float()
        last = gc[..., -1]
        kd = k[:, :, chunk] * (last[..., None] - gc).exp()[..., None]
        state = last.exp()[..., None, None] * state + kd.transpose(-1, -2) @ r
        residuals.append(r)
    return torch.stack(states, dim=2), torch.stack(residuals, dim=2), state


def chunk_output(q, cumulative_g, causal_qk, entering_states, residuals, q_scale=1.0):
    q = expand_heads(q, residuals.shape[1])
    q = q * q_scale
    qs = q * cumulative_g.float().exp()[..., None]
    return qs @ entering_states.float() + causal_qk.float() @ residuals.float()


def chunked(q, k, v, g, beta, state, q_scale=1.0):
    gc = g.float().cumsum(dim=-1)
    system, qk = chunk_matrices(q, k, gc, beta, q_scale=q_scale)
    transform = triangle_transform(system)
    w, u = wy(transform, k, v, gc, beta)
    states, r, final = chunk_scan(k, gc, w, u, state)
    output = chunk_output(q, gc, qk, states, r, q_scale=q_scale)
    return output, final, {"gc": gc, "system": system, "qk": qk,
                          "transform": transform, "w": w, "u": u,
                          "entering_states": states, "residuals": r}
