# E8P Quantization Technical Note

Orinfer uses **E8P lattice vector quantization** with input Hadamard rotation for both **2-bit MoE experts (E8P Q2)** and **4-bit two-stage dense/projection weights (E8P Q4)**. All compressed weights stay single-resident in memory and decode on the fly inside Tensor Core tiles.

---

## 1. Codebook Goal & Input Rotation

Encoding 8 weights in 16 bits provides `2^16 = 65,536` reconstruction vectors. Instead of an 8-dimensional Cartesian grid of independent 2-bit scalars, E8P arranges reconstruction points on a shifted **E8 lattice** and exploits lattice symmetry so the lookup table shrinks from 512 KiB down to **1–2 KiB**.

### Signed Block-128 Hadamard Rotation

Input channels are grouped into blocks of 128, multiplied by fixed random signs `D`, and transformed by a normalized block-diagonal Hadamard matrix `H`:

$$
R = HD, \qquad \tilde{x} = Rx, \qquad \tilde{W} = WR^T.
$$

Because `R` is orthogonal, `Wx = W̃x̃`. Weights are rotated offline (spreading outlier magnitudes across coordinates), and activations are rotated online in 7 butterfly stages before A8 quantization. Each output row shares one FP16 scale.

---

## 2. Lattice Construction & 256 Basis Patterns

Following [QuIP# E8P](https://arxiv.org/html/2402.04396v2#S4.SS2), start from the half-integer lattice:

$$
\widehat{D}_8 = \left\{ t \in (\mathbb{Z} + \tfrac{1}{2})^8 : \sum_j t_j \in 2\mathbb{Z} \right\}, \qquad
(\widehat{D}_8 - \tfrac{1}{4}\mathbf{1}) \cup (\widehat{D}_8 + \tfrac{1}{4}\mathbf{1}) = E_8 + \tfrac{1}{4}\mathbf{1}.
$$

Scaling coordinates by 4 converts this to pure integers (the factor of 4 is absorbed into the row scale):

$$
B = 4t, \qquad q = B \pm \mathbf{1}.
$$

Every basis coordinate `B_j` is **even**, so shifting by `+1` or `-1` produces **odd** integer coordinates `q_j`.

### Enumerating the 256 Basis Patterns

Magnitude patterns `A_j ∈ {2, 6, 10}` with `Σ A_j² ≤ 160` satisfy `n_6 + 3 n_10 ≤ 4`:

| Pattern Class | Count |
| --- | ---: |
| No `10`, at most four `6`s | `C(8,0) + … + C(8,4) = 163` |
| One `10`, at most one `6` | `8 + 8 × 7 = 64` |
| **Inner patterns** | **227** |
| **Outer padding** (five `6`s, three `2`s; `‖A‖² = 192`, selected from `C(8,5) = 56`) | **29** |
| **Total ("Padded" E8P)** | **256** |

If `Σ(A/4)` is odd, the last coordinate is negated so the canonical basis representative `B` has even `Σ(B/4)`. Flipping any **even** number of coordinate signs preserves validity (`2^7 = 128` sign patterns). Combined with 2 shifts (`±1`):

$$
256 \text{ bases} \times 128 \text{ sign flips} \times 2 \text{ shifts} = 65,536 \text{ codewords (16 bits)}.
$$

---

## 3. U16 Bit Layout & Nearest-Neighbor Encoding

- **High byte (`b = code >> 8`)**: 8-bit basis index (`0..255`).
- **Low byte (`u = m XOR p`)**: Packs the 7-bit even-parity sign-flip mask `m` and 1-bit shift `p` (`0 → +1`, `1 → -1`). Because `popcount(m)` is even, `p = popcount(u) & 1` and `m = u XOR p`.
- **Permutation**: `P = [0, 4, 1, 5, 2, 6, 3, 7]` maps coordinates to mask bits without changing parity.

```text
b = code >> 8
u = code & 0xff
p = popcount(u) & 1
m = u XOR p
q[j] = B[b, j] * (1 - 2 * ((m >> P[j]) & 1)) + (1 - 2 * p)
```

### Nearest-Neighbor Search & Scale Fitting

Given normalized `z = w̃ / s`, for each of the `256 × 2 = 512` `(basis, shift)` pairs:
1. Subtract shift: `y = z - (1 - 2p)`.
2. Align signs to `y`; if the sign-flip count relative to `B` is odd, flip the coordinate minimizing `4 |y_j| |B_j|`.
3. Pick the global minimum squared distance across all 512 pairs (1 warp per 8D vector on GPU).
4. Update the FP16 row scale via least squares `s = (Σ w̃_k q_k) / (Σ q_k²)` and alternate until convergence.

---

## 4. Storage Layout & Full-A8 INT4 Tensor Core Execution

For an `N × K` weight matrix:

| Tensor | Logical Shape | Description |
| --- | --- | --- |
| U16 codewords | `[N, K/8]` | 1 codeword per 8 input weights |
| FP16 scale | `[N]` | 1 scale per output row |
| INT8 rotation signs | `[K]` | Input Hadamard sign vector |
| INT8 basis / U32 selector | `[256, 8]` / `[256]` | 2 KiB basis or 1 KiB `PRMT` selector table |

At build time, codewords are losslessly repacked into K-major tiles (`[E, K/128, N, 16]` or `[E, K/64, N, 8]`).

### Lossless INT4 Bit-Plane Decomposition

Because every `q ∈ [-11, 11]` is odd, `q = 2v + 1` with signed INT4 `v = (q - 1) / 2 ∈ [-6, 5]`. Splitting signed INT8 activation `a` into unsigned low nibble `a_lo = a & 0x0f` and signed high nibble `a_hi = a >> 4` gives `a = a_lo + 16 a_hi`, so:

$$
\sum q a = 2 \left( \sum v a_{lo} + 16 \sum v a_{hi} \right) + \sum a.
$$

Two INT4 Tensor Core MMAs (`S4×U4` and `S4×S4`) plus the reused activation row sum recover the **exact INT32 dot product** at full A8 precision, followed by `(FP32(dot) * FP32(weight_scale)) * FP32(token_scale)`.

---

## 5. Comparison with `Q2_K` / `Q2_0` (Flash Next)

| Dimension | E8P Q2 | `Q2_K` (llama.cpp) |
| --- | --- | --- |
| Representation | Joint 8D lattice vector per 8 weights | Independent 2-bit scalar per weight |
| Effective bitwidth | ~2.02 bits/weight (incl. row scale, signs, LUT) | 2.625 bits/weight (84 B / 256 weights) |
| Scaling | Single FP16 row scale after full-K INT32 accumulation | Per-16-weight scale & min with local rescaling |

### Flash Next Quality & Speed Snapshot (Jetson AGX Orin 64GB, 1300.5 MHz)

- **Weight L2 error (`‖Ŵ - W‖₂ / ‖W‖₂` vs. BF16)**: E8P Q2 gate/up = **0.3026** vs. GSQ-RCO `Q2_0` = **0.5360** (10-expert avg).
- **Whole-model 48-layer teacher-forced probabilities (64 positions, seed `20261002`)**: **62/64 (96.88%)** top-1 match, **64/64** inside BF16 top-3, non-thinking ΔNLL **+0.0224 nats/token**.
- **Single-stream speed (512 / 8192 input)**: Orinfer E8P Q2 + Q4/W8 (MTP ≤ 3) achieves **925.6 / 858.1 prefill TPS** and **56.35 / 48.03 decode TPS**, vs. `llama.cpp` `Q2_0` (MTP off) at **269.0 / 251.8 prefill TPS** and **16.26 / 15.93 decode TPS**.

---

## 6. Two-Stage E8P Q4 Extension

Dense models (`Qwen3.8-27B`, `Qwen3.5-4B`, `Index-Translate-2B/9B/35B`, `Qwen3-ASR-1.7B`, `Nemotron-3.5-Embed-1B`, `Qwen3-Reranker-0.6B`) encode coarse and residual codewords using the **same** E8P table and fixed integer coefficients `[4, 1]`, packed as one `U32` per 8 weights (`4 bits/weight`):

$$
\hat{\tilde{w}} = s (4 q_0 + q_1) = s \left( 2 (4 v_0 + v_1) + 5 \right).
$$

Offline encoding uses **beam-8 search** over coarse candidates jointly with residual error and alternating least-squares FP16 scale refinement. The direct INT4 path executes two bit-plane MMAs per stage plus a shared `5 × Σa` correction, recovering exact INT32 dot products.
