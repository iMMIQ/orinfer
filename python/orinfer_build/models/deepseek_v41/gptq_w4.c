/* Fast C11 OpenMP kernels for Act-Order Blocked GPTQ + Hessian-Weighted SSZ INT4 [-8, 7].
 *
 * Pair `orinfer_gptq_diag_block` (128-col triangular elimination in C11 OpenMP) with BLAS/NPU GEMM
 * for the off-diagonal block update `W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]`.
 */

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float clamp_i4_f(float x) {
    float r = nearbyintf(x);
    if (r < -8.0f) return -8.0f;
    if (r > 7.0f) return 7.0f;
    return r;
}

void orinfer_hdiag_ssz_scale(
    const float *w,
    const float *h_diag,
    size_t n,
    size_t k,
    int n_threads,
    float *s_out
) {
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t r = 0; r < n; ++r) {
        const float *row = w + r * k;
        float amax = 0.0f;
        for (size_t j = 0; j < k; ++j) {
            float a = fabsf(row[j]);
            if (a > amax) amax = a;
        }
        if (amax <= 1e-30f) {
            s_out[r] = 1e-30f;
            continue;
        }
        double best_score = -1.0;
        float best_alpha = 0.80f;
        double best_s = (double)amax / 7.5;

        for (int ai = 0; ai <= 10; ++ai) {
            float alpha = 0.50f + 0.05f * (float)ai;
            float inv_s = 7.5f / (amax * alpha);
            double num = 0.0, den = 0.0;
            for (size_t j = 0; j < k; ++j) {
                double q = (double)clamp_i4_f(row[j] * inv_s);
                double hd = (double)h_diag[j];
                num += hd * (double)row[j] * q;
                den += hd * q * q;
            }
            if (den >= 1e-30) {
                double sc = (num * num) / den;
                if (sc > best_score) {
                    best_score = sc;
                    best_alpha = alpha;
                    best_s = num / den;
                }
            }
        }

        for (int di = -3; di <= 3; ++di) {
            float alpha = best_alpha + 0.01f * (float)di;
            if (alpha < 0.45f || alpha > 1.02f) continue;
            float inv_s = 7.5f / (amax * alpha);
            double num = 0.0, den = 0.0;
            for (size_t j = 0; j < k; ++j) {
                double q = (double)clamp_i4_f(row[j] * inv_s);
                double hd = (double)h_diag[j];
                num += hd * (double)row[j] * q;
                den += hd * q * q;
            }
            if (den < 1e-30) continue;
            double s1 = num / den;
            if (s1 < 1e-30) s1 = 1e-30;
            float inv_s1 = (float)(1.0 / s1);
            double num2 = 0.0, den2 = 0.0;
            for (size_t j = 0; j < k; ++j) {
                double q = (double)clamp_i4_f(row[j] * inv_s1);
                double hd = (double)h_diag[j];
                num2 += hd * (double)row[j] * q;
                den2 += hd * q * q;
            }
            if (den2 >= 1e-30) {
                double sc2 = (num2 * num2) / den2;
                if (sc2 > best_score) {
                    best_score = sc2;
                    best_s = num2 / den2;
                }
            }
        }
        if (best_s < 1e-30) best_s = 1e-30;
        s_out[r] = (float)best_s;
    }
}

/* Process one contiguous block of B columns [N, B] in-place:
 *   w_blk   [N, B] float32 (modified in-place)
 *   hinv_b  [B, B] float32 (upper-triangular diagonal block of Hinv_ch)
 *   scales  [N]    float32
 *   q_blk   [N, B] int8    (output quantized codes)
 *   err_blk [N, B] float32 (output normalized errors for GEMM update)
 */
void orinfer_gptq_diag_block(
    float *w_blk,
    const float *hinv_b,
    const float *scales,
    size_t n,
    size_t b,
    int n_threads,
    int8_t *q_blk,
    float *err_blk
) {
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t r = 0; r < n; ++r) {
        float *wr = w_blk + r * b;
        int8_t *qr = q_blk + r * b;
        float *er = err_blk + r * b;
        float s = scales[r];
        if (s < 1e-30f) s = 1e-30f;
        float inv_s = 1.0f / s;
        for (size_t j = 0; j < b; ++j) {
            float wv = wr[j];
            float qf = clamp_i4_f(wv * inv_s);
            qr[j] = (int8_t)qf;
            const float *hrow = hinv_b + j * b;
            float e = (wv - qf * s) / hrow[j];
            er[j] = e;
            for (size_t c = j + 1; c < b; ++c) {
                wr[c] -= e * hrow[c];
            }
        }
    }
}

/* Sparse Gauss-Seidel Coordinate Descent pass on Q in H-order, parallel over rows N. */
void orinfer_gptq_cd_pass(
    const float *h_mat,
    const float *h_diag,
    const int32_t *perm,
    const float *scales,
    size_t n,
    size_t k,
    int n_threads,
    int8_t *q_io,
    float *r_io
) {
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t row = 0; row < n; ++row) {
        int8_t *q = q_io + row * k;
        float *r = r_io + row * k;
        float s = scales[row];
        if (s < 1e-30f) continue;
        float inv_s = 1.0f / s;
        for (size_t pi = 0; pi < k; ++pi) {
            size_t j = (size_t)perm[pi];
            float hd = h_diag[j];
            if (hd <= 1e-30f) continue;
            float q_old = (float)q[j];
            float q_new = clamp_i4_f(q_old + (r[j] * inv_s) / hd);
            float dq = q_new - q_old;
            if (dq != 0.0f) {
                q[j] = (int8_t)q_new;
                float scale_dq = s * dq;
                const float *hcol = h_mat + j * k;
                for (size_t c = 0; c < k; ++c) {
                    r[c] -= scale_dq * hcol[c];
                }
            }
        }
    }
}
