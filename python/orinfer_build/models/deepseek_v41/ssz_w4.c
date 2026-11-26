/* Per-output-channel SSZ signed INT4 [-8, 7] quantizer with LS scale refinement. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline int8_t clamp_i4(float x) {
    float r = nearbyintf(x);
    if (r < -8.0f) return -8;
    if (r > 7.0f) return 7;
    return (int8_t)r;
}

static inline double eval_pass(const float *row, size_t k, float inv_s, double *out_s) {
    double num = 0.0;
    double den = 0.0;
    for (size_t j = 0; j < k; ++j) {
        int8_t q = clamp_i4(row[j] * inv_s);
        double qd = (double)q;
        num += (double)row[j] * qd;
        den += qd * qd;
    }
    if (den < 1e-30) {
        *out_s = 1e-30;
        return 0.0;
    }
    *out_s = num / den;
    if (*out_s < 1e-30) *out_s = 1e-30;
    return (num * num) / den;
}

void orinfer_ssz_w4_rows(
    const float *w,
    size_t n,
    size_t k,
    int8_t *q_out,
    float *s_out,
    double *err_out,
    double *energy_out
) {
    double total_err = 0.0;
    double total_energy = 0.0;

#pragma omp parallel for reduction(+:total_err,total_energy) schedule(static)
    for (size_t r = 0; r < n; ++r) {
        const float *row = w + r * k;
        int8_t *qrow = q_out + r * k;

        float amax = 0.0f;
        double norm2 = 0.0;
        for (size_t j = 0; j < k; ++j) {
            float v = row[j];
            float a = fabsf(v);
            if (a > amax) amax = a;
            norm2 += (double)v * (double)v;
        }
        total_energy += norm2;

        if (amax <= 1e-30f) {
            s_out[r] = 1e-30f;
            for (size_t j = 0; j < k; ++j) qrow[j] = 0;
            continue;
        }

        double best_score = -1.0;
        float best_alpha = 0.80f;
        double best_s = (double)amax / 7.5;

        /* Coarse grid: 11 clipping ratios from 0.50 to 1.00. */
        for (int ai = 0; ai <= 10; ++ai) {
            float alpha = 0.50f + 0.05f * (float)ai;
            float inv_s = 7.5f / (amax * alpha);
            double s1 = 0.0;
            double sc = eval_pass(row, k, inv_s, &s1);
            if (sc > best_score) {
                best_score = sc;
                best_alpha = alpha;
                best_s = s1;
            }
        }

        /* Fine grid around best_alpha (+/- 0.03, step 0.01) + coordinate refinement. */
        for (int di = -3; di <= 3; ++di) {
            float alpha = best_alpha + 0.01f * (float)di;
            if (alpha < 0.45f || alpha > 1.02f) continue;
            float inv_s = 7.5f / (amax * alpha);
            double s1 = 0.0;
            eval_pass(row, k, inv_s, &s1);
            double s2 = 0.0;
            double sc2 = eval_pass(row, k, (float)(1.0 / s1), &s2);
            if (sc2 > best_score) {
                best_score = sc2;
                best_s = s2;
            }
        }

        double s_final = best_s;
        double sc_final = eval_pass(row, k, (float)(1.0 / best_s), &s_final);
        if (sc_final >= best_score) {
            best_score = sc_final;
            best_s = s_final;
        }

        float sf = (float)best_s;
        if (sf < 1e-30f) sf = 1e-30f;
        float inv_sf = 1.0f / sf;
        double num = 0.0;
        double den = 0.0;
        for (size_t j = 0; j < k; ++j) {
            int8_t q = clamp_i4(row[j] * inv_sf);
            qrow[j] = q;
            double qd = (double)q;
            num += (double)row[j] * qd;
            den += qd * qd;
        }
        if (den >= 1e-30) {
            sf = (float)(num / den);
            if (sf < 1e-30f) sf = 1e-30f;
        }
        s_out[r] = sf;

        double err = 0.0;
        for (size_t j = 0; j < k; ++j) {
            double d = (double)row[j] - (double)sf * (double)qrow[j];
            err += d * d;
        }
        total_err += err;
    }

    *err_out = total_err;
    *energy_out = total_energy;
}
