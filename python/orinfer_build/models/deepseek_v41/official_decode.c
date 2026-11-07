/* SPDX-License-Identifier: LGPL-3.0-or-later
 * Offline FP4 decoding; one FP32 scale per output row and 32 inputs.
 */
#include <stddef.h>
#include <stdint.h>

void orinfer_official_fp4(const uint8_t *packed, const float *scales,
                         float *out, size_t rows, size_t columns) {
    static const float values[16] = {
        0, .5f, 1, 1.5f, 2, 3, 4, 6,
        -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6
    };
    const size_t groups = columns / 32;
    #pragma omp parallel for num_threads(8) if(rows * columns >= 1048576)
    for (size_t row = 0; row < rows; ++row) {
        for (size_t group = 0; group < groups; ++group) {
            const float scale = scales[row * groups + group];
            const uint8_t *source = packed + row * (columns / 2) + group * 16;
            float *destination = out + row * columns + group * 32;
            for (size_t i = 0; i < 16; ++i) {
                const uint8_t code = source[i];
                destination[2 * i] = values[code & 15] * scale;
                destination[2 * i + 1] = values[code >> 4] * scale;
            }
        }
    }
}
