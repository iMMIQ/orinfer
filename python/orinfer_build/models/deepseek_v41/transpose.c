/* SPDX-License-Identifier: LGPL-3.0-or-later
 * Offline packed-byte transpose. INT4 nibbles are preserved verbatim.
 */
#include <stddef.h>
#include <stdint.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

void orinfer_transpose_u8(const uint8_t *src, uint8_t *dst,
                          size_t rows, size_t columns, size_t dst_stride) {
    for (size_t rb = 0; rb < rows; rb += 64) {
        const size_t re = rb + 64 < rows ? rb + 64 : rows;
        for (size_t cb = 0; cb < columns; cb += 64) {
            const size_t ce = cb + 64 < columns ? cb + 64 : columns;
            for (size_t r = rb; r < re; r += 8) {
                for (size_t c = cb; c < ce; c += 8) {
#if defined(__aarch64__)
                    if (r + 8 <= re && c + 8 <= ce) {
                        uint8x8x2_t a = vtrn_u8(vld1_u8(src + r * columns + c),
                                               vld1_u8(src + (r + 1) * columns + c));
                        uint8x8x2_t b = vtrn_u8(vld1_u8(src + (r + 2) * columns + c),
                                               vld1_u8(src + (r + 3) * columns + c));
                        uint8x8x2_t d = vtrn_u8(vld1_u8(src + (r + 4) * columns + c),
                                               vld1_u8(src + (r + 5) * columns + c));
                        uint8x8x2_t e = vtrn_u8(vld1_u8(src + (r + 6) * columns + c),
                                               vld1_u8(src + (r + 7) * columns + c));
                        uint16x4x2_t f = vtrn_u16(vreinterpret_u16_u8(a.val[0]), vreinterpret_u16_u8(b.val[0]));
                        uint16x4x2_t g = vtrn_u16(vreinterpret_u16_u8(a.val[1]), vreinterpret_u16_u8(b.val[1]));
                        uint16x4x2_t h = vtrn_u16(vreinterpret_u16_u8(d.val[0]), vreinterpret_u16_u8(e.val[0]));
                        uint16x4x2_t i = vtrn_u16(vreinterpret_u16_u8(d.val[1]), vreinterpret_u16_u8(e.val[1]));
                        uint32x2x2_t j = vtrn_u32(vreinterpret_u32_u16(f.val[0]), vreinterpret_u32_u16(h.val[0]));
                        uint32x2x2_t k = vtrn_u32(vreinterpret_u32_u16(g.val[0]), vreinterpret_u32_u16(i.val[0]));
                        uint32x2x2_t l = vtrn_u32(vreinterpret_u32_u16(f.val[1]), vreinterpret_u32_u16(h.val[1]));
                        uint32x2x2_t m = vtrn_u32(vreinterpret_u32_u16(g.val[1]), vreinterpret_u32_u16(i.val[1]));
                        vst1_u8(dst + c * dst_stride + r, vreinterpret_u8_u32(j.val[0]));
                        vst1_u8(dst + (c + 1) * dst_stride + r, vreinterpret_u8_u32(k.val[0]));
                        vst1_u8(dst + (c + 2) * dst_stride + r, vreinterpret_u8_u32(l.val[0]));
                        vst1_u8(dst + (c + 3) * dst_stride + r, vreinterpret_u8_u32(m.val[0]));
                        vst1_u8(dst + (c + 4) * dst_stride + r, vreinterpret_u8_u32(j.val[1]));
                        vst1_u8(dst + (c + 5) * dst_stride + r, vreinterpret_u8_u32(k.val[1]));
                        vst1_u8(dst + (c + 6) * dst_stride + r, vreinterpret_u8_u32(l.val[1]));
                        vst1_u8(dst + (c + 7) * dst_stride + r, vreinterpret_u8_u32(m.val[1]));
                        continue;
                    }
#endif
                    for (size_t x = r; x < re && x < r + 8; ++x)
                        for (size_t y = c; y < ce && y < c + 8; ++y)
                            dst[y * dst_stride + x] = src[x * columns + y];
                }
            }
        }
    }
}
