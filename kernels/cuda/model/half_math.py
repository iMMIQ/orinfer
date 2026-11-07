"""Stream sigmoid with the existing FP32-to-FP16 rounding on SM87.

The SFU path matches the original exp/divide expression for every finite FP16
input except two midpoint cases. Their original rounded outputs are explicit.
The GPU operator check exhausts this domain; this is not an FP32 approximation
interface. BF16 streams keep the reference expression.
"""

import tilelang.language as T

SOURCE = r"""
__device__ __forceinline__ float orinfer_sigmoid_half(float u) {
  if (u == 0.0068359375f) return 0.501953125f;
  if (u == -0.00634765625f) return 0.49853515625f;
  float x = -u * 1.4426950408889634f, e, r;
  asm volatile("ex2.approx.ftz.f32 %0,%1;" : "=f"(e) : "f"(x));
  float d = 1.f + e;
  asm volatile("rcp.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(d));
  return r;
}
"""


def stream_sigmoid(value, dtype):
    """value must already be rounded to the specified stream dtype."""
    if dtype == "float16":
        return T.cast(T.call_pure_extern("float32", "orinfer_sigmoid_half", value), T.float16)
    if dtype != "bfloat16":
        raise ValueError("Sigmoid requires an FP16 or BF16 stream")
    return T.cast(1 / (1 + T.exp(-value)), T.bfloat16)
