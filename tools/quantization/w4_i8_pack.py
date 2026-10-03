"""Lossless F16-fragment W4 to I8-fragment W4 permutation, one packed copy.

Both layouts contain [N/64, K/128, 128 lanes, 8 uint32 words]. I8 words
hold two four-code quartets for the K halves of one N8 fragment; adjacent
words hold the two N8 fragments. No code, scale or zero-point changes.
"""
import numpy as np

LAYOUT = 'u4_warp_n64_k128_mma_i8'


def pack_array(source, *, verify=True):
    assert source.dtype in (np.dtype('uint32'), np.dtype('int32')) and np.little_endian
    assert source.ndim == 4 and source.shape[2:] == (128, 8)
    assert source.shape[0] > 0 and source.shape[1] > 0
    source = source.view(np.uint32)
    result = np.empty_like(source)
    lane = np.arange(128, dtype=np.uint32)
    first = (lane//4)*4+(lane%2)*2
    second = first+1
    for ki in range(4):
        for ni in range(2):
            shift = ni*16+((lane%4)//2)*8
            word = np.zeros(source.shape[:-1], dtype=np.uint32)
            for half in range(2):
                lo = (source[:, :, first, ki*2+half] >> shift) & 255
                hi = (source[:, :, second, ki*2+half] >> shift) & 255
                word |= (lo | (hi << 8)) << (half*16)
            result[:, :, :, ki*2+ni] = word
    if verify:
        assert np.array_equal(unpack_array(result), source)
    assert result.nbytes == source.nbytes
    return result


def unpack_array(source):
    """Inverse to the exact original packed words, including all nibble bits."""
    assert source.dtype in (np.dtype('uint32'), np.dtype('int32')) and np.little_endian
    assert source.ndim == 4 and source.shape[2:] == (128, 8)
    source = source.view(np.uint32)
    lane = np.arange(128, dtype=np.uint32)
    tid = lane%4
    result = np.empty_like(source)
    for original_word in range(8):
        word = np.zeros(source.shape[:-1], dtype=np.uint32)
        for ni in range(2):
            for high_k in range(2):
                target_lane = lane//4*4+tid//2+high_k*2
                encoded = source[:, :, target_lane, original_word//2*2+ni]
                for pair in range(2):
                    shift = (original_word%2)*16+(tid%2)*8+pair*4
                    code = (encoded >> shift) & 15
                    word |= code << ((ni*4+high_k*2+pair)*4)
        result[:, :, :, original_word] = word
    return result
