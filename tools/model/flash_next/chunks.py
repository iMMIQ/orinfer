"""Prefill buckets shared by offline Flash Next request runners."""


def chunks(tokens, maximum):
    """Yield full chunks and one exact tail, without padding private state."""
    if type(maximum) is not int or not 1 <= maximum <= 4096:
        raise ValueError('Flash Next chunk must be in 1..4096')
    if not tokens:
        raise ValueError('Prefill requires input tokens')
    cursor = 0
    while cursor < len(tokens):
        size = min(maximum, len(tokens) - cursor)
        yield tokens[cursor:cursor + size]
        cursor += size


def index_capacity(end, capacity):
    """Bound index workspace by the live prefix, retaining power-of-two plans."""
    if any(type(x) is not int for x in (end, capacity)) or not 1 <= end <= capacity <= 262144:
        raise ValueError('Invalid live index prefix')
    return min(capacity, 1 << (max(512, end) - 1).bit_length())
