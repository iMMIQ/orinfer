"""Prefill buckets shared by offline Flash Next request runners."""


def chunks(tokens, maximum):
    """Yield exact slices, using maximum, eight and one without padding state."""
    if type(maximum) is not int or not 1 <= maximum <= 512:
        raise ValueError('Flash Next chunk must be in 1..512')
    if not tokens:
        raise ValueError('Prefill requires input tokens')
    cursor = 0
    while cursor < len(tokens):
        remaining = len(tokens) - cursor
        size = maximum if remaining >= maximum else 8 if maximum >= 8 and remaining >= 8 else 1
        yield tokens[cursor:cursor + size]
        cursor += size
