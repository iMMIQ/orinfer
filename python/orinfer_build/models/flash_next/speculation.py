"""CPU-only acceptance and workload guards for Flash's greedy MTP recipe."""

DEFAULT_DRAFTS = 3


def greedy_commit(drafts, target):
    if len(target) != len(drafts) + 1 or not target:
        raise ValueError("Verification must include the target bonus token")
    accepted = 0
    while accepted < len(drafts) and drafts[accepted] == target[accepted]:
        accepted += 1
    return list(drafts[:accepted]) + [target[accepted]]


def clip_outputs(tokens, eos, remaining):
    if type(remaining) is not int or remaining < 1 or not tokens:
        raise ValueError("Positive output budget required")
    result = []
    reason = None
    for token in tokens[:remaining]:
        result.append(token)
        if token in eos:
            reason = "stop"
            break
    if reason is None and len(result) == remaining:
        reason = "length"
    return result, reason


def verification_size(drafts, remaining, capacity):
    if type(drafts) is not int or not 1 <= drafts <= 7:
        raise ValueError("Draft depth must be in 1..7")
    if type(remaining) is not int or remaining < 1 or type(capacity) is not int or capacity < 1:
        raise ValueError("No remaining output or context capacity")
    available = min(remaining, capacity)
    if available >= drafts + 1:
        return drafts + 1
    # Reuse power-of-two tails, avoiding a recurrent-prefix buffer for every
    # possible remaining length. All processed inputs are real causal tokens.
    return 1 << (available.bit_length() - 1)
