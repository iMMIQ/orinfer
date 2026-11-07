"""Bounded AIV launch lowering for the pinned A2 compiler's paired Vector frame."""

import re


def lower_independent_vector(source, *, grid_limit=None):
    """Keep the original arithmetic and pair indices; omit the unused Cube task.

    The pinned code generator emits MIX_AIC_1_2 even for independent UB-only
    kernels. Its Vector block index already spans twice the logical grid.
    AIV-only launches therefore double the host grid and derive the pair lane
    from that physical index. Unrecognized or coupled programs keep their
    original launch. A dynamic grid requires the logical bound established by
    the shared TIR grid pass. No cross-block synchronization is introduced.
    """
    mixed = "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)"
    if source.count(mixed) != 1:
        return source
    if any(x in source for x in ("ASCEND_IS_AIC", "CrossCore", "SyncAll", "GlobalBarrier")):
        return source
    if len(re.findall(r"\bfftsAddr\b", source)) != 4:
        return source
    if source.count("AscendC::GetSubBlockIdx()") != 1:
        return source
    block = re.search(r"auto (\w+) = AscendC::GetBlockIdx\(\);", source)
    if block is None or f"{block[1]} = {block[1]} / 2;" not in source:
        return source
    pattern = r"(\w+)<<<([^,\n]+), nullptr, stream>>>"
    launches = list(re.finditer(pattern, source))
    if len(launches) != 1:
        return source
    grid = launches[0][2].strip()
    if grid.isdecimal():
        if not 0 < int(grid) <= 32767:
            return source
        physical = str(int(grid) * 2)
    else:
        # Only the shared m scalar may appear in the compiler's integer grid.
        # Its producer has already clamped the logical grid to grid_limit.
        if (
            type(grid_limit) is not int
            or not 1 <= grid_limit <= 32767
            or set(re.findall(r"[a-zA-Z_]\w*", grid)) != {"m"}
            or re.fullmatch(r"[0-9m\s()+*/%<>=?:-]+", grid) is None
        ):
            return source
        physical = f"(2 * ({grid}))"
    unused = ("ascend_l0a", "ascend_l0b", "ascend_l1", "ascend_l0c")
    if any(len(re.findall(r"\b" + name + r"\b", source)) != 2 for name in unused):
        return source
    lines = source.splitlines(keepends=True)
    # Each removed line contains only the compiler's unused allocation setup.
    removed = [line for line in lines if any(re.search(r"\b" + n + r"\b", line) for n in unused)]
    if any(
        re.fullmatch(r"\s*(?:AscendC::TBuf<[^>]+> \w+;\s*)?(?:pipe.InitBuffer\([^;]+\);\s*)?", line)
        is None
        for line in removed
    ):
        return source
    lowered = "".join(line for line in lines if line not in removed)
    lowered = lowered.replace(mixed, "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)")
    lowered = lowered.replace("AscendC::GetSubBlockIdx()", "(AscendC::GetBlockIdx() % 2)")
    return re.sub(pattern, lambda m: f"{m[1]}<<<{physical}, nullptr, stream>>>", lowered)


def lower_independent_cube(source):
    """Omit an unused Vector task from a recognized Cube-only stage."""
    mixed = "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)"
    if source.count(mixed) != 1 or "ASCEND_IS_AIC" not in source:
        return source
    if source.count("ASCEND_IS_AIV") != 1 or any(
        text in source for text in ("CrossCore", "SyncAll", "GlobalBarrier")
    ):
        return source
    block = re.search(r"auto (\w+) = AscendC::GetBlockIdx\(\);", source)
    lane = re.search(r"auto (\w+) = AscendC::GetSubBlockIdx\(\);", source)
    if block is None or lane is None or len(re.findall(r"\b" + lane[1] + r"\b", source)) != 1:
        return source
    pair = rf"\s*if ASCEND_IS_AIV\s*\{{\s*{block[1]} = {block[1]} / 2;\s*\}}"
    if len(re.findall(pair, source)) != 1 or len(re.findall(r"\bascend_ub\b", source)) != 2:
        return source
    lines = source.splitlines(keepends=True)
    removed = [line for line in lines if re.search(r"\bascend_ub\b", line)]
    if any(
        re.fullmatch(r"\s*(?:AscendC::TBuf<[^>]+> \w+;\s*)?(?:pipe.InitBuffer\([^;]+\);\s*)?", line)
        is None
        for line in removed
    ):
        return source
    lowered = "".join(line for line in lines if line not in removed)
    lowered = re.sub(pair, "", lowered)
    lowered = lowered.replace(lane[0], "")
    return lowered.replace(mixed, "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)")
