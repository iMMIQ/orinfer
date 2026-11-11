"""Pack the generated CANN host arguments without the per-argument array stub."""

import re


def lower_host_arguments(source):
    """Keep device code, tiling, grid and stream; return launch errors directly."""
    calls = list(re.finditer(r"(\w+)<<<(.*?)>>>\(([^;]*)\);", source, re.DOTALL))
    if len(calls) != 1:
        raise ValueError("Packed host launch requires one generated device call")
    call = calls[0]
    config = [part.strip() for part in call[2].rsplit(",", 2)]
    arguments = [part.strip() for part in call[3].split(",")]
    if (
        len(config) != 3
        or config[1:] != ["nullptr", "stream"]
        or not arguments
        or any(re.fullmatch(r"\w+", argument) is None for argument in arguments)
    ):
        raise ValueError("Unsupported generated host launch arguments")
    fields = " ".join(f"decltype({argument}) p{i};" for i, argument in enumerate(arguments))
    replacement = f"""aclrtFuncHandle function{{nullptr}};
  int32_t status = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(&{call[1]}), &function);
  if (status) return status;
  struct {{ {fields} }} parameters{{{", ".join(arguments)}}};
  return aclrtLaunchKernelWithHostArgs(function, static_cast<uint32_t>({config[0]}),
      stream, nullptr, &parameters, sizeof(parameters), nullptr, 0);"""
    source = source[: call.start()] + replacement + source[call.end() :]
    source, count = re.subn(r'extern "C" void call\(', 'extern "C" int32_t call(', source)
    if count != 1:
        raise ValueError("Packed host launch requires one generated host entry")
    return source
