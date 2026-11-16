"""Explicit symbolic axes for offline compiler tasks and cache identity."""


def dynamic_arguments(parameters, fallback, capacity):
    """Recover explicit row multipliers, retaining legacy inference when absent."""
    dynamic = parameters.get("_dynamic_arguments", fallback)
    if not isinstance(dynamic, dict):
        raise ValueError("Dynamic arguments must name integer row multipliers")
    for name, factor in dynamic.items():
        if (
            not isinstance(name, str)
            or name.startswith("_")
            or type(factor) is not int
            or factor < 1
            or type(capacity) is not int
            or capacity < 1
            or type(parameters.get(name)) is not int
            or parameters[name] != capacity * factor
        ):
            raise ValueError("Dynamic argument differs from its capacity geometry")
    return dict(dynamic)


def dynamic_parameters(parameters, multipliers, capacity):
    """Include symbolic axes in AOT identity even when capacity shapes coincide."""
    result = dict(parameters, _dynamic_arguments=dict(multipliers))
    dynamic_arguments(result, {}, capacity)
    return result
