"""Shared staging, parallel compilation, validation, and CLI base for DSv4.1 plan transforms."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.common.shared_json import decode
from orinfer_build.models.deepseek_v41.assembly import (
    compact_buffers,
    publish_package,
    selected_packet_ranges,
    validate_rows,
)


def plan_parser(
    description,
    *,
    default_workers=4,
    min_rows=None,
    max_rows=False,
    model_dir=True,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=default_workers)
    if min_rows is not None:
        parser.add_argument("--min-rows", type=int, default=min_rows)
    if max_rows:
        parser.add_argument("--max-rows", type=int)
    if model_dir:
        parser.add_argument("--model-dir", type=Path)
    return parser


def make_describer(base):
    descriptions = {}

    def describe(step):
        if step.get("op") != "kernel":
            return None
        name = step["manifest"]
        if name not in descriptions:
            descriptions[name] = json.loads((base / name).read_text())
        return descriptions[name]

    return describe


def stage_plan(
    plan,
    output,
    *,
    workers=1,
    allow_resume=False,
    minimum=None,
    maximum=None,
    require_batch=False,
    validate_source=False,
    model_dir=None,
):
    if workers < 1:
        raise ValueError("Require a positive compiler count")
    if allow_resume:
        if (output / "package").exists():
            raise ValueError("Require an unpublished output directory")
    elif output.exists():
        raise ValueError("Require a new output directory")
    model = decode(json.loads(plan.read_text()))
    if minimum is not None:
        for rank in model["ranks"]:
            rank["packets"] = selected_packet_ranges(rank["packets"], minimum, maximum)
    if require_batch and not model.get("batch"):
        raise ValueError("Require the owner-row native A2 contract")
    if validate_source:
        validate_rows(model, plan.parent, model_dir=model_dir)
    base = output / "source"
    if allow_resume and base.exists():
        if json.loads((base / "model.json").read_text()) != model:
            raise ValueError("Transform build input changed")
    else:
        output.mkdir(parents=True, exist_ok=True)
        publish_package(model, plan.parent, base)
    return model, base, make_describer(base)


def define_buffer(rank, definitions, name, size, *, label="transform", **kwargs):
    definition = dict(name=name, bytes=size, **kwargs)
    if name not in definitions:
        rank["buffers"].append(definition)
        definitions[name] = definition
    elif definitions[name] != definition:
        raise ValueError(f"Conflicting {label} buffer: {name}")
    return name


def compile_and_replace(
    model,
    tasks,
    base,
    workers,
    *,
    compiler,
    include_head=False,
    include_uncompacted=False,
):
    jobs = [(task, str(base)) for task in tasks.values()]
    if workers == 1:
        replacements = dict(compiler(job) for job in jobs)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            replacements = dict(pool.map(compiler, jobs))
    fields = ("manifest", "uncompacted_manifest") if include_uncompacted else ("manifest",)
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            steps = packet["body"] + packet["head"] if include_head else packet["body"]
            for step in steps:
                for field in fields:
                    if step.get(field) in replacements:
                        step[field] = replacements[step[field]]
    return replacements


def finalize_plan(
    model,
    base,
    output,
    *,
    owner_tables=False,
    model_dir=None,
    save_base=False,
    report_name=None,
    report=None,
):
    compact_buffers(model, owner_tables=owner_tables)
    validate_rows(model, base, model_dir=model_dir)
    if save_base:
        atomic_json(base / "model.json", model)
    assets = publish_package(model, base, output / "package")
    package_model = output / "package/model.json"
    if report_name is not None:
        payload = dict(model=str(package_model), assets=assets, **(report or {}))
        atomic_json(output / report_name, payload)
        return payload
    return assets, package_model
