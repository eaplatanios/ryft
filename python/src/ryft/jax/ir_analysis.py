"""Reusable helpers for normalizing and summarizing JAX and MLIR IR artifacts."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


MLIR_OPERATION_PATTERN = re.compile(
    r"^\s*(?:%\S+(?:,\s*%\S+)*\s*=\s*)?([A-Za-z_][\w.]+)\b",
    flags=re.MULTILINE,
)


def normalize_op_name(name: str) -> str:
    """Normalizes one primitive name onto the shared comparison vocabulary."""

    if name in {"add", "add_any"}:
        return "add"
    if name == "mul":
        return "mul"
    if name == "neg":
        return "neg"
    if name == "sin":
        return "sin"
    if name == "cos":
        return "cos"
    if name in {"matmul", "dot_general", "left_matmul", "right_matmul"}:
        return "matmul"
    if name in {"matrix_transpose", "linear_matrix_transpose", "transpose"}:
        return "transpose"
    if name == "scale":
        return "scale"
    if name in {"const", "constant"}:
        return "const"
    if name in {"shard_map", "linear_shard_map"}:
        return "shard_map"
    return f"unknown:{name}"


def normalize_mlir_op_name(name: str) -> str | None:
    """Normalizes one MLIR operation name onto the shared comparison vocabulary."""

    if name in {"module", "func.func", "return", "sdy.return", "sdy.mesh"}:
        return None
    if name == "sdy.manual_computation":
        return "shard_map"

    _, _, bare_name = name.partition(".")
    bare_name = bare_name or name
    if bare_name == "multiply":
        bare_name = "mul"
    elif bare_name == "negate":
        bare_name = "neg"
    elif bare_name == "sine":
        bare_name = "sin"
    elif bare_name == "cosine":
        bare_name = "cos"
    elif bare_name == "dot_general":
        bare_name = "matmul"

    return normalize_op_name(bare_name)


def strip_mlir_location_markers(module_text: str) -> str:
    """Removes textual MLIR location annotations from one module rendering."""

    def strip_locations_from_line(line: str) -> str:
        fragments: list[str] = []
        start_index = 0

        while True:
            location_index = line.find(" loc(", start_index)
            if location_index == -1:
                fragments.append(line[start_index:])
                break

            fragments.append(line[start_index:location_index])

            cursor = location_index + len(" loc(")
            depth = 1
            in_string = False
            while cursor < len(line) and depth > 0:
                character = line[cursor]
                if in_string:
                    if character == "\\" and cursor + 1 < len(line):
                        cursor += 2
                        continue
                    if character == '"':
                        in_string = False
                else:
                    if character == '"':
                        in_string = True
                    elif character == "(":
                        depth += 1
                    elif character == ")":
                        depth -= 1
                cursor += 1
            start_index = cursor

        return "".join(fragments).rstrip()

    stripped_lines: list[str] = []
    for line in module_text.splitlines():
        if line.lstrip().startswith("#loc"):
            continue

        stripped_line = strip_locations_from_line(line.rstrip())
        if stripped_line:
            stripped_lines.append(stripped_line)

    if not stripped_lines:
        return ""
    return "\n".join(stripped_lines) + "\n"


def summarize_mlir(module_text: str) -> dict[str, Any]:
    """Builds a lightweight structural summary from textual MLIR."""

    operation_names = MLIR_OPERATION_PATTERN.findall(module_text)
    op_histogram: Counter[str] = Counter()
    for operation_name in operation_names:
        normalized_name = normalize_mlir_op_name(operation_name)
        if normalized_name is not None:
            op_histogram[normalized_name] += 1

    return {
        "input_leaf_count": 0,
        "output_leaf_count": 0,
        "equation_count": sum(op_histogram.values()),
        "constant_count": op_histogram.get("const", 0),
        "op_histogram": dict(sorted(op_histogram.items())),
        "nested_region_count": module_text.count("sdy.manual_computation"),
        "nested_regions": [],
        "max_dependency_depth": 0,
    }


def normalize_mlir_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replaces per-record summaries with MLIR-derived summaries."""

    normalized_records: list[dict[str, Any]] = []
    for record in records:
        normalized_record = dict(record)
        normalized_record["summary"] = summarize_mlir(record["raw_ir"])
        normalized_records.append(normalized_record)
    normalized_records.sort(key=lambda record: (record["case_id"], record["surface"]))
    return normalized_records


def make_mlir_record(case_id: str, category: str, surface: str, raw_ir: str) -> dict[str, Any]:
    """Builds one benchmark record from textual MLIR."""

    return {
        "case_id": case_id,
        "category": category,
        "surface": surface,
        "raw_ir": raw_ir,
        "summary": summarize_mlir(raw_ir),
    }


def is_literal(jax_core: Any, variable: Any) -> bool:
    """Returns whether one JAX variable is a literal."""

    literal_type = getattr(jax_core, "Literal", None)
    if literal_type is not None and isinstance(variable, literal_type):
        return True
    return type(variable).__name__ == "Literal" and hasattr(variable, "val")


def unwrap_jaxpr_payload(payload: Any) -> tuple[Any, tuple[Any, ...]]:
    """Returns a `(jaxpr, consts)` pair from a JAX region payload."""

    if hasattr(payload, "jaxpr") and hasattr(payload, "consts"):
        return payload.jaxpr, tuple(payload.consts)
    if hasattr(payload, "jaxpr"):
        return payload.jaxpr, ()
    return payload, ()


def collect_nested_region_payloads(value: Any, label: str) -> list[tuple[str, Any]]:
    """Collects nested JAX region payloads recursively from one equation parameter."""

    if value is None:
        return []

    if hasattr(value, "jaxpr") or (
        hasattr(value, "eqns") and hasattr(value, "invars") and hasattr(value, "outvars")
    ):
        return [(label, value)]

    if isinstance(value, dict):
        nested_regions: list[tuple[str, Any]] = []
        for key, nested_value in value.items():
            nested_regions.extend(collect_nested_region_payloads(nested_value, f"{label}.{key}"))
        return nested_regions

    if isinstance(value, (list, tuple)):
        nested_regions = []
        for index, nested_value in enumerate(value):
            nested_regions.extend(collect_nested_region_payloads(nested_value, f"{label}[{index}]"))
        return nested_regions

    return []


def summarize_jaxpr(payload: Any, jax_core: Any) -> dict[str, Any]:
    """Summarizes one `ClosedJaxpr` or `Jaxpr` payload."""

    jaxpr, _consts = unwrap_jaxpr_payload(payload)
    op_histogram: Counter[str] = Counter()
    nested_regions: list[dict[str, Any]] = []
    depth_by_var: dict[int, int] = {id(variable): 0 for variable in getattr(jaxpr, "invars", ())}
    depth_by_var.update({id(variable): 0 for variable in getattr(jaxpr, "constvars", ())})
    literal_count = 0

    for equation in getattr(jaxpr, "eqns", ()):
        primitive_label = normalize_op_name(equation.primitive.name)
        op_histogram[primitive_label] += 1

        input_depth = 0
        for input_var in equation.invars:
            if is_literal(jax_core, input_var):
                literal_count += 1
            else:
                input_depth = max(input_depth, depth_by_var.get(id(input_var), 0))

        output_depth = input_depth + 1
        for output_var in equation.outvars:
            depth_by_var[id(output_var)] = output_depth

        for parameter_name, parameter_value in equation.params.items():
            for nested_label, nested_payload in collect_nested_region_payloads(
                parameter_value, f"{primitive_label}.{parameter_name}"
            ):
                nested_summary = summarize_jaxpr(nested_payload, jax_core)
                nested_regions.append(
                    {
                        "label": nested_label,
                        "input_leaf_count": nested_summary["input_leaf_count"],
                        "output_leaf_count": nested_summary["output_leaf_count"],
                        "equation_count": nested_summary["equation_count"],
                        "constant_count": nested_summary["constant_count"],
                        "op_histogram": nested_summary["op_histogram"],
                        "nested_region_count": nested_summary["nested_region_count"],
                        "max_dependency_depth": nested_summary["max_dependency_depth"],
                    }
                )

    nested_region_count = len(nested_regions) + sum(region["nested_region_count"] for region in nested_regions)
    max_dependency_depth = max(
        (
            depth_by_var.get(id(output_var), 0)
            for output_var in getattr(jaxpr, "outvars", ())
            if not is_literal(jax_core, output_var)
        ),
        default=0,
    )

    return {
        "input_leaf_count": len(getattr(jaxpr, "invars", ())),
        "output_leaf_count": len(getattr(jaxpr, "outvars", ())),
        "equation_count": len(getattr(jaxpr, "eqns", ())),
        "constant_count": len(getattr(jaxpr, "constvars", ())) + literal_count,
        "op_histogram": dict(sorted(op_histogram.items())),
        "nested_region_count": nested_region_count,
        "nested_regions": nested_regions,
        "max_dependency_depth": max_dependency_depth,
    }


def make_jaxpr_record(case_id: str, category: str, surface: str, closed_jaxpr: Any, jax_core: Any) -> dict[str, Any]:
    """Builds one benchmark record from a `ClosedJaxpr`."""

    return {
        "case_id": case_id,
        "category": category,
        "surface": surface,
        "raw_ir": str(closed_jaxpr),
        "summary": summarize_jaxpr(closed_jaxpr, jax_core),
    }


__all__ = [
    "collect_nested_region_payloads",
    "is_literal",
    "make_jaxpr_record",
    "make_mlir_record",
    "normalize_mlir_op_name",
    "normalize_mlir_records",
    "normalize_op_name",
    "strip_mlir_location_markers",
    "summarize_jaxpr",
    "summarize_mlir",
    "unwrap_jaxpr_payload",
]
