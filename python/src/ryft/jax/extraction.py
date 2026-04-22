from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax


@dataclass(frozen=True)
class ProgramCase:
    """Describes a named JAX program together with the inputs used to inspect it."""

    name: str
    description: str
    function: Callable[..., Any]
    example_args: tuple[Any, ...] | None = None
    abstract_args: tuple[Any, ...] | None = None
    jit_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def require_example_args(self) -> tuple[Any, ...]:
        """Returns the concrete example arguments required for JAXPR extraction."""

        if self.example_args is None:
            raise ValueError(f"program case `{self.name}` does not define example arguments")
        return self.example_args

    def lowering_args(self) -> tuple[Any, ...]:
        """Returns the abstract arguments required for StableHLO lowering."""

        if self.abstract_args is not None:
            return self.abstract_args
        return self.require_example_args()


@dataclass(frozen=True)
class JaxprInspection:
    """Captures the textual JAXPR plus any closed-over constants for a JAX program."""

    text: str
    consts: tuple[Any, ...]


@dataclass(frozen=True)
class ProgramInspection:
    """Captures the extracted textual artifacts for one named JAX program."""

    case: ProgramCase
    jaxpr: JaxprInspection | None = None
    stablehlo: str | None = None


def extract_jaxpr(function: Callable[..., Any], *args: Any) -> JaxprInspection:
    """Extracts the textual JAXPR for `function` at the provided concrete arguments."""

    closed_jaxpr = jax.make_jaxpr(function)(*args)
    return JaxprInspection(text=str(closed_jaxpr), consts=tuple(closed_jaxpr.consts))


def lower_to_stablehlo(
    function: Callable[..., Any],
    *args: Any,
    jit_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Lowers `function` to StableHLO using the provided abstract or concrete arguments."""

    lowered = jax.jit(function, **dict(jit_kwargs or {})).lower(*args)
    return str(lowered.compiler_ir("stablehlo"))


def inspect_program(
    case: ProgramCase,
    *,
    include_jaxpr: bool = True,
    include_stablehlo: bool = True,
) -> ProgramInspection:
    """Extracts the requested textual artifacts for the provided program case."""

    jaxpr = extract_jaxpr(case.function, *case.require_example_args()) if include_jaxpr else None
    stablehlo = (
        lower_to_stablehlo(case.function, *case.lowering_args(), jit_kwargs=case.jit_kwargs)
        if include_stablehlo
        else None
    )
    return ProgramInspection(case=case, jaxpr=jaxpr, stablehlo=stablehlo)


def render_program_inspection(inspection: ProgramInspection) -> str:
    """Renders the extracted program artifacts into a readable text block."""

    lines = [inspection.case.name, inspection.case.description]

    if inspection.jaxpr is not None:
        lines.extend(
            [
                "JAXPR:",
                inspection.jaxpr.text,
            ]
        )
        if inspection.jaxpr.consts:
            lines.append(f"JAXPR consts: {inspection.jaxpr.consts}")

    if inspection.stablehlo is not None:
        lines.extend(
            [
                "StableHLO:",
                inspection.stablehlo,
            ]
        )

    return "\n".join(lines)
