"""Render JAXPR and StableHLO for the named JAX programs tracked by `ryft`."""

from __future__ import annotations

import argparse

from ryft.jax.examples import program_cases_by_name
from ryft.jax.extraction import inspect_program, render_program_inspection


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments for the inspection script."""

    parser = argparse.ArgumentParser(
        description="Render JAXPR and StableHLO for the named JAX programs tracked by ryft."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        default=[],
        help="Restrict output to the named case. Repeat this flag to inspect multiple cases.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available program cases and exit.",
    )
    parser.add_argument(
        "--jaxpr-only",
        action="store_true",
        help="Only render JAXPR output.",
    )
    parser.add_argument(
        "--stablehlo-only",
        action="store_true",
        help="Only render StableHLO output.",
    )
    return parser.parse_args()


def main() -> int:
    """Runs the requested JAX program inspection workflow."""

    args = parse_args()
    cases_by_name = program_cases_by_name()

    if args.list:
        for case_name, case in cases_by_name.items():
            print(f"{case_name}: {case.description}")
        return 0

    if args.jaxpr_only and args.stablehlo_only:
        raise ValueError("cannot request both --jaxpr-only and --stablehlo-only")

    include_jaxpr = not args.stablehlo_only
    include_stablehlo = not args.jaxpr_only

    selected_case_names = args.case_names or list(cases_by_name.keys())
    unknown_case_names = sorted(set(selected_case_names) - cases_by_name.keys())
    if unknown_case_names:
        available_case_names = ", ".join(cases_by_name.keys())
        missing_case_names = ", ".join(unknown_case_names)
        raise ValueError(
            f"unknown program case(s): {missing_case_names}; available cases: {available_case_names}"
        )

    for index, case_name in enumerate(selected_case_names):
        if index > 0:
            print()
        print(
            render_program_inspection(
                inspect_program(
                    cases_by_name[case_name],
                    include_jaxpr=include_jaxpr,
                    include_stablehlo=include_stablehlo,
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
