"""Exposes the pinned source revisions to the native library."""

def _source_revisions_repository_impl(ctx):
    """Writes the revision header after verifying the Triton pin in the resolved XLA tree."""
    # OpenXLA can substitute the Intel fork before fetching Triton.
    if ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1":
        fail("ENABLE_INTEL_XPU_TRITON selects a different Triton source than the TRITON_COMMIT pin")

    workspace = ctx.attr.xla_triton_workspace
    assignments = []
    for line in ctx.read(workspace).splitlines():
        fields = line.split("#", 1)[0].split("=", 1)
        if len(fields) == 2 and fields[0].strip() == "TRITON_COMMIT":
            assignments.append(fields[1].strip())
    if len(assignments) != 1:
        fail("expected exactly one TRITON_COMMIT assignment in {}".format(workspace))

    expected = ctx.attr.expected_triton_commit
    selected = assignments[0]
    if selected != '"{}"'.format(expected):
        fail("Triton revision mismatch in {} (expected \"{}\", found {})".format(workspace, expected, selected))

    ctx.file(
        "include/ryft_xla_sys/source_revisions.h",
        """\
#pragma once

#define RYFT_XLA_COMMIT "{xla_commit}"
#define RYFT_JAX_COMMIT "{jax_commit}"
#define RYFT_TRITON_COMMIT {triton_commit}
""".format(
            xla_commit = ctx.attr.xla_commit,
            jax_commit = ctx.attr.jax_commit,
            triton_commit = selected,
        ),
        executable = False,
    )
    ctx.file("BUILD.bazel", """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "source_revisions",
    hdrs = ["include/ryft_xla_sys/source_revisions.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""", executable = False)

source_revisions_repository = repository_rule(
    implementation = _source_revisions_repository_impl,
    attrs = {
        "expected_triton_commit": attr.string(mandatory = True),
        "jax_commit": attr.string(mandatory = True),
        "xla_commit": attr.string(mandatory = True),
        "xla_triton_workspace": attr.label(mandatory = True),
    },
    doc = "Provides the pinned XLA and JAX revisions and the verified Triton revision in a C++ header.",
)
