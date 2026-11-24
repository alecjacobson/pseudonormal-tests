"""Expose two runtime tolerances in a generated copy of libigl's 3D classifier."""

import argparse
import hashlib
import json
from pathlib import Path


def generate(source, output):
    """Keep the upstream classifier body, replacing only its name and constants."""
    text = source.read_text()
    begin = text.index("template <")
    end = text.index("\ntemplate <", begin + 1)
    function = text[begin:end]
    # Accept both the original constant and the precision-dependent defaults.
    epsilon_begin = function.index("  const Scalar epsilon = ")
    epsilon_end = function.index(";\n", epsilon_begin) + len(";\n")
    epsilon_declaration = function[epsilon_begin:epsilon_end]
    replacements = {
        "igl::pseudonormal_test(": "pseudonormal_test_with_tolerances(",
        "Eigen::PlainObjectBase<Derivedn> & n)": (
            "Eigen::PlainObjectBase<Derivedn> & n,\n"
            "  const Scalar minimum_double_area,\n"
            "  const Scalar epsilon)"
        ),
        "  const Scalar MIN_DOUBLE_AREA = Scalar(1e-4);\n": "",
        epsilon_declaration: "",
        "if(area>MIN_DOUBLE_AREA)": "if(area>minimum_double_area)",
    }
    for before, after in replacements.items():
        if function.count(before) != 1:
            raise ValueError(f"Unexpected source: expected one occurrence of {before!r}")
        function = function.replace(before, after)
    # Retain upstream copyright/license text when generating the local function.
    license_text = text[:text.index('#include "pseudonormal_test.h"')]
    generated = license_text + """#pragma once
#include <igl/barycentric_coordinates.h>
#include <igl/doublearea.h>
#include <igl/project_to_line_segment.h>
#include <cassert>
#include <cmath>
namespace igl {
""" + function + "\n}\n"
    output.write_text(generated)
    output.with_suffix(".json").write_text(json.dumps({
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "replacements": replacements,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.source, args.output)
