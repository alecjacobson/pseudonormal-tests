"""Copy the measured normal accumulators, removing only final normalization."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def generate(source_dir, output):
    """Extract the overloads taking face normals and retain their exact summation."""
    pieces = [
        "#pragma once\n#include <igl/per_vertex_normals.h>\n"
        "#include <igl/per_edge_normals.h>\nnamespace igl {\n"
    ]
    provenance = []
    for name in ("per_vertex_normals", "per_edge_normals"):
        source = source_dir / "include/igl" / f"{name}.cpp"
        text = source.read_text()
        starts = [match.start() for match in re.finditer(r"(?m)^template\s*<", text)]
        candidates = []
        for start, end in zip(starts, starts[1:] + [len(text)], strict=True):
            function = text[start:end]
            if "const Eigen::MatrixBase<DerivedFN>& FN" in function and "weighting," in function:
                candidates.append(function)
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one weighted {name} overload taking FN")
        function = candidates[0]
        if function.count("N.rowwise().normalize();") != 1:
            raise ValueError(f"Expected exactly one normalization in {name}")
        function = function.replace(f"igl::{name}(", f"{name}_unnormalized(")
        function = function.replace("N.rowwise().normalize();", "// Retain the accumulated vector.")
        pieces.append(text[: text.index("#include")] + function)
        provenance.append(
            {
                "source": str(source.resolve()),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    pieces.append("}\n")
    output.write_text("\n".join(pieces))
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sources": provenance,
                "generated_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "changes": "Rename functions; remove only final N.rowwise().normalize()",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.source_dir, args.output)
