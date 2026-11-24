"""Record the provenance of a freshly built comparison extension."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    """Hash one build input or output."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_state(path):
    """Capture a clean repository revision."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, timeout=30
        ).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Build inputs must be clean: {path}")
    return {"commit": git("rev-parse", "HEAD"), "diff": ""}


def record(module, core, bindings):
    """Write a manifest immediately after building against the named checkouts."""
    build = module.parent
    manifest = {
        "module_sha256": sha256(module),
        "core": source_state(core),
        "bindings": source_state(bindings),
        "cmake_cache_sha256": sha256(build / "CMakeCache.txt"),
        "sources": {
            str(path): sha256(path)
            for path in (
                Path("native_precision/module.cpp"),
                Path("native_precision/CMakeLists.txt"),
                bindings / "src/signed_distance.cpp",
                bindings / "src/fast_winding_number.cpp",
            )
        },
        "core_source_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD:include/igl"], cwd=core, text=True, timeout=30
        ).strip(),
        "compiler": subprocess.check_output(
            ["c++", "--version"], text=True, timeout=30
        ).splitlines()[0],
        "flags": "Release; -ffp-contract=off; native_precision CMake defaults",
        "build_log_retained": False,
    }
    (build / "build_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--libigl-source", type=Path, required=True)
    parser.add_argument("--bindings-source", type=Path, required=True)
    args = parser.parse_args()
    record(args.module, args.libigl_source, args.bindings_source)
