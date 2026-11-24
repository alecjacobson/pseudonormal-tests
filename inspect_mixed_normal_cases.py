"""Export reproducible local geometry for representative mixed-normal cases."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from benchmark import load_mesh
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from validate_tolerances import file_hash, import_module, read_rows


def run(args):
    """Inspect saved queries and compare the selected normals and barycentrics."""
    config = json.loads((args.results / "config.json").read_text())
    _, mesh_dir, _ = prepare_dataset(DEFAULT_CACHE)
    module = import_module(args.module)
    cases = read_rows(args.results / "cases.csv")
    examples = []
    for file_id, sample_index in (
        ("109130", 60),
        ("153100", 23),
        ("57854", 4939),
        ("1233195", 1521),
        ("1452672", 1),
    ):
        vertices, faces = load_mesh(mesh_dir / f"{file_id}.npz")
        with np.load(Path(config["reference"]) / "points" / f"{file_id}.npz") as archive:
            reference = dict(archive)
        with np.load(Path(config["baseline"]) / "points" / f"{file_id}.npz") as archive:
            saved = dict(archive)
        face = int(saved["float32_closest_face_index"][sample_index])
        triangle = vertices[faces[face]]
        rounded_triangle = triangle.astype(np.float32).astype(np.float64)
        closest_float = saved["float32_closest_point"][sample_index]
        closest_double = reference["closest_point"][sample_index]
        data = np.concatenate([triangle.ravel(), closest_float, closest_double])
        output = subprocess.run(
            [str(args.triangle_tool.resolve())],
            input=" ".join(format(x, ".17g") for x in data),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        diagnostic = json.loads(output.stdout)
        selected_cases = [
            r for r in cases if r["file_id"] == file_id and int(r["sample_index"]) == sample_index
        ]
        mesh = module.PreparedMeshDouble(vertices, faces)
        fn, vn, en, _, _, edges, edge_map = mesh.normals()
        double_normal = saved["float64_tuned_pseudonormal"][sample_index]
        matches = {}
        for name, table in (("face", fn), ("edge", en), ("vertex", vn)):
            indices = np.flatnonzero(np.linalg.norm(table - double_normal, axis=1) < 1e-12)
            matches[name] = indices.tolist()
        edge_details = {}
        for edge in matches["edge"]:
            edge_details[str(edge)] = {
                "vertices": edges[edge].tolist(),
                "faces": (np.flatnonzero(edge_map == edge) % len(faces)).tolist(),
            }
        incident_geometry = {}
        incident_faces = {face}
        for row in selected_cases:
            incident_faces.update(json.loads(row["incident_faces"]))
        for incident in sorted(incident_faces):
            original = vertices[faces[incident]]
            rounded = original.astype(np.float32).astype(np.float64)
            incident_geometry[str(incident)] = {
                "vertices": original.tolist(),
                "original_face_normal": fn[incident].tolist(),
                "cross_norm_original": float(
                    np.linalg.norm(np.cross(original[1] - original[0], original[2] - original[0]))
                ),
                "cross_norm_rounded": float(
                    np.linalg.norm(np.cross(rounded[1] - rounded[0], rounded[2] - rounded[0]))
                ),
            }
        examples.append(
            {
                "file_id": file_id,
                "sample_index": sample_index,
                "closest_face_float": face,
                "closest_face_double": int(reference["closest_face_index"][sample_index]),
                "triangle_vertex_indices": faces[face].tolist(),
                "triangle_original": triangle.tolist(),
                "triangle_float": rounded_triangle.tolist(),
                "closest_float": closest_float.tolist(),
                "closest_double": closest_double.tolist(),
                "cross_norm_original": float(
                    np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
                ),
                "cross_norm_rounded": float(
                    np.linalg.norm(
                        np.cross(
                            rounded_triangle[1] - rounded_triangle[0],
                            rounded_triangle[2] - rounded_triangle[0],
                        )
                    )
                ),
                "libigl_area_barycentric": diagnostic,
                "double_normal": double_normal.tolist(),
                "double_normal_table_matches": matches,
                "double_matched_edges": edge_details,
                "incident_face_geometry": incident_geometry,
                "variant_cases": selected_cases,
            }
        )
    write_json(
        args.results / "examples.json",
        {
            "triangle_tool_sha256": file_hash(args.triangle_tool),
            "normal_table_module_sha256": file_hash(args.module),
            "examples": examples,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--triangle-tool", type=Path, required=True)
    run(parser.parse_args())
