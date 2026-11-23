"""Check geometric sign conventions and the benchmark's measurement contract."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["IGL_NUM_THREADS"] = "4"

import numpy as np

from benchmark import (
    append_result,
    disagreement_statistics,
    query_mesh,
    sample_points,
    select_models,
    write_csv,
)


def cube():
    """Return an outward-oriented unit cube."""
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


class BenchmarkTests(unittest.TestCase):
    def test_append_checkpoints_preserve_errors_and_new_fields(self):
        """Retain earlier failures and later result fields without rewriting records."""
        rows = [
            {"file_id": "1", "status": "error", "error": "invalid mesh"},
            {"file_id": "2", "status": "ok", "error": "", "disagreements": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "per_model.jsonl"
            append_result(journal, rows[0])
            first_record = journal.read_bytes()
            append_result(journal, rows[1])
            self.assertTrue(journal.read_bytes().startswith(first_record))
            recovered = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(recovered, rows)
            with self.assertRaises(ValueError):
                append_result(journal, {"file_id": "3", "distance": float("nan")})
            self.assertEqual(len(journal.read_text().splitlines()), 2)
            output = Path(directory) / "per_model.csv"
            write_csv(output, recovered)
            with output.open(newline="") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(exported[0]["error"], "invalid mesh")
            self.assertEqual(exported[0]["disagreements"], "")
            self.assertEqual(exported[1]["disagreements"], "3")

    def test_cube_face_edge_vertex_distances_and_signs(self):
        """Verify analytic distances and inside-positive signs on a unit cube."""
        vertices, faces = cube()
        points = np.array(
            [[0.5, 0.5, 0.5], [2, 0.4, 0.6], [2, 2, 0.5], [2, 2, 2], [0.0001, 0.4, 0.6]]
        )
        arrays, _ = query_mesh(vertices, faces, points)
        np.testing.assert_allclose(arrays["d"], [0.5, 1, np.sqrt(2), np.sqrt(3), 0.0001])
        np.testing.assert_array_equal(arrays["sign_p"], [1, -1, -1, -1, 1])
        np.testing.assert_array_equal(arrays["sign_w"], [1, -1, -1, -1, 1])

    def test_cavity_with_disconnected_boundary_components(self):
        """Treat an inward-oriented inner shell as an empty cavity."""
        vertices, faces = cube()
        cavity_vertices = vertices * 0.5 + 0.25
        combined_vertices = np.vstack([vertices, cavity_vertices])
        combined_faces = np.vstack([faces, faces[:, ::-1] + len(vertices)])
        points = np.array([[0.5, 0.5, 0.5], [0.1, 0.5, 0.5], [2, 0.5, 0.5]])
        arrays, _ = query_mesh(combined_vertices, combined_faces, points)
        np.testing.assert_allclose(arrays["d"], [0.25, 0.1, 1])
        np.testing.assert_array_equal(arrays["sign_p"], [-1, 1, -1])
        np.testing.assert_array_equal(arrays["sign_w"], [-1, 1, -1])

    def test_strict_distance_cutoff_and_denominators(self):
        """Exclude cutoff ties and use all points as the primary denominator."""
        arrays = {
            "d": np.array([0, 0.01, 0.0101, 0.02]),
            "sign_p": np.array([1, 1, 1, -1]),
            "sign_w": np.array([-1, -1, -1, -1]),
            "winding": np.zeros(4),
            "rounded_winding": np.zeros(4, dtype=np.int64),
        }
        stats = disagreement_statistics(arrays, 10, 0.001)
        self.assertEqual(stats["num_counted_disagreements"], 1)
        self.assertEqual(stats["disagreement_percent_all"], 25)
        self.assertEqual(stats["disagreement_percent_away"], 50)
        np.testing.assert_array_equal(arrays["counted_disagreement"], [False, False, True, False])

    def test_filter_and_reproducible_point_stream(self):
        """Include multiple components and reproduce samples independently by file ID."""
        rows = [
            {
                "file_id": str(i),
                "solid": "1",
                "vertex_manifold": "1",
                "edge_manifold": "1",
                "num_connected_components": str(i),
            }
            for i in range(1, 5)
        ]
        rows[2]["solid"] = "0"
        rows[3]["vertex_manifold"] = "0"
        chosen, count = select_models(rows, 2, 123)
        self.assertEqual(count, 2)
        self.assertEqual({row["file_id"] for row in chosen}, {"1", "2"})
        self.assertEqual(chosen, select_models(list(reversed(rows)), 2, 123)[0])
        vertices, _ = cube()
        first, _, _, _ = sample_points(vertices, 20, 123, "2")
        repeated, _, _, _ = sample_points(vertices, 20, 123, "2")
        other, _, _, _ = sample_points(vertices, 20, 123, "1")
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.all((first >= 0) & (first <= 1)))


if __name__ == "__main__":
    unittest.main()
