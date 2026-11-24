"""Check the independent geometry oracle on exact known and exhaustive cases."""

import unittest
from fractions import Fraction

import mpmath as mp
import numpy as np

from audit_double_diagnostics import strict_crossing
from exact_mesh_diagnostics import ExactMesh, direct_winding, rational_point


def mesh(vertices, faces):
    vertices, faces = np.array(vertices, dtype=float), np.array(faces, dtype=int)
    edges, lookup, edge_map = [], {}, []
    for opposite in range(3):
        for face in faces:
            edge = tuple(sorted((face[(opposite + 1) % 3], face[(opposite + 2) % 3])))
            if edge not in lookup:
                lookup[edge] = len(edges)
                edges.append(edge)
            edge_map.append(lookup[edge])
    return ExactMesh(vertices, faces, np.array(edges), np.array(edge_map))


class ExactGeometryTests(unittest.TestCase):
    def test_strict_crossing_near_vertex(self):
        """Distinguish an arbitrarily close interior crossing from a vertex touch."""
        triangle = tuple(rational_point(p) for p in ([0, 0, 0], [1, 0, 0], [0, 1, 0]))
        hit = strict_crossing(
            rational_point([1e-20, 1e-20, -1]), rational_point([1e-20, 1e-20, 1]), triangle
        )
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], Fraction(1, 2))
        self.assertTrue(all(x > 0 for x in hit[2]))
        self.assertEqual(sum(hit[2]), 1)
        self.assertIsNone(
            strict_crossing(rational_point([0, 0, -1]), rational_point([0, 0, 1]), triangle)
        )

    def test_skinny_face_projection(self):
        """Recover an interior feature when a binary64 Gram determinant vanishes."""
        width = 2.0**-60
        value = mesh([[0, 0, 0], [1, 0, 0], [1, width, 0]], [[0, 1, 2]])
        q = rational_point([0.75, width / 4, 1])
        result = value.closest_triangle(q, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0], (Fraction(1), (Fraction(3, 4), Fraction(1, 2**62), Fraction(0)), (0, 0))
        )

    def test_edge_and_vertex_regions(self):
        """Identify exact edge and vertex regions without a feature epsilon."""
        value = mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        edge = value.closest_triangle(rational_point([0.5, -0.2, 1]), 0)[0]
        vertex = value.closest_triangle(rational_point([-1, -1, 1]), 0)[0]
        self.assertEqual(edge[2], value.feature(0, 1, 2))
        self.assertEqual(edge[1], (Fraction(1, 2), Fraction(0), Fraction(0)))
        self.assertEqual(vertex[2], (2, 0))

    def test_degenerate_triangle(self):
        """Return the segment minimum for an exactly collinear triangle."""
        value = mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]])
        results = value.closest_triangle(rational_point([0.5, 1, 0]), 0)
        self.assertTrue(all(r[0] == 1 and r[1] == rational_point([0.5, 0, 0]) for r in results))
        self.assertIsNone(value.face_normal(0))

    def test_pruning_matches_exhaustive_search(self):
        """Match an exhaustive rational search for a seeded collection of triangles."""
        rng = np.random.default_rng(24517)
        vertices = rng.normal(size=(90, 3))
        value = mesh(vertices, np.arange(90).reshape(-1, 3))
        for point in rng.normal(size=(16, 3)):
            q = rational_point(point)
            exhaustive = [r for face in range(30) for r in value.closest_triangle(q, face)]
            best_distance = min(r[0] for r in exhaustive)
            expected = {r[2] for r in exhaustive if r[0] == best_distance}
            actual, _ = value.closest_global(q, 0)
            self.assertEqual(actual[0][0], best_distance)
            self.assertEqual({r[2] for r in actual}, expected)

    def test_winding_and_normals_on_tetrahedron(self):
        """Recover known winding numbers and outward normals on a tetrahedron."""
        value = mesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
        winding = direct_winding(
            value.vertices, value.faces, np.array([[0.1, 0.1, 0.1], [2, 2, 2]])
        )
        np.testing.assert_allclose(winding, [1, 0], atol=1e-15)
        self.assertEqual(value.normal(0, 0), (mp.mpf(0), mp.mpf(0), mp.mpf(-1)))


if __name__ == "__main__":
    unittest.main()
