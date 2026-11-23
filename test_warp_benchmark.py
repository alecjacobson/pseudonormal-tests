"""Check Warp query integration and comparison against the fixed libigl reference."""

import unittest

import numpy as np

try:
    import warp as wp
except ImportError as error:
    raise unittest.SkipTest(
        "Build and install the pinned Warp source to run Warp integration tests."
    ) from error

from test_benchmark import cube
from warp_benchmark import comparison_statistics
from warp_queries import WARP_METHODS, query_warp


class WarpBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Warp and exercise CPU plus every available CUDA device."""
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_cube_queries(self):
        """Match analytic face, edge, and vertex distances with inside-positive signs."""
        vertices, faces = cube()
        points = np.array(
            [[0.5, 0.5, 0.5], [2, 0.4, 0.6], [2, 2, 0.5], [2, 2, 2], [0.0001, 0.4, 0.6]]
        )
        for device in self.devices:
            with self.subTest(device=str(device)):
                result, _ = query_warp(vertices, faces, points, device)
                for method in WARP_METHODS:
                    np.testing.assert_array_equal(result[f"warp_valid_{method}"], 1)
                    np.testing.assert_allclose(
                        result[f"warp_d_{method}"],
                        [0.5, 1, np.sqrt(2), np.sqrt(3), 0.0001],
                        atol=1e-7,
                    )
                    np.testing.assert_array_equal(result[f"warp_sign_{method}"], [1, -1, -1, -1, 1])

    def test_cavity_queries(self):
        """Respect the empty cavity bounded by a second inward-oriented shell."""
        vertices, faces = cube()
        vertices = np.vstack([vertices, vertices * 0.5 + 0.25])
        faces = np.vstack([faces, faces[:, ::-1] + 8])
        points = np.array([[0.51, 0.52, 0.53], [0.1, 0.5, 0.5], [2, 0.5, 0.5]])
        for device in self.devices:
            with self.subTest(device=str(device)):
                result, _ = query_warp(vertices, faces, points, device)
                for method in WARP_METHODS:
                    np.testing.assert_array_equal(result[f"warp_valid_{method}"], 1)
                    np.testing.assert_allclose(
                        result[f"warp_d_{method}"], [0.22, 0.1, 1], atol=1e-7
                    )
                    np.testing.assert_array_equal(result[f"warp_sign_{method}"], [-1, 1, -1])

    def test_common_reference_mask(self):
        """Use the original strict cutoff even when Warp returns different distances."""
        reference = {
            "d": np.array([0, 0.01, 0.0101, 0.02]),
            "sign_p": np.array([-1, -1, -1, -1]),
            "sign_w": np.array([-1, -1, -1, -1]),
        }
        result = {
            f"warp_{name}_{method}": value.copy()
            for method in WARP_METHODS
            for name, value in {
                "d": np.array([1.0, 1.0, 0.0, 0.0]),
                "sign": np.ones(4, dtype=np.int32),
                "valid": np.ones(4, dtype=np.int32),
            }.items()
        }
        stats, masks, _ = comparison_statistics(reference, result, 10.0, 0.001)
        self.assertEqual(stats["warp_p_disagreements"], 2)
        self.assertEqual(stats["warp_p_percent_all"], 50)
        self.assertEqual(stats["warp_p_percent_away"], 100)
        np.testing.assert_array_equal(masks["warp_p"], [False, False, True, True])
        result["warp_valid_p"][0] = 0
        with self.assertRaisesRegex(ValueError, "missed 1 points"):
            comparison_statistics(reference, result, 10.0, 0.001)

    @unittest.expectedFailure
    def test_pseudonormal_at_symmetric_cavity_center(self):
        """Expose the pinned Warp revision's wrong sign at an empty cavity's center."""
        vertices, faces = cube()
        vertices = np.vstack([vertices, vertices * 0.5 + 0.25])
        faces = np.vstack([faces, faces[:, ::-1] + 8])
        signs = []
        for device in self.devices:
            result, _ = query_warp(vertices, faces, np.array([[0.5, 0.5, 0.5]]), device)
            signs.append(result["warp_sign_p"][0])
        # Both CPU and CUDA currently return +1 here. Keep the analytic expectation
        # visible as an expected failure instead of treating that result as correct.
        np.testing.assert_array_equal(signs, np.full(len(signs), -1))


if __name__ == "__main__":
    unittest.main()
