"""Use exact rational closest features and high-precision pseudonormals for diagnosis."""

from fractions import Fraction
from functools import cache

import mpmath as mp
import numpy as np


def add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def scale(a, s):
    return tuple(x * s for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rational_point(point):
    return tuple(Fraction(float(x)) for x in point)


def precise(x):
    if isinstance(x, Fraction):
        return mp.mpf(x.numerator) / x.denominator
    return mp.mpf(float(x))


def unit(vector):
    vector = tuple(precise(x) if isinstance(x, Fraction) else x for x in vector)
    length = mp.sqrt(dot(vector, vector))
    return scale(vector, 1 / length) if length else None


def sign_of_dot(query, closest, normal):
    if normal is None:
        return None, None
    value = dot(
        tuple(precise(x) - precise(y) for x, y in zip(query, closest, strict=True)),
        tuple(precise(x) if isinstance(x, (Fraction, float, np.floating)) else x for x in normal),
    )
    return (-1 if value >= 0 else 1), value


class ExactMesh:
    """Keep input binary64 coordinates exact during all closest-point predicates."""

    def __init__(self, vertices, faces, edges, edge_map):
        self.vertices, self.faces, self.edges, self.edge_map = vertices, faces, edges, edge_map
        triangles = vertices[faces]
        self.lower, self.upper = triangles.min(axis=1), triangles.max(axis=1)
        for name in ("vertex", "triangle", "face_normal", "incident_faces", "normal"):
            setattr(self, name, cache(getattr(self, name)))

    def vertex(self, index):
        return rational_point(self.vertices[index])

    def triangle(self, face):
        return tuple(self.vertex(int(i)) for i in self.faces[face])

    def feature(self, face, local_kind, local_index):
        if local_kind == 0:
            return 0, int(face)
        if local_kind == 1:
            return 1, int(self.edge_map[local_index * len(self.faces) + face])
        return 2, int(self.faces[face, local_index])

    def closest_triangle(self, query, face):
        """Enumerate face/edge/vertex candidates using exact rational arithmetic."""
        triangle = self.triangle(face)
        candidates = []
        for i, point in enumerate(triangle):
            candidates.append(
                (dot(sub(query, point), sub(query, point)), point, self.feature(face, 2, i))
            )
        for opposite in range(3):
            a, b = triangle[(opposite + 1) % 3], triangle[(opposite + 2) % 3]
            edge = sub(b, a)
            length2 = dot(edge, edge)
            if not length2:
                continue
            t = dot(sub(query, a), edge) / length2
            if 0 < t < 1:
                point = add(a, scale(edge, t))
                candidates.append(
                    (
                        dot(sub(query, point), sub(query, point)),
                        point,
                        self.feature(face, 1, opposite),
                    )
                )
        a, b, c = triangle
        ab, ac, aq = sub(b, a), sub(c, a), sub(query, a)
        normal = cross(ab, ac)
        norm2 = dot(normal, normal)
        if norm2:
            beta = dot(cross(aq, ac), normal) / norm2
            gamma = dot(cross(ab, aq), normal) / norm2
            if beta > 0 and gamma > 0 and beta + gamma < 1:
                point = add(a, add(scale(ab, beta), scale(ac, gamma)))
                candidates.append(
                    (dot(sub(query, point), sub(query, point)), point, (0, int(face)))
                )
        best = min(x[0] for x in candidates)
        return [x for x in candidates if x[0] == best]

    def closest_global(self, query, initial_face):
        """Prune with outward-rounded box bounds, then compare rational distances."""
        best = self.closest_triangle(query, initial_face)
        upper = np.nextafter(float(best[0][0]), np.inf)
        q = np.array([float(x) for x in query])
        delta = np.maximum(
            np.maximum(
                np.nextafter(self.lower - q, -np.inf), np.nextafter(q - self.upper, -np.inf)
            ),
            0,
        )
        squares = np.maximum(np.nextafter(delta * delta, -np.inf), 0)
        lower = np.maximum(np.nextafter(squares[:, 0] + squares[:, 1], -np.inf), 0)
        lower = np.maximum(np.nextafter(lower + squares[:, 2], -np.inf), 0)
        candidates = np.flatnonzero(lower <= upper)
        evaluated = 0
        for face in candidates[np.argsort(lower[candidates])]:
            if lower[face] > upper:
                break
            if face == initial_face:
                continue
            current = self.closest_triangle(query, int(face))
            evaluated += 1
            if current[0][0] < best[0][0]:
                best = current
                upper = np.nextafter(float(best[0][0]), np.inf)
            elif current[0][0] == best[0][0]:
                best.extend(current)
        unique = {x[2]: x for x in best}
        return list(unique.values()), evaluated

    def face_normal(self, face):
        a, b, c = self.triangle(face)
        return unit(cross(sub(b, a), sub(c, a)))

    def incident_faces(self, kind, index):
        if kind == 0:
            return [index]
        if kind == 1:
            return (np.flatnonzero(self.edge_map == index) % len(self.faces)).tolist()
        return np.flatnonzero((self.faces == index).any(axis=1)).tolist()

    def normal(self, kind, index):
        """Compute unit-face sums and vertex angles with the active mpmath precision."""
        if kind == 0:
            return self.face_normal(index)
        total = (mp.mpf(0),) * 3
        for face in self.incident_faces(kind, index):
            normal = self.face_normal(face)
            if normal is None:
                continue
            weight = mp.mpf(1)
            if kind == 2:
                local = int(np.flatnonzero(self.faces[face] == index)[0])
                triangle = self.triangle(face)
                a = sub(triangle[(local + 1) % 3], triangle[local])
                b = sub(triangle[(local + 2) % 3], triangle[local])
                raw_cross = cross(a, b)
                weight = mp.atan2(mp.sqrt(precise(dot(raw_cross, raw_cross))), precise(dot(a, b)))
            total = add(total, scale(normal, weight))
        return unit(total)


def direct_winding(vertices, faces, queries):
    """Sum all triangle solid angles in extended precision, without approximation."""
    triangles = vertices[faces].astype(np.longdouble)
    result = []
    pi = np.arccos(np.longdouble(-1))
    for query in queries:
        rays = triangles - query.astype(np.longdouble)
        rays /= np.sqrt(np.sum(rays * rays, axis=2))[:, :, None]
        a, b, c = rays[:, 0], rays[:, 1], rays[:, 2]
        numerator = np.sum(a * np.cross(b, c), axis=1)
        denominator = 1 + np.sum(a * b, axis=1) + np.sum(b * c, axis=1) + np.sum(c * a, axis=1)
        result.append(float(np.sum(np.arctan2(numerator, denominator)) / (2 * pi)))
    return np.array(result)
