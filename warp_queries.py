"""Evaluate Warp's native pseudonormal and winding query paths on identical points."""

from time import perf_counter

import numpy as np
import warp as wp

WARP_METHODS = ("p", "w", "parity")


@wp.kernel
def pseudonormal_query_kernel(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_distance: float,
    epsilon: float,
    distances: wp.array[float],
    signs: wp.array[int],
    faces: wp.array[int],
    closest_points: wp.array[wp.vec3],
    valid: wp.array[int],
):
    index = wp.tid()
    query = wp.mesh_query_point_sign_normal(mesh_id, points[index], max_distance, epsilon)
    valid[index] = int(query.result)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[index] = wp.length(points[index] - closest)
        signs[index] = wp.where(query.sign < 0.0, 1, -1)
        faces[index] = query.face
        closest_points[index] = closest
    else:
        distances[index] = 0.0
        signs[index] = 0
        faces[index] = -1
        closest_points[index] = wp.vec3(0.0)


@wp.kernel
def winding_query_kernel(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_distance: float,
    accuracy: float,
    distances: wp.array[float],
    signs: wp.array[int],
    faces: wp.array[int],
    closest_points: wp.array[wp.vec3],
    valid: wp.array[int],
):
    index = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(
        mesh_id, points[index], max_distance, accuracy, 0.5
    )
    valid[index] = int(query.result)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[index] = wp.length(points[index] - closest)
        signs[index] = wp.where(query.sign < 0.0, 1, -1)
        faces[index] = query.face
        closest_points[index] = closest
    else:
        distances[index] = 0.0
        signs[index] = 0
        faces[index] = -1
        closest_points[index] = wp.vec3(0.0)


@wp.kernel
def parity_query_kernel(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_distance: float,
    ray_count: int,
    perturbation_scale: float,
    distances: wp.array[float],
    signs: wp.array[int],
    faces: wp.array[int],
    closest_points: wp.array[wp.vec3],
    valid: wp.array[int],
):
    index = wp.tid()
    query = wp.mesh_query_point_sign_parity(
        mesh_id, points[index], max_distance, ray_count, perturbation_scale
    )
    valid[index] = int(query.result)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[index] = wp.length(points[index] - closest)
        signs[index] = wp.where(query.sign < 0.0, 1, -1)
        faces[index] = query.face
        closest_points[index] = closest
    else:
        distances[index] = 0.0
        signs[index] = 0
        faces[index] = -1
        closest_points[index] = wp.vec3(0.0)


def query_warp(
    vertices,
    faces,
    points,
    device,
    *,
    epsilon=0.001,
    accuracy=2.0,
    parity_rays=1,
    parity_perturbation=0.1,
):
    """Return both query results with separate setup, query, and transfer timings."""
    if len(points) == 0:
        raise ValueError("At least one query point is required.")
    if faces.min() < 0 or faces.max() >= len(vertices) or faces.max() > np.iinfo(np.int32).max:
        raise ValueError("Warp requires valid int32 triangle indices.")
    started = perf_counter()
    vertices32 = np.asarray(vertices, dtype=np.float32)
    points32 = np.asarray(points, dtype=np.float32)
    if not np.isfinite(vertices32).all() or not np.isfinite(points32).all():
        raise ValueError("Coordinates must remain finite after float32 conversion.")
    bbd = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not np.isfinite(bbd) or bbd <= 0:
        raise ValueError("The bounding-box diagonal must be positive and finite.")
    mesh_points = wp.array(vertices32, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(faces.astype(np.int32).ravel(), dtype=wp.int32, device=device)
    query_points = wp.array(points32, dtype=wp.vec3, device=device)
    wp.synchronize_device(device)
    timings = {"input_conversion_upload_seconds": perf_counter() - started}
    started = perf_counter()
    mesh = wp.Mesh(points=mesh_points, indices=mesh_indices, support_winding_number=True)
    wp.synchronize_device(device)
    timings["mesh_build_seconds"] = perf_counter() - started
    if not mesh.support_winding_number:
        raise ValueError("Winding support is required; ray-voting fallback is not allowed.")
    started = perf_counter()
    outputs = {}
    for method in WARP_METHODS:
        outputs[method] = [
            wp.empty(len(points), dtype=dtype, device=device)
            for dtype in (wp.float32, wp.int32, wp.int32, wp.vec3, wp.int32)
        ]
    wp.synchronize_device(device)
    timings["output_allocation_seconds"] = perf_counter() - started
    # These synchronization boundaries are intentional: timings measure completed work.
    started = perf_counter()
    wp.launch(
        pseudonormal_query_kernel,
        len(points),
        inputs=[mesh.id, query_points, 2.0 * bbd, epsilon],
        outputs=outputs["p"],
        device=device,
    )
    wp.synchronize_device(device)
    timings["pseudonormal_query_seconds"] = perf_counter() - started
    started = perf_counter()
    wp.launch(
        winding_query_kernel,
        len(points),
        inputs=[mesh.id, query_points, 2.0 * bbd, accuracy],
        outputs=outputs["w"],
        device=device,
    )
    wp.synchronize_device(device)
    timings["winding_query_seconds"] = perf_counter() - started
    started = perf_counter()
    wp.launch(
        parity_query_kernel,
        len(points),
        inputs=[mesh.id, query_points, 2.0 * bbd, parity_rays, parity_perturbation],
        outputs=outputs["parity"],
        device=device,
    )
    wp.synchronize_device(device)
    timings["parity_query_seconds"] = perf_counter() - started
    started = perf_counter()
    result = {"warp_points": points32}
    for method in WARP_METHODS:
        for name, array in zip(
            ("d", "sign", "face", "closest_point", "valid"), outputs[method], strict=True
        ):
            result[f"warp_{name}_{method}"] = array.numpy()
    timings["result_download_seconds"] = perf_counter() - started
    result["max_vertex_rounding_over_bbd"] = (
        np.max(np.linalg.norm(vertices - vertices32, axis=1)) / bbd
    )
    result["max_query_rounding_over_bbd"] = np.max(np.linalg.norm(points - points32, axis=1)) / bbd
    return result, timings
