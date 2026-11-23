# Warp comparison findings

Run completed 2026-11-23T21:09:14.094283+00:00. Warp `1.21.0.dev0`, [main commit `915c3d4f5643`](https://github.com/NVIDIA/warp/commit/915c3d4f56432e557339752eb619a7930c813d66). Device: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb (`cuda:0`).

Reused the libigl baseline's 4963 models and 49,630,000 saved samples, with no resampling. Every dataset asset was reused from the local cache. All comparisons use saved `igl_sign_w` as the reference and the same mask `igl_d > 0.001 * bbd` (48,001,158 points). Signs are inside +1, outside -1. The reference is a numerical algorithm, not independently established ground truth.

| Method | Disagreements | % of all samples | % beyond cutoff | Models affected | Before cutoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| libigl pseudonormal | 1271 | 0.002560951 | 0.00264785279 | 177 | 1282 |
| Warp pseudonormal | 11817 | 0.0238102 | 0.0246181561 | 2346 | 12058 |
| Warp winding | 0 | 0 | 0 | 0 | 8 |
| Warp parity | 0 | 0 | 0 | 0 | 10 |

Warp pseudonormal's counted disagreements comprise 5364 inside predictions and 6453 outside predictions relative to libigl winding. Methods agreeing with the reference even before the cutoff: none.

## Method and parameters

Warp calls `mesh_query_point_sign_normal`, `mesh_query_point_sign_winding_number`, and `mesh_query_point_sign_parity` separately. Each distance is `length(p - mesh_eval_position(mesh, face, u, v))` from that call. Mesh winding support is explicitly enabled, preventing the winding API's ray-vote fallback. Vertices and samples are converted to float32 for Warp without normalization; libigl uses float64. The original connectivity and orientation remain.

Pseudonormal epsilon: `0.001`; winding accuracy: `2.0`, threshold: `0.5`. Parity uses **1 ray(s) per query**, perturbation scale `0.1` and Warp's fixed internal RNG seed 42. This is a one-ray parity test through the voting API. Use `--parity-rays 3` or another positive odd count for multi-ray majority voting. Warp winding uses its native threshold; libigl rounds its raw winding values.

## Distance agreement

Absolute distance differences are normalized by each model's bounding-box diagonal.

| Warp distance path | Mean / bbd | 99th percentile / bbd | Maximum / bbd | Count > 1e-5 bbd |
| --- | ---: | ---: | ---: | ---: |
| p | 4.17285941e-08 | 3.65181779e-07 | 0.000762837795 | 22381 |
| w | 4.17248671e-08 | 3.6514526e-07 | 0.000762837795 | 22378 |
| parity | 4.17248671e-08 | 3.6514526e-07 | 0.000762837795 | 22378 |

[distance_outliers.csv](distance_outliers.csv) records all 22,382 samples exceeding `1e-5 * bbd` in any path, across 763 models. [distance_diagnostics.csv](distance_diagnostics.csv) recomputes the worst case per affected model using libigl's double-precision distance query with exactly Warp's float32-rounded geometry and samples.

| file_id | Point index | Original libigl d | libigl d on float32 inputs | Warp d |
| --- | ---: | ---: | ---: | ---: |
| 472084 | 6971 | 0.0009103952866 | 0.0009103820975 | 0.07603135705 |
| 384087 | 6474 | 0.0003944603838 | 0.0003944623797 | 0.004713279661 |

The largest discrepancies shown above persist with identical rounded inputs, so input rounding alone does not explain them. The returned nearest triangles differ in 2 of these 2 cases. A root cause in the distance query has not been established.

## Timings

Sums over 4963 batches of 10,000 points, one batch per method per model. Each GPU query timing includes launch and synchronization plus distance, sign, and closest-point outputs. No per-model warmup or repetitions. Mesh construction, transfers, and module loading are measured separately.

| Step | Seconds |
| --- | ---: |
| reference load | 8.972038 |
| mesh load | 1.783457 |
| input conversion upload | 3.219986 |
| mesh build | 2.057846 |
| output allocation | 0.406806 |
| pseudonormal query | 3.189852 |
| winding query | 16.040080 |
| parity query | 3.033867 |
| result download | 2.043512 |
| point output | 137.915638 |
| model total | 189.372530 |
| checkpoint | 0.470548 |
| csv export | 0.148677 |
| Module loading (before queries) | 0.100547 |
| Whole run | 192.226605 |

Warp kernels had been compiled during validation before this run; module loading reused the shared kernel cache. Native source build time is outside the benchmark. The original libigl CPU timings include acceleration-structure construction and normals inside each call. These timing scopes and devices differ; ratios are not an isolated algorithmic speed comparison.

## Validation and retained data

The full point audit passed; see [validation.json](validation.json). Analytic integration tests exercise CPU and CUDA cube distances, cavity signs, and the shared cutoff. One analytic case is marked as an expected failure: at the exact center of a symmetric empty cube cavity, the pinned Warp pseudonormal query returns inside on both CPU and CUDA. This observed behavior is preserved in `test_warp_benchmark.py`; it is separate from the random benchmark.

Join [per_model.csv](per_model.csv) to the dataset's `geometry_data.csv` on `file_id` or to `input_summary.csv` on `ID`. Original metadata is retained in [selected_metadata.csv](../full-4963/selected_metadata.csv). [disagreements.csv](disagreements.csv) contains the union of counted cases with coordinates, raw libigl winding, signs, distances and Warp closest-face indices. Full point arrays remain cached locally and are excluded from Git. [config.json](config.json) records source commit, native library hashes, baseline hashes, precision, parameters and timing scopes. [cache_access.json](cache_access.json) confirms reuse of every dataset asset.

Prepared with OpenAI Codex.
