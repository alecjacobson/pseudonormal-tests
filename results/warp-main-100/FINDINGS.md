# Warp comparison findings

Run completed 2026-11-23T20:34:07.679007+00:00. Warp `1.21.0.dev0`, [main commit `915c3d4f5643`](https://github.com/NVIDIA/warp/commit/915c3d4f56432e557339752eb619a7930c813d66). Device: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb (`cuda:0`).

Reused the initial experiment's 100 models and 1,000,000 saved samples, with no resampling. Every dataset asset was reused from the local cache. All comparisons use saved `igl_sign_w` as the reference and the same mask `igl_d > 0.001 * bbd` (966,656 points). Signs are inside +1, outside -1. The reference is a numerical algorithm, not independently established ground truth.

| Method | Disagreements | % of all samples | % beyond cutoff | Models affected | Before cutoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| libigl pseudonormal | 7 | 0.0007 | 0.000724145922 | 4 | 7 |
| Warp pseudonormal | 291 | 0.0291 | 0.0301037805 | 48 | 295 |
| Warp winding | 0 | 0 | 0 | 0 | 0 |
| Warp parity | 0 | 0 | 0 | 0 | 0 |

Warp pseudonormal's counted disagreements comprise 86 inside predictions and 205 outside predictions relative to libigl winding. Methods agreeing with the reference even before the cutoff: Warp winding, Warp parity.

## Method and parameters

Warp calls `mesh_query_point_sign_normal`, `mesh_query_point_sign_winding_number`, and `mesh_query_point_sign_parity` separately. Each distance is `length(p - mesh_eval_position(mesh, face, u, v))` from that call. Mesh winding support is explicitly enabled, preventing the winding API's ray-vote fallback. Vertices and samples are converted to float32 for Warp without normalization; libigl uses float64. The original connectivity and orientation remain.

Pseudonormal epsilon: `0.001`; winding accuracy: `2.0`, threshold: `0.5`. Parity uses **1 ray(s) per query**, perturbation scale `0.1` and Warp's fixed internal RNG seed 42. This is a one-ray parity test through the voting API. Use `--parity-rays 3` or another positive odd count for multi-ray majority voting. Warp winding uses its native threshold; libigl rounds its raw winding values.

## Distance agreement

Absolute distance differences are normalized by each model's bounding-box diagonal.

| Warp distance path | Mean / bbd | 99th percentile / bbd | Maximum / bbd | Count > 1e-5 bbd |
| --- | ---: | ---: | ---: | ---: |
| p | 8.45684348e-08 | 2.2266572e-07 | 0.000355729024 | 1467 |
| w | 8.45686675e-08 | 2.22671431e-07 | 0.000355729024 | 1467 |
| parity | 8.45686675e-08 | 2.22671431e-07 | 0.000355729024 | 1467 |

[distance_outliers.csv](distance_outliers.csv) records all 1,467 samples exceeding `1e-5 * bbd` in any path, across 10 models. [distance_diagnostics.csv](distance_diagnostics.csv) recomputes the worst case per affected model using libigl's double-precision distance query with exactly Warp's float32-rounded geometry and samples.

| file_id | Point index | Original libigl d | libigl d on float32 inputs | Warp d |
| --- | ---: | ---: | ---: | ---: |
| 472081 | 8116 | 0.002065981008 | 0.002065938761 | 0.08041140437 |
| 1063854 | 3163 | 0.02647093203 | 0.02647111545 | 0.08699116111 |

The two largest discrepancies persist with identical rounded inputs, so input rounding alone does not explain them. The returned nearest triangles differ. A root cause in the distance query has not been established.

## Timings

Sums over 100 batches of 10,000 points, one batch per method per model. Each GPU query timing includes launch and synchronization plus distance, sign, and closest-point outputs. No per-model warmup or repetitions. Mesh construction, transfers, and module loading are measured separately.

| Step | Seconds |
| --- | ---: |
| reference load | 0.181616 |
| mesh load | 0.044498 |
| input conversion upload | 0.076858 |
| mesh build | 0.051173 |
| output allocation | 0.008115 |
| pseudonormal query | 0.061536 |
| winding query | 0.351243 |
| parity query | 0.060395 |
| result download | 0.065656 |
| point output | 2.783339 |
| model total | 3.912491 |
| Module loading (before queries) | 0.136267 |
| Whole run | 4.730490 |

Warp kernels had been compiled during validation before this run; module loading reused the shared kernel cache. Native source build time is outside the benchmark. The original libigl CPU timings include acceleration-structure construction and normals inside each call. These timing scopes and devices differ; ratios are not an isolated algorithmic speed comparison.

## Validation and retained data

The full point audit passed; see [validation.json](validation.json). Analytic integration tests exercise CPU and CUDA cube distances, cavity signs, and the shared cutoff. One analytic case is marked as an expected failure: at the exact center of a symmetric empty cube cavity, the pinned Warp pseudonormal query returns inside on both CPU and CUDA. This observed behavior is preserved in `test_warp_benchmark.py`; it is separate from the random benchmark.

Join [per_model.csv](per_model.csv) to the dataset's `geometry_data.csv` on `file_id` or to `input_summary.csv` on `ID`. Original metadata is retained in [selected_metadata.csv](../initial-100/selected_metadata.csv). [disagreements.csv](disagreements.csv) contains the union of counted cases with coordinates, raw libigl winding, signs, distances and Warp closest-face indices. Full point arrays remain cached locally and are excluded from Git. [config.json](config.json) records source commit, native library hashes, baseline hashes, precision, parameters and timing scopes. [cache_access.json](cache_access.json) confirms reuse of every dataset asset.

Prepared with OpenAI Codex.
