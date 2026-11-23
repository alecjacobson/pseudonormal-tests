# Benchmark findings

Run completed 2026-11-23T21:05:37.099038+00:00. Dataset revision `2d5d3b2f3cd3711028ad75b12788c13b25559ec6`; seed `20261123`. Sampled 4963 of 4,963 eligible models and 10,000 points per model. Eligibility requires solid, vertex-manifold, and edge-manifold metadata flags; multiple components are allowed.

**1271 disagreements beyond the cutoff across 177 models**, out of 49,630,000 sampled points: **0.00256095%**. Among the 48,001,158 points with `d > 0.001 * bbd`, the disagreement rate is 0.00264785279%. There were 1282 sign disagreements before applying the cutoff. There were 0 failed models and 0 models with nonbinary rounded winding values.

## Models with disagreements

| file_id | Faces | Components | Points beyond cutoff | Disagreements | % of all points | % beyond cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 153100 | 216 | 1 | 9,763 | 277 | 2.77 | 2.837243 |
| 37285 | 1,254 | 1 | 9,856 | 141 | 1.41 | 1.430601 |
| 1535137 | 5,156 | 1 | 9,746 | 65 | 0.65 | 0.6669403 |
| 1452672 | 302 | 1 | 9,852 | 59 | 0.59 | 0.5988632 |
| 41360 | 10,058 | 1 | 9,836 | 56 | 0.56 | 0.5693371 |
| 85770 | 80 | 4 | 9,807 | 51 | 0.51 | 0.5200367 |
| 472183 | 25,288 | 1 | 9,633 | 44 | 0.44 | 0.4567632 |
| 472179 | 14,548 | 1 | 9,437 | 39 | 0.39 | 0.4132669 |
| 472185 | 25,332 | 1 | 9,629 | 30 | 0.3 | 0.3115588 |
| 42040 | 7,532 | 1 | 9,636 | 27 | 0.27 | 0.2801993 |
| 111008 | 256 | 1 | 9,869 | 25 | 0.25 | 0.2533185 |
| 111011 | 1,536 | 6 | 9,811 | 24 | 0.24 | 0.2446234 |
| 472189 | 11,688 | 1 | 9,692 | 22 | 0.22 | 0.2269913 |
| 74730 | 46,632 | 1 | 9,847 | 18 | 0.18 | 0.1827968 |
| 109130 | 7,600 | 3 | 9,798 | 15 | 0.15 | 0.1530925 |
| 111009 | 274 | 1 | 9,903 | 15 | 0.15 | 0.1514693 |
| 472172 | 14,548 | 1 | 9,431 | 15 | 0.15 | 0.1590499 |
| 39159 | 24,284 | 1 | 9,816 | 12 | 0.12 | 0.1222494 |
| 107041 | 3,450 | 1 | 9,719 | 12 | 0.12 | 0.1234695 |
| 62286 | 4,680 | 1 | 9,729 | 10 | 0.1 | 0.1027855 |

Showing the 20 largest counts among 177 affected models. All affected models are retained in per_model.csv.

The complete [per-model CSV](per_model.csv) joins to the dataset geometry CSV on `file_id`, or to `input_summary.csv` on `ID`. The [selected metadata](selected_metadata.csv) retains all geometry columns. Every counted case, including coordinates and raw winding value, is in [disagreements.csv](disagreements.csv).

Observed `(sign_p, sign_w)` pairs: `[(-1, 1), (1, -1)]`. The disagreement points have `d / bbd` between 0.001063232 and 0.2680578. This comparison does not independently establish which method is correct.

## Timings

| Step | Seconds |
| --- | ---: |
| Initial download (archive and metadata) | 15.432130 |
| Initial archive extraction | 34.233328 |
| Current run cache access | 0.000911 |
| Metadata CSV loading | 0.051941 |
| Mesh loading, summed over models | 2.264397 |
| Pseudonormal signed-distance calls, summed | 178.839664 |
| Fast winding calls, summed | 174.109069 |
| Per-point output compression/writing, summed | 129.082624 |
| Benchmark wall time with dataset cached | 491.580032 |
| Append-only checkpoints | 0.361848 |
| Final CSV export | 0.062431 |

Environment: Python 3.12.3, libigl 2.6.3, NumPy 2.5.5, AMD EPYC 9B45; `IGL_NUM_THREADS=4`, cgroup CPU limit `400000 100000`. The run uses the libigl CPU implementation.

Each method was called once per model, without warmup. Both call timings include internal acceleration-structure setup; pseudonormal timing also includes the distance query. These are batch costs, not isolated sign evaluation or precomputed-tree query timings. Mesh loading excludes the separately timed download and extraction. OS file caching and shared machine load can affect timings.

## Validation and reproducibility

The repository's analytic tests cover cube distances and signs, a two-component cavity boundary, the strict cutoff, denominator choices, and deterministic sampling. The point-output audit passed for every retained sample; see [validation.json](validation.json). Cache hit records for this run are in [cache_access.json](cache_access.json).

Reproduce with the committed lockfile, default parameters, and a fresh output directory. Exact selected IDs and source hashes are retained in the result files. Full per-point arrays are cached locally in `points/`; they are not committed.

Prepared with OpenAI Codex.
