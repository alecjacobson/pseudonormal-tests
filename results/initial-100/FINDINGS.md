# Benchmark findings

Run completed 2026-11-23T20:14:31.791539+00:00. Dataset revision `2d5d3b2f3cd3711028ad75b12788c13b25559ec6`; seed `20261123`. Sampled 100 of 4,963 eligible models and 10,000 points per model. Eligibility requires solid, vertex-manifold, and edge-manifold metadata flags; multiple components are allowed.

**7 disagreements beyond the cutoff across 4 models**, out of 1,000,000 sampled points: **0.0007%**. Among the 966,656 points with `d > 0.001 * bbd`, the disagreement rate is 0.000724145922%. There were 7 sign disagreements before applying the cutoff. There were 0 failed models and 0 models with nonbinary rounded winding values.

## Models with disagreements

| file_id | Faces | Components | Points beyond cutoff | Disagreements | % of all points | % beyond cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 79195 | 39,996 | 1 | 9,939 | 2 | 0.02 | 0.02012275 |
| 47851 | 4,630 | 1 | 9,807 | 1 | 0.01 | 0.0101968 |
| 65282 | 5,252 | 1 | 9,895 | 2 | 0.02 | 0.02021223 |
| 472038 | 50,726 | 1 | 9,787 | 2 | 0.02 | 0.02043527 |

The complete [per-model CSV](per_model.csv) joins to the dataset geometry CSV on `file_id`, or to `input_summary.csv` on `ID`. The [selected metadata](selected_metadata.csv) retains all geometry columns. Every counted case, including coordinates and raw winding value, is in [disagreements.csv](disagreements.csv).

Observed `(sign_p, sign_w)` pairs: `[(1, -1)]`. The disagreement points have `d / bbd` between 0.02528343 and 0.1251916. This comparison does not independently establish which method is correct.

## Timings

| Step | Seconds |
| --- | ---: |
| Initial download (archive and metadata) | 15.432130 |
| Initial archive extraction | 34.233328 |
| Current run cache access | 0.000804 |
| Metadata CSV loading | 0.053366 |
| Mesh loading, summed over models | 0.060798 |
| Pseudonormal signed-distance calls, summed | 5.494726 |
| Fast winding calls, summed | 4.617520 |
| Per-point output compression/writing, summed | 2.609989 |
| Benchmark wall time with dataset cached | 13.118680 |

Environment: Python 3.12.3, libigl 2.6.3, NumPy 2.5.5, AMD EPYC 9B45; `IGL_NUM_THREADS=4`, cgroup CPU limit `400000 100000`. The run uses the libigl CPU implementation.

Each method was called once per model, without warmup. Both call timings include internal acceleration-structure setup; pseudonormal timing also includes the distance query. These are batch costs, not isolated sign evaluation or precomputed-tree query timings. Mesh loading excludes the separately timed download and extraction. OS file caching and shared machine load can affect timings.

## Validation and reproducibility

The repository's analytic tests cover cube distances and signs, a two-component cavity boundary, the strict cutoff, denominator choices, and deterministic sampling. The point-output audit passed for every retained sample; see [validation.json](validation.json). Cache hit records for this run are in [cache_access.json](cache_access.json).

Reproduce with the committed lockfile, default parameters, and a fresh output directory. Exact selected IDs and source hashes are retained in the result files. Full per-point arrays are cached locally in `points/`; they are not committed.

Prepared with OpenAI Codex.
