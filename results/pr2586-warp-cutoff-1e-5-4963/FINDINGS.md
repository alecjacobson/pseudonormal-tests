# Full comparison with the surface cutoff reduced 100-fold

Fresh queries on **4,963 eligible meshes × 10,000 points = 49,630,000 points**.
The primary cutoff is now **`d > 0.00001 * bbd`**, down from `d > 0.001 * bbd`.
It includes **49,613,294** points, adding **1,612,136** closer points.
All methods use the same original double distance and bounding-box diagonal.

| Method | Old cutoff | New cutoff | Newly included | % of all samples | Affected models |
| --- | ---: | ---: | ---: | ---: | ---: |
| libigl PR pseudonormal, double | 62 | 62 | 0 | 0.000124924441% | 2 |
| libigl PR pseudonormal, float | 150 | 158 | 8 | 0.000318355833% | 22 |
| Warp pseudonormal | 11,817 | 12,016 | 199 | 0.0242111626% | 2,359 |
| Warp winding | 0 | 0 | 0 | 0% | 0 |
| Warp parity (1 ray) | 0 | 0 | 0 | 0% | 0 |

Percentages divide by all samples; `summary.json` also reports percentages among eligible points.
The reference is freshly computed libigl fast winding with double inputs and NumPy rounding
(inside +1, outside -1). Its implementation internally uses float arithmetic. It is a numerical
reference, not an independent proof of the correct sign.

## Reproducibility and checks

- Core PR [#2586](https://github.com/libigl/libigl/pull/2586), commit `e810369e42090e66fc805370f9969771e93b431f`.
- Actual Python binding sources at `d3b61ac35fbbe4e0e77af588306389418973dfdd`; both signed-distance and fast-winding bindings rebuilt against the PR.
- Warp main revision `915c3d4f56432e557339752eb619a7930c813d66` on `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb`; original settings retained (pseudonormal epsilon 0.001, winding accuracy 2, parity one ray).
- Original scales, same dataset revision and eligible mask, seed 20261123. Every cached point array was verified against its per-model seeded generator.
- Both libigl pseudonormal routes, libigl winding, and all three Warp methods were rerun for every model. Libigl signs/normal bytes/validity match the earlier feature replay; double distances and closest geometry match the original reference. All Warp distances, signs, faces, closest points, and validity flags match the earlier run.
- Maximum raw winding difference from the cached reference: `0.0`. All reference signs match.
- Independent point-file audit reproduces every per-model count and aggregate; no model failed.
- Fresh binding passed all 26 precision tests. Four-model smoke test includes the known float degeneracy and double residual models; a strict-boundary fixture checks both distance thresholds and invalid-normal counts.

## Newly included points

The added band contains **201 distinct disagreement points** across the methods.
Eight disagree for libigl float and 199 for Warp pseudonormals; six points occur
in both groups. All eight new float cases have finite outputs:

| Model | Original point indices |
| --- | --- |
| 384087 | 6474 |
| 99467 | 570, 5927 |
| 97489 | 59 |
| 49545 | 7463 |
| 135218 | 4456, 6759, 7804 |

Warp's eight raw winding and ten raw parity disagreements all remain within
`d <= 1e-5 * bbd`; neither method gains a counted disagreement.

## Invalid normals

Libigl float has **90 invalid-normal queries**. Its new-cutoff disagreement total contains **73 finite-output cases** and **85 invalid-normal cases**.
Returned signs are retained for continuity with previous experiments; invalid outputs are also identified per point and per model. Other methods' invalid-query counts are in `summary.json`.

## Timings

The full fresh run took **2067.95 seconds (34.47 minutes)**. The independent audit took **57.79 seconds**.
Single passes on shared hardware; no timing repetitions or speedup claim. Libigl times include
internal tree/normal construction; Warp kernel times exclude separately measured setup/transfers.

| Operation | Seconds |
| --- | ---: |
| libigl double signed distance | 800.740 |
| libigl float signed distance | 804.898 |
| libigl fast winding | 102.391 |
| Warp mesh builds | 2.573 |
| Warp pseudonormal kernels | 3.313 |
| Warp winding kernels | 16.131 |
| Warp parity kernels | 3.044 |
| Mesh load/hash | 3.762 |
| Reference/control loading | 42.883 |
| Compressed output | 259.717 |

Dataset cache access took 0.000706 seconds; all downloads/extraction were cache hits. No dataset download occurred.
Checkpointing and CSV export took 0.642 seconds combined.

## Distance agreement

Errors below are relative to the original double unsigned distance and normalized by bbd.

| Method | Maximum absolute error / bbd | Mean absolute error / bbd |
| --- | ---: | ---: |
| libigl PR pseudonormal, double | 0 | 0 |
| libigl PR pseudonormal, float | 0.000762837795 | 4.2285861e-08 |
| Warp pseudonormal | 0.000762837795 | 4.17285941e-08 |
| Warp winding | 0.000762837795 | 4.17248671e-08 |
| Warp parity (1 ray) | 0.000762837795 | 4.17248671e-08 |

## Artifacts

- `per_model.csv` and `selected_metadata.csv` join to the dataset on `file_id` (or `input_summary.csv` on `ID`).
- `disagreements.csv` contains every counted disagreement point across methods, with coordinates, signs, validity, distances, and nearest faces.
- `newly_included_disagreements.csv` restricts this to the newly admitted distance band; `affected_models.csv` lists each affected method/model.
- `distance_outliers.csv` records distance errors exceeding the new relative cutoff.
- `config.json`, `cache_access.json`, `summary.json`, and `audit.json` preserve provenance, timings, and verification. `build_commands.txt` records the exact compiler commands, and `cached_recount.json` records the separate recount of earlier outputs.
- Fresh per-point outputs remain in local ignored `points/<file_id>.npz`; the original query coordinates remain in the referenced baseline cache.

## CI follow-up

Core commit `625e4341c085f338758a2bb529f1c335add89831` fixes missing static-library instantiations and a Windows macro collision in a test. It changes no query arithmetic; the full-run binary and recorded benchmark revision remain unchanged. The expanded suite passes **22 cases / 1,186 assertions** with `-Dnear=`. A targeted 33-translation-unit static build passes, while the same driver against the previous AABB source fails to link the three affected signatures. See [ci_followup.json](ci_followup.json).
