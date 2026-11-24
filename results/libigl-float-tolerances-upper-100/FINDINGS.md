# Float pseudonormal tolerance sweep

The same 100 Thingi10K models and 1,000,000 cached points were evaluated at original scale. The saved double-input winding signs and `original_d > 0.001 * original_bbd` mask remain fixed.

A single compiled experiment exposes both tolerances as function arguments. The classifier is generated mechanically from the patched libigl source, changing only the function name and replacing its two constants with parameters. Mesh normals are computed once per model; closest faces and points come from the cached patched-float run. No recompilation or nearest-point search occurs between settings.

**All 1,000,000 default classifications reproduce the saved float baseline:** zero sign changes, zero differing normal bytes, and zero validity changes.

Swept **80 parameter pairs**: 8 epsilon values × 10 area thresholds. A model-level split (seed `20261124`) assigns 50 models to tuning and 50 to holdout before evaluation. The selected parameters minimize the tuning disagreement count; the holdout half does not select them.

| Setting | Minimum double area | Epsilon | Tuning / 500,000 | Holdout / 500,000 | All / 1,000,000 | All % | Invalid queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 0.0001 | 1e-12 | 505 | 257 | 762 | 0.0762% | 1 |
| Selected on tuning half | 0.0001 | 0.003 | 0 | 3 | 3 | 0.0003% | 1 |
| Best over all models (exploratory) | 1e-06 | 0.01 | 0 | 2 | 2 | 0.0002% | 1 |

The tuning-selected setting reduces the full-sample count by **99.606%**. It resolves 759 default disagreements and introduces 0 others. It has 2 counted disagreements with finite outputs and 1 invalid queries. The double baseline is 7 disagreements (0.0007%).

![Tolerance grid](tolerance_grid.png)

[Download PDF](tolerance_grid.pdf). Complete grids: [all](grid_all.csv), [tuning](grid_tuning.csv), [holdout](grid_holdout.csv).

## Epsilon sweep at the default area threshold

| Epsilon | Disagreements | Percent of all points | Invalid queries |
| --- | ---: | ---: | ---: |
| 1e-12 | 762 | 0.0762% | 1 |
| 0.003 | 3 | 0.0003% | 1 |
| 0.01 | 5 | 0.0005% | 1 |
| 0.03 | 10 | 0.001% | 1 |
| 0.1 | 45 | 0.0045% | 0 |
| 0.2 | 162 | 0.0162% | 0 |
| 0.3 | 385 | 0.0385% | 1 |
| 1 | 3673 | 0.3673% | 0 |

## Interpretation and timings

Epsilon is dimensionless in the barycentric branch, but measures coordinate distance in the small-area vertex/edge checks. The area threshold compares twice triangle area in squared coordinate units. These absolute tolerances were tested at original mesh scales; this does not establish scale-independent defaults. No upstream defaults or classification algorithm were changed.

Normal preprocessing: 3.841 s total. Classifier calls across all 80 settings: 8.107 s total. Sweep wall time including loading, verification and compressed output: 18.081 s. These classifier-only timings exclude nearest-point queries and are not comparable to the earlier full-query timings.

See [config.json](config.json) for exact float32 parameter values, source/module hashes and split IDs; [validation.json](validation.json) for the independent point-array recount; [per_model.csv](per_model.csv) and [selected_cases.csv](selected_cases.csv) for model/point diagnostics. All setting signs and validity flags remain cached locally by `(file_id, setting_id, sample_index)`.

Prepared by OpenAI Codex.
