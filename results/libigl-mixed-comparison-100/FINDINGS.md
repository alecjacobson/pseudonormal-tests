# Libigl scalar precision benchmark

The same 100 models and 1,000,000 cached query points were evaluated at their original scale and position. The winding reference and `original_d > 0.001 * original_bbd` mask were frozen. There are 966,656 points beyond that cutoff. The dataset filter allows multiple components.

| Variant | Counted returned-sign disagreements | % of all points | Affected models | Invalid queries |
| --- | ---: | ---: | ---: | ---: |
| original_double | 7 | 0.0007% | 4 | 0 |
| patched_double | 7 | 0.0007% | 4 | 0 |
| patched_single | 762 | 0.0762% | 45 | 1 |
| quantized_double | 9 | 0.0009% | 5 | 0 |
| original_single_mixed | 764 | 0.0764% | 45 | 1 |

Single precision changes the returned-sign disagreement rate by **0.0755 percentage points** (**108.857×** the double rate).

- `original_vs_wheel`: **0 differing bytes** across every signed distance, closest-face index, closest point, pseudonormal, and sign.
- `patched_double_vs_original`: **0 differing bytes** across every signed distance, closest-face index, closest point, pseudonormal, and sign.

## Original C++ float versus patched float

`original_single_mixed` uses the untouched pre-PR C++ source with float32 inputs and float outputs. The new Python float overload is used only to expose that existing C++ instantiation. Its `signed_distance.cpp` wrapper is identical to the wrapper used for the patched-float run. The original hardcoded double intermediates and math calls remain intact. This differs from the old Python behavior, which converted float32 inputs to double.

Changing from the old mixed arithmetic to the patched float arithmetic changes the rate by **-0.0002 percentage points**. There are **761 shared disagreements**, **1 newly introduced**, and **3 resolved** beyond the fixed cutoff. Signs change at 4 points overall, including 4 beyond the cutoff.

Nonfinite query outputs: 1 before and 1 after; 1 shared, 0 introduced, 0 resolved. The old mixed-precision result has 763 counted disagreements with finite outputs.

Closest-face indices change for 0 points. The maximum unsigned-distance difference between the two float variants is 0 × original `bbd` (p99 0; median 0). Per-model changes and diagnostic point identities are retained in the CSV files.

| Model | Sample index | Original d / bbd | Old mixed sign | Patched float sign | Winding reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42194 | 11 | 0.03476199 | +1 | -1 | -1 |
| 42194 | 2736 | 0.042307 | -1 | +1 | -1 |
| 42194 | 3956 | 0.02016988 | +1 | -1 | -1 |
| 42194 | 7703 | 0.006960485 | +1 | -1 | -1 |

Single precision returned nonfinite outputs at **1 query/queries**, including 1 beyond the cutoff. These are retained and flagged, not silently dropped. There are **761 disagreements with finite query outputs**. The table counts signs returned by the library; invalid normals do not provide trustworthy pseudonormal classifications.

- Invalid case: `file_id=1233195`, `sample_index=1521`, `d/bbd=0.00307145302`.

## Distance differences and timings

Errors below compare unsigned distances with the original double result and divide by original `bbd`.

| Variant | Median error / bbd | p99 error / bbd | Max error / bbd | Batch query seconds |
| --- | ---: | ---: | ---: | ---: |
| original_double | 0 | 0 | 0 | 21.446 |
| patched_double | 0 | 0 | 0 | 21.334 |
| patched_single | 4.1936e-09 | 2.14248e-07 | 0.000375773 | 21.398 |
| quantized_double | 3.10985e-09 | 9.37104e-08 | 1.09269e-06 | 21.170 |
| original_single_mixed | 4.1936e-09 | 2.14248e-07 | 0.000375773 | 21.229 |

Timings sum one batch call per mesh, including AABB and normal preprocessing. They do not isolate steady-state query throughput. Loading, conversion, writing, and cache access are recorded separately in each run directory. The dataset was reused locally, with no download. All source comparisons use the same compiler flags, including disabled floating-point contraction; the small extension builds the actual upstream `signed_distance.cpp` binding.

## Reproducibility and interpretation

Seed `20261123`; dataset revision `2d5d3b2f3cd3711028ad75b12788c13b25559ec6`; Python 3.12.3; Eigen 3.4.0; nanobind 2.7.0; GCC 13.3.0; four libigl threads. Model IDs, mesh hashes, original points, source commits, module hashes, and configurations are retained. See [summary.json](summary.json), [per_model.csv](per_model.csv), and [changed_cases.csv](changed_cases.csv). Per-model tables join to dataset geometry metadata on `file_id`; diagnostic points additionally use `sample_index`.

- `original_double`: libigl `b41f1944a703653ae0da81e6588650ba14c270ea`, bindings `6893d1b18f7fa4f5abf374f54d5e6b0c53c7d716`.
- `patched_double`: libigl `0641c7029519a6ad4e66b3823b40055c91115f12`, bindings `eac43e8fa1b29fb91645200287a17798774dce74`.
- `patched_single`: libigl `0641c7029519a6ad4e66b3823b40055c91115f12`, bindings `eac43e8fa1b29fb91645200287a17798774dce74`.
- `quantized_double`: libigl `0641c7029519a6ad4e66b3823b40055c91115f12`, bindings `eac43e8fa1b29fb91645200287a17798774dce74`.
- `original_single_mixed`: libigl `b41f1944a703653ae0da81e6588650ba14c270ea`, bindings `51d6dee06139682c08163012b00b5945095c38f1`.

The quantized-input control rounds vertices and points to float32 once, then evaluates their exact float64 representations in double. It separates input quantization from arithmetic precision. The original winding results remain a comparison reference, not independently proven ground truth. The scalar patch does not retune the feature classifier or its tolerances; that investigation is tracked in issue 2.

Prepared by OpenAI Codex.
