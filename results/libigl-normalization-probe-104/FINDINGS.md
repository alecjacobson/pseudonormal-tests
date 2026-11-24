# Final vertex/edge normalization probe

All **369 nonfinite tuned-float query normals** in the full validation originate
in the final `N.rowwise().normalize()` in **`per_edge_normals`**. None originate
in vertex normalization. Face normals and the accumulated vertex/edge sums are
finite on the 104 tested models.

The immediate division failure has two causes:

- **368 queries:** the selected edge's accumulated normal is exactly `[0,0,0]`
  in float. Normalization divides by zero and returns NaNs.
- **1 query:** model `1233195`, sample `1521`, edge `3324` (vertices `1046,1057`),
  adjacent faces `308,307`. Their normals nearly cancel, leaving
  `[2.6072526112518294e-20, 1.5881867761018131e-21, 0]`. Its float squared norm
  is `6.82299229e-40`, a positive subnormal. Eigen 3.4's default SSE fast square
  root flushes this to zero during vectorized rowwise normalization, producing
  `[inf,inf,nan]`. Scalar square root instead gives `2.6120858e-20`.

The standalone [reproducer](../../native_precision/normalization_reproducer.cpp)
checks both cases. With `EIGEN_FAST_MATH=0`, the nonzero vector normalizes to
approximately `[0.998149693,0.0608014762,0]`; the exactly zero vector still
produces NaNs. This is Eigen's own default fast-math behavior; the experiment
does **not** use the compiler's `-ffast-math` option. Exact outputs and compile
flags are in [validation.json](validation.json).

## Ablation on cached samples

The original 100 models plus the five models with nonfinite queries overlap in
one model, giving **104 models and 1,040,000 points per variant**. This is a
deliberately enriched diagnostic subset, not a new estimate of full-set error.
Float/double geometry, closest points, face normals, weights, original scales,
samples, winding reference, and the distance cutoff all remain unchanged.
Epsilons remain `0.003` for float and `1e-9` for double, with area threshold
`1e-4`. Default tolerances were also replayed.

| Tuned setting | Float disagreements | Nonfinite query normals | Exactly zero query normals | Double disagreements |
| --- | ---: | ---: | ---: | ---: |
| Original normalization | 365 | 369 | 0 | 277 |
| Omit vertex normalization | 365 | 369 | 0 | 277 |
| Omit edge normalization | 8 | 0 | 368 | 277 |
| Omit both | 8 | 0 | 368 | 277 |

Every previously finite classification stays unchanged, for both precisions,
both tolerance settings, and all four normalization variants. The normalized
variant reproduces the saved signs and normal bytes exactly. Renormalizing
the copied raw accumulators also reproduces the original normal-table bytes.
The run took **20.46 seconds**, with no mesh downloads or distance queries.

Of the 369 sign changes with edge normalization omitted, 363 resolve a counted
disagreement and six introduce one. **This does not establish that the zero
normal classifications are repaired.** A NaN dot product previously took the
inside branch; a zero dot product now takes the outside branch. Of the 368 zero
vectors, 362 happen to match winding and six disagree. The remaining two of
the eight disagreements have finite nonzero normals.

The tiny nonzero edge normal is a distinct case: omitting normalization retains
its direction and gives the winding-matching outside result. In exact arithmetic,
dividing a finite nonzero final pseudonormal by a positive length is unnecessary
for the dot-product sign test. Unit **face** normals are still required to
preserve the intended uniform edge and angle-weighted vertex sums. Removing
face normalization would change those weights.

## Scope and retained evidence

This experiment changes only local generated copies of the final vertex/edge
normal accumulators; upstream libigl and its PRs are unchanged. Removing
normalization globally from normal-producing APIs would also change their
returned-vector contract. Zero/cancelled pseudonormals still need explicit
treatment in a robust classifier.

Double had no nonfinite **query** normals in this subset, but model `57854`
contains two nonfinite double edge-normal table entries that these queries
do not select. Absence of invalid returned queries does not imply that every
precomputed normal is finite.

- [Per-model comparisons](per_model.csv), [aggregates](aggregate.csv)
- [Normal-table diagnostics](normal_diagnostics.csv)
- [All 369 traced invalid queries](invalid_queries.csv), including matched edge IDs
- [Configuration/provenance](config.json), [point audit and reproducer](validation.json)

All 16,640,000 point/variant results were independently recounted. Cached point
arrays remain local in `points/`, excluded from Git. The normal generators keep
the upstream summation code and remove only the final normalization, recording
source and generated-header hashes.

Prepared by OpenAI Codex at the repository owner's request.
