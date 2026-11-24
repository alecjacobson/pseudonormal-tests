# Closest-feature transport: full-mask validation

**Preserving the nearest query's feature fixes all 379 diagnosed double
classification failures, with no new double disagreements anywhere in the
49.63 million samples.** Float also improves substantially, with four new
disagreements on an exactly collapsed triangle.

| Route | Tuned baseline | Closest-feature selection | Resolved | Introduced | Models with disagreements, before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Double | 441 | **62** | 379 | **0** | 17 → 2 |
| Float | 1,061 | **150** | 915 | **4** | 147 → 18 |

These primary counts use the original `d > 0.001 * bbd` mask. Both routes use
all **4,963 models**, **10,000 cached points per model**, original mesh scales,
and the same saved double winding reference. There are 49,630,000 total samples
and 48,001,158 samples beyond the cutoff. New disagreement rates as percentages
of all samples are **0.0001249244%** for double and **0.0003022366%** for float.

The original 100-model subset gives double **0 → 0** and float **3 → 1**;
the remaining float disagreement has an invalid normal. The additional 4,863
models give double **441 → 62** and float **1,058 → 149**.

## Implementation

Core commit
[`4a4a31137ab5e4d8973a284e1708fa911367242c`](https://github.com/libigl/libigl/commit/4a4a31137ab5e4d8973a284e1708fa911367242c)
is on branch `fix/pseudonormal-closest-feature`, based on the scalar/tolerance
commit `a72d3b843889289d2a8fdab6756d10c1e4714597` from PR 2583.
The change is available as [draft PR 2586](https://github.com/libigl/libigl/pull/2586),
stacked on PR 2583. The exact diff is also saved as [libigl.patch](libigl.patch).

1. New AABB query overloads return the winning closest point's barycentric
   coordinates. Traversal carries them with the face index and closest point,
   updating all outputs under the same strict distance comparison.
2. A new `pseudonormal_test` overload uses their zero coordinates to select the
   face, edge, or vertex normal. It does not recompute barycentrics from a
   rounded Cartesian point and applies no area or feature epsilon.
3. Both 3D production signed-distance entry points use this information.
   Normal construction, the nearest-triangle arithmetic, and distance
   comparisons retain their previous behavior. Existing query signatures and
   the standalone tolerance-based classifier remain available; 2D behavior
   is unchanged.

The core change has no Python signature change. The experiment compiles the
actual binding source against the new core and compares it with the cached
production module for the previous tuned implementation. The existing Python
binding PR still pins its previous core revision.

## Double: the predicted 379 fixes and no regressions

All **277** double failures on `153100` disappear. Across the 15 affected
classification-error models, the new route fixes all 281 wrong-edge choices
and all 98 face-instead-of-edge choices identified by the earlier exact
diagnostic study.

Only the two previously identified non-classifier cases remain:

| Model | Remaining double disagreements | Prior diagnosis |
| --- | ---: | --- |
| [1452672](https://ten-thousand-models.appspot.com/detail.html?file_id=1452672) | 59 | Residuals on cached geometry with exact self-intersections |
| [1652976](https://ten-thousand-models.appspot.com/detail.html?file_id=1652976) | 3 | Native global nearest search chooses the wrong feature among almost-equal distances |

This implementation does not repair either of those upstream geometry/search
conditions. See the [17-model diagnostic report](../libigl-double-diagnostics-17/FINDINGS.md).

## Float: four regressions on an exact geometric tie

All four new disagreements occur on **91147**, samples **470, 3717, 7295,
and 8348**, whose cached closest face is triangle **70**. Rounding its three
vertices to float produces:

```text
[-8.754549,  1.4190838, 11.655327]
[-8.754549,  7.4190836, 11.655327]
[-8.754549, -4.7809160, 11.655327]
```

The exact binary32 values are collinear. Exact rational mesh-wide queries find
**two equally closest edge features**, 42 and 58, at the same point and
distance. Their pseudonormals disagree about inside/outside:

| Choice | Edge | Stored pseudonormal | Sign on these samples |
| --- | ---: | --- | --- |
| Previous tolerance classifier | 42 | approximately `[-0.508638, 0, 0.860981]` | Outside, matching winding |
| Feature retained by nearest routine | 58 | `[1, 0, 0]` | Inside |

Normals computed at 80 decimal digits retain these opposite signs. Direct
solid-angle sums on the float-rounded mesh confirm the outside winding sign.
Thus these four regressions are exposed by carrying the nearest routine's
choice through a degenerate, nonunique feature configuration. They are not
fixed by higher-precision construction of those normals. No special-case
fallback or tolerance was added to conceal them. The exact tie distances,
coordinates, features, and normals are in
[regression_diagnosis.json](regression_diagnosis.json).

## Invalid normals

Float queries with nonfinite selected normals decrease from **369 to 90**:
279 resolved, zero introduced. All 90 are beyond the distance cutoff.

| Model | Remaining invalid queries | Counted disagreements among them |
| --- | ---: | ---: |
| 153100 | 84 | 84 |
| 1233195 | 1 | 1 |
| 423825 | 5 | 0 |

Consequently, float's 150 counted disagreements comprise **65 with finite
normals and 85 with invalid normals**. The other five invalid outputs happen
to return a sign matching winding; they remain explicitly flagged as invalid.
The normal tables were not changed: the improvement comes from selecting
different entries. Double has no invalid outputs. All returned barycentrics
in both precisions were finite, nonnegative, and had positive support.

## Checks beyond the distance cutoff

The saved outputs were also recounted without filtering out near-surface
samples:

| Route | All-point baseline disagreements | All-point new disagreements | Resolved | Introduced |
| --- | ---: | ---: | ---: | ---: |
| Double | 452 | 73 | 379 | 0 |
| Float | 1,120 | 186 | 938 | 4 |

There are no newly introduced disagreements hidden by the cutoff in either
precision. Float resolves another 23 disagreements within the excluded band.
The double signs change on exactly the 379 diagnosed failures.

## Validation and timings

- Every cached point was replayed in both precisions. Old signs and normal
  bytes reproduce the tuned baseline exactly, and nearest-point bytes and
  unsigned distances are unchanged.
- Complete old/new signed-distance calls were additionally compared on the
  original 100 models plus all 17 double-failure models: **117 models and
  1.17 million queries per precision**. AABB output face indices, closest
  points, distances, signs, and normal bytes agree with the cached replay.
- An independent auditor recounts all saved point arrays, verifies every
  per-model metric, and exports changes and remaining failures.
- **19 C++ test cases / 1,166 assertions pass.** The thin-triangle regression
  fails for both float and double against the previous implementation.
  Tests cover explicit feature support, closest-feature transport through
  multi-node AABB traversal, equal-distance bounds, points/segments, and the
  existing scalar and 2D behavior.
- A targeted static archive of **33 translation units** links and runs the
  production float and double queries at two scales. This is a targeted
  static-link check, not a full-library build.

The full replay took **540.47 seconds (9 minutes)**. It includes normal setup,
baseline controls, output caching, and the 117-model end-to-end comparison;
the independent recount and regression diagnosis are separate. The dataset
was entirely cached, with no download. Aggregate measured times:

| Operation | Float seconds | Double seconds |
| --- | ---: | ---: |
| Normal setup, 4,963 models | 147.95 | 150.70 |
| Old classifier using cached geometry, 49.63M queries | 6.14 | 5.98 |
| Winning-triangle replay plus new classifier, 49.63M queries | 4.37 | 4.57 |
| Complete old pipeline, 117 models | 24.42 | 24.05 |
| Complete new pipeline, 117 models | 23.93 | 23.86 |

Loading took 28.27 seconds and writing compressed outputs 83.67 seconds.
The complete-query timings are similar. These are single-pass measurements
on a shared machine, with some compilation concurrent with the run; the small
timing differences do not establish a speedup.

## Reproduction and artifacts

See the [README instructions](../../README.md#carrying-the-closest-feature-into-normal-selection).

- [config.json](config.json): measured module/source hashes, original subset,
  selection rules, dataset revision, and paths.
- [validation.json](validation.json): core commit, C++ checks, static build,
  regression-before-fix check, and final-source provenance. The only changes
  to arithmetic-source files after the measured modules were built are static
  instantiations; removing those reproduces the recorded source hashes.
- [per_model.csv](per_model.csv), [timings.csv](timings.csv),
  [summary.json](summary.json), and [audit.json](audit.json).
- [changed_and_remaining_points.csv](changed_and_remaining_points.csv),
  [remaining_points.csv](remaining_points.csv), and
  [introduced.json](introduced.json), indexed by dataset file ID and original
  sample index. In the CSVs, `new_vs_default` / `resolved_vs_default` compare
  against the **tuned** baseline, inherited names from the shared metrics helper.
- Full per-point normals, barycentrics, signs, and validity arrays remain
  cached locally under `points/`; dataset meshes and original samples are
  reused from their existing caches.

Prepared by OpenAI Codex at the repository owner's request.
