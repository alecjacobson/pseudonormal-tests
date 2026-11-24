# Production validation of tuned pseudonormal defaults

Libigl PR [2583](https://github.com/libigl/libigl/pull/2583), commit
`a72d3b843889289d2a8fdab6756d10c1e4714597`, now uses a 3D feature epsilon of
`0.003` for float and `1e-9` for double. The area threshold remains `1e-4`.
Other scalar types and the 2D overload retain epsilon `1e-12`. No normalization
or feature-selection algorithm changes are included.

The new production `signed_distance` implementation was evaluated on the
original 100 models plus all five models with invalid outputs in the full
validation (104 unique models). It recomputed both nearest geometry and
pseudonormal classification on the same cached points, at original scales.

For **1,040,000 points per precision**, every sign, normal byte, and validity
flag matches the saved tuned runtime-classifier experiment. Unsigned distances,
closest points, and closest-face indices match the pre-tolerance results exactly.

| Group | Float disagreements | Float invalid queries | Double disagreements | Double invalid queries |
| --- | ---: | ---: | ---: | ---: |
| Original 100 models | 3 | 1 | 0 | 0 |
| All 104 validation models | 365 | 369 | 277 | 0 |

Both float rows contain only two disagreements with finite normal outputs.
The larger subset intentionally contains every known invalid model; its rate
is not an estimate for the complete dataset. The previous
[4,963-model frozen-tolerance validation](../libigl-tolerance-validation-4963/FINDINGS.md)
remains the broader evidence: float 48,268 → 1,061 disagreements and double
1,271 → 441, with float invalid queries increasing from 224 to 369.

The full production replay took **51.12 seconds**. Source revisions and the
tested patch are recorded in [config.json](config.json); the production binary
is retained locally. [Per-model results](per_model.csv) and
[pointwise audit summary](summary.json) are tracked. No points were resampled
or meshes downloaded, and reproduced point arrays were not duplicated.

The focused C++ suite passes **13 cases / 312 assertions**. New float/double
near-feature regressions fail with the old tolerances. The tiny-scale fixture
explicitly checks the new float behavior: all tested locations in its
1/1024-scale triangle lie within the absolute vertex tolerance. All **26 Python
precision tests** pass through the freshly compiled signed-distance binding,
and the float helper audit passes with `-Werror=double-promotion`.

The previously documented normal-normalization failures remain unchanged;
see the [normalization probe](../libigl-normalization-probe-104/FINDINGS.md).

Prepared by OpenAI Codex at the repository owner's request.
