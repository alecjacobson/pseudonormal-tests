# Pseudonormal tests

Compare libigl and Warp mesh distances and signs on Thingi10K. Both the initial
100-model subset and the full set of 4,963 eligible models use 10,000 points
sampled uniformly in each model's axis-aligned bounding box. Warp reuses the exact
saved libigl samples.

The [full-set comparison](results/warp-main-full-4963/FINDINGS.md) covers
**49,630,000 points**, including 48,001,158 with `igl_d > 0.001 * bbd`. It uses
libigl winding as the reference and that same original distance mask for every
method. Percentages below use all sampled points as the denominator:

| Method | Disagreements | % of all samples | Models affected |
| --- | ---: | ---: | ---: |
| libigl pseudonormal | 1,271 | 0.00256095% | 177 |
| Warp pseudonormal | 11,817 | 0.0238102% | 2,346 |
| Warp winding | 0 | 0% | 0 |
| Warp parity (native default: 1 ray) | 0 | 0% | 0 |

Warp winding and parity have 8 and 10 disagreements respectively before the
distance cutoff; all are excluded by the cutoff. For Warp distances, the 99th
percentile absolute difference is about `3.65e-7 * bbd`; the maximum is
`7.63e-4 * bbd`.

Both passes completed without failures, reusing the dataset cache offline.
The libigl pass took **491.58 seconds**, and all three Warp methods plus their
loading and output took **192.23 seconds**: **11 minutes 24 seconds combined**,
excluding the subsequent audits. Appending checkpoints and exporting both final
CSVs cost **1.04 seconds total**. The libigl-only details are in
[full baseline findings](results/full-4963/FINDINGS.md).

The original [libigl subset findings](results/initial-100/FINDINGS.md) and
[Warp subset comparison](results/warp-main-100/FINDINGS.md) remain available.
The full run reproduced all original 100 models' samples, distances and signs
exactly; see [full-set validation](results/warp-main-full-4963/full_set_validation.json).

## Run

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked
uv run python -m unittest -v
uv run python benchmark.py --output results/my-run
uv run python summarize.py results/my-run
```

Defaults: 100 models, 10,000 points per model, seed `20261123`, distance cutoff
`0.001 * bbd`, and `IGL_NUM_THREADS=4`. `bbd` is the Euclidean length of the bounding
box diagonal. Use `--models`, `--points`, `--seed`, `--relative-cutoff`, or
`--threads` to change these values. Each run requires a fresh output directory;
existing results are not overwritten. Models run sequentially, with native libigl
parallelism limited by `IGL_NUM_THREADS`.

## Dataset and cache

The script uses the original NumPy representation of
[Thingi10K/Thingi10K on Hugging Face](https://huggingface.co/datasets/Thingi10K/Thingi10K),
pinned to commit `2d5d3b2f3cd3711028ad75b12788c13b25559ec6`. It downloads the full
4.09 GB `Thingi10K_npz.tar.gz` archive and three metadata CSVs, and extracts the
archive once. Both the Hugging Face download cache and the extracted meshes stay
under `.cache/thingi10k/` in this checkout. Use `--cache-dir /persistent/path` to
share an existing cache. A completed cache is reused without network requests;
the next run does not download or extract again. Extraction is locked and marked
complete only after it succeeds. Keep at least 20 GiB free before extraction.

The geometry metadata contains 9,997 rows. Eligibility is exactly:

```python
solid == 1 and vertex_manifold == 1 and edge_manifold == 1
```

There is no connected-component restriction. This selects 4,963 models in the
pinned snapshot. See the [mesh-arrangements definitions](https://www.cs.columbia.edu/cg/mesh-arrangements/).
Eligible rows are sorted numerically by `file_id`, then selected uniformly without
replacement with NumPy PCG64. Points use a separate PCG64 stream initialized with
`SeedSequence([seed, file_id])`, so a model's points do not depend on processing order.
The script preserves coordinates, connectivity, and orientation as provided in the
NPZ files, with float64 vertices and int64 indices passed to libigl. It does not
repair, weld, normalize, or reorient meshes.

## Measurement

For each batch of query points `P`:

```python
S, I, C, N = igl.signed_distance(P, V, F, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
d = abs(S)
sign_p = np.where(S < 0, 1, -1)
w = igl.fast_winding_number(V, F, P)
sign_w = 2 * np.rint(w).astype(np.int64) - 1
counted = (sign_p != sign_w) & (d > 0.001 * bbd)
percentage = 100 * np.count_nonzero(counted) / len(P)
```

Signs use **inside +1, outside -1**. NumPy rounds winding numbers to the nearest
integer, with half-integer ties to even. Exact zero signed distances are assigned
outside; the strict distance cutoff excludes them. Unexpected rounded winding
numbers outside `{0, 1}` are preserved, counted in `nonbinary_winding_count`, and
flagged with a warning status, never silently clipped into valid signs. Nonfinite
queries and load failures are recorded as errors, and make the program exit with
failure after processing the remaining models. Failed models are not replaced.

The main metric, `disagreement_percent_all`, uses **all sampled points** as its
denominator. `disagreement_percent_away` additionally reports the percentage among
points strictly beyond the cutoff. Both methods are numerical algorithms; their
disagreement is not an independent ground-truth classification of which is wrong.

Query timings measure one entire Python API call per method per model, including
the internally built acceleration structures and normals. The pseudonormal timing
also includes distance and closest-feature computation; the winding timing only
computes winding numbers. These are useful end-to-end costs, not isolated signing
costs or precomputed-tree query timings. There are no warmup calls or timing
repetitions. The triangle-soup fast winding overload uses libigl's defaults
(expansion order 2, accuracy scale 2).

## Outputs

- `per_model.csv`: results and timings, keyed by `file_id`. Includes bounding box,
  diagonal, cutoff, loaded mesh sizes, mesh SHA-256, disagreement counts and both
  percentages, winding diagnostics, and load/query/output timings.
- `selected_metadata.csv`: all original geometry metadata columns for the selected
  models, in processing order. Join to `geometry_data.csv` on `file_id`, or to
  `input_summary.csv` on its `ID` column.
- `config.json`: seeds, filter, dataset revision, metadata and script hashes,
  package versions, CPU details, thread setting, and timing conventions.
- `download.json`: first cache acquisition, with separate download and archive
  extraction timings and byte counts. Preserved when the cache is reused.
- `cache_access.json`: cache hits and access/extraction timings for the current run.
- `summary.json`: aggregate results and timings; failed model count is explicit.
- `FINDINGS.md`, `disagreements.csv`, `validation.json`: generated by `summarize.py`,
  which audits retained point outputs and exports all counted disagreement cases.
- `points/<file_id>.npz`: every query point, unsigned and signed distances,
  `sign_p`, raw and rounded winding values, `sign_w`, closest face/point/normal,
  and the three comparison masks. These larger artifacts stay local and are
  excluded from Git. They can be regenerated from the locked environment and seeds.

Each completed model is appended to `per_model.jsonl` and the file is closed to
flush the record. `per_model.csv` is exported atomically once at completion.
Appending avoids repeatedly rewriting all previous rows as the run grows;
error records and fields introduced by later rows are preserved. Console progress
prints every 50 models by default, plus the first/last model and every warning or
error; change this with `--progress-every`. Both scripts report checkpoint and
final CSV export times separately. This does not add automatic resumption.

The [logging measurement](results/logging/summary.json), generated by
`benchmark_logging.py`, compares the previous and current paths using repeated
recorded rows with numeric fields restored. For 1,000 Warp rows, checkpointing
fell from 12.793 seconds to 0.0574 seconds; the new path logged and exported all
4,963 rows in 0.282 seconds. These are single wall-clock measurements of logging
only, without geometry queries.

The journal remains readable during a run. To export completed records after an
interruption, without rerunning any queries:

```sh
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
from benchmark import write_csv
output = Path("results/my-run")
rows = [json.loads(line) for line in (output / "per_model.jsonl").read_text().splitlines()]
write_csv(output / "per_model.csv", rows)
PY
```

A partial last JSON line from an interrupted write must be inspected before
exporting; malformed records are rejected. Journals, downloaded meshes,
environments, and per-point arrays are ignored by Git; final CSVs, scripts,
summary files, and findings are versioned. Individual mesh licenses are recorded
in the source dataset; this repository does not redistribute those meshes.

## Validation

`uv run python -m unittest -v` checks analytic cube distances (face, edge, and vertex
closest features), inside-positive signs, a solid with a cavity represented by two
boundary components, the strict cutoff and both denominators, and deterministic
sampling with the intended filter.

## Warp from GitHub main

The comparison pins the main commit fetched on 2026-11-23,
[`915c3d4f56432e557339752eb619a7930c813d66`](https://github.com/NVIDIA/warp/commit/915c3d4f56432e557339752eb619a7930c813d66),
version `1.21.0.dev0`, in [warp_source.json](warp_source.json). Reuse an existing
clean Warp checkout when available. On another machine, clone
`https://github.com/NVIDIA/warp.git` only if needed, check out that revision, and
build its native libraries according to Warp's source-build requirements:

```sh
# From the Warp checkout; use an installed CUDA toolkit supported by your GPU.
git checkout 915c3d4f56432e557339752eb619a7930c813d66
uv sync --extra dev --no-install-project
uv run --no-sync build_lib.py --quick --cuda-path /path/to/cuda --jobs 4
```

The recorded build used CUDA 12.9.1 and driver 13.2. `--quick` requires a driver
new enough for the toolkit. Then, from this benchmark repository:

```sh
uv sync --locked
uv pip install --python .venv/bin/python --no-deps --editable /path/to/warp
uv run --no-sync python -m unittest -v

# Regenerate the seeded baseline if its ignored points/ files are absent.
uv run --no-sync python benchmark.py --output results/libigl-reproduced
uv run --no-sync python warp_benchmark.py \
  --baseline results/libigl-reproduced --output results/warp-reproduced
uv run --no-sync python summarize_warp.py results/warp-reproduced
```

On the original machine, the baseline point files are already retained, so run
`warp_benchmark.py --output results/another-warp-run` directly. Set
`HF_HUB_OFFLINE=1` to verify that a run uses the existing dataset cache.

Warp's source build is pinned separately from the PyPI dependencies in `uv.lock`.
Use **`uv run --no-sync` after installing it**; environment synchronization can
remove the separately installed package. The comparison checks the imported
checkout's Git revision, clean tracked Warp source, and version, and records
native library SHA-256 hashes. Update the pin and rebuild matching native
libraries deliberately when testing another main revision.

`warp_queries.py` calls these APIs independently on `cuda:0` by default:

- `mesh_query_point_sign_normal`: epsilon `0.001`.
- `mesh_query_point_sign_winding_number`: accuracy `2.0`, threshold `0.5`;
  `Mesh(support_winding_number=True)` prevents a ray-vote fallback.
- `mesh_query_point_sign_parity`: native default `n_sample=1`, perturbation
  scale `0.1`. This uses one ray through the voting API. Pass `--parity-rays 3`
  for majority voting with three rays. Warp initializes its internal RNG with
  fixed seed 42, so ray directions are deterministic.

Use `--device cpu` for CPU queries. The wrapper converts all native signs to
inside `+1`, outside `-1`, and derives each method's distance from its returned
closest point. The same float64 samples are cast to float32 for Warp without
normalization. Every comparison uses **the saved libigl distance mask**, so Warp
distance differences cannot change which points are counted. Invalid queries
are reported as errors. Libigl winding is the chosen reference, not independently
verified ground truth.

Warp results have a separate `per_model.csv`, `summary.json`, and `config.json`.
The CSV joins to the original metadata on `file_id` and includes all three Warp
sign comparisons, distance errors, and separate loading, mesh construction,
upload, query, download, and output timings. Queries are timed with device
synchronization; module loading is timed outside the query batches. Libigl's
original CPU calls include their internal setup, so their timings have a
different scope from Warp's GPU queries.

`summarize_warp.py` audits the retained arrays and generates `FINDINGS.md` and
`validation.json`. `disagreements.csv` exports every counted case with all signs,
distances, and coordinates. `distance_outliers.csv` records every point whose
absolute distance difference exceeds `1e-5 * bbd`; `distance_diagnostics.csv`
recomputes the worst case per affected model in libigl using float32-rounded
inputs. Full Warp arrays, including closest points, faces and validity flags,
remain in ignored local `points/<file_id>.npz` files.

The combined suite has eight passing tests and one expected failure at the
pinned revision. CPU and CUDA integration tests verify cube distances, cavity
signs and cutoff handling. The expected failure records Warp pseudonormal's
inside prediction at the exact center of a symmetric empty cube cavity; the
analytic expectation is outside. It is separate from the random benchmark.

## Full eligible set

The pinned metadata has 4,963 eligible models. With the dataset and pinned Warp
already installed, run all 49,630,000 samples with:

```sh
HF_HUB_OFFLINE=1 uv run --no-sync python benchmark.py \
  --models 4963 --points 10000 --seed 20261123 --output results/full-4963
HF_HUB_OFFLINE=1 uv run --no-sync python warp_benchmark.py \
  --baseline results/full-4963 --output results/warp-main-full-4963
uv run --no-sync python summarize.py results/full-4963
HF_HUB_OFFLINE=1 uv run --no-sync python summarize_warp.py results/warp-main-full-4963
HF_HUB_OFFLINE=1 uv run --no-sync python validate_full_run.py
```

The per-model point seed is unchanged from the initial subset, so every model
shared between runs receives identical samples. Query algorithms and default
parameters, including one-ray parity, are unchanged.
`validate_full_run.py` checks full-mask membership, the journal-to-CSV exports,
and exact reproduction of the initial 100 models' saved samples, distances and
signs. It requires the locally retained point arrays from both initial runs.

The scripts and experiment documentation were prepared with OpenAI Codex.

## Libigl scalar precision experiment

[The 100-model scalar comparison](results/libigl-scalar-100/FINDINGS.md) reuses the
original 1,000,000 points, mesh coordinates, winding signs, and distance cutoff.
The patched double implementation is bitwise identical to the original source
build and the installed wheel for all retained outputs. Single precision returns
762 counted disagreements (0.0762%), compared with 7 (0.0007%) in double. One
single-precision query has a nonfinite normal and is explicitly flagged; 761
counted disagreements have finite outputs. Rounding the inputs to float32 and
then computing in double gives 9 disagreements (0.0009%).

[The original C++ float comparison](results/libigl-mixed-comparison-100/FINDINGS.md)
adds the pre-PR implementation with its existing mix of float and double
arithmetic: **764 disagreements (0.0764%)**, versus **762 (0.0762%)** after the
scalar patch. Only four signs change, all on model `42194`: three disagreements
are resolved and one is introduced. Unsigned distances are bitwise identical and
closest-face indices match everywhere. Both float paths have the same one
nonfinite-normal query. Thus the large difference from double was already
present in the original C++ float path.

The upstream changes are [libigl PR #2583](https://github.com/libigl/libigl/pull/2583)
and [Python bindings PR #321](https://github.com/libigl/libigl-python-bindings/pull/321).
They preserve the existing tolerance values and classifier.
`native_precision/` builds the actual Python `signed_distance.cpp` wrapper from
specified libigl and Python-binding checkouts. The patched wrapper dispatches
float32 geometry and query arrays to a float C++ instantiation. Dataset downloads
and the original point sampling are not repeated.

To reproduce, use clean source checkouts at the commits recorded in the run's
`config.json`. Build the original extension before applying or checking out the
patches, or use isolated Git worktrees. Keep each build directory associated with
its source revisions; rebuilding the original directory against patched sources
would replace the baseline binary. All build and result directories must be fresh.

```bash
# Set these paths to checkouts at the desired original or patched revisions.
# For the original build, use b41f1944 (libigl) and 6893d1b1 (bindings).
cmake -S native_precision -B .cache/precision/original -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLIBIGL_SOURCE_DIR=/path/to/original/libigl \
  -DBINDINGS_SOURCE_DIR=/path/to/original/libigl-python-bindings \
  -DPRECISION_MODULE=_precision_original
cmake --build .cache/precision/original -j 4
uv run --no-sync python precision_benchmark.py \
  --variant original_double \
  --module .cache/precision/original/_precision_original*.so \
  --libigl-source /path/to/original/libigl \
  --bindings-source /path/to/original/libigl-python-bindings \
  --output results/libigl-original-double-100

# Use the patched commits from results/libigl-scalar-100/summary.json.
cmake -S native_precision -B .cache/precision/patched -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLIBIGL_SOURCE_DIR=/path/to/patched/libigl \
  -DBINDINGS_SOURCE_DIR=/path/to/patched/libigl-python-bindings \
  -DPRECISION_MODULE=_precision_patched \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$PWD/.cache/precision/original/_deps/nanobind-src" \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN="$PWD/.cache/precision/original/_deps/eigen-src"
cmake --build .cache/precision/patched -j 4
uv run --no-sync python precision_benchmark.py \
  --variant patched_double \
  --module .cache/precision/patched/_precision_patched*.so \
  --libigl-source /path/to/patched/libigl \
  --bindings-source /path/to/patched/libigl-python-bindings \
  --output results/libigl-scalar-double-100
```

Repeat the last command with `--variant patched_single` and output
`results/libigl-scalar-single-100`, then with `--variant quantized_double` and
output `results/libigl-quantized-double-100`. Generate the comparison with
`uv run --no-sync python summarize_precision.py`. Individual run directories hold
per-model measurements, cache timings, configuration, and locally retained point
outputs. The comparison directory holds a complete output audit and a diagnostic
CSV keyed by `(file_id, sample_index)`.

The patched C++ tests can be built using
`cmake -S native_precision -B .cache/precision/patched -DPRECISION_BUILD_TESTS=ON`,
then `cmake --build .cache/precision/patched --target test_precision -j 4` and
`.cache/precision/patched/test_precision`. The compiler audit for the float query
helpers is:

```bash
c++ -std=c++17 -Werror=double-promotion \
  -I/path/to/patched/libigl/include \
  -isystem .cache/precision/original/_deps/eigen-src \
  -fsyntax-only native_precision/audit_float.cpp
```

See [validation.json](results/libigl-scalar-100/validation.json) for the C++ tests,
sanitizer results, Python binding tests, and the regression check against the
original implementation. Classification improvements are tracked separately in
[issue 2](https://github.com/alecjacobson/pseudonormal-tests/issues/2).

The follow-up libigl commit `8e59c1897bb33a03d06a79b23c803e299cf890b0` adds
the static-library instantiations required by the new float intermediates,
extends test coverage, and adjusts formatting. It does not change the arithmetic
measured above. A separate static-link smoke test passed against an archive of
the 36 required libigl translation units, with function-section garbage
collection. This was a targeted static build, not a full-library build. To run
the same queries against a libigl static archive:

```bash
c++ -std=c++17 -DIGL_STATIC_LIBRARY \
  -I/path/to/patched/libigl/include -isystem /path/to/eigen \
  native_precision/static_query.cpp /path/to/libigl.a \
  -Wl,--gc-sections -pthread -o /path/to/test_static_query
/path/to/test_static_query
```

To reproduce the old mixed-precision float variant, use the untouched libigl
source at `b41f1944a703653ae0da81e6588650ba14c270ea` with the float-capable wrapper
at bindings commit `51d6dee06139682c08163012b00b5945095c38f1`. The wrapper's
`src/signed_distance.cpp` is byte-identical to the wrapper used in the patched
float experiment (`eac43e8`). The harness selects libigl through
`LIBIGL_SOURCE_DIR`; it does not use the binding repository's libigl dependency
pin. This exposes the existing C++ float instantiation without changing its
internal arithmetic.

```bash
cmake -S native_precision -B .cache/precision/mixed -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLIBIGL_SOURCE_DIR=/path/to/original/libigl \
  -DBINDINGS_SOURCE_DIR=/path/to/patched/libigl-python-bindings \
  -DPRECISION_MODULE=_precision_mixed \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$PWD/.cache/precision/original/_deps/nanobind-src" \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN="$PWD/.cache/precision/original/_deps/eigen-src"
cmake --build .cache/precision/mixed -j 4
uv run --no-sync python precision_benchmark.py \
  --variant original_single_mixed \
  --module .cache/precision/mixed/_precision_mixed*.so \
  --libigl-source /path/to/original/libigl \
  --bindings-source /path/to/patched/libigl-python-bindings \
  --output results/libigl-original-mixed-100
uv run --no-sync python summarize_precision.py \
  --mixed results/libigl-original-mixed-100 \
  --output results/libigl-mixed-comparison-100
```

The new run's `build_record.json` records configure/build commands and times.
The comparison's `validation.json` records an independent point-array audit,
including the four changed signs and the shared invalid query. The original
four-variant experiment and its cached outputs remain available unchanged.

## Float tolerance sweep

[The runtime tolerance sweep](results/libigl-float-tolerances-100/FINDINGS.md)
evaluates 230 distinct `(minimum_double_area, epsilon)` pairs across two grids,
using the same 100 meshes and 1,000,000 points. With the original area threshold
`1e-4`, increasing epsilon from `1e-12` to `0.003` reduces disagreements from
**762 to 3**. A fixed 50/50 model split selected this pair on the tuning half
(505 → 0 disagreements); the holdout half improves from 257 to 3. One of the
three remaining cases has the same pre-existing nonfinite normal.

The best pair over all models is area `1e-6`, epsilon `0.01`, with two returned-sign
disagreements (one with finite outputs). This is an exploratory result. Larger
epsilon values eventually increase disagreement. Epsilon has different units
in the two classification branches, so these results at original mesh scales
do not establish scale-independent defaults.

The experiment compiles once. A generator mechanically copies the existing 3D
classifier body and replaces its constants with function arguments. The native
wrapper computes mesh normals once per model and reuses cached closest faces
and points. Every default sign and normal matches the saved float result
byte-for-byte on all 1,000,000 points. The second grid uses the same binary;
its 30 repeated parameter pairs reproduce every point's sign and validity flag.
No upstream API or default tolerance is changed.

```bash
cmake -S native_precision -B .cache/precision/tolerances -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLIBIGL_SOURCE_DIR=/path/to/patched/libigl \
  -DBINDINGS_SOURCE_DIR=/path/to/patched/libigl-python-bindings \
  -DPRECISION_BUILD_TOLERANCE_SWEEP=ON \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$PWD/.cache/precision/original/_deps/nanobind-src" \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN="$PWD/.cache/precision/original/_deps/eigen-src"
cmake --build .cache/precision/tolerances --target _tolerance_sweep -j 4
uv run --no-sync python sweep_tolerances.py \
  --module .cache/precision/tolerances/_tolerance_sweep*.so \
  --libigl-source /path/to/patched/libigl \
  --bindings-source /path/to/patched/libigl-python-bindings \
  --output results/libigl-float-tolerances-100
uv run --no-sync python sweep_tolerances.py \
  --module .cache/precision/tolerances/_tolerance_sweep*.so \
  --libigl-source /path/to/patched/libigl \
  --bindings-source /path/to/patched/libigl-python-bindings \
  --epsilons 1e-12,3e-3,1e-2,3e-2,1e-1,2e-1,3e-1,1 \
  --output results/libigl-float-tolerances-upper-100
# Plotting is optional; install it without syncing away an existing Warp setup.
uv pip install --python .venv/bin/python 'matplotlib>=3.10,<4'
uv run --no-sync python summarize_tolerances.py results/libigl-float-tolerances-upper-100
uv run --no-sync python summarize_tolerances.py results/libigl-float-tolerances-100 \
  --upper-results results/libigl-float-tolerances-upper-100
```

Use libigl commit `8e59c1897bb33a03d06a79b23c803e299cf890b0` and bindings commit
`51d6dee06139682c08163012b00b5945095c38f1`. The native generator changes only the
temporary generated function, leaving these source checkouts clean. Run
directories contain exact configurations, per-model grids, timings, validation,
PDF/PNG heatmaps and the selected diagnostic cases. All point signs and validity
flags for every setting remain cached locally. The two sweeps take approximately
54 seconds combined on this machine, including compressed output.

## Double tolerance sweep

[The double sweep](results/libigl-double-tolerances-100/FINDINGS.md) evaluates the
same 230 distinct parameter pairs in `float64`, on the same cached million points
at original scale. With the area threshold kept at `1e-4`:

| Epsilon | Double disagreements | Percent of all points |
| --- | ---: | ---: |
| `1e-12` (default) | 7 | 0.0007% |
| `1e-10` | 1 | 0.0001% |
| Every tested value from `1e-9` through `0.003` | 0 | 0% |
| `0.01` | 2 | 0.0002% |
| `0.03` | 10 | 0.001% |

There are no nonfinite output normals anywhere in this double grid. The default
classifier reproduces all saved double signs and normal bytes exactly. The
smallest swept epsilon that eliminates disagreements at the default area is
`1e-9`; the float-selected `0.003` also works on this double sample.

All seven original double failures lie in the fixed holdout half. The tuning
half has zero default failures, so the selection rule retains the defaults.
Choosing `1e-9` from the combined results is exploratory and needs validation
on additional models. The split is unchanged from the float study.

The runtime wrapper now supports both precisions, preserving the existing float
interface. [A compatibility check](results/libigl-double-tolerances-100/float_compatibility.json)
reproduces all million float default signs and normal bytes, and every point's
sign and validity at epsilon `1e-12`, `0.003`, and `0.01` against the earlier
float module. The double sweep takes 44.3 seconds including I/O, with 23.5
seconds in classifier calls. All 230 settings were independently recounted
from the saved point arrays. Upstream defaults remain unchanged.

To reproduce, use the same pinned source commits and configure/build commands
above with build directory `.cache/precision/tolerances-double`, then run:

```bash
uv run --no-sync python sweep_tolerances.py \
  --module .cache/precision/tolerances-double/_tolerance_sweep*.so \
  --libigl-source /path/to/patched/libigl \
  --bindings-source /path/to/patched/libigl-python-bindings \
  --precision float64 \
  --epsilons 0,1e-14,1e-12,1e-10,1e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,2e-1,3e-1,1 \
  --output results/libigl-double-tolerances-100
uv run --no-sync python summarize_tolerances.py results/libigl-double-tolerances-100
```

`--precision` defaults to `float32`; each precision selects its corresponding
cached scalar baseline. `--baseline` can override that path, and the existing
`--float-baseline` option remains an alias. Tolerances are converted to the
selected scalar type before evaluation and recorded exactly in `config.json`.

## Broader validation of fixed tolerances

[The full-mask validation](results/libigl-tolerance-validation-4963/FINDINGS.md)
uses all 4,963 eligible models and the same 49,630,000 cached sample points.
The area threshold stays at `1e-4`; epsilon is fixed at `0.003` for float and
`1e-9` for double before this run. Results distinguish the original 100 models
from the additional 4,863 models, which were not used in the tolerance sweeps.
The original full baseline was already available, so this is broader regression
validation rather than a completely uninspected holdout.

| Precision | Default disagreements | Tuned disagreements | Tuned percent of all points | Invalid queries, default → tuned |
| --- | ---: | ---: | ---: | ---: |
| Float | 48,268 | 1,061 | 0.00213782% | 224 → 369 |
| Double | 1,271 | 441 | 0.000888575% | 0 → 0 |

Float tuning resolves 47,504 disagreements and introduces 297, for a 97.80%
net reduction. It also introduces 145 invalid queries; only 698 of its 1,061
remaining disagreements have finite outputs. Double tuning resolves 830 and
introduces none, for a 65.30% reduction, with all outputs finite. The additional
4,863 models alone give float **47,506 → 1,058** and double **1,264 → 441**.
The original 100-model results reproduce exactly.

Model `153100` retains 277 double disagreements and accounts for much of the
remaining error. Aggregate improvement does not eliminate individual model
regressions or establish universal tolerances. No defaults in the upstream PRs
were changed. The run took 20 minutes 37 seconds; all 49.63 million point
results in each of the four variants were independently recounted.

`validate_tolerances.py` reuses the compiled dual-precision runtime module and
the patched signed-distance extension. It computes float nearest points once
per mesh and reuses cached double nearest points. In each precision, the
default and tuned classifiers share those closest features. Every default sign
and normal byte must reproduce its baseline, including nonfinite values.
Per-model metrics and timings are appended to journals; CSV is exported once
at completion. Nonfinite outputs remain visible alongside returned-sign counts.

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python validate_tolerances.py \
  --runtime-module .cache/precision/tolerances-double/_tolerance_sweep*.so \
  --precision-module .cache/precision/patched/_precision_patched*.so \
  --output results/libigl-tolerance-validation-4963
uv run --no-sync python summarize_tolerance_validation.py \
  results/libigl-tolerance-validation-4963
```

Use the cached modules whose hashes are recorded by the earlier double sweep
and scalar-float run; the runner checks those hashes before starting. The
optional `--runtime-study` and `--precision-study` paths select their provenance
records. The auditor checks full-mask coverage, recounts all four variants
from cached point arrays, verifies the original 100-model results, and exports
per-model comparisons, remaining diagnostic cases, and a PDF/PNG figure.

## Diagnosing nonfinite pseudonormals

[The normalization probe](results/libigl-normalization-probe-104/FINDINGS.md)
traces all 369 nonfinite tuned-float queries to final **edge** normalization:
368 select exactly zero accumulated edge normals; one selects a tiny nonzero
normal whose subnormal squared length Eigen's default SIMD square root flushes
to zero. Omitting vertex normalization alone changes none of these cases.

On the original 100 models plus all affected models (104 total), omitting final
edge normalization changes tuned float disagreements from 365 to 8 and removes
nonfinite outputs, with every previously finite sign unchanged. However, 368
returned normals are now exactly zero: they take the outside branch by tie
breaking, so this is not a complete classification fix. Double's 277 subset
disagreements stay unchanged. Unit face normals and their weighting remain.

Build the isolated probe using the existing dependency caches:

```bash
cmake -S native_precision -B .cache/precision/normalization \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLIBIGL_SOURCE_DIR=/home/horde/projects/libigl \
  -DBINDINGS_SOURCE_DIR=/home/horde/projects/libigl-python-bindings \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN="$PWD/.cache/precision/original/_deps/eigen-src" \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$PWD/.cache/precision/original/_deps/nanobind-src" \
  -DPRECISION_BUILD_NORMALIZATION_PROBE=ON
cmake --build .cache/precision/normalization --target _normalization_probe -j 4
HF_HUB_OFFLINE=1 uv run --no-sync python probe_normalization.py \
  --module .cache/precision/normalization/_normalization_probe*.so \
  --output results/libigl-normalization-probe-104
```

Use a fresh output directory for each replay. The standalone
`native_precision/normalization_reproducer.cpp` additionally isolates the
subnormal and exact-zero cases; compiling with `-DEIGEN_FAST_MATH=0` resolves
the former but leaves the latter. This experiment does not modify upstream
libigl or its default behavior.

## Tuned defaults in the upstream PR

Libigl PR [2583](https://github.com/libigl/libigl/pull/2583) now incorporates
the validated 3D epsilons: `0.003` for float and `1e-9` for double, retaining
area threshold `1e-4`. The 2D overload and other scalar types retain `1e-12`.
Normalization is unchanged. The earlier reports describe their pinned
historical source revisions, including the previous defaults.

[Production validation](results/libigl-tuned-defaults-104/FINDINGS.md) on the
original 100 models plus all invalid-output models reproduces every tuned
sign and normal byte on 1.04 million points per precision, with identical
unsigned distances and nearest geometry. The original 100 models give three
float disagreements (including one invalid normal) and zero double disagreements.

Build a fresh precision module against the updated core commit using the
`native_precision` CMake project, then run:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python validate_tuned_defaults.py \
  --module .cache/precision/tuned-defaults/_precision_tuned*.so \
  --output results/libigl-tuned-defaults-104
```

The runtime-classifier generator accepts both the original and new epsilon
declarations, so future parameter experiments can continue to use a single
compiled module. Existing baseline binaries remain cached unchanged.

## Double-built normals with a float classifier

[The full mixed-normal experiment](results/libigl-mixed-normals-4963/FINDINGS.md)
changes only the construction and normalization of face, angle-weighted vertex,
and uniform-weighted edge normals. Geometry, queries, cached closest points,
closest-face indices, and `pseudonormal_test` stay float32, with area `1e-4` and
epsilon `0.003`. Normal tables are built and normalized in double, then cast
to float. Two variants distinguish double arithmetic on float-rounded vertices
from using the original double vertices. Every model's pure-float control must
reproduce the prior tuned signs, validity masks, and normal bytes exactly.

Build and replay the cached full mask:

```bash
cmake -S native_precision -B .cache/precision/mixed-normals \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLIBIGL_SOURCE_DIR=/home/horde/projects/libigl \
  -DBINDINGS_SOURCE_DIR=/home/horde/projects/libigl-python-bindings \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN="$PWD/.cache/precision/original/_deps/eigen-src" \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$PWD/.cache/precision/original/_deps/nanobind-src" \
  -DPRECISION_BUILD_MIXED_NORMALS=ON
cmake --build .cache/precision/mixed-normals --target _mixed_normals inspect_triangle -j 4
HF_HUB_OFFLINE=1 uv run --no-sync python benchmark_mixed_normals.py \
  --module .cache/precision/mixed-normals/_mixed_normals*.so \
  --output results/libigl-mixed-normals-4963
uv run --no-sync python summarize_mixed_normals.py results/libigl-mixed-normals-4963
HF_HUB_OFFLINE=1 uv run --no-sync python inspect_mixed_normal_cases.py \
  results/libigl-mixed-normals-4963 \
  --module .cache/precision/normalization/_normalization_probe*.so \
  --triangle-tool .cache/precision/mixed-normals/inspect_triangle
HF_HUB_OFFLINE=1 uv run --no-sync python plot_mixed_normal_case.py \
  results/libigl-mixed-normals-4963
```

Use a fresh output directory. The examples exporter reuses the normalization
probe built in the section above to identify matching double normal-table
entries. The standalone triangle inspector reads triangle vertices, the float
closest point, and the double closest point from stdin (15 numbers), and reports
libigl's area and barycentrics in both precisions. It can also be used for other
cached cases. `cases.csv` records selected features, normals, dots, incident
faces, and coordinates; `points/` retains full per-query outputs locally.
Normal setup, casting, classification, diagnostic work, loading, and output
are timed separately. Distance and winding queries are reused, so these timings
are not an end-to-end signed-distance benchmark.

## Diagnosing the 17 tuned-double failure models

[The double diagnostic study](results/libigl-double-diagnostics-17/FINDINGS.md)
examines all 441 disagreements remaining with area `1e-4` and epsilon `1e-9`:
281 select the wrong edge in the small-area fallback, 98 select a face instead
of an edge, three select the wrong global nearest feature, and 59 persist on
model `1452672`, whose cached coordinates have exact self-intersections.
Higher-precision normal construction alone fixes none. All 441 winding signs
are confirmed by direct solid-angle sums and ordinary libigl winding.

The instrumented classifier reproduces all 170,000 baseline signs and normal
bytes on these meshes. The diagnostic substitutions are evaluated on the 441
original failures; they are not a full regression test of a replacement
classifier. All input meshes and points remain at their original scales.

Install the optional diagnostic dependency into the existing environment
without replacing the editable Warp installation:

```bash
uv pip install --python .venv/bin/python 'mpmath>=1.3,<2'
cmake -S native_precision -B .cache/precision/double-diagnostics -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DLIBIGL_SOURCE_DIR=/home/horde/projects/libigl \
  -DBINDINGS_SOURCE_DIR=/home/horde/projects/libigl-python-bindings \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN="$PWD/.cache/precision/original/_deps/eigen-src" \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$PWD/.cache/precision/original/_deps/nanobind-src" \
  -DPRECISION_BUILD_DOUBLE_DIAGNOSTICS=ON
cmake --build .cache/precision/double-diagnostics --target _double_diagnostics -j 4
HF_HUB_OFFLINE=1 uv run --no-sync python diagnose_double_failures.py \
  --module .cache/precision/double-diagnostics/_double_diagnostics*.so \
  --output results/libigl-double-diagnostics-17
HF_HUB_OFFLINE=1 uv run --no-sync python diagnose_double_failures.py \
  --module .cache/precision/double-diagnostics/_double_diagnostics*.so \
  --digits 120 --output .cache/precision/double-diagnostics-check120
HF_HUB_OFFLINE=1 uv run --no-sync python audit_double_diagnostics.py \
  results/libigl-double-diagnostics-17 \
  --check .cache/precision/double-diagnostics-check120
uv run --no-sync python plot_double_diagnostics.py results/libigl-double-diagnostics-17
uv run --no-sync python -m unittest test_exact_mesh_diagnostics -v
```

Choose fresh output paths when replaying. `--file-ids` restricts the diagnosis
to particular models. Exact geometry uses rational representations of the
cached binary64 inputs, with outward-rounded bounding-box pruning. Normals use
exact face cross products followed by mpmath normalization and angle weights.
The audit compares 80/120-digit signs and features and exports rational
witnesses of strict crossings in `1452672`; it is not a complete intersection
detector. Per-point evidence, per-model timings, source hashes, and a diagnosis
figure are saved in the result directory.
