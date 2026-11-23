# Pseudonormal tests

Compare libigl and Warp mesh distances and signs on Thingi10K. The experiment
selects 100 models and samples 10,000 points uniformly in each model's axis-aligned
bounding box. Warp reuses the exact saved libigl samples.

The [initial findings](results/initial-100/FINDINGS.md) contain **7 disagreements
beyond the distance cutoff across 4 models**, or **0.0007% of all 1,000,000 samples**.
All four analytic tests passed, and the full run reused the cache with
`HF_HUB_OFFLINE=1`.

The [Warp main comparison](results/warp-main-100/FINDINGS.md) uses libigl winding
as the reference and the same original distance cutoff:

| Method | Disagreements | % of all samples | Models affected |
| --- | ---: | ---: | ---: |
| libigl pseudonormal | 7 | 0.0007% | 4 |
| Warp pseudonormal | 291 | 0.0291% | 48 |
| Warp winding | 0 | 0% | 0 |
| Warp parity (native default: 1 ray) | 0 | 0% | 0 |

For Warp distances, the 99th percentile absolute difference is about
`2.23e-7 * bbd`; the maximum is `3.56e-4 * bbd`. Saved outlier diagnostics show
that input rounding alone does not explain the largest discrepancies.

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
