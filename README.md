# Pseudonormal tests

Compare libigl's pseudonormal distance sign with its fast winding number sign on
Thingi10K. The initial experiment selects 100 models and samples 10,000 points
uniformly in each model's axis-aligned bounding box.

The [initial findings](results/initial-100/FINDINGS.md) contain **7 disagreements
beyond the distance cutoff across 4 models**, or **0.0007% of all 1,000,000 samples**.
All four analytic tests passed, and the full run reused the cache with
`HF_HUB_OFFLINE=1`.

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

`per_model.csv` is checkpointed after every model. Downloaded meshes, environments,
and per-point arrays are ignored by Git; scripts, small summary files, and findings
are versioned. Individual mesh licenses are recorded in the source dataset; this
repository does not redistribute those meshes.

## Validation

`uv run python -m unittest -v` checks analytic cube distances (face, edge, and vertex
closest features), inside-positive signs, a solid with a cavity represented by two
boundary components, the strict cutoff and both denominators, and deterministic
sampling with the intended filter.

The scripts and experiment documentation were prepared with OpenAI Codex.
