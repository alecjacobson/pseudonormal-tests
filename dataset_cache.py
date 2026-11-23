"""Cache one immutable Thingi10K snapshot and record acquisition timings."""

import argparse
import json
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from filelock import FileLock
from huggingface_hub import hf_hub_download, try_to_load_from_cache

REPOSITORY = "Thingi10K/Thingi10K"
REVISION = "2d5d3b2f3cd3711028ad75b12788c13b25559ec6"
DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache" / "thingi10k"
METADATA_FILES = (
    "metadata/geometry_data.csv",
    "metadata/input_summary.csv",
    "metadata/contextual_data.csv",
)


def write_json(path, data):
    """Write JSON atomically without nonstandard NaN or infinity literals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_dataset(cache_dir=DEFAULT_CACHE, revision=REVISION):
    """Download and extract once, then reuse the snapshot without network access."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Use a full, immutable Hugging Face commit SHA.")
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    report = {
        "repository": REPOSITORY,
        "revision": revision,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "cache_dir": str(cache_dir),
        "files": [],
    }
    paths = {}
    with FileLock(cache_dir / "prepare.lock"):
        for filename in (*METADATA_FILES, "Thingi10K_npz.tar.gz"):
            step_started = perf_counter()
            cached = try_to_load_from_cache(
                REPOSITORY,
                filename,
                cache_dir=cache_dir / "hub",
                revision=revision,
                repo_type="dataset",
            )
            cache_hit = isinstance(cached, str)
            print(f"{'Cached' if cache_hit else 'Downloading'} {filename}", flush=True)
            path = (
                Path(cached)
                if cache_hit
                else Path(
                    hf_hub_download(
                        REPOSITORY,
                        filename,
                        cache_dir=cache_dir / "hub",
                        revision=revision,
                        repo_type="dataset",
                    )
                )
            )
            report["files"].append(
                {
                    "filename": filename,
                    "cache_hit": cache_hit,
                    "seconds": perf_counter() - step_started,
                    "file_bytes": path.stat().st_size,
                }
            )
            paths[filename] = path
        extracted = cache_dir / "extracted" / revision
        marker = extracted / "complete.json"
        step_started = perf_counter()
        extraction_hit = marker.is_file()
        if not extraction_hit:
            free_bytes = shutil.disk_usage(cache_dir).free
            if free_bytes < 20 * 1024**3:
                raise OSError("Less than 20 GiB free; stop before extracting the dataset.")
            if extracted.exists():
                raise FileExistsError(f"Incomplete extraction at {extracted}; inspect it first.")
            extracted.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix="extract-", dir=extracted.parent))
            print("Extracting the full NPZ dataset (cached after completion)", flush=True)
            try:
                with tarfile.open(paths["Thingi10K_npz.tar.gz"], "r:gz") as archive:
                    archive.extractall(staging, filter="data")
                count = sum(1 for _ in (staging / "npz").glob("*.npz"))
                if count == 0:
                    raise ValueError("Archive contained no npz/<file_id>.npz files.")
                write_json(
                    staging / "complete.json",
                    {"revision": revision, "npz_files": count},
                )
                staging.replace(extracted)
            except BaseException:
                shutil.rmtree(staging)
                raise
        report["extraction"] = {
            "cache_hit": extraction_hit,
            "seconds": perf_counter() - step_started,
            **json.loads(marker.read_text()),
        }
    report["total_seconds"] = perf_counter() - started
    # Preserve the first acquisition record even when subsequent runs use the cache.
    acquisition = cache_dir / f"acquisition-{revision}.json"
    if not acquisition.exists():
        write_json(acquisition, report)
    return paths["metadata/geometry_data.csv"], extracted / "npz", report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, _, report = prepare_dataset(args.cache_dir, args.revision)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
