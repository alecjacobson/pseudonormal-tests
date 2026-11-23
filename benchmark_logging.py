"""Measure CSV rewrites versus append-only checkpoints using recorded result rows."""

import argparse
import csv
import json
import tempfile
from pathlib import Path
from time import perf_counter

from benchmark import append_result, write_csv
from dataset_cache import write_json


def run(output):
    """Compare the old and new logging paths without running mesh queries."""
    if output.exists():
        raise FileExistsError(f"Choose a fresh output file; {output} already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    measurements = []
    for source in ("results/initial-100/per_model.csv", "results/warp-main-100/per_model.csv"):
        with Path(source).open(newline="") as stream:
            original = list(csv.DictReader(stream))
        # Restore number types so CSV/JSON formatting costs match live result rows.
        for row in original:
            for key, value in row.items():
                try:
                    row[key] = int(value)
                except ValueError:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass
        for count in (100, 1000, 4963):
            rows = (original * ((count + len(original) - 1) // len(original)))[:count]
            for method in ("rewrite_csv", "append_jsonl_then_csv"):
                # The full-size old path is quadratic; measure it up to 1,000 records.
                if count == 4963 and method == "rewrite_csv":
                    continue
                with tempfile.TemporaryDirectory(dir=output.parent) as directory:
                    root = Path(directory)
                    started = perf_counter()
                    if method == "rewrite_csv":
                        for index in range(count):
                            write_csv(root / "per_model.csv", rows[: index + 1])
                    else:
                        for row in rows:
                            append_result(root / "per_model.jsonl", row)
                        write_csv(root / "per_model.csv", rows)
                    seconds = perf_counter() - started
                measurements.append(
                    {"source": source, "models": count, "method": method, "seconds": seconds}
                )
                print(json.dumps(measurements[-1]), flush=True)
    write_json(
        output,
        {
            "methodology": "single wall-clock measurement per case on this machine; original 100 rows repeated with numeric types restored; no geometry queries; temporary files removed",
            "measurements": measurements,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/logging/summary.json"))
    run(parser.parse_args().output)
