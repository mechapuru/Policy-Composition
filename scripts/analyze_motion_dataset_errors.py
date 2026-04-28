#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os

import numpy as np


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def count_above(rows, key, thresholds):
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return
    print(f"{key} threshold counts:")
    for threshold in thresholds:
        count = sum(float(value) > threshold for value in values)
        pct = 100.0 * count / len(values)
        print(f"  > {threshold}: {count}/{len(values)} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Summarize motion-planning raw dataset error metadata.")
    parser.add_argument("--data-dir", default="/scratch3/cross-emb/dataset_mp_poco")
    parser.add_argument("--output-csv", default="motion_dataset_errors.csv")
    parser.add_argument("--pos-thresholds", type=float, nargs="*", default=[0.01, 0.02, 0.03, 0.04, 0.05])
    parser.add_argument("--tracking-thresholds", type=float, nargs="*", default=[0.02, 0.05, 0.10, 0.15])
    parser.add_argument("--orn-thresholds", type=float, nargs="*", default=[1.0, 2.0, 5.0, 10.0])
    args = parser.parse_args()

    rows = []
    for json_path in sorted(glob.glob(os.path.join(args.data_dir, "iter_*", "state_action.json"))):
        with open(json_path, "r") as f:
            data = json.load(f)

        row = {
            "episode": os.path.basename(os.path.dirname(json_path)),
            "num_frames": data.get("num_frames"),
            "success": data.get("success"),
            "pos_error_m": data.get("pos_error_m"),
            "orn_error_deg": data.get("orn_error_deg"),
            "tracking_err_rad": data.get("tracking_err_rad"),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No state_action.json files found under {args.data_dir}")

    fieldnames = ["episode", "num_frames", "success", "pos_error_m", "orn_error_deg", "tracking_err_rad"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote per-episode error metadata to {args.output_csv}")
    for key in ["pos_error_m", "orn_error_deg", "tracking_err_rad"]:
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            print(f"{key}: no values found")
            continue
        print(f"{key} summary:")
        for stat, value in summarize(values).items():
            print(f"  {stat}: {value}")
        if key == "pos_error_m":
            count_above(rows, key, args.pos_thresholds)
        elif key == "tracking_err_rad":
            count_above(rows, key, args.tracking_thresholds)
        elif key == "orn_error_deg":
            count_above(rows, key, args.orn_thresholds)


if __name__ == "__main__":
    main()
