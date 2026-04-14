#!/usr/bin/env python3
"""
Output Explorer
===============
Read and understand the harmonized Zarr store output.
Shows you exactly what's inside, and exports a CSV you can open in Excel.

Run:
    python3 explore_output.py
"""

import os
import sys
import glob
import json

import xarray as xr
import numpy as np
import pandas as pd


def find_zarr_store(output_dir):
    """Find the most recent Zarr store in the output directory."""
    candidates = glob.glob(os.path.join(output_dir, "*.zarr"))
    if not candidates:
        return None
    # Return the most recently modified one
    return max(candidates, key=os.path.getmtime)


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    zarr_dir    = os.path.join(project_dir, "output", "zarr_store")
    audit_dir   = os.path.join(project_dir, "output", "audit_logs")

    # ── Find the Zarr store ───────────────────────────────────────────
    zarr_path = find_zarr_store(zarr_dir)
    if zarr_path is None:
        print("ERROR: No Zarr store found.")
        print("Run 'python3 run_pipeline.py' first.")
        sys.exit(1)

    print("=" * 65)
    print("  Zarr Output Explorer")
    print("=" * 65)
    print(f"\n  Zarr store: {zarr_path}\n")

    # ── Load the Zarr store ───────────────────────────────────────────
    ds = xr.open_zarr(zarr_path)

    # ── 1. High-level summary ─────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 1: What is inside the Zarr store?")
    print("━" * 65)
    print(f"\n  Total observations (rows): {len(ds.obs):,}")
    print(f"\n  Variables (the data columns):")
    for v in ds.data_vars:
        arr = ds[v]
        print(f"    • {v}")
        print(f"        dtype : {arr.dtype}")
        if np.issubdtype(arr.dtype, np.number):
            vals = arr.values
            print(f"        min   : {float(np.nanmin(vals)):.4f}")
            print(f"        max   : {float(np.nanmax(vals)):.4f}")
            print(f"        mean  : {float(np.nanmean(vals)):.4f}")
        for k, v2 in arr.attrs.items():
            print(f"        {k:6s}: {v2}")

    print(f"\n  Coordinates (index columns):")
    for c in ds.coords:
        arr = ds.coords[c]
        print(f"    • {c}  (dtype: {arr.dtype})")

    # ── 2. First 10 rows ──────────────────────────────────────────────
    print("\n" + "━" * 65)
    print("  SECTION 2: First 10 rows (like a spreadsheet preview)")
    print("━" * 65)
    df_preview = ds.isel(obs=slice(0, 10)).to_dataframe().reset_index(drop=True)
    # Make it more readable
    if "time" in df_preview.columns:
        df_preview["time"] = pd.to_datetime(df_preview["time"]).dt.strftime("%Y-%m-%d %H:%M")
    if "sea_water_temperature" in df_preview.columns:
        df_preview["temp_celsius"] = (df_preview["sea_water_temperature"] - 273.15).round(3)
    print(df_preview.to_string(index=False))

    # ── 3. Source breakdown ───────────────────────────────────────────
    print("\n" + "━" * 65)
    print("  SECTION 3: Where did each row come from?")
    print("━" * 65)

    # Read the source encoding from the dataset_idx variable
    if "source_dataset_idx" in ds:
        idx_var   = ds["source_dataset_idx"]
        encoding  = json.loads(idx_var.attrs.get("encoding", "{}"))
        idx_vals  = idx_var.values

        int_to_name = {v: k for k, v in encoding.items()}
        print(f"\n  Dataset ID encoding: {encoding}")
        print(f"\n  Observations per source:")
        unique, counts = np.unique(idx_vals, return_counts=True)
        for u, c in zip(unique, counts):
            name = int_to_name.get(int(u), f"dataset_{u}")
            pct  = 100 * c / len(idx_vals)
            bar  = "█" * int(pct / 2)
            print(f"    [{u}] {name}")
            print(f"        {c:,} rows  ({pct:.1f}%)  {bar}")

    # ── 4. Time range per source ───────────────────────────────────────
    print("\n" + "━" * 65)
    print("  SECTION 4: Time ranges (shows partial overlap)")
    print("━" * 65)
    if "source_dataset_idx" in ds and "time" in ds.coords:
        idx_vals  = ds["source_dataset_idx"].values
        time_vals = ds.coords["time"].values
        encoding  = json.loads(ds["source_dataset_idx"].attrs.get("encoding", "{}"))
        int_to_name = {v: k for k, v in encoding.items()}

        for uid in np.unique(idx_vals):
            mask      = idx_vals == uid
            name      = int_to_name.get(int(uid), f"dataset_{uid}")
            t_subset  = time_vals[mask]
            t_min     = pd.Timestamp(t_subset.min()).strftime("%Y-%m-%d")
            t_max     = pd.Timestamp(t_subset.max()).strftime("%Y-%m-%d")
            print(f"    {name}")
            print(f"        {t_min}  →  {t_max}")

    # ── 5. Temperature stats per source ───────────────────────────────
    print("\n" + "━" * 65)
    print("  SECTION 5: Temperature statistics per source (Kelvin & °C)")
    print("━" * 65)
    if "source_dataset_idx" in ds and "sea_water_temperature" in ds:
        idx_vals  = ds["source_dataset_idx"].values
        temp_vals = ds["sea_water_temperature"].values
        encoding  = json.loads(ds["source_dataset_idx"].attrs.get("encoding", "{}"))
        int_to_name = {v: k for k, v in encoding.items()}

        for uid in np.unique(idx_vals):
            mask   = idx_vals == uid
            name   = int_to_name.get(int(uid), f"dataset_{uid}")
            subset = temp_vals[mask]
            print(f"\n    {name}:")
            print(f"        Min  : {np.nanmin(subset):.2f} K  ({np.nanmin(subset)-273.15:.2f} °C)")
            print(f"        Max  : {np.nanmax(subset):.2f} K  ({np.nanmax(subset)-273.15:.2f} °C)")
            print(f"        Mean : {np.nanmean(subset):.2f} K  ({np.nanmean(subset)-273.15:.2f} °C)")

    # ── 6. Lineage report ─────────────────────────────────────────────
    print("\n" + "━" * 65)
    print("  SECTION 6: Lineage / Traceability")
    print("  (Where did 'sea_water_temperature' come from?)")
    print("━" * 65)
    lineage_path = os.path.join(audit_dir, "lineage_report.json")
    if os.path.exists(lineage_path):
        with open(lineage_path) as f:
            lineage = json.load(f)
        for did, record in lineage.get("datasets", {}).items():
            for deriv in record.get("derivations", []):
                if deriv["derived_variable"] == "sea_water_temperature":
                    act_id = deriv["activity_id"]
                    act    = next(
                        (a for a in record["activities"] if a["activity_id"] == act_id),
                        {}
                    )
                    print(f"\n    Source dataset : {did}")
                    print(f"    Source file    : {record['entity']['source_file']}")
                    print(f"    Original var   : {deriv['source_variable']}")
                    print(f"    Institution    : {record['entity']['institution']}")
                    print(f"    Transform used : {act.get('transform_applied','?')}")
                    print(f"    Description    : {act.get('description','?')}")
    else:
        print("  (Run pipeline first to generate lineage report)")

    # ── 7. Integrity chain ────────────────────────────────────────────
    print("\n" + "━" * 65)
    print("  SECTION 7: Integrity Hash Chains")
    print("  (Proves data was not tampered with)")
    print("━" * 65)
    integrity_path = os.path.join(audit_dir, "integrity_records.json")
    if os.path.exists(integrity_path):
        with open(integrity_path) as f:
            records = json.load(f)
        for fname, rec in records.items():
            chain   = rec.get("chain", [])
            valid   = rec.get("is_valid", False)
            status  = "✓ VALID" if valid else "✗ BROKEN"
            print(f"\n    {fname}: {status}  ({len(chain)} steps)")
            for step in chain:
                action = step.get("action", "?")
                ts     = step.get("timestamp", "")[:19]
                h      = step.get("chain_hash", "")[:16]
                print(f"      step {step['step']}: {action:40s}  hash={h}...  [{ts}]")

    # ── 8. Export to CSV and Excel ────────────────────────────────────
    print("\n" + "━" * 65)
    print("  SECTION 8: Exporting to CSV and Excel for easy viewing")
    print("━" * 65)

    df = ds.to_dataframe().reset_index(drop=True)

    # Add human-readable columns
    if "sea_water_temperature" in df.columns:
        df["temperature_celsius"] = (df["sea_water_temperature"] - 273.15).round(4)
        df["temperature_kelvin"]  = df["sea_water_temperature"].round(4)
        df.drop(columns=["sea_water_temperature"], inplace=True)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Decode source dataset name
    if "source_dataset_idx" in df.columns and "source_dataset_idx" in ds:
        encoding    = json.loads(ds["source_dataset_idx"].attrs.get("encoding", "{}"))
        int_to_name = {v: k for k, v in encoding.items()}
        df["source_dataset"] = df["source_dataset_idx"].map(
            lambda x: int_to_name.get(int(x), f"dataset_{x}")
        )
        df.drop(columns=["source_dataset_idx"], inplace=True)

    # Reorder columns nicely
    preferred = ["time", "latitude", "longitude", "depth",
                 "temperature_celsius", "temperature_kelvin",
                 "source_dataset", "source_obs_index"]
    cols = [c for c in preferred if c in df.columns]
    df   = df[cols]

    # Save CSV (all rows)
    csv_path = os.path.join(zarr_dir, "harmonized_output.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ CSV exported ({len(df):,} rows):  {csv_path}")

    # Save Excel (first 5000 rows — Excel has row limits and can be slow)
    try:
        excel_path = os.path.join(zarr_dir, "harmonized_output_preview.xlsx")
        df.head(5000).to_excel(excel_path, index=False, sheet_name="Harmonized Data")
        print(f"  ✓ Excel preview (first 5,000 rows): {excel_path}")
        print(f"    (Full data is in the CSV above)")
    except ImportError:
        print("  ℹ Excel export skipped (install openpyxl: pip3 install openpyxl)")

    # ── Done ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXPLORATION COMPLETE")
    print("=" * 65)
    print(f"\n  Your output files are in: {zarr_dir}")
    print(f"  Open 'harmonized_output_preview.xlsx' in Excel to browse the data.")
    print(f"\n  What each column means:")
    print(f"    time               → when the measurement was taken (UTC)")
    print(f"    latitude           → where (north/south, degrees)")
    print(f"    longitude          → where (east/west, degrees)")
    print(f"    depth              → how deep in meters (0 = surface/satellite)")
    print(f"    temperature_celsius→ sea water temperature in °C")
    print(f"    temperature_kelvin → same in Kelvin (unit used by ML models)")
    print(f"    source_dataset     → which original file this row came from")
    print(f"    source_obs_index   → row number in the original source file")


if __name__ == "__main__":
    main()
