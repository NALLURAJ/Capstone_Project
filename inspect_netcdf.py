#!/usr/bin/env python3
"""
NetCDF File Inspector
=====================
Run this on any .nc file to see exactly what's inside it —
all variable names, units, fill values, actual data ranges.

Usage:
    python3 inspect_netcdf.py path/to/your_file.nc
"""

import sys
import os
import numpy as np
import xarray as xr


def inspect(filepath):
    print("=" * 65)
    print(f"  Inspecting: {os.path.basename(filepath)}")
    print("=" * 65)

    # xarray automatically decodes scale_factor and fill values
    ds = xr.open_dataset(filepath, mask_and_scale=True)

    print(f"\n  File size : {os.path.getsize(filepath):,} bytes")
    print(f"  Dimensions: {dict(ds.sizes)}")

    print("\n  ── COORDINATES ──────────────────────────────────────────")
    for name, coord in ds.coords.items():
        vals = coord.values
        print(f"\n  [{name}]")
        print(f"    dtype  : {vals.dtype}")
        print(f"    shape  : {vals.shape}")
        if np.issubdtype(vals.dtype, np.number):
            print(f"    min    : {np.nanmin(vals)}")
            print(f"    max    : {np.nanmax(vals)}")
        elif np.issubdtype(vals.dtype, np.datetime64):
            import pandas as pd
            print(f"    first  : {pd.Timestamp(vals.flat[0])}")
            print(f"    last   : {pd.Timestamp(vals.flat[-1])}")
        if coord.attrs:
            for k, v in coord.attrs.items():
                print(f"    {k:12s}: {v}")

    print("\n  ── DATA VARIABLES ───────────────────────────────────────")
    for name, var in ds.data_vars.items():
        vals = var.values
        non_nan = vals[~np.isnan(vals)] if np.issubdtype(vals.dtype, np.floating) else vals.flatten()

        print(f"\n  [{name}]")
        print(f"    dtype        : {vals.dtype}")
        print(f"    shape        : {vals.shape}")
        print(f"    total values : {vals.size:,}")
        print(f"    non-NaN vals : {len(non_nan):,}")
        print(f"    NaN count    : {vals.size - len(non_nan):,}")

        if len(non_nan) > 0 and np.issubdtype(vals.dtype, np.number):
            print(f"    min          : {np.nanmin(vals):.4f}")
            print(f"    max          : {np.nanmax(vals):.4f}")
            print(f"    mean         : {np.nanmean(vals):.4f}")
            print(f"    sample vals  : {non_nan.flat[:5].tolist()}")
        elif len(non_nan) == 0:
            print(f"    *** ALL VALUES ARE NaN / EMPTY ***")
            print(f"    Raw fill value in file:")
            # Try reading WITHOUT decode to see raw values
            ds_raw = xr.open_dataset(filepath, mask_and_scale=False)
            raw_vals = ds_raw[name].values
            print(f"    raw min: {np.nanmin(raw_vals.astype(float))}")
            print(f"    raw max: {np.nanmax(raw_vals.astype(float))}")
            print(f"    raw sample: {raw_vals.flat[:5].tolist()}")

        if var.attrs:
            print(f"    ── attributes:")
            for k, v in var.attrs.items():
                print(f"      {k:20s}: {v}")

    print("\n  ── GLOBAL ATTRIBUTES ────────────────────────────────────")
    for k, v in ds.attrs.items():
        print(f"  {k:20s}: {v}")

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Variables found: {list(ds.data_vars.keys())}")
    print(f"  Coordinates    : {list(ds.coords.keys())}")

    # Tell them which variable to use for temperature
    temp_candidates = [v for v in list(ds.data_vars) + list(ds.coords)
                       if any(t in v.lower() for t in
                              ['temp', 'sst', 'temperature'])]
    if temp_candidates:
        print(f"\n  ✓ Temperature variable(s) found: {temp_candidates}")
        for tc in temp_candidates:
            var = ds[tc]
            vals = var.values
            non_nan = vals[~np.isnan(vals)] if np.issubdtype(
                vals.dtype, np.floating) else vals.flatten()
            if len(non_nan) > 0:
                print(f"    '{tc}' has {len(non_nan):,} real values — USE THIS ONE")
            else:
                print(f"    '{tc}' is ALL NaN — do NOT use this")
    else:
        print(f"\n  ✗ No temperature variable found!")
        print(f"    Available variables: {list(ds.data_vars.keys())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # If no argument, inspect all .nc files in source_data/
        source_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "source_data"
        )
        nc_files = [f for f in os.listdir(source_dir) if f.endswith(".nc")]
        if not nc_files:
            print("Usage: python3 inspect_netcdf.py yourfile.nc")
            print("Or place .nc files in source_data/ and run without arguments.")
            sys.exit(1)
        for f in nc_files:
            inspect(os.path.join(source_dir, f))
            print("\n")
    else:
        inspect(sys.argv[1])
