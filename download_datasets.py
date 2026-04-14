#!/usr/bin/env python3
"""
ERDDAP Dataset Downloader for Capstone Project
================================================
Downloads 3 heterogeneous ocean temperature datasets as NetCDF files
using direct ERDDAP URLs (no erddapy dependency needed).

Instructions:
    1. Install dependencies:  pip3 install xarray netCDF4 requests
    2. Run this script:       python3 download_datasets.py
    3. The 3 .nc files will appear in a 'source_data/' folder

The 3 datasets chosen:
    Dataset 1: Barkley Canyon CTD (Ocean Networks Canada)
               - Fixed deep-ocean sensor
               - Variable: "Temperature" (°C)
               - Tabular: timeseries_id, time, Temperature, latitude, longitude, depth

    Dataset 2: NOAA Sea Surface Temperature (CoastWatch ERDDAP)
               - Satellite-derived gridded data
               - Variable: "sst" (°C)
               - Gridded: time, latitude, longitude

    Dataset 3: Baynes Sound CTD (Ocean Networks Canada)
               - Coastal shallow-water sensor
               - Variable: "Temperature" (°C)
               - Same variable name as Dataset 1 but different location/depth

These 3 give you:
    - Different variable names (Temperature vs sst)
    - Different platforms (fixed sensor vs satellite vs coastal sensor)
    - Different spatial coverage (deep ocean vs sea surface vs coastal)
    - Partial overlap in time
    - Different metadata quality/completeness
"""

import os
import sys
import hashlib
import json
import urllib.request
import ssl
from datetime import datetime, timezone


# Disable SSL verification (some ERDDAP servers have cert issues)
ssl._create_default_https_context = ssl._create_unverified_context


def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url, filepath, description):
    """Download a file from a URL with progress indication."""
    print(f"  Downloading from: {url[:80]}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"  ✓ Saved: {os.path.basename(filepath)} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_dataset_1(output_dir):
    """
    Dataset 1: Barkley Canyon CTD - Deep ocean fixed sensor
    Source: Ocean Networks Canada ERDDAP (tabledap)
    Direct URL approach - downloads as NetCDF without erddapy.
    """
    print("\n[Dataset 1] Barkley Canyon CTD (Deep Ocean, Fixed Sensor)")
    print("  Source: Ocean Networks Canada")

    filepath = os.path.join(output_dir, "dataset_01_barkley_canyon_ctd.nc")
    if os.path.exists(filepath):
        print(f"  ✓ Already exists: {os.path.basename(filepath)}")
        return filepath

    # Try multiple dataset IDs (deployments change over time)
    dataset_ids = ["scalar_1211656", "scalar_1214292", "scalar_1207539"]
    time_ranges = [
        ("2023-01-01T00:00:00Z", "2023-06-30T23:59:59Z"),
        ("2023-09-01T00:00:00Z", "2024-03-31T23:59:59Z"),
        ("2021-01-01T00:00:00Z", "2021-06-30T23:59:59Z"),
    ]

    for did, (t_start, t_end) in zip(dataset_ids, time_ranges):
        url = (
            f"https://data.oceannetworks.ca/erddap/tabledap/{did}.nc"
            f"?timeseries_id,time,Temperature,latitude,longitude,depth"
            f"&time>={t_start}&time<={t_end}"
        )
        if download_file(url, filepath, "Barkley Canyon CTD"):
            return filepath
        print(f"  Trying next dataset ID...")

    print("  ✗ All attempts failed for Dataset 1")
    return None


def download_dataset_2(output_dir):
    """
    Dataset 2: NOAA SST - Satellite-derived sea surface temperature (gridded)
    Source: NOAA CoastWatch ERDDAP (griddap)
    This gives a VERY different structure from the CTD datasets.
    """
    print("\n[Dataset 2] NOAA Sea Surface Temperature (Satellite, Gridded)")
    print("  Source: NOAA CoastWatch ERDDAP")

    filepath = os.path.join(output_dir, "dataset_02_noaa_sst_gridded.nc")
    if os.path.exists(filepath):
        print(f"  ✓ Already exists: {os.path.basename(filepath)}")
        return filepath

    # HadISST Monthly SST - subset to NE Pacific near Barkley Canyon
    # griddap URL format: dataset.nc?variable[(time_start):(time_end)][(lat_start):(lat_end)][(lon_start):(lon_end)]
    urls = [
        # HadISST (monthly, global, 1°)
        (
            "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdHadISST.nc"
            "?sst[(2023-01-15T00:00:00Z):(2023-06-15T00:00:00Z)]"
            "[(47.0):(50.0)][(-128.0):(-124.0)]"
        ),
        # NOAA ERSST v4 (monthly, 2°)
        (
            "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nceiErsstv4.nc"
            "?sst[(2020-01-15T00:00:00Z):(2020-06-15T00:00:00Z)][(0.0):(0.0)]"
            "[(46.0):(50.0)][(232.0):(236.0)]"
        ),
    ]

    for url in urls:
        if download_file(url, filepath, "NOAA SST Gridded"):
            return filepath
        print(f"  Trying alternative dataset...")

    print("  ✗ All attempts failed for Dataset 2")
    return None


def download_dataset_3(output_dir):
    """
    Dataset 3: Baynes Sound CTD - Coastal shallow-water sensor
    Source: Ocean Networks Canada ERDDAP (tabledap)
    Same 'Temperature' variable name as Dataset 1 but different location/depth.
    """
    print("\n[Dataset 3] Baynes Sound CTD (Coastal, Shallow Sensor)")
    print("  Source: Ocean Networks Canada")

    filepath = os.path.join(output_dir, "dataset_03_baynes_sound_ctd.nc")
    if os.path.exists(filepath):
        print(f"  ✓ Already exists: {os.path.basename(filepath)}")
        return filepath

    dataset_ids = ["scalar_1203269", "scalar_1203272", "scalar_1203275"]
    time_ranges = [
        ("2020-01-01T00:00:00Z", "2020-06-30T23:59:59Z"),
        ("2020-01-01T00:00:00Z", "2020-06-30T23:59:59Z"),
        ("2020-01-01T00:00:00Z", "2020-06-30T23:59:59Z"),
    ]

    for did, (t_start, t_end) in zip(dataset_ids, time_ranges):
        url = (
            f"https://data.oceannetworks.ca/erddap/tabledap/{did}.nc"
            f"?timeseries_id,time,Temperature,latitude,longitude,depth"
            f"&time>={t_start}&time<={t_end}"
        )
        if download_file(url, filepath, "Baynes Sound CTD"):
            return filepath
        print(f"  Trying next dataset ID...")

    print("  ✗ All attempts failed for Dataset 3")
    return None


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source_data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  ERDDAP Dataset Downloader - Capstone Project")
    print("  Downloading 3 heterogeneous ocean temperature datasets")
    print("  (Using direct ERDDAP URLs - no erddapy needed)")
    print("=" * 65)

    # Download all 3 datasets
    f1 = download_dataset_1(output_dir)
    f2 = download_dataset_2(output_dir)
    f3 = download_dataset_3(output_dir)

    downloaded = [f for f in [f1, f2, f3] if f is not None]

    # Compute SHA-256 hashes
    print("\n" + "=" * 65)
    print("  Computing SHA-256 hashes for integrity tracking")
    print("=" * 65)

    manifest = {
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "files": {}
    }

    for filepath in downloaded:
        file_hash = compute_sha256(filepath)
        filename = os.path.basename(filepath)
        manifest["files"][filename] = {
            "path": filepath,
            "sha256": file_hash,
            "size_bytes": os.path.getsize(filepath)
        }
        print(f"  {filename}")
        print(f"    SHA-256: {file_hash}")
        print(f"    Size:    {os.path.getsize(filepath):,} bytes")

    # Save manifest
    manifest_path = os.path.join(output_dir, "source_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  ✓ Manifest saved: {manifest_path}")

    # Summary
    print("\n" + "=" * 65)
    print(f"  DONE: {len(downloaded)}/3 datasets downloaded successfully")
    print("=" * 65)

    if len(downloaded) < 3:
        print("\n  Some downloads failed. MANUAL ALTERNATIVE:")
        print("  ─────────────────────────────────────────────")
        print("  For Ocean Networks Canada datasets:")
        print("    1. Go to https://data.oceannetworks.ca/erddap/search/index.html")
        print("    2. Search 'temperature'")
        print("    3. Click 'data' next to any CTD dataset")
        print("    4. Check the variable boxes (time, Temperature, latitude, longitude, depth)")
        print("    5. Set a time range (e.g., 2023-01-01 to 2023-06-30)")
        print("    6. Change file type dropdown to .nc")
        print("    7. Click Submit → save the .nc file")
        print("    8. Rename it to match: dataset_01_barkley_canyon_ctd.nc")
        print()
        print("  For NOAA SST (gridded, different structure):")
        print("    1. Go to https://coastwatch.pfeg.noaa.gov/erddap/search/index.html")
        print("    2. Search 'sst'")
        print("    3. Click 'data' for 'HadISST Average Sea Surface Temperature'")
        print("    4. Set time range and lat/lon bounds (47-50°N, 128-124°W)")
        print("    5. Change file type to .nc, click Submit")
        print("    6. Rename to: dataset_02_noaa_sst_gridded.nc")
        print()
        print("  Place all .nc files in the source_data/ folder.")

    if len(downloaded) == 3:
        print("\n  ✓ All datasets ready!")
        print("  Next step: python3 run_pipeline.py")
    else:
        print("\n  You can also use sample data instead:")
        print("  python3 create_sample_data.py")
        print("  python3 run_pipeline.py")


if __name__ == "__main__":
    main()
