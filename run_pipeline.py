#!/usr/bin/env python3
"""
Ocean Data Harmonization Pipeline - Main Orchestrator
=====================================================
Capstone Project: Traceable Harmonization of Ocean Data for ML Readiness

This is the main entry point. Run it after placing your NetCDF source
files in the source_data/ directory.

Usage:
    python run_pipeline.py

What it does:
    1. Reads configuration from pipeline/configs/dataset_mappings.yaml
    2. Ingests source NetCDF files (immutable, hashed)
    3. Harmonizes variables, units, and coordinates via config-driven transforms
    4. Writes a unified Zarr store with embedded provenance and integrity metadata
    5. Produces audit logs, lineage reports, and integrity chain records
    6. Runs verification checks

Directory Structure:
    Capstone_Project/
    ├── source_data/                  # Raw NetCDF files (IMMUTABLE)
    │   ├── dataset_01_*.nc
    │   ├── dataset_02_*.nc
    │   └── dataset_03_*.nc
    ├── pipeline/
    │   ├── configs/
    │   │   └── dataset_mappings.yaml # THE config (edit this, not code!)
    │   └── modules/
    │       ├── ingestor.py           # Reads source files
    │       ├── harmonizer.py         # Config-driven transformations
    │       ├── transforms.py         # Explicit transform functions
    │       ├── zarr_writer.py        # Zarr output with provenance
    │       ├── integrity.py          # SHA-256 hashing, hash chains
    │       └── provenance.py         # Lineage tracking (W3C PROV-inspired)
    ├── output/
    │   ├── zarr_store/
    │   │   └── harmonized.zarr       # THE ML-ready output
    │   └── audit_logs/
    │       ├── lineage_report.json   # Complete data lineage
    │       ├── integrity_records.json# Hash chains per dataset
    │       └── pipeline_run_log.json # Timestamped run log
    ├── run_pipeline.py               # THIS FILE
    └── download_datasets.py          # Script to download from ERDDAP
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import yaml


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # =========================================================================
    # Setup paths
    # =========================================================================
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, "source_data")
    config_path = os.path.join(project_dir, "pipeline", "configs", "dataset_mappings.yaml")
    output_dir = os.path.join(project_dir, "output", "zarr_store")
    audit_dir = os.path.join(project_dir, "output", "audit_logs")

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)

    # =========================================================================
    # Start pipeline
    # =========================================================================
    run_log = {
        "pipeline_version": "1.0.0",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_file": config_path,
        "source_dir": source_dir,
        "output_dir": output_dir,
        "steps": [],
    }

    print("=" * 65)
    print("  Ocean Data Harmonization Pipeline")
    print("  Traceable | Immutable | Config-Driven | ML-Ready")
    print("=" * 65)

    # =========================================================================
    # Step 1: Load Configuration
    # =========================================================================
    print("\n[STEP 1] Loading configuration...")
    start = time.time()

    config = load_config(config_path)
    datasets_config = config["datasets"]

    print(f"  Config loaded: {len(datasets_config)} datasets defined")
    for did, dcfg in datasets_config.items():
        print(f"    - {did}: {dcfg['source_file']} ({dcfg['institution']})")

    run_log["steps"].append({
        "step": 1,
        "name": "load_config",
        "duration_s": round(time.time() - start, 3),
        "datasets_configured": list(datasets_config.keys()),
    })

    # =========================================================================
    # Step 2: Ingest Source Files
    # =========================================================================
    print("\n[STEP 2] Ingesting source NetCDF files...")
    start = time.time()

    # Import pipeline modules
    sys.path.insert(0, project_dir)
    from pipeline.modules.provenance import LineageTracker
    from pipeline.modules.ingestor import DatasetIngestor

    lineage_tracker = LineageTracker()
    ingestor = DatasetIngestor(source_dir, lineage_tracker)

    # Check which files actually exist
    available_files = os.listdir(source_dir) if os.path.exists(source_dir) else []
    nc_files = [f for f in available_files if f.endswith(".nc")]
    print(f"  Found {len(nc_files)} NetCDF files in source_data/: {nc_files}")

    if not nc_files:
        print("\n  ✗ ERROR: No .nc files found in source_data/")
        print("  Please run 'python download_datasets.py' first,")
        print("  or manually download NetCDF files from ERDDAP and place them in source_data/")
        sys.exit(1)

    raw_datasets = ingestor.ingest_all(datasets_config)

    if not raw_datasets:
        print("\n  ✗ ERROR: No datasets could be ingested!")
        print("  Check that your .nc filenames match those in dataset_mappings.yaml")
        sys.exit(1)

    run_log["steps"].append({
        "step": 2,
        "name": "ingest",
        "duration_s": round(time.time() - start, 3),
        "datasets_ingested": list(raw_datasets.keys()),
        "source_hashes": {
            fn: entry["sha256"]
            for fn, entry in ingestor.manifest.entries.items()
        },
    })

    # Save source manifest
    manifest_path = os.path.join(audit_dir, "source_manifest.json")
    ingestor.manifest.save(manifest_path)
    print(f"\n  ✓ Source manifest saved: {manifest_path}")

    # =========================================================================
    # Step 3: Harmonize Datasets
    # =========================================================================
    print("\n[STEP 3] Harmonizing datasets (config-driven)...")
    start = time.time()

    from pipeline.modules.harmonizer import DatasetHarmonizer

    harmonizer = DatasetHarmonizer(config, lineage_tracker)
    harmonized_datasets = {}

    for dataset_id, raw_ds in raw_datasets.items():
        filename = datasets_config[dataset_id]["source_file"]
        integrity_record = ingestor.integrity_records.get(filename)
        if integrity_record is None:
            print(f"  WARNING: No integrity record for {filename}, skipping")
            continue

        harmonized = harmonizer.harmonize_dataset(
            dataset_id=dataset_id,
            ds=raw_ds,
            integrity_record=integrity_record,
        )
        harmonized_datasets[dataset_id] = harmonized

    run_log["steps"].append({
        "step": 3,
        "name": "harmonize",
        "duration_s": round(time.time() - start, 3),
        "datasets_harmonized": list(harmonized_datasets.keys()),
        "observations_per_dataset": {
            did: int(len(ds.obs)) for did, ds in harmonized_datasets.items()
        },
    })

    # =========================================================================
    # Step 4: Write Zarr Store
    # =========================================================================
    print("\n[STEP 4] Writing ML-ready Zarr store...")
    start = time.time()

    from pipeline.modules.zarr_writer import ZarrWriter

    writer = ZarrWriter(output_dir, lineage_tracker)
    zarr_path = writer.write(harmonized_datasets, ingestor.integrity_records)

    run_log["steps"].append({
        "step": 4,
        "name": "write_zarr",
        "duration_s": round(time.time() - start, 3),
        "zarr_path": zarr_path,
    })

    # =========================================================================
    # Step 5: Verify Source Integrity (confirm nothing was modified)
    # =========================================================================
    print("\n[STEP 5] Verifying source file integrity...")
    start = time.time()

    integrity_results = ingestor.verify_source_integrity()

    all_intact = all(integrity_results.values())
    run_log["steps"].append({
        "step": 5,
        "name": "verify_integrity",
        "duration_s": round(time.time() - start, 3),
        "all_sources_intact": all_intact,
        "per_file": integrity_results,
    })

    # =========================================================================
    # Step 6: Verify hash chains
    # =========================================================================
    print("\n[STEP 6] Verifying integrity hash chains...")

    for filename, record in ingestor.integrity_records.items():
        is_valid = record.verify_chain()
        status = "✓ VALID" if is_valid else "✗ BROKEN"
        print(f"  {filename}: chain {status} ({len(record.chain)} steps)")

    # =========================================================================
    # Step 7: Save lineage to audit logs
    # =========================================================================
    print("\n[STEP 7] Saving audit logs...")

    lineage_path = os.path.join(audit_dir, "lineage_report.json")
    lineage_tracker.save_report(lineage_path)
    print(f"  ✓ Lineage report: {lineage_path}")

    # Save integrity records
    integrity_path = os.path.join(audit_dir, "integrity_records.json")
    integrity_data = {
        fn: rec.to_dict() for fn, rec in ingestor.integrity_records.items()
    }
    with open(integrity_path, "w") as f:
        json.dump(integrity_data, f, indent=2)
    print(f"  ✓ Integrity records: {integrity_path}")

    # =========================================================================
    # Step 8: Demo - Trace a variable back to source
    # =========================================================================
    print("\n[STEP 8] Demo: Tracing 'sea_water_temperature' to its sources...")

    lineage = lineage_tracker.trace_variable("sea_water_temperature")
    for path in lineage:
        print(f"\n  Source: {path['source_dataset']}")
        print(f"    File:        {path['source_file']}")
        print(f"    Variable:    {path['source_variable']}")
        print(f"    Institution: {path['institution']}")
        print(f"    Platform:    {path['platform']}")
        print(f"    Transform:   {path['transform_applied']}")
        print(f"    Description: {path['description']}")

    # =========================================================================
    # Finish
    # =========================================================================
    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_log["status"] = "SUCCESS"

    run_log_path = os.path.join(audit_dir, "pipeline_run_log.json")
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Zarr store:        {zarr_path}")
    print(f"  Audit logs:        {audit_dir}")
    print(f"  Source integrity:  {'ALL INTACT' if all_intact else 'ISSUES DETECTED'}")
    print(f"  Run log:           {run_log_path}")
    print("=" * 65)

    # =========================================================================
    # Quick ML usage demo
    # =========================================================================
    print("\n  Quick ML Usage Demo:")
    print("  >>> import xarray as xr")
    print(f'  >>> ds = xr.open_zarr("{zarr_path}")')
    print('  >>> temps = ds["sea_water_temperature"].values  # Kelvin, ready for ML')
    print('  >>> print(ds)  # Rich metadata preserved')


if __name__ == "__main__":
    main()
