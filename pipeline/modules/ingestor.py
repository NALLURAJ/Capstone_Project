"""
Ingestion Module
================
Reads source NetCDF files without modifying them. Computes integrity
hashes and registers provenance metadata.

Addresses Constraint #1: Source immutability
Addresses Constraint #2: Re-derivability
"""

import os
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Optional

from .integrity import compute_file_sha256, IntegrityRecord, SourceManifest
from .provenance import LineageTracker


class DatasetIngestor:
    """
    Ingests source NetCDF files into the pipeline.
    - Reads data into xarray Datasets (in memory, source file untouched)
    - Computes SHA-256 hashes of source files
    - Creates integrity records
    - Registers provenance
    """

    def __init__(self, source_dir: str, lineage_tracker: LineageTracker):
        self.source_dir = source_dir
        self.lineage_tracker = lineage_tracker
        self.manifest = SourceManifest()
        self.ingested_datasets: Dict[str, xr.Dataset] = {}
        self.integrity_records: Dict[str, IntegrityRecord] = {}

    def ingest_file(self, filename: str, dataset_config: dict) -> xr.Dataset:
        """
        Ingest a single NetCDF file.

        Parameters
        ----------
        filename : str
            Name of the .nc file in source_dir
        dataset_config : dict
            Configuration for this dataset from dataset_mappings.yaml

        Returns
        -------
        xr.Dataset
            The loaded dataset (read-only copy in memory)
        """
        filepath = os.path.join(self.source_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Source file not found: {filepath}\n"
                f"Run download_datasets.py first, or place your .nc files in {self.source_dir}"
            )

        # Step 1: Compute hash BEFORE reading (establishes source integrity)
        file_hash = compute_file_sha256(filepath)

        # Step 2: Register in manifest
        self.manifest.register_file(filepath)

        # Step 3: Read the NetCDF file (source file remains untouched on disk).
        # We use LAZY loading (no .load()) here so that large files (e.g., the
        # 507 MB NOAA global SST grid) are NOT fully read into RAM at this point.
        # The harmonizer will first squeeze dimensions and spatially subset the
        # data to the region of interest; ONLY THEN will the small subset be
        # materialized into memory. This prevents out-of-memory crashes.
        ds = xr.open_dataset(filepath)

        # Step 4: Create integrity record (start of hash chain)
        integrity = IntegrityRecord(
            source_file=filename,
            source_hash=file_hash
        )
        self.integrity_records[filename] = integrity

        # Step 5: Register provenance
        dataset_id = os.path.splitext(filename)[0]
        self.lineage_tracker.register_dataset(
            dataset_id=dataset_id,
            source_file=filename,
            institution=dataset_config.get("institution", "Unknown"),
            platform=dataset_config.get("platform", "Unknown"),
        )

        # Step 6: Store reference
        self.ingested_datasets[filename] = ds

        print(f"  [Ingestor] Ingested: {filename}")
        print(f"             SHA-256:  {file_hash[:16]}...")
        print(f"             Dims:     {dict(ds.dims)}")
        print(f"             Vars:     {list(ds.data_vars)}")

        return ds

    def ingest_all(self, datasets_config: dict) -> Dict[str, xr.Dataset]:
        """
        Ingest all datasets defined in the config.

        Parameters
        ----------
        datasets_config : dict
            The 'datasets' section from dataset_mappings.yaml

        Returns
        -------
        Dict mapping dataset_id -> xr.Dataset
        """
        results = {}
        for dataset_id, config in datasets_config.items():
            filename = config["source_file"]
            try:
                ds = self.ingest_file(filename, config)
                results[dataset_id] = ds
            except FileNotFoundError as e:
                print(f"  [Ingestor] WARNING: {e}")
                continue
        return results

    def verify_source_integrity(self) -> Dict[str, bool]:
        """
        Re-verify all source files haven't been modified since ingestion.
        This supports Constraint #1 (immutability verification).
        """
        results = {}
        for filename, entry in self.manifest.entries.items():
            filepath = entry["filepath"]
            if os.path.exists(filepath):
                is_valid = self.manifest.verify_file(filepath)
                results[filename] = is_valid
                status = "✓ INTACT" if is_valid else "✗ MODIFIED!"
                print(f"  [Integrity] {filename}: {status}")
            else:
                results[filename] = False
                print(f"  [Integrity] {filename}: ✗ FILE MISSING!")
        return results
