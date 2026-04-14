"""
Zarr Output Module
==================
Writes harmonized datasets to a Zarr store with embedded provenance,
lineage metadata, and integrity information.

Addresses Constraint #5: ML usability without domain hardcoding
- The Zarr store is directly loadable by ML practitioners
- Rich metadata preserves domain context for auditing

Addresses Constraint #3: Traceability through aggregation
- Each variable in the Zarr store carries source attribution
"""

import json
import os
import shutil
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from typing import Dict, List

from .integrity import IntegrityRecord, compute_array_hash
from .provenance import LineageTracker


class ZarrWriter:
    """
    Writes harmonized datasets into a Zarr store, combining multiple
    datasets while preserving per-observation source attribution.
    """

    def __init__(self, output_dir: str, lineage_tracker: LineageTracker):
        self.output_dir = output_dir
        self.zarr_path = os.path.join(output_dir, "harmonized.zarr")
        self.lineage_tracker = lineage_tracker

    def write(
        self,
        harmonized_datasets: Dict[str, xr.Dataset],
        integrity_records: Dict[str, IntegrityRecord],
    ) -> str:
        """
        Combine all harmonized datasets and write to Zarr.

        Parameters
        ----------
        harmonized_datasets : dict
            Mapping of dataset_id -> harmonized xr.Dataset
        integrity_records : dict
            Mapping of filename -> IntegrityRecord

        Returns
        -------
        str
            Path to the created Zarr store
        """
        print("\n" + "=" * 60)
        print("  Writing Harmonized Zarr Store")
        print("=" * 60)

        if not harmonized_datasets:
            raise ValueError("No harmonized datasets to write!")

        # Step 1: Add source_dataset_id as a variable for traceability
        datasets_with_source = []
        for dataset_id, ds in harmonized_datasets.items():
            ds_copy = ds.copy()
            # Add a source tracking variable: which dataset each obs came from
            n_obs = len(ds_copy.obs)
            ds_copy["source_dataset_id"] = (
                "obs",
                np.array([dataset_id] * n_obs, dtype=object),
                {"description": "ID of the source dataset this observation came from"}
            )
            # Add observation index within source
            ds_copy["source_obs_index"] = (
                "obs",
                np.arange(n_obs, dtype=np.int64),
                {"description": "Original observation index within the source dataset"}
            )
            datasets_with_source.append(ds_copy)
            print(f"  Added source tracking for: {dataset_id} ({n_obs} obs)")

        # Step 2: Concatenate all datasets along the obs dimension
        print(f"\n  Concatenating {len(datasets_with_source)} datasets...")
        combined = xr.concat(datasets_with_source, dim="obs")

        # Step 3: Add global metadata
        combined.attrs = {
            "title": "Harmonized Ocean Temperature Data for ML",
            "description": (
                "Combined and harmonized ocean temperature observations "
                "from multiple heterogeneous sources. All temperatures in Kelvin, "
                "coordinates in WGS84, times in UTC."
            ),
            "pipeline_version": "1.0.0",
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_datasets": json.dumps(list(harmonized_datasets.keys())),
            "canonical_schema_version": "v1",
            "total_observations": int(len(combined.obs)),
            "conventions": "CF-1.8",
            "history": (
                f"Created by Ocean Data Harmonization Pipeline on "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            ),
        }

        # Step 4: Encode string variables for Zarr compatibility
        if "source_dataset_id" in combined:
            # Zarr doesn't handle object arrays well; encode as categories
            unique_ids = list(harmonized_datasets.keys())
            id_to_int = {sid: i for i, sid in enumerate(unique_ids)}
            encoded_ids = np.array(
                [id_to_int[str(v)] for v in combined["source_dataset_id"].values],
                dtype=np.int32
            )
            combined["source_dataset_idx"] = (
                "obs", encoded_ids,
                {
                    "description": "Integer-encoded source dataset ID",
                    "encoding": json.dumps(id_to_int),
                }
            )
            combined = combined.drop_vars("source_dataset_id")

        # Step 5: Clean up attributes that conflict with xarray encoding
        # xarray uses 'units' internally for time encoding, so we move
        # our custom units to 'canonical_units' to avoid conflicts
        for var_name in list(combined.coords) + list(combined.data_vars):
            if var_name in combined:
                attrs = dict(combined[var_name].attrs)
                if "units" in attrs and np.issubdtype(combined[var_name].dtype, np.datetime64):
                    attrs["canonical_units"] = attrs.pop("units")
                    combined[var_name].attrs = attrs

        # Write to Zarr
        # Remove existing store if present, then write fresh
        if os.path.exists(self.zarr_path):
            try:
                shutil.rmtree(self.zarr_path)
            except PermissionError:
                # If permission denied, use timestamped path
                import time as _time
                self.zarr_path = os.path.join(
                    self.output_dir,
                    f"harmonized_{int(_time.time())}.zarr"
                )

        combined.to_zarr(self.zarr_path, mode="w")

        print(f"\n  ✓ Zarr store written to: {self.zarr_path}")
        print(f"    Total observations: {len(combined.obs)}")
        print(f"    Variables: {list(combined.data_vars)}")
        print(f"    Coordinates: {list(combined.coords)}")

        # Step 6: Save integrity records alongside Zarr
        self._save_integrity_records(integrity_records)

        # Step 7: Save lineage report alongside Zarr
        lineage_path = os.path.join(self.output_dir, "lineage_report.json")
        self.lineage_tracker.save_report(lineage_path)
        print(f"  ✓ Lineage report: {lineage_path}")

        return self.zarr_path

    def _save_integrity_records(self, integrity_records: Dict[str, IntegrityRecord]):
        """Save all integrity hash chains to a JSON file."""
        records = {}
        for filename, record in integrity_records.items():
            records[filename] = record.to_dict()

        filepath = os.path.join(self.output_dir, "integrity_records.json")
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  ✓ Integrity records: {filepath}")

    @staticmethod
    def load_and_verify(zarr_path: str) -> xr.Dataset:
        """
        Load the Zarr store for ML use.
        This is what an ML practitioner would call.
        """
        ds = xr.open_zarr(zarr_path)
        print(f"Loaded Zarr store: {zarr_path}")
        print(f"  Observations: {len(ds.obs)}")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Time range: {ds.time.values.min()} to {ds.time.values.max()}")
        return ds
