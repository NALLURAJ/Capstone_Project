"""
Harmonization Module
====================
Config-driven transformation of heterogeneous datasets into a
canonical schema. All transformations are explicit, logged, and
traceable.

Addresses Constraint #4: Explicit transformations
Addresses Constraint #5: ML usability without domain hardcoding

KEY DESIGN: The Guiding Question answer lives here.
Changing how a variable is interpreted only requires editing the YAML
config file — not this code. This module reads the config and applies
whatever mappings/transforms are specified there.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .transforms import get_transform
from .integrity import IntegrityRecord, compute_array_hash
from .provenance import ProvenanceRecord, LineageTracker


class DatasetHarmonizer:
    """
    Transforms datasets from their source schema to the canonical schema
    based entirely on configuration (no hardcoded variable names).
    """

    def __init__(self, config: dict, lineage_tracker: LineageTracker):
        """
        Parameters
        ----------
        config : dict
            Full parsed config from dataset_mappings.yaml
        lineage_tracker : LineageTracker
            Shared lineage tracker across all datasets
        """
        self.config = config
        self.canonical_schema = config["canonical_schema"]
        self.lineage_tracker = lineage_tracker

    def harmonize_dataset(
        self,
        dataset_id: str,
        ds: xr.Dataset,
        integrity_record: IntegrityRecord
    ) -> xr.Dataset:
        """
        Harmonize a single dataset according to its config.

        Parameters
        ----------
        dataset_id : str
            ID matching a key in config['datasets']
        ds : xr.Dataset
            The raw ingested dataset
        integrity_record : IntegrityRecord
            Hash chain for this dataset

        Returns
        -------
        xr.Dataset
            Harmonized dataset with canonical variable names and units
        """
        ds_config = self.config["datasets"][dataset_id]
        provenance = self.lineage_tracker.records.get(dataset_id)

        print(f"\n  [Harmonizer] Processing: {dataset_id}")

        # Step 0: Preprocessing (e.g., flatten gridded data to tabular)
        preprocessing = ds_config.get("preprocessing", {})

        # 0a. Squeeze single-element dimensions (e.g., altitude dim in NOAA 4D grid)
        squeeze_dims = preprocessing.get("squeeze_dims", [])
        if squeeze_dims:
            ds = self._squeeze_dimensions(ds, squeeze_dims, dataset_id, integrity_record)

        # 0b. Subset to a spatial region BEFORE flattening (critical for large global files)
        subset_region = preprocessing.get("subset_region", {})
        if subset_region:
            ds = self._subset_region(ds, subset_region, dataset_id, integrity_record)

        # 0c. Flatten gridded (time x lat x lon) data into tabular rows
        if preprocessing.get("flatten_grid", False):
            ds = self._flatten_grid(ds, dataset_id, integrity_record)

        # Step 1: Apply variable mappings
        harmonized_vars = {}
        mappings = ds_config.get("variable_mappings", {})

        for source_var, mapping in mappings.items():
            target_var = mapping["target"]
            transform_name = mapping["transform"]

            if source_var not in ds:
                # Check if it's a coordinate
                if source_var in ds.coords:
                    data = ds.coords[source_var].values
                else:
                    print(f"    WARNING: Variable '{source_var}' not found in dataset, skipping")
                    continue
            else:
                data = ds[source_var].values

            # Apply the transform
            transform_fn = get_transform(transform_name)
            transformed_data, description = transform_fn(data)

            # Record in integrity chain
            data_hash = compute_array_hash(transformed_data)
            integrity_record.add_step(
                action=f"transform_{source_var}_to_{target_var}",
                description=description,
                data_hash=data_hash,
                metadata={
                    "source_variable": source_var,
                    "target_variable": target_var,
                    "transform": transform_name,
                    "source_units": mapping.get("source_units", "unknown"),
                    "target_units": mapping.get("target_units", "unknown"),
                }
            )

            # Record in provenance
            if provenance:
                activity_id = provenance.log_activity(
                    activity_name=f"harmonize_{source_var}",
                    description=description,
                    input_variables=[source_var],
                    output_variables=[target_var],
                    transform_name=transform_name,
                    parameters={
                        "source_units": mapping.get("source_units"),
                        "target_units": mapping.get("target_units"),
                    }
                )
                provenance.log_derivation(
                    source_variable=source_var,
                    derived_variable=target_var,
                    activity_id=activity_id,
                )

            harmonized_vars[target_var] = transformed_data
            print(f"    {source_var} -> {target_var}: {description}")

        # Step 2: Add synthetic variables (e.g., depth=0 for satellite data)
        synthetics = ds_config.get("synthetic_variables", {})
        for var_name, syn_config in synthetics.items():
            target_var = syn_config["target"]
            fill_value = syn_config["fill_value"]

            # Determine the size from an existing variable
            ref_size = len(next(iter(harmonized_vars.values())))
            synthetic_data = np.full(ref_size, fill_value, dtype=np.float64)

            harmonized_vars[target_var] = synthetic_data

            # Record provenance for synthetic variable
            if provenance:
                activity_id = provenance.log_activity(
                    activity_name=f"synthesize_{target_var}",
                    description=f"Synthetic variable: {syn_config.get('rationale', '')}",
                    input_variables=[],
                    output_variables=[target_var],
                    transform_name="synthetic_fill",
                    parameters={"fill_value": fill_value},
                )
                provenance.log_derivation(
                    source_variable="<synthetic>",
                    derived_variable=target_var,
                    activity_id=activity_id,
                )

            data_hash = compute_array_hash(synthetic_data)
            integrity_record.add_step(
                action=f"synthesize_{target_var}",
                description=f"Created synthetic variable with fill_value={fill_value}",
                data_hash=data_hash,
                metadata={"rationale": syn_config.get("rationale", "")},
            )
            print(f"    <synthetic> -> {target_var}: filled with {fill_value} ({syn_config.get('rationale', '')})")

        # Step 3: Build harmonized xarray Dataset
        harmonized_ds = self._build_harmonized_dataset(
            harmonized_vars, dataset_id, ds_config
        )

        # Step 4a: Drop NaN temperature values if configured
        if preprocessing.get("drop_nan_temperature", False):
            temp_var = "sea_water_temperature"
            if temp_var in harmonized_ds:
                before = len(harmonized_ds["time"])
                harmonized_ds = harmonized_ds.where(
                    harmonized_ds[temp_var].notnull(), drop=True
                )
                after = len(harmonized_ds["time"])
                print(f"    Dropped NaN temperature rows: {before} -> {after}")

        # Step 4b: Drop extreme/fill-value temperatures (e.g., NOAA sst fill = -32M°F)
        if preprocessing.get("drop_extreme_temperature", False):
            temp_var = "sea_water_temperature"
            if temp_var in harmonized_ds:
                schema = self.canonical_schema["variables"].get(temp_var, {})
                valid_range = schema.get("valid_range", [270.0, 320.0])
                vmin, vmax = valid_range[0], valid_range[1]
                before = int(harmonized_ds["time"].size)
                mask = (
                    (harmonized_ds[temp_var] >= vmin) &
                    (harmonized_ds[temp_var] <= vmax)
                )
                harmonized_ds = harmonized_ds.where(mask, drop=True)
                after = int(harmonized_ds["time"].size)
                print(f"    Dropped extreme temperature rows (outside [{vmin}, {vmax}] K): {before} -> {after}")

        # Step 5: Validate against canonical schema
        self._validate_against_schema(harmonized_ds, dataset_id)

        return harmonized_ds

    def _squeeze_dimensions(
        self,
        ds: xr.Dataset,
        dims_to_squeeze: list,
        dataset_id: str,
        integrity_record: IntegrityRecord
    ) -> xr.Dataset:
        """
        Remove dimensions of size 1 (e.g., the 'altitude' dim in the NOAA 4D
        satellite file which has shape (time, altitude=1, latitude, longitude)).
        After squeezing, the file is effectively 3D and can be flattened normally.
        """
        squeezed_dims = []
        for dim in dims_to_squeeze:
            if dim in ds.dims and ds.dims[dim] == 1:
                ds = ds.squeeze(dim, drop=True)
                squeezed_dims.append(dim)
                print(f"    Squeezed dimension '{dim}' (size=1) out of dataset")
            elif dim in ds.dims:
                print(f"    WARNING: Cannot squeeze '{dim}' — size is {ds.dims[dim]}, not 1")
            else:
                print(f"    INFO: Dimension '{dim}' not present — nothing to squeeze")

        if squeezed_dims:
            integrity_record.add_step(
                action="squeeze_dims",
                description=f"Squeezed single-element dimensions: {squeezed_dims}",
                data_hash=compute_array_hash(
                    np.array([ds.dims[k] for k in ds.dims], dtype=np.int64)
                ),
                metadata={"squeezed_dims": squeezed_dims, "remaining_dims": dict(ds.dims)}
            )

        return ds

    def _subset_region(
        self,
        ds: xr.Dataset,
        region: dict,
        dataset_id: str,
        integrity_record: IntegrityRecord
    ) -> xr.Dataset:
        """
        Spatially subset a gridded dataset to a lat/lon bounding box.

        This is CRITICAL for large global files (e.g., the 507 MB NOAA SST file
        with 63 million grid points). Loading the full global grid into memory
        would likely cause an out-of-memory crash. We subset to the NE Pacific
        region of interest before doing anything else.

        The region dict looks like:
            lat_min: 44.0  lat_max: 52.0
            lon_min: -132.0  lon_max: -122.0
        """
        lat_min = region.get("lat_min")
        lat_max = region.get("lat_max")
        lon_min = region.get("lon_min")
        lon_max = region.get("lon_max")

        # Identify latitude and longitude coordinate names (flexible naming)
        lat_name = None
        lon_name = None
        for name in ds.coords:
            nl = name.lower()
            if nl in ("latitude", "lat") and lat_name is None:
                lat_name = name
            if nl in ("longitude", "lon") and lon_name is None:
                lon_name = name

        if lat_name is None or lon_name is None:
            print(f"    WARNING: Cannot subset — lat/lon coords not found in {list(ds.coords)}")
            return ds

        lat_vals = ds.coords[lat_name].values
        lon_vals = ds.coords[lon_name].values

        before_lat = len(lat_vals)
        before_lon = len(lon_vals)

        lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)

        ds = ds.isel(**{lat_name: lat_mask, lon_name: lon_mask})

        after_lat = int(ds.dims[lat_name])
        after_lon = int(ds.dims[lon_name])

        print(f"    Spatial subset: lat {before_lat} → {after_lat} points "
              f"({lat_min}°N – {lat_max}°N)")
        print(f"    Spatial subset: lon {before_lon} → {after_lon} points "
              f"({lon_min}°E – {lon_max}°E)")
        print(f"    Grid reduced: {before_lat * before_lon:,} → "
              f"{after_lat * after_lon:,} grid cells per time step")

        integrity_record.add_step(
            action="subset_region",
            description=(
                f"Spatial subset to lat [{lat_min}, {lat_max}], "
                f"lon [{lon_min}, {lon_max}]"
            ),
            data_hash=compute_array_hash(
                np.array([after_lat, after_lon], dtype=np.int64)
            ),
            metadata={
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max,
                "grid_before": before_lat * before_lon,
                "grid_after": after_lat * after_lon,
            }
        )

        return ds

    def _flatten_grid(self, ds: xr.Dataset, dataset_id: str,
                      integrity_record: IntegrityRecord) -> xr.Dataset:
        """
        Flatten a gridded dataset (time x lat x lon) into tabular form
        so it can be combined with tabular CTD data.

        After squeeze_dims and subset_region the dataset should be 3D.
        We stack all three dimensions into a single 'obs' dimension.
        """
        print(f"    Flattening gridded data to tabular format...")
        print(f"    Current dims before flatten: {dict(ds.dims)}")

        # Discover the actual dimension names (flexible: lat or latitude, lon or longitude)
        all_dims = list(ds.dims)
        time_dim = next((d for d in all_dims if d.lower() == "time"), None)
        lat_dim  = next((d for d in all_dims if d.lower() in ("latitude", "lat")), None)
        lon_dim  = next((d for d in all_dims if d.lower() in ("longitude", "lon")), None)

        if not all([time_dim, lat_dim, lon_dim]):
            raise ValueError(
                f"Cannot flatten grid for {dataset_id}: "
                f"expected time/lat/lon dims, found {all_dims}"
            )

        stacked = ds.stack(obs=(time_dim, lat_dim, lon_dim))
        flat = stacked.reset_index("obs")

        # Convert to a dataset with a single 'obs' dimension
        flat_ds = xr.Dataset()
        for var in ds.data_vars:
            if var in flat:
                flat_ds[var] = ("obs", flat[var].values)
        for coord_name in [time_dim, lat_dim, lon_dim]:
            if coord_name in flat.coords:
                flat_ds[coord_name] = ("obs", flat[coord_name].values)

        integrity_record.add_step(
            action="flatten_grid",
            description=f"Flattened gridded data from dims {dict(ds.dims)} to tabular ({len(flat_ds.obs)} observations)",
            data_hash=compute_array_hash(
                np.array([len(flat_ds.obs)], dtype=np.int64)
            ),
            metadata={"original_dims": {k: v for k, v in ds.dims.items()}}
        )

        return flat_ds

    def _build_harmonized_dataset(
        self,
        harmonized_vars: Dict[str, np.ndarray],
        dataset_id: str,
        ds_config: dict
    ) -> xr.Dataset:
        """Build a clean xarray Dataset from harmonized variables."""

        # Use 'obs' as the single dimension (observation-oriented)
        n_obs = len(next(iter(harmonized_vars.values())))

        data_vars = {}
        coords = {}

        # Separate coordinates from data variables
        coord_names = {"time", "latitude", "longitude", "depth"}

        for var_name, values in harmonized_vars.items():
            # Get canonical metadata
            canonical = self.canonical_schema["variables"].get(var_name, {})

            attrs = {
                "standard_name": canonical.get("standard_name", var_name),
                "units": canonical.get("units", "unknown"),
                "source_dataset": dataset_id,
                "source_institution": ds_config.get("institution", "Unknown"),
                "source_platform": ds_config.get("platform", "Unknown"),
                "source_file": ds_config.get("source_file", "Unknown"),
            }

            if var_name in coord_names:
                coords[var_name] = ("obs", values, attrs)
            else:
                data_vars[var_name] = ("obs", values, attrs)

        harmonized_ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Add global attributes
        harmonized_ds.attrs = {
            "source_dataset_id": dataset_id,
            "source_institution": ds_config.get("institution", "Unknown"),
            "source_platform": ds_config.get("platform", "Unknown"),
            "source_platform_description": ds_config.get("platform_description", ""),
            "source_region": ds_config.get("region", "Unknown"),
            "harmonization_schema": "canonical_v1",
            "pipeline_version": "1.0.0",
        }

        return harmonized_ds

    def _validate_against_schema(self, ds: xr.Dataset, dataset_id: str) -> None:
        """Validate harmonized dataset against canonical schema."""
        schema_vars = self.canonical_schema["variables"]
        issues = []

        for var_name, spec in schema_vars.items():
            if var_name not in ds and var_name not in ds.coords:
                issues.append(f"Missing variable: {var_name}")
                continue

            data = ds[var_name].values if var_name in ds else ds.coords[var_name].values

            # Check valid range
            if "valid_range" in spec and np.issubdtype(data.dtype, np.number):
                vmin, vmax = spec["valid_range"]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    actual_min = float(np.nanmin(valid_data))
                    actual_max = float(np.nanmax(valid_data))
                    if actual_min < vmin or actual_max > vmax:
                        issues.append(
                            f"{var_name}: range [{actual_min:.2f}, {actual_max:.2f}] "
                            f"outside expected [{vmin}, {vmax}]"
                        )

        if issues:
            print(f"    ⚠ Validation warnings for {dataset_id}:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"    ✓ Validation passed for {dataset_id}")
