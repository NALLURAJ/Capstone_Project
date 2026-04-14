#!/usr/bin/env python3
"""
Create Sample NetCDF Files for Testing
=======================================
Generates 3 synthetic NetCDF files that EXACTLY mimic the structure of the
real ERDDAP datasets discovered via inspect_netcdf.py.

Dataset 01 - Ocean Networks Canada (Endeavour South):
  - Variable: "Temperature"  Units: Kelvin
  - Dimension: obs (tabular)  Fixed location: 47.93N, 129.11W, depth 1966m
  - Transform needed: identity (already in Kelvin)

Dataset 02 - NOAA CoastWatch MODIS Aqua SST:
  - Variable: "sst"  Units: degree_F (Fahrenheit)
  - Dimensions: time x altitude(=1) x latitude x longitude   (4D grid!)
  - Transform needed: fahrenheit_to_kelvin + squeeze altitude + subset region

Dataset 03 - Ocean Observatories Initiative (OOI) Cabled Array:
  - Variable: "sea_water_temperature_01"  Units: degree_Celsius
  - Dimension: obs (tabular)  Uses "z" (altitude, positive up, all 0.0)
  - Transform needed: celsius_to_kelvin + altitude_to_depth
"""

import os
import numpy as np
import xarray as xr


def create_dataset_01_onc_endeavour(output_dir: str):
    """
    ONC Endeavour South - deep-sea hydrothermal vent sensor.
    Variable 'Temperature' already in Kelvin (unusual but real).
    Fixed point: 47.93N, 129.11W, 1966m depth.
    """
    np.random.seed(42)

    times = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-07-01"),
        np.timedelta64(10, "m")   # every 10 minutes
    )
    n_obs = len(times)

    # Hydrothermal vent: mostly cool (~278 K) with brief spikes to ~285 K
    temperature_k = 278.0 + 1.5 * np.sin(
        np.arange(n_obs) * 2 * np.pi / (6 * 24 * 6)
    ) + np.random.normal(0, 0.3, n_obs)

    ds = xr.Dataset(
        {
            "Temperature": ("obs", temperature_k.astype(np.float64), {
                "units": "K",
                "long_name": "Sea Water Temperature",
                "standard_name": "sea_water_temperature",
                "sensor": "Nortek Aquadopp",
            }),
            "timeseries_id": ("obs", np.full(n_obs, "Endeavour_South_001", dtype=object)),
        },
        coords={
            "time": ("obs", times),
            "latitude": ("obs", np.full(n_obs, 47.93)),
            "longitude": ("obs", np.full(n_obs, -129.11)),
            "depth": ("obs", np.full(n_obs, 1966.0)),
        },
        attrs={
            "institution": "Ocean Networks Canada",
            "title": "ONC Endeavour South Hydrothermal Vent Current Meter",
            "source": "In-situ fixed seafloor sensor",
            "Conventions": "CF-1.6",
            "featureType": "timeSeries",
        }
    )

    filepath = os.path.join(output_dir, "dataset_01_onc_endeavour.nc")
    ds.to_netcdf(filepath)
    print(f"  Created: {filepath} ({n_obs} observations, Temperature in Kelvin)")
    return filepath


def create_dataset_02_noaa_sst_gridded(output_dir: str):
    """
    NOAA CoastWatch MODIS Aqua SST - satellite-derived, gridded.
    Variable 'sst' in FAHRENHEIT (unusual for ocean data!).
    4D structure: (time, altitude=1, latitude, longitude).
    Subset of the real global file — NE Pacific region.
    """
    np.random.seed(123)

    # 8-day composite periods covering NE Pacific
    times = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-07-01"),
        np.timedelta64(8, "D")   # 8-day composite
    )

    # NE Pacific subset (matches dataset_mappings.yaml subset region)
    lats = np.arange(44.0, 52.025, 0.025)   # 0.025 deg resolution
    lons = np.arange(-132.0, -122.025, 0.025)

    nt = len(times)
    nlat = len(lats)
    nlon = len(lons)

    # SST in Fahrenheit: NE Pacific is roughly 44–64°F (7–18°C)
    # Values outside 35–95°F would be physically impossible; fill = -32767
    sst_f = np.zeros((nt, nlat, nlon), dtype=np.float32)
    for t in range(nt):
        base_f = 50.0 + 8.0 * np.sin(t * 2 * np.pi / nt)   # seasonal cycle
        sst_f[t, :, :] = base_f + np.random.normal(0, 1.5, (nlat, nlon)).astype(np.float32)

    # Introduce some NaN patches (clouds) to simulate real satellite gaps
    rng = np.random.default_rng(42)
    cloud_mask = rng.random((nt, nlat, nlon)) < 0.15   # 15% cloud cover
    sst_f[cloud_mask] = np.nan

    # The 4D array has an altitude dimension of size 1 (altitude = 0.0)
    sst_4d = sst_f[:, np.newaxis, :, :]   # shape: (time, altitude=1, lat, lon)

    ds = xr.Dataset(
        {
            "sst": (["time", "altitude", "latitude", "longitude"],
                    sst_4d.astype(np.float32), {
                "units": "degree_F",
                "long_name": "Sea Surface Temperature",
                "standard_name": "sea_surface_temperature",
                "source": "MODIS Aqua 8-day composite",
                "_FillValue": -32767.0,
            }),
        },
        coords={
            "time": times,
            "altitude": np.array([0.0]),   # single altitude level = sea surface
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "institution": "NOAA NMFS SWFSC ERD / CoastWatch",
            "title": "MODIS Aqua 8-day SST 0.025 degree",
            "source": "Satellite-derived",
            "Conventions": "CF-1.6",
            "comment": "8-day composite SST on a 0.025 degree grid, Fahrenheit units",
        }
    )

    filepath = os.path.join(output_dir, "dataset_02_noaa_sst_gridded.nc")
    ds.to_netcdf(filepath)
    print(f"  Created: {filepath} "
          f"({nt} times × 1 altitude × {nlat} lats × {nlon} lons, SST in °F)")
    return filepath


def create_dataset_03_ooi_vent(output_dir: str):
    """
    OOI Regional Cabled Array - seafloor temperature sensor.
    Variable 'sea_water_temperature_01' in Celsius.
    Uses 'z' (altitude, positive up) instead of 'depth' — all 0.0.
    Location: 45.93N, 129.98W (Axial Seamount area).
    """
    np.random.seed(99)

    times = np.arange(
        np.datetime64("2024-02-01"),
        np.datetime64("2024-08-01"),
        np.timedelta64(1, "h")
    )
    n_obs = len(times)

    # Hydrothermal vent: ~3-5°C background with episodic heating events
    temp_c = 3.5 + 0.8 * np.sin(
        np.arange(n_obs) * 2 * np.pi / (30 * 24)
    ) + np.random.normal(0, 0.2, n_obs)

    # Also create the other temperature sensors that OOI files include
    # (these are listed in drop_variables in the config)
    temp_c_02 = temp_c + np.random.normal(0, 0.05, n_obs)
    temp_c_03 = temp_c + np.random.normal(0, 0.08, n_obs)

    ds = xr.Dataset(
        {
            "sea_water_temperature_01": ("obs", temp_c.astype(np.float64), {
                "units": "degree_Celsius",
                "long_name": "Sea Water Temperature",
                "standard_name": "sea_water_temperature",
                "sensor_serial_number": "01",
            }),
            "sea_water_temperature_01_qc_agg": ("obs", np.ones(n_obs, dtype=np.int8), {
                "long_name": "Sea Water Temperature QC Aggregate",
            }),
            "sea_water_temperature_01_qc_tests": ("obs", np.ones(n_obs, dtype=np.int8), {
                "long_name": "Sea Water Temperature QC Tests",
            }),
            "sea_water_temperature_02": ("obs", temp_c_02.astype(np.float64), {
                "units": "degree_Celsius",
                "long_name": "Sea Water Temperature (sensor 2)",
            }),
            "sea_water_temperature_02_qc_agg": ("obs", np.ones(n_obs, dtype=np.int8)),
            "sea_water_temperature_02_qc_tests": ("obs", np.ones(n_obs, dtype=np.int8)),
            "sea_water_temperature_03": ("obs", temp_c_03.astype(np.float64), {
                "units": "degree_Celsius",
                "long_name": "Sea Water Temperature (sensor 3)",
            }),
            "sea_water_temperature_03_qc_agg": ("obs", np.ones(n_obs, dtype=np.int8)),
            "sea_water_temperature_03_qc_tests": ("obs", np.ones(n_obs, dtype=np.int8)),
        },
        coords={
            "time": ("obs", times),
            "latitude": ("obs", np.full(n_obs, 45.93)),
            "longitude": ("obs", np.full(n_obs, -129.98)),
            "z": ("obs", np.zeros(n_obs)),   # altitude (positive up), all 0.0
        },
        attrs={
            "institution": "Ocean Observatories Initiative (OOI)",
            "title": "Regional Cabled Array MJ03C Temperature",
            "source": "Cabled seafloor sensor",
            "Conventions": "CF-1.6",
            "featureType": "timeSeries",
            "platform": "MJ03C",
        }
    )

    filepath = os.path.join(output_dir, "dataset_03_ooi_vent.nc")
    ds.to_netcdf(filepath)
    print(f"  Created: {filepath} ({n_obs} observations, temp in °C, z=altitude)")
    return filepath


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "source_data"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Creating sample NetCDF datasets...")
    print("(These now exactly mimic the real ERDDAP dataset structures)\n")

    create_dataset_01_onc_endeavour(output_dir)
    create_dataset_02_noaa_sst_gridded(output_dir)
    create_dataset_03_ooi_vent(output_dir)

    print("\n✓ All 3 sample datasets created in source_data/")
    print("  These mimic:")
    print("    01: ONC Endeavour  → Temperature in Kelvin   (identity transform)")
    print("    02: NOAA CoastWatch → sst in Fahrenheit (4D) (fahrenheit_to_kelvin)")
    print("    03: OOI Vent        → sea_water_temperature_01 in Celsius + z altitude")
    print("\n  Now run: python run_pipeline.py")


if __name__ == "__main__":
    main()
