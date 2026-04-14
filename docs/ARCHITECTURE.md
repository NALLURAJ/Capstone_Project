# Architecture Document: Traceable Harmonization of Ocean Data for ML Readiness

## 1. Overview

This system implements a data pipeline that ingests heterogeneous ocean observation datasets in NetCDF format and produces a harmonized Zarr store suitable for machine learning. The architecture prioritizes **traceability**, **source immutability**, **config-driven transformations**, and **data integrity** throughout.

```
Source NetCDF Files (Immutable)
        │
        ▼
┌─────────────────┐
│   INGESTOR       │  → SHA-256 hashing, manifest creation, provenance registration
│   (ingestor.py)  │
└────────┬────────┘
         │  Raw xarray Datasets (in-memory only)
         ▼
┌─────────────────┐
│  HARMONIZER      │  → Config-driven variable mapping, unit conversion, validation
│  (harmonizer.py) │  ← Reads from dataset_mappings.yaml (NOT hardcoded)
└────────┬────────┘
         │  Canonical-schema xarray Datasets
         ▼
┌─────────────────┐
│  ZARR WRITER     │  → Combines datasets, embeds provenance, writes Zarr + audit logs
│  (zarr_writer.py)│
└────────┬────────┘
         │
         ▼
   ML-Ready Zarr Store + Lineage Report + Integrity Records
```

## 2. Directory Structure

```
Capstone_Project/
├── source_data/                      # Raw NetCDF files (NEVER modified)
│   ├── dataset_01_barkley_canyon_ctd.nc
│   ├── dataset_02_noaa_sst_gridded.nc
│   └── dataset_03_baynes_sound_ctd.nc
├── pipeline/
│   ├── configs/
│   │   └── dataset_mappings.yaml     # THE config file (edit this, not code)
│   └── modules/
│       ├── ingestor.py               # Source file reading + hashing
│       ├── harmonizer.py             # Config-driven transformation engine
│       ├── transforms.py             # Explicit, named transform functions
│       ├── zarr_writer.py            # Zarr output with embedded provenance
│       ├── integrity.py              # SHA-256 hashing, hash chains
│       └── provenance.py             # W3C PROV-inspired lineage tracking
├── output/
│   ├── zarr_store/
│   │   ├── harmonized.zarr           # THE ML-ready output
│   │   ├── lineage_report.json
│   │   └── integrity_records.json
│   └── audit_logs/
│       ├── source_manifest.json      # SHA-256 hashes of all source files
│       ├── lineage_report.json       # Full provenance report
│       ├── integrity_records.json    # Hash chains per dataset
│       └── pipeline_run_log.json     # Timestamped execution log
├── docs/
│   └── ARCHITECTURE.md               # This document
├── run_pipeline.py                   # Main entry point
├── create_sample_data.py             # Generates synthetic test data
└── download_datasets.py              # Downloads real data from ERDDAP
```

## 3. How Each Constraint Is Addressed

### Constraint 1: Source Immutability

Source NetCDF files are treated as read-only inputs. The pipeline:
- Computes SHA-256 hashes of each source file **before** reading any data
- Stores hashes in `source_manifest.json`
- Loads data into memory with `xr.open_dataset()` + `.load()` — no write handles
- After pipeline completion, **re-verifies** source file hashes to confirm nothing changed
- If a hash mismatch is detected, the pipeline reports it as an integrity violation

### Constraint 2: Re-derivability

The pipeline can be re-run at any time to produce a different output:
- Source files are never consumed or altered
- All transformation logic is driven by `dataset_mappings.yaml`
- To produce a different harmonized output, edit the YAML config and re-run
- The YAML file is versioned alongside the code — different configs = different outputs from the same sources

### Constraint 3: Traceability Through Aggregation

When the Zarr store combines data from 3 different sources, every observation carries:
- `source_dataset_idx`: integer ID mapping back to the originating dataset
- `source_obs_index`: the observation's index within the original source file
- **Lineage report** (`lineage_report.json`): for each harmonized variable, traces back to:
  - Which source file it came from
  - What the original variable name was
  - Which institution produced it
  - What transform was applied
  - Exact timestamp of the transformation

The `LineageTracker.trace_variable("sea_water_temperature")` method returns all source paths contributing to any harmonized variable.

### Constraint 4: Explicit Transformations

Every transformation is:
- **Named**: registered in `transforms.py` (e.g., `celsius_to_kelvin`)
- **Documented**: each function has a docstring and returns a human-readable description
- **Referenced in config**: the YAML maps source variables to transforms by name
- **Inspectable**: `list_transforms()` returns all available transforms
- **No black boxes**: no hidden preprocessing steps; even grid-flattening is logged

### Constraint 5: ML Usability Without Domain Hardcoding

The Zarr store is designed for direct ML consumption:
- Standard variable names (`sea_water_temperature` in Kelvin)
- Single dimension (`obs`) — simple array indexing
- NumPy-compatible dtypes
- No domain-specific knowledge required to load and use

Meanwhile, domain scientists can audit via:
- Per-variable `attrs` showing `source_dataset`, `source_institution`, `source_platform`
- The lineage report for full transformation chain
- The config YAML showing all mapping decisions

### Constraint 6: Data Integrity and Security Awareness

The `IntegrityRecord` class implements a **hash chain**:
- Step 0: SHA-256 of the source file
- Each subsequent step: SHA-256 of the transformed data
- Chain hash = hash(previous_chain_hash + current_data_hash)
- This creates a **tamper-evident linked chain** — if any intermediate step is altered, the chain validation fails

The `verify_chain()` method checks the entire chain for consistency. This is conceptually similar to blockchain integrity verification.

### Constraint 7: Data Provenance and Sovereignty

The `ProvenanceRecord` follows a W3C PROV-inspired model:
- **Entity**: the dataset (source file, institution, ownership)
- **Activity**: each transformation (what was done, when, to which variables)
- **Derivation**: links derived variables back to their sources

The `institution` and `ownership` fields preserve data sovereignty — even after harmonization, each observation is attributable to its originating institution.

## 4. Answering the Guiding Question

> How would you design the system so that changing the interpretation of a variable does not require rewriting the ingestion code or modifying the original source data?

**Answer: Config-driven semantic mapping with a transform registry.**

The key architectural decision is the **complete separation of ingestion from interpretation**:

1. **Ingestion** (`ingestor.py`) reads source files byte-for-byte and doesn't know or care what the variables mean. It just hashes them and loads them into memory.

2. **Interpretation** lives entirely in `dataset_mappings.yaml`. Each source variable is mapped to a canonical target via a named transform. For example:
   ```yaml
   Temperature:
     target: "sea_water_temperature"
     source_units: "celsius"
     target_units: "kelvin"
     transform: "celsius_to_kelvin"
   ```

3. **Transforms** are registered functions in `transforms.py`. Adding a new unit conversion means adding one function and referencing it in the config.

To change an interpretation (e.g., switching from Kelvin to Fahrenheit output), you:
- Edit the YAML config: change `target_units` and `transform`
- Optionally add a new transform function
- Re-run the pipeline

No ingestion code changes. No source data modifications. The same source files produce a different harmonized output.

## 5. Trade-offs and Limitations

**Scalability**: The current prototype loads entire datasets into memory. For production-scale data (terabytes), this would need chunked/streaming processing with Dask.

**Schema evolution**: If the canonical schema changes (e.g., adding a new variable like `salinity`), existing Zarr stores aren't automatically updated. A versioning strategy for the canonical schema would be needed.

**Synthetic variables**: Adding `depth=0.0` for satellite data is a simplification. A more sophisticated approach might use the actual sensor specification or leave depth as NaN with metadata explaining why.

**Time alignment**: The current approach concatenates observations without temporal interpolation. Datasets at different sampling frequencies are combined as-is, which is appropriate for observation-oriented ML but would need modification for regular-grid ML inputs.

**Security**: The hash chain provides tamper evidence but not prevention. A production system might add digital signatures or integrate with a trusted timestamping service.

## 6. How to Extend

**Add a new dataset**: Add an entry to `dataset_mappings.yaml` under `datasets:` with the new file's variable mappings. No code changes needed.

**Add a new unit conversion**: Write a function in `transforms.py`, register it in `TRANSFORM_REGISTRY`, and reference it in the YAML config.

**Change the canonical schema**: Edit `canonical_schema` in the YAML. Update the `valid_range` and `units` as needed.

**Add a new output format**: Create a new writer module (like `zarr_writer.py`) for Parquet, CSV, or another format. The harmonized xarray Dataset is format-agnostic.
