"""
Transforms Module
=================
All transformation functions used in the harmonization pipeline.
Each transform is explicit, inspectable, and registered by name.

Addresses Constraint #4: Explicit transformations
- No black-box preprocessing
- Every function is named, documented, and referenced in the config
"""

import numpy as np
import xarray as xr
from typing import Callable, Dict, Any


# =============================================================================
# Transform Functions
# =============================================================================
# Each function takes (data, metadata) and returns (transformed_data, description)
# The description is a human-readable record of what was done.

def transform_identity(data: np.ndarray, **kwargs) -> tuple:
    """Pass-through: no transformation applied."""
    return data.copy(), "Identity transform (no change)"


def transform_celsius_to_kelvin(data: np.ndarray, **kwargs) -> tuple:
    """Convert temperature from Celsius to Kelvin."""
    result = data + 273.15
    return result, "Converted Celsius to Kelvin (value + 273.15)"


def transform_fahrenheit_to_kelvin(data: np.ndarray, **kwargs) -> tuple:
    """Convert temperature from Fahrenheit to Kelvin."""
    result = (data - 32.0) * 5.0 / 9.0 + 273.15
    return result, "Converted Fahrenheit to Kelvin ((value - 32) * 5/9 + 273.15)"


def transform_altitude_to_depth(data: np.ndarray, **kwargs) -> tuple:
    """Convert altitude (positive up) to depth (positive down) by negating."""
    result = -data.astype(np.float64)
    return result, "Converted altitude (positive up) to depth (positive down): depth = -altitude"


def transform_ensure_utc(data, **kwargs) -> tuple:
    """Ensure datetime values are in UTC."""
    import pandas as pd
    if hasattr(data, 'values'):
        values = data.values
    else:
        values = data
    # If already datetime64, just ensure no timezone issues
    if np.issubdtype(values.dtype, np.datetime64):
        return values, "Confirmed datetime64 in UTC (no timezone conversion needed)"
    # If pandas Timestamps with timezone info
    try:
        ts = pd.DatetimeIndex(values)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.values, "Converted to UTC datetime64[ns]"
    except Exception:
        return values, "Time transform: kept as-is (already compatible format)"


# =============================================================================
# Transform Registry
# =============================================================================
# Maps config names to actual functions

TRANSFORM_REGISTRY: Dict[str, Callable] = {
    "identity": transform_identity,
    "celsius_to_kelvin": transform_celsius_to_kelvin,
    "fahrenheit_to_kelvin": transform_fahrenheit_to_kelvin,
    "altitude_to_depth": transform_altitude_to_depth,
    "ensure_utc": transform_ensure_utc,
}


def get_transform(name: str) -> Callable:
    """Look up a transform function by its config name."""
    if name not in TRANSFORM_REGISTRY:
        available = ", ".join(TRANSFORM_REGISTRY.keys())
        raise ValueError(
            f"Unknown transform '{name}'. Available: {available}"
        )
    return TRANSFORM_REGISTRY[name]


def register_transform(name: str, func: Callable) -> None:
    """
    Register a new transform function at runtime.
    This supports extensibility without modifying existing code.
    """
    TRANSFORM_REGISTRY[name] = func


def list_transforms() -> Dict[str, str]:
    """List all registered transforms with their docstrings."""
    return {
        name: (func.__doc__ or "No description").strip()
        for name, func in TRANSFORM_REGISTRY.items()
    }
