"""
Integrity Module
================
Handles cryptographic hashing, tamper detection, and data integrity
verification throughout the pipeline.

Addresses Constraint #6: Data integrity and security awareness
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of an entire file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_data_hash(data_bytes: bytes) -> str:
    """Compute SHA-256 hash of arbitrary data (e.g., a numpy array's bytes)."""
    return hashlib.sha256(data_bytes).hexdigest()


def compute_array_hash(array) -> str:
    """Compute SHA-256 hash of a numpy/xarray array's raw bytes."""
    import numpy as np
    if hasattr(array, 'values'):
        # xarray DataArray
        raw = np.ascontiguousarray(array.values).tobytes()
    else:
        raw = np.ascontiguousarray(array).tobytes()
    return compute_data_hash(raw)


class IntegrityRecord:
    """
    A tamper-evident record that tracks the hash chain for a piece of data.
    Each transformation appends a new entry to the chain.
    """

    def __init__(self, source_file: str, source_hash: str):
        self.chain = []
        self.chain.append({
            "step": 0,
            "action": "source_ingestion",
            "source_file": source_file,
            "file_hash": source_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chain_hash": source_hash  # Initial chain hash = file hash
        })

    def add_step(self, action: str, description: str, data_hash: str,
                 metadata: dict = None) -> None:
        """
        Add a transformation step to the integrity chain.
        The chain_hash includes the previous chain_hash, creating a
        tamper-evident linked chain (similar to blockchain concept).
        """
        prev_chain_hash = self.chain[-1]["chain_hash"]
        # New chain hash = hash(previous_chain_hash + current_data_hash)
        combined = f"{prev_chain_hash}{data_hash}".encode()
        new_chain_hash = hashlib.sha256(combined).hexdigest()

        entry = {
            "step": len(self.chain),
            "action": action,
            "description": description,
            "data_hash": data_hash,
            "chain_hash": new_chain_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata
        self.chain.append(entry)

    def verify_chain(self) -> bool:
        """Verify the integrity of the hash chain (no tampering)."""
        for i in range(1, len(self.chain)):
            prev_hash = self.chain[i - 1]["chain_hash"]
            data_hash = self.chain[i]["data_hash"]
            expected = hashlib.sha256(
                f"{prev_hash}{data_hash}".encode()
            ).hexdigest()
            if self.chain[i]["chain_hash"] != expected:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "chain": self.chain,
            "is_valid": self.verify_chain()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SourceManifest:
    """
    Tracks all source files and their integrity hashes.
    This is the first thing created when source files are ingested.
    """

    def __init__(self):
        self.entries = {}

    def register_file(self, filepath: str) -> dict:
        """Register a source file and compute its hash."""
        filename = os.path.basename(filepath)
        file_hash = compute_file_sha256(filepath)
        entry = {
            "filepath": os.path.abspath(filepath),
            "filename": filename,
            "sha256": file_hash,
            "size_bytes": os.path.getsize(filepath),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self.entries[filename] = entry
        return entry

    def verify_file(self, filepath: str) -> bool:
        """Check if a file still matches its registered hash."""
        filename = os.path.basename(filepath)
        if filename not in self.entries:
            raise ValueError(f"File {filename} not in manifest")
        current_hash = compute_file_sha256(filepath)
        return current_hash == self.entries[filename]["sha256"]

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump({
                "manifest_created": datetime.now(timezone.utc).isoformat(),
                "files": self.entries
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SourceManifest":
        manifest = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
        manifest.entries = data.get("files", {})
        return manifest
