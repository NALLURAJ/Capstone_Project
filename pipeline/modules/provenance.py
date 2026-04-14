"""
Provenance & Lineage Module
============================
Tracks the complete lineage of every data point from source to output.
Supports the ability to answer: "Where did this value come from?"

Addresses Constraint #3: Traceability through aggregation
Addresses Constraint #7: Data provenance and sovereignty
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


class ProvenanceRecord:
    """
    A provenance record for a single dataset as it moves through the pipeline.
    Captures: who produced it, where it came from, what was done to it.
    Follows W3C PROV-inspired model (Entity, Activity, Agent).
    """

    def __init__(self, dataset_id: str, source_file: str,
                 institution: str, platform: str):
        self.record_id = str(uuid.uuid4())
        self.dataset_id = dataset_id
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Entity: the data itself
        self.entity = {
            "dataset_id": dataset_id,
            "source_file": source_file,
            "institution": institution,
            "platform": platform,
            "ownership": institution,  # Data sovereignty
        }

        # Activities: transformations applied
        self.activities: List[Dict] = []

        # Derivation chain: links derived data to source
        self.derivations: List[Dict] = []

    def log_activity(self, activity_name: str, description: str,
                     input_variables: List[str],
                     output_variables: List[str],
                     transform_name: str,
                     parameters: Dict = None) -> str:
        """Log a transformation activity."""
        activity_id = str(uuid.uuid4())
        activity = {
            "activity_id": activity_id,
            "name": activity_name,
            "description": description,
            "input_variables": input_variables,
            "output_variables": output_variables,
            "transform_applied": transform_name,
            "parameters": parameters or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.activities.append(activity)
        return activity_id

    def log_derivation(self, source_variable: str, derived_variable: str,
                       activity_id: str, source_dataset: str = None):
        """Record that a derived variable came from a source variable."""
        self.derivations.append({
            "source_variable": source_variable,
            "source_dataset": source_dataset or self.dataset_id,
            "derived_variable": derived_variable,
            "activity_id": activity_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "entity": self.entity,
            "activities": self.activities,
            "derivations": self.derivations,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class LineageTracker:
    """
    Central lineage tracker that aggregates provenance from all datasets.
    Supports queries like: "trace this variable back to its source."
    """

    def __init__(self):
        self.records: Dict[str, ProvenanceRecord] = {}
        self.global_log: List[Dict] = []

    def register_dataset(self, dataset_id: str, source_file: str,
                         institution: str, platform: str) -> ProvenanceRecord:
        """Create and register a provenance record for a dataset."""
        record = ProvenanceRecord(
            dataset_id=dataset_id,
            source_file=source_file,
            institution=institution,
            platform=platform
        )
        self.records[dataset_id] = record
        self._log_global(f"Registered dataset: {dataset_id} from {source_file}")
        return record

    def trace_variable(self, variable_name: str) -> List[Dict]:
        """
        Trace a harmonized variable back to all source datasets and
        transformations that contributed to it.

        Returns a list of lineage paths, one per source dataset.
        """
        lineage_paths = []
        for dataset_id, record in self.records.items():
            for derivation in record.derivations:
                if derivation["derived_variable"] == variable_name:
                    # Find the activity that produced this
                    activity = next(
                        (a for a in record.activities
                         if a["activity_id"] == derivation["activity_id"]),
                        None
                    )
                    lineage_paths.append({
                        "source_dataset": dataset_id,
                        "source_file": record.entity["source_file"],
                        "source_variable": derivation["source_variable"],
                        "institution": record.entity["institution"],
                        "platform": record.entity["platform"],
                        "transform_applied": activity["transform_applied"] if activity else "unknown",
                        "description": activity["description"] if activity else "",
                    })
        return lineage_paths

    def get_dataset_lineage(self, dataset_id: str) -> Optional[Dict]:
        """Get complete lineage for a specific dataset."""
        if dataset_id in self.records:
            return self.records[dataset_id].to_dict()
        return None

    def get_full_report(self) -> Dict:
        """Generate a complete lineage report for all datasets."""
        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "total_datasets": len(self.records),
            "datasets": {
                did: rec.to_dict() for did, rec in self.records.items()
            },
            "global_log": self.global_log,
        }

    def save_report(self, filepath: str) -> None:
        """Save the full lineage report to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.get_full_report(), f, indent=2)

    def _log_global(self, message: str):
        self.global_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
        })
