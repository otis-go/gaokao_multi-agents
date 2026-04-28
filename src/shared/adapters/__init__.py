# src/shared/adapters/__init__.py
"""
Stage Adapters Package

This package provides adapter functions to bridge between Stage1 and Stage2.

The adapters ensure loose coupling between stages by:
1. Extracting only needed fields from Stage1 output
2. Validating required fields
3. Providing structured reasons for skip scenarios
"""

from src.shared.adapters.stage1_to_stage2 import (
    build_stage2_record,
    validate_stage2_record,
    Stage2RecordBuildResult,
)

__all__ = [
    "build_stage2_record",
    "validate_stage2_record",
    "Stage2RecordBuildResult",
]
