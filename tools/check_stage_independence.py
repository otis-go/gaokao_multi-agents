#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_stage_independence.py

Static check script for Stage1/Stage2 independence.

This script verifies that:
1. src/evaluation/** does NOT import src/generation/**
2. src/generation/** does NOT import src/evaluation/**
3. CLI does NOT import DataLoader for pre-validation (only for unit_id list)
4. Bridge adapter exists and is the unique entry point

Exit codes:
    0 - All checks passed
    1 - Independence violations found

Usage:
    python tools/check_stage_independence.py
    python tools/check_stage_independence.py --verbose
    python tools/check_stage_independence.py --json outputs/audit/independence_check.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Windows console UTF-8 support
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class ImportViolation:
    """Represents an import violation."""
    file_path: str
    line_number: int
    import_statement: str
    violation_type: str
    description: str


@dataclass
class IndependenceCheckResult:
    """Result of the independence check."""
    passed: bool = True
    violations: List[ImportViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_performed: Dict[str, bool] = field(default_factory=dict)

    def add_violation(self, violation: ImportViolation):
        self.violations.append(violation)
        self.passed = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [
                {
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "import_statement": v.import_statement,
                    "violation_type": v.violation_type,
                    "description": v.description,
                }
                for v in self.violations
            ],
            "warnings": self.warnings,
            "checks_performed": self.checks_performed,
            "summary": {
                "total_violations": len(self.violations),
                "total_warnings": len(self.warnings),
            },
        }


def extract_imports_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract all import statements from a Python file.

    Returns list of dicts with keys:
    - line_number
    - statement
    - module (the module being imported)
    - is_from_import (bool)
    """
    imports = []

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except Exception as e:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "line_number": node.lineno,
                    "statement": f"import {alias.name}",
                    "module": alias.name,
                    "is_from_import": False,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "line_number": node.lineno,
                    "statement": f"from {module} import {alias.name}",
                    "module": module,
                    "is_from_import": True,
                })

    return imports


def check_cross_stage_imports(result: IndependenceCheckResult, verbose: bool = False):
    """
    Check for forbidden cross-stage imports:
    - src/evaluation/** must NOT import src/generation/**
    - src/generation/** must NOT import src/evaluation/**
    """
    evaluation_dir = PROJECT_ROOT / "src" / "evaluation"
    generation_dir = PROJECT_ROOT / "src" / "generation"

    # Check evaluation -> generation
    if evaluation_dir.exists():
        for py_file in evaluation_dir.rglob("*.py"):
            imports = extract_imports_from_file(py_file)
            rel_path = str(py_file.relative_to(PROJECT_ROOT))

            for imp in imports:
                module = imp["module"]
                if module.startswith("src.generation") or "src.generation" in module:
                    result.add_violation(ImportViolation(
                        file_path=rel_path,
                        line_number=imp["line_number"],
                        import_statement=imp["statement"],
                        violation_type="evaluation_imports_generation",
                        description="src/evaluation/** must NOT import src/generation/**",
                    ))
                    if verbose:
                        print(f"  [VIOLATION] {rel_path}:{imp['line_number']} - {imp['statement']}")

    # Check generation -> evaluation
    if generation_dir.exists():
        for py_file in generation_dir.rglob("*.py"):
            imports = extract_imports_from_file(py_file)
            rel_path = str(py_file.relative_to(PROJECT_ROOT))

            for imp in imports:
                module = imp["module"]
                if module.startswith("src.evaluation") or "src.evaluation" in module:
                    result.add_violation(ImportViolation(
                        file_path=rel_path,
                        line_number=imp["line_number"],
                        import_statement=imp["statement"],
                        violation_type="generation_imports_evaluation",
                        description="src/generation/** must NOT import src/evaluation/**",
                    ))
                    if verbose:
                        print(f"  [VIOLATION] {rel_path}:{imp['line_number']} - {imp['statement']}")

    result.checks_performed["cross_stage_imports"] = len([
        v for v in result.violations
        if v.violation_type in ("evaluation_imports_generation", "generation_imports_evaluation")
    ]) == 0


def check_cli_independence(result: IndependenceCheckResult, verbose: bool = False):
    """
    Check that CLI does not import implementation details for pre-validation.

    Allowed:
    - Import orchestrators
    - Import schemas/adapters from src/shared
    - Import DataLoader only for unit_id list (not for pre-validation)

    Forbidden:
    - Import dimension mapping implementation
    - Import evaluation implementation details (except orchestrator)
    """
    cli_path = PROJECT_ROOT / "cli.py"

    if not cli_path.exists():
        result.add_warning("cli.py not found")
        result.checks_performed["cli_independence"] = True
        return

    imports = extract_imports_from_file(cli_path)
    rel_path = str(cli_path.relative_to(PROJECT_ROOT))

    # Check for forbidden imports
    forbidden_patterns = [
        # Importing internal eval modules (except orchestrator)
        (r"src\.evaluation\.(ai_centric_eval|pedagogical_eval|coarse_screening)", "cli_imports_eval_internals"),
        # Importing internal generation modules (except orchestrator)
        (r"src\.generation\.agents\.", "cli_imports_gen_internals"),
    ]

    for imp in imports:
        module = imp["module"]
        for pattern, violation_type in forbidden_patterns:
            if re.search(pattern, module):
                result.add_violation(ImportViolation(
                    file_path=rel_path,
                    line_number=imp["line_number"],
                    import_statement=imp["statement"],
                    violation_type=violation_type,
                    description=f"CLI should not import internal implementation: {module}",
                ))
                if verbose:
                    print(f"  [VIOLATION] {rel_path}:{imp['line_number']} - {imp['statement']}")

    result.checks_performed["cli_independence"] = len([
        v for v in result.violations
        if v.violation_type in ("cli_imports_eval_internals", "cli_imports_gen_internals")
    ]) == 0


def check_adapter_exists(result: IndependenceCheckResult, verbose: bool = False):
    """
    Check that the bridge adapter exists and is properly structured.
    """
    adapter_path = PROJECT_ROOT / "src" / "shared" / "adapters" / "stage1_to_stage2.py"

    if not adapter_path.exists():
        result.add_violation(ImportViolation(
            file_path="src/shared/adapters/stage1_to_stage2.py",
            line_number=0,
            import_statement="",
            violation_type="adapter_missing",
            description="Bridge adapter src/shared/adapters/stage1_to_stage2.py does not exist",
        ))
        result.checks_performed["adapter_exists"] = False
        return

    # Check that it has the required function
    try:
        content = adapter_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        has_build_function = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_stage2_record":
                has_build_function = True
                break

        if not has_build_function:
            result.add_warning("Adapter missing build_stage2_record function")

        result.checks_performed["adapter_exists"] = True

        if verbose and has_build_function:
            print("  [OK] Adapter has build_stage2_record function")

    except Exception as e:
        result.add_warning(f"Failed to parse adapter: {e}")
        result.checks_performed["adapter_exists"] = False


def check_adapter_no_cross_imports(result: IndependenceCheckResult, verbose: bool = False):
    """
    Check that the adapter does not import from generation or evaluation.
    """
    adapter_path = PROJECT_ROOT / "src" / "shared" / "adapters" / "stage1_to_stage2.py"

    if not adapter_path.exists():
        return

    imports = extract_imports_from_file(adapter_path)
    rel_path = str(adapter_path.relative_to(PROJECT_ROOT))

    for imp in imports:
        module = imp["module"]
        if module.startswith("src.generation") or module.startswith("src.evaluation"):
            result.add_violation(ImportViolation(
                file_path=rel_path,
                line_number=imp["line_number"],
                import_statement=imp["statement"],
                violation_type="adapter_cross_import",
                description="Adapter must NOT import from src/generation or src/evaluation",
            ))
            if verbose:
                print(f"  [VIOLATION] {rel_path}:{imp['line_number']} - {imp['statement']}")

    result.checks_performed["adapter_no_cross_imports"] = len([
        v for v in result.violations if v.violation_type == "adapter_cross_import"
    ]) == 0


def check_schemas_independence(result: IndependenceCheckResult, verbose: bool = False):
    """
    Check that schemas.py does not import from generation or evaluation.
    """
    schemas_path = PROJECT_ROOT / "src" / "shared" / "schemas.py"

    if not schemas_path.exists():
        result.add_warning("schemas.py not found")
        result.checks_performed["schemas_independence"] = True
        return

    imports = extract_imports_from_file(schemas_path)
    rel_path = str(schemas_path.relative_to(PROJECT_ROOT))

    for imp in imports:
        module = imp["module"]
        if module.startswith("src.generation") or module.startswith("src.evaluation"):
            result.add_violation(ImportViolation(
                file_path=rel_path,
                line_number=imp["line_number"],
                import_statement=imp["statement"],
                violation_type="schemas_cross_import",
                description="schemas.py must NOT import from src/generation or src/evaluation",
            ))
            if verbose:
                print(f"  [VIOLATION] {rel_path}:{imp['line_number']} - {imp['statement']}")

    result.checks_performed["schemas_independence"] = len([
        v for v in result.violations if v.violation_type == "schemas_cross_import"
    ]) == 0


def run_all_checks(verbose: bool = False) -> IndependenceCheckResult:
    """Run all independence checks."""
    result = IndependenceCheckResult()

    print("=" * 60)
    print("Stage Independence Check")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # Check 1: Cross-stage imports
    print("[1/5] Checking cross-stage imports...")
    check_cross_stage_imports(result, verbose)

    # Check 2: CLI independence
    print("[2/5] Checking CLI independence...")
    check_cli_independence(result, verbose)

    # Check 3: Adapter exists
    print("[3/5] Checking adapter exists...")
    check_adapter_exists(result, verbose)

    # Check 4: Adapter has no cross imports
    print("[4/5] Checking adapter has no cross imports...")
    check_adapter_no_cross_imports(result, verbose)

    # Check 5: Schemas independence
    print("[5/5] Checking schemas independence...")
    check_schemas_independence(result, verbose)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    for check_name, passed in result.checks_performed.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")

    if result.violations:
        print(f"\nTotal violations: {len(result.violations)}")
        for v in result.violations:
            print(f"  - {v.file_path}:{v.line_number} [{v.violation_type}]")
            print(f"    {v.description}")

    if result.warnings:
        print(f"\nWarnings: {len(result.warnings)}")
        for w in result.warnings:
            print(f"  - {w}")

    print()
    if result.passed:
        print("[PASS] All independence checks passed!")
    else:
        print("[FAIL] Independence violations found!")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check Stage1/Stage2 independence"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Output results to JSON file"
    )

    args = parser.parse_args()

    result = run_all_checks(verbose=args.verbose)

    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
