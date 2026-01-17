#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tools/self_check.py
# Runtime call trace and dead code detection tool (stdlib only)

"""
【工具说明】
用于生成运行时调用链跟踪和死代码候选列表。

【功能】
1. 使用 sys.setprofile 记录函数调用（只记录 cli.py 和 src/ 下的函数）
2. 输出 JSONL 到 outputs/<exp_id>/audit/runtime_call_trace.jsonl
3. 静态扫描：用 ast 遍历 src/ 和 cli.py 提取所有 def/class
4. 对比运行时命中集合，输出 dead_code_candidates.json
5. 【2025-12 新增】记录调用边（caller -> callee）输出 call_graph_edges.json

【使用方式】
1. 作为模块导入，在 cli.py 中启用 trace：
   from tools.self_check import enable_trace, finalize_trace
   enable_trace(output_dir)
   # ... run pipeline ...
   summary = finalize_trace()

2. 独立运行静态扫描：
   python tools/self_check.py --static-only --output-dir outputs/<exp_id>/audit

3. 检查环境变量 SELF_CHECK_TRACE_ENABLED=1 自动启用（供 suite 使用）
"""

import ast
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

# Global state for tracing
_trace_enabled = False
_trace_records: List[Dict[str, Any]] = []
_trace_hit_set: Set[str] = set()
_trace_output_dir: Optional[Path] = None
_trace_start_time: float = 0.0
_project_root: Path = Path(__file__).resolve().parent.parent

# 【2025-12 新增】调用边跟踪
_trace_edges: Set[Tuple[str, str]] = set()  # (caller_key, callee_key)
_trace_call_stack: Dict[int, List[str]] = {}  # thread_id -> call stack


def _is_project_file(filename: str) -> bool:
    """Check if the file is within the project (cli.py or src/)"""
    if not filename:
        return False
    try:
        fpath = Path(filename).resolve()
        # Check if it's cli.py
        if fpath.name == "cli.py" and fpath.parent == _project_root:
            return True
        # Check if it's under src/
        src_dir = _project_root / "src"
        try:
            fpath.relative_to(src_dir)
            return True
        except ValueError:
            return False
    except Exception:
        return False


def _get_module_key(filename: str, func_name: str) -> str:
    """Get module:func key from filename and function name"""
    try:
        fpath = Path(filename).resolve()
        if fpath.name == "cli.py":
            module = "cli"
        else:
            rel = fpath.relative_to(_project_root / "src")
            module = "src." + str(rel.with_suffix("")).replace(os.sep, ".")
    except Exception:
        module = filename
    return f"{module}:{func_name}"


def _trace_calls(frame, event: str, arg):
    """Profile function to trace calls and returns"""
    global _trace_records, _trace_hit_set, _trace_edges, _trace_call_stack

    if event not in ("call", "return"):
        return _trace_calls

    filename = frame.f_code.co_filename
    if not _is_project_file(filename):
        return _trace_calls

    func_name = frame.f_code.co_name
    lineno = frame.f_lineno
    thread_id = threading.get_ident()

    # Build module path relative to project
    try:
        fpath = Path(filename).resolve()
        if fpath.name == "cli.py":
            module = "cli"
        else:
            rel = fpath.relative_to(_project_root / "src")
            module = "src." + str(rel.with_suffix("")).replace(os.sep, ".")
    except Exception:
        module = filename

    # Record the call
    ts = time.time() - _trace_start_time
    record = {
        "ts": round(ts, 4),
        "module": module,
        "func": func_name,
        "lineno": lineno,
        "event": event,
        "thread": thread_id,
    }
    _trace_records.append(record)

    # Track hit set for dead code analysis
    hit_key = f"{module}:{func_name}"
    _trace_hit_set.add(hit_key)

    # 【2025-12 新增】记录调用边
    if thread_id not in _trace_call_stack:
        _trace_call_stack[thread_id] = []

    if event == "call":
        # 记录 caller -> callee 边
        stack = _trace_call_stack[thread_id]
        if stack:
            caller_key = stack[-1]
            callee_key = hit_key
            _trace_edges.add((caller_key, callee_key))
        # 压栈
        stack.append(hit_key)
    elif event == "return":
        # 出栈
        stack = _trace_call_stack[thread_id]
        if stack:
            stack.pop()

    return _trace_calls


def enable_trace(output_dir: str) -> None:
    """
    Enable runtime call tracing.

    Args:
        output_dir: Output directory for trace files
    """
    global _trace_enabled, _trace_records, _trace_hit_set, _trace_output_dir
    global _trace_start_time, _trace_edges, _trace_call_stack

    _trace_enabled = True
    _trace_records = []
    _trace_hit_set = set()
    _trace_edges = set()
    _trace_call_stack = {}
    _trace_output_dir = Path(output_dir)
    _trace_start_time = time.time()

    # Create audit directory
    audit_dir = _trace_output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    sys.setprofile(_trace_calls)
    print(f"[self_check] Runtime tracing enabled, output_dir={output_dir}")


def disable_trace() -> None:
    """Disable runtime call tracing"""
    global _trace_enabled
    sys.setprofile(None)
    _trace_enabled = False


def get_trace_hit_set() -> Set[str]:
    """Get the current trace hit set (for merging across scenarios)"""
    return _trace_hit_set.copy()


def get_trace_edges() -> Set[Tuple[str, str]]:
    """Get the current trace edges (for merging across scenarios)"""
    return _trace_edges.copy()


def finalize_trace() -> Dict[str, Any]:
    """
    Finalize tracing: write call trace and generate dead code candidates.
    Returns summary dict with file paths.
    """
    global _trace_records, _trace_hit_set, _trace_output_dir, _trace_edges

    disable_trace()

    if _trace_output_dir is None:
        return {"error": "No output directory configured"}

    audit_dir = _trace_output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Write runtime call trace as JSONL
    trace_file = audit_dir / "runtime_call_trace.jsonl"
    with open(trace_file, "w", encoding="utf-8") as f:
        for record in _trace_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[self_check] Runtime call trace written: {trace_file} ({len(_trace_records)} records)")

    # 【2025-12 新增】写入调用边
    edges_file = audit_dir / "call_graph_edges.json"
    edges_list = [{"caller": c, "callee": e} for c, e in sorted(_trace_edges)]
    with open(edges_file, "w", encoding="utf-8") as f:
        json.dump(edges_list, f, ensure_ascii=False, indent=2)
    print(f"[self_check] Call graph edges written: {edges_file} ({len(edges_list)} edges)")

    # Generate Mermaid diagram
    mermaid_file = audit_dir / "call_graph.mermaid"
    mermaid_content = generate_mermaid_graph(_trace_edges)
    with open(mermaid_file, "w", encoding="utf-8") as f:
        f.write(mermaid_content)
    print(f"[self_check] Mermaid diagram written: {mermaid_file}")

    # Static scan for all definitions
    all_definitions = static_scan_definitions()

    # Compute dead code candidates
    dead_candidates = []
    for defn in all_definitions:
        hit_key = f"{defn['module']}:{defn['name']}"
        if hit_key not in _trace_hit_set:
            dead_candidates.append({
                "module": defn["module"],
                "name": defn["name"],
                "type": defn["type"],
                "file": defn["file"],
                "lineno": defn["lineno"],
                "category": _categorize_dead_code(defn),
            })

    # Write dead code candidates
    dead_file = audit_dir / "dead_code_candidates.json"
    with open(dead_file, "w", encoding="utf-8") as f:
        json.dump(dead_candidates, f, ensure_ascii=False, indent=2)
    print(f"[self_check] Dead code candidates written: {dead_file} ({len(dead_candidates)} candidates)")

    # Write hit set for merging
    hit_set_file = audit_dir / "hit_set.json"
    with open(hit_set_file, "w", encoding="utf-8") as f:
        json.dump(sorted(_trace_hit_set), f, ensure_ascii=False, indent=2)

    return {
        "trace_records": len(_trace_records),
        "hit_functions": len(_trace_hit_set),
        "total_definitions": len(all_definitions),
        "dead_candidates": len(dead_candidates),
        "call_edges": len(_trace_edges),
        "trace_file": str(trace_file),
        "dead_file": str(dead_file),
        "edges_file": str(edges_file),
        "mermaid_file": str(mermaid_file),
        "hit_set_file": str(hit_set_file),
    }


def generate_mermaid_graph(edges: Set[Tuple[str, str]], max_edges: int = 200) -> str:
    """
    Generate a Mermaid flowchart from call edges.

    Args:
        edges: Set of (caller, callee) tuples
        max_edges: Maximum edges to include (to avoid huge diagrams)

    Returns:
        Mermaid diagram as string
    """
    lines = ["flowchart TD"]

    # Build node set and sanitize names
    nodes: Set[str] = set()
    for c, e in edges:
        nodes.add(c)
        nodes.add(e)

    # Create node ID mapping (sanitize for Mermaid)
    node_ids: Dict[str, str] = {}
    for i, node in enumerate(sorted(nodes)):
        safe_id = f"N{i}"
        node_ids[node] = safe_id

    # Add node definitions with labels
    for node, safe_id in sorted(node_ids.items(), key=lambda x: x[1]):
        # Truncate long labels
        label = node if len(node) <= 40 else node[:37] + "..."
        label = label.replace('"', "'")
        lines.append(f'    {safe_id}["{label}"]')

    # Add edges (limit count)
    edge_list = sorted(edges)[:max_edges]
    for caller, callee in edge_list:
        caller_id = node_ids.get(caller, "?")
        callee_id = node_ids.get(callee, "?")
        lines.append(f"    {caller_id} --> {callee_id}")

    if len(edges) > max_edges:
        lines.append(f"    %% Note: Truncated to {max_edges} edges (total: {len(edges)})")

    return "\n".join(lines)


def _categorize_dead_code(defn: Dict[str, Any]) -> str:
    """Categorize dead code candidate"""
    name = defn["name"]
    module = defn["module"]

    # Test-related
    if "test" in name.lower() or "test" in module.lower():
        return "possibly_test"

    # Private/internal
    if name.startswith("_") and not name.startswith("__"):
        return "private_helper"

    # Dunder methods
    if name.startswith("__") and name.endswith("__"):
        return "dunder_method"

    # Compatibility/deprecated
    if "compat" in name.lower() or "deprecated" in name.lower() or "legacy" in name.lower():
        return "compatibility_layer"

    # Factory functions
    if name.startswith("create_") or name.startswith("build_"):
        return "factory_function"

    return "unknown"


def static_scan_definitions(root_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Scan all Python files in cli.py and src/ for function and class definitions.
    Returns list of definition dicts with module, name, type, file, lineno.
    """
    if root_dir is None:
        root_dir = _project_root

    definitions: List[Dict[str, Any]] = []

    # Scan cli.py
    cli_file = root_dir / "cli.py"
    if cli_file.exists():
        definitions.extend(_scan_file(cli_file, "cli"))

    # Scan src/ directory
    src_dir = root_dir / "src"
    if src_dir.exists():
        for py_file in src_dir.rglob("*.py"):
            try:
                rel = py_file.relative_to(src_dir)
                module = "src." + str(rel.with_suffix("")).replace(os.sep, ".")
                definitions.extend(_scan_file(py_file, module))
            except Exception as e:
                print(f"[self_check] Warning: Failed to scan {py_file}: {e}")

    return definitions


def _scan_file(filepath: Path, module: str) -> List[Dict[str, Any]]:
    """Scan a single Python file for definitions"""
    definitions: List[Dict[str, Any]] = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                definitions.append({
                    "module": module,
                    "name": node.name,
                    "type": "function",
                    "file": str(filepath),
                    "lineno": node.lineno,
                })
            elif isinstance(node, ast.AsyncFunctionDef):
                definitions.append({
                    "module": module,
                    "name": node.name,
                    "type": "async_function",
                    "file": str(filepath),
                    "lineno": node.lineno,
                })
            elif isinstance(node, ast.ClassDef):
                definitions.append({
                    "module": module,
                    "name": node.name,
                    "type": "class",
                    "file": str(filepath),
                    "lineno": node.lineno,
                })
    except Exception as e:
        print(f"[self_check] Warning: Failed to parse {filepath}: {e}")

    return definitions


# ============================================================================
# 【2025-12 新增】静态引用扫描
# ============================================================================

def scan_static_references(
    candidates: List[Dict[str, Any]],
    root_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Scan for static references to dead code candidates.

    For each candidate, searches the codebase for:
    - AST Name nodes matching the symbol name
    - AST Attribute nodes for method calls
    - Text-based fallback search

    Args:
        candidates: List of dead code candidate dicts
        root_dir: Project root directory

    Returns:
        Updated candidates with static_refs_count and static_ref_examples
    """
    if root_dir is None:
        root_dir = _project_root

    # Build name -> candidate mapping
    name_to_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        name = c["name"]
        if name not in name_to_candidates:
            name_to_candidates[name] = []
        name_to_candidates[name].append(c)

    # Initialize reference counts
    for c in candidates:
        c["static_refs_count"] = 0
        c["static_ref_examples"] = []

    # Collect all Python files
    py_files: List[Path] = []
    cli_file = root_dir / "cli.py"
    if cli_file.exists():
        py_files.append(cli_file)
    src_dir = root_dir / "src"
    if src_dir.exists():
        py_files.extend(src_dir.rglob("*.py"))

    # Scan each file for references
    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(py_file))

            # Walk AST for Name and Attribute nodes
            for node in ast.walk(tree):
                name = None
                lineno = 0

                if isinstance(node, ast.Name):
                    name = node.id
                    lineno = node.lineno
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                    lineno = node.lineno

                if name and name in name_to_candidates:
                    for c in name_to_candidates[name]:
                        # Don't count self-reference (definition line)
                        if str(py_file) == c["file"] and lineno == c["lineno"]:
                            continue
                        c["static_refs_count"] += 1
                        if len(c["static_ref_examples"]) < 3:
                            c["static_ref_examples"].append({
                                "file": str(py_file),
                                "lineno": lineno,
                            })
        except Exception as e:
            print(f"[self_check] Warning: Failed to scan refs in {py_file}: {e}")

    return candidates


def check_whitelist(
    candidates: List[Dict[str, Any]],
    root_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Check candidates against whitelist criteria.

    Whitelist includes:
    - Symbols exported via __init__.py
    - Names in __all__
    - Names used in reflection calls (getattr/locals/globals)
    - Plugin registry patterns

    Args:
        candidates: List of dead code candidate dicts
        root_dir: Project root directory

    Returns:
        Updated candidates with whitelist_reason field
    """
    if root_dir is None:
        root_dir = _project_root

    # Collect __init__.py exports
    init_exports: Set[str] = set()
    all_exports: Set[str] = set()
    reflection_names: Set[str] = set()
    registry_names: Set[str] = set()

    # Scan __init__.py files
    src_dir = root_dir / "src"
    if src_dir.exists():
        for init_file in src_dir.rglob("__init__.py"):
            try:
                with open(init_file, "r", encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source, filename=str(init_file))

                # Check for __all__ definition
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "__all__":
                                if isinstance(node.value, (ast.List, ast.Tuple)):
                                    for elt in node.value.elts:
                                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                            all_exports.add(elt.value)

                # Check for direct imports/exports
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            name = alias.asname or alias.name
                            init_exports.add(name)
            except Exception:
                pass

    # Scan for reflection patterns
    py_files: List[Path] = []
    cli_file = root_dir / "cli.py"
    if cli_file.exists():
        py_files.append(cli_file)
    if src_dir.exists():
        py_files.extend(src_dir.rglob("*.py"))

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()

            # Text-based search for reflection patterns
            # getattr(x, "name"), locals(), globals()
            for match in re.finditer(r'getattr\s*\([^,]+,\s*["\'](\w+)["\']', source):
                reflection_names.add(match.group(1))

            # Registry patterns
            for match in re.finditer(r'register\s*\(["\'](\w+)["\']', source):
                registry_names.add(match.group(1))
            for match in re.finditer(r'registry\s*\[["\'](\w+)["\']\]', source):
                registry_names.add(match.group(1))

        except Exception:
            pass

    # Apply whitelist to candidates
    for c in candidates:
        name = c["name"]
        reasons = []

        if name in init_exports:
            reasons.append("exported_in_init")
        if name in all_exports:
            reasons.append("in___all__")
        if name in reflection_names:
            reasons.append("reflection_target")
        if name in registry_names:
            reasons.append("registry_pattern")

        # Dunder methods are always whitelisted
        if name.startswith("__") and name.endswith("__"):
            reasons.append("dunder_method")

        c["whitelist_reasons"] = reasons
        c["is_whitelisted"] = len(reasons) > 0

    return candidates


def compute_deletion_eligibility(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute deletion eligibility for each candidate.

    Deletion threshold:
    - runtime not-hit (implicit since they're candidates)
    - static_refs_count == 0
    - is_whitelisted == False

    Args:
        candidates: List of dead code candidate dicts

    Returns:
        Updated candidates with can_delete and delete_evidence fields
    """
    for c in candidates:
        refs_count = c.get("static_refs_count", 0)
        is_whitelisted = c.get("is_whitelisted", False)

        can_delete = (refs_count == 0) and (not is_whitelisted)

        evidence = []
        evidence.append("runtime_not_hit=True")
        evidence.append(f"static_refs_count={refs_count}")
        evidence.append(f"is_whitelisted={is_whitelisted}")
        if is_whitelisted:
            evidence.append(f"whitelist_reasons={c.get('whitelist_reasons', [])}")

        c["can_delete"] = can_delete
        c["delete_evidence"] = evidence

    return candidates


# ============================================================================
# 【2025-12 新增】合并多场景 trace
# ============================================================================

def merge_traces(
    trace_dirs: List[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Merge traces from multiple scenario runs.

    Args:
        trace_dirs: List of directories containing audit/ subdirs
        output_dir: Output directory for merged results

    Returns:
        Summary dict with merged results
    """
    merged_hit_set: Set[str] = set()
    merged_edges: Set[Tuple[str, str]] = set()
    total_records = 0

    for trace_dir in trace_dirs:
        audit_dir = trace_dir / "audit"

        # Load hit set
        hit_set_file = audit_dir / "hit_set.json"
        if hit_set_file.exists():
            with open(hit_set_file, "r", encoding="utf-8") as f:
                hits = json.load(f)
                merged_hit_set.update(hits)

        # Load edges
        edges_file = audit_dir / "call_graph_edges.json"
        if edges_file.exists():
            with open(edges_file, "r", encoding="utf-8") as f:
                edges = json.load(f)
                for e in edges:
                    merged_edges.add((e["caller"], e["callee"]))

        # Count records
        trace_file = audit_dir / "runtime_call_trace.jsonl"
        if trace_file.exists():
            with open(trace_file, "r", encoding="utf-8") as f:
                total_records += sum(1 for _ in f)

    # Create merged output directory
    merged_audit = output_dir / "audit"
    merged_audit.mkdir(parents=True, exist_ok=True)

    # Write merged hit set
    merged_hit_file = merged_audit / "merged_hit_set.json"
    with open(merged_hit_file, "w", encoding="utf-8") as f:
        json.dump(sorted(merged_hit_set), f, ensure_ascii=False, indent=2)

    # Write merged edges
    merged_edges_file = merged_audit / "merged_call_graph_edges.json"
    edges_list = [{"caller": c, "callee": e} for c, e in sorted(merged_edges)]
    with open(merged_edges_file, "w", encoding="utf-8") as f:
        json.dump(edges_list, f, ensure_ascii=False, indent=2)

    # Generate merged Mermaid diagram
    merged_mermaid_file = merged_audit / "merged_call_graph.mermaid"
    mermaid_content = generate_mermaid_graph(merged_edges, max_edges=300)
    with open(merged_mermaid_file, "w", encoding="utf-8") as f:
        f.write(mermaid_content)

    # Static scan for all definitions
    all_definitions = static_scan_definitions()

    # Compute merged dead code candidates
    dead_candidates = []
    for defn in all_definitions:
        hit_key = f"{defn['module']}:{defn['name']}"
        if hit_key not in merged_hit_set:
            dead_candidates.append({
                "module": defn["module"],
                "name": defn["name"],
                "type": defn["type"],
                "file": defn["file"],
                "lineno": defn["lineno"],
                "category": _categorize_dead_code(defn),
            })

    # Enhance with static reference scan
    dead_candidates = scan_static_references(dead_candidates)

    # Check whitelist
    dead_candidates = check_whitelist(dead_candidates)

    # Compute deletion eligibility
    dead_candidates = compute_deletion_eligibility(dead_candidates)

    # Write merged dead code candidates
    merged_dead_file = merged_audit / "dead_code_candidates_merged.json"
    with open(merged_dead_file, "w", encoding="utf-8") as f:
        json.dump(dead_candidates, f, ensure_ascii=False, indent=2)

    return {
        "scenarios_merged": len(trace_dirs),
        "total_records": total_records,
        "merged_hit_functions": len(merged_hit_set),
        "merged_call_edges": len(merged_edges),
        "total_definitions": len(all_definitions),
        "dead_candidates": len(dead_candidates),
        "deletable_candidates": sum(1 for c in dead_candidates if c.get("can_delete")),
        "merged_hit_file": str(merged_hit_file),
        "merged_edges_file": str(merged_edges_file),
        "merged_mermaid_file": str(merged_mermaid_file),
        "merged_dead_file": str(merged_dead_file),
    }


def main():
    """Command line interface for static-only analysis"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-check tool for runtime tracing and dead code detection"
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Only run static scan (no runtime tracing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/audit",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: auto-detect)"
    )
    parser.add_argument(
        "--with-refs",
        action="store_true",
        help="Include static reference scan in output"
    )

    args = parser.parse_args()

    if args.project_root:
        global _project_root
        _project_root = Path(args.project_root).resolve()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.static_only:
        # Static scan only
        definitions = static_scan_definitions()

        # Categorize definitions
        categorized = []
        for defn in definitions:
            defn["category"] = _categorize_dead_code(defn)
            categorized.append(defn)

        # Write all definitions
        all_defs_file = output_dir / "all_definitions.json"
        with open(all_defs_file, "w", encoding="utf-8") as f:
            json.dump(categorized, f, ensure_ascii=False, indent=2)

        print(f"[self_check] Static scan complete:")
        print(f"  Total definitions: {len(definitions)}")
        print(f"  Output: {all_defs_file}")

        # Summary by type
        type_counts: Dict[str, int] = {}
        for d in definitions:
            t = d["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        print("  By type:")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c}")

        # Optional: static reference scan
        if args.with_refs:
            print("\n[self_check] Running static reference scan...")
            # Create pseudo-candidates (all definitions)
            candidates = []
            for defn in definitions:
                candidates.append({
                    "module": defn["module"],
                    "name": defn["name"],
                    "type": defn["type"],
                    "file": defn["file"],
                    "lineno": defn["lineno"],
                    "category": defn["category"],
                })

            candidates = scan_static_references(candidates)
            candidates = check_whitelist(candidates)

            refs_file = output_dir / "definitions_with_refs.json"
            with open(refs_file, "w", encoding="utf-8") as f:
                json.dump(candidates, f, ensure_ascii=False, indent=2)
            print(f"  Output with refs: {refs_file}")
    else:
        print("[self_check] For runtime tracing, import this module and use enable_trace()/finalize_trace()")
        print("             Or use --static-only for static analysis")


if __name__ == "__main__":
    main()
