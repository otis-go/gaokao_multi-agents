# src/shared/llm_logger.py
"""
LLM Prompt and Response Logger

Features:
1. Record input prompts and output responses for each LLM call
2. Support multiple output formats (JSON, JSONL, TXT)
3. Save grouped by experiment/run
4. Configurable enable/disable
5. Support statistical analysis (token estimation, call count, etc.)

Usage:
    from src.shared.llm_logger import LLMLogger, get_global_logger

    # Global usage
    logger = get_global_logger()
    logger.log_call(
        model_name="GLM-4.5-Flash",
        messages=[{"role": "user", "content": "..."}],
        response="...",
        metadata={"agent": "Agent4", "unit_id": "1"}
    )

    # Save logs
    logger.save()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading


@dataclass
class LLMCallRecord:
    """Single LLM call record"""
    timestamp: str
    model_name: str
    messages: List[Dict[str, str]]
    response: str
    temperature: float = 1.0  # 2025-12-27 update: consistent with api_config.py
    max_tokens: int = 16384   # 2025-12-26 update: consistent with api_config.py
    latency_ms: Optional[float] = None
    prompt_tokens_est: Optional[int] = None
    completion_tokens_est: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "messages": self.messages,
            "response": self.response,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "latency_ms": self.latency_ms,
            "prompt_tokens_est": self.prompt_tokens_est,
            "completion_tokens_est": self.completion_tokens_est,
            "metadata": self.metadata,
            "error": self.error,
        }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough estimation of token count.
        Chinese approximately 1.5-2 chars/token, English approximately 4 chars/token.
        Using simple mixed estimation here.
        """
        if not text:
            return 0
        # Simple estimation: Chinese character count + English word count
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


@dataclass
class LLMLogger:
    """
    LLM Call Logger

    Features:
    - Thread-safe
    - Support multiple output formats
    - Automatically group by experiment
    """
    output_dir: Path = field(default_factory=lambda: Path("outputs/llm_logs"))
    experiment_id: Optional[str] = None
    enabled: bool = True
    format: str = "jsonl"  # "json", "jsonl", "txt"

    # Internal state
    _records: List[LLMCallRecord] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _call_count: int = field(default=0, init=False)
    _start_time: Optional[datetime] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize"""
        self.output_dir = Path(self.output_dir)
        if not self.experiment_id:
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start_time = datetime.now()

        # Ensure output directory exists
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[LLMLogger] Initialization completed")
            print(f"  - Experiment ID: {self.experiment_id}")
            print(f"  - Output directory: {self.output_dir}")
            print(f"  - Output format: {self.format}")

    def log_call(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        response: str,
        temperature: float = 1.0,   # 2025-12-27 update: consistent with api_config.py
        max_tokens: int = 16384,    # 2025-12-26 update: consistent with api_config.py
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record an LLM call

        Args:
            model_name: Model name
            messages: Input message list
            response: Model response
            temperature: Temperature parameter
            max_tokens: Maximum token count
            latency_ms: Latency in milliseconds
            metadata: Additional metadata (e.g. agent name, unit_id, etc.)
            error: If call failed, record error message
        """
        if not self.enabled:
            return

        # Estimate token count
        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens_est = LLMCallRecord.estimate_tokens(prompt_text)
        completion_tokens_est = LLMCallRecord.estimate_tokens(response)

        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            messages=messages,
            response=response,
            temperature=temperature,
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            prompt_tokens_est=prompt_tokens_est,
            completion_tokens_est=completion_tokens_est,
            metadata=metadata or {},
            error=error,
        )

        with self._lock:
            self._records.append(record)
            self._call_count += 1

        # If JSONL format, append immediately
        if self.format == "jsonl":
            self._append_jsonl(record)

    def _append_jsonl(self, record: LLMCallRecord) -> None:
        """Append to JSONL file"""
        filepath = self.output_dir / f"llm_calls_{self.experiment_id}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save all log records

        Args:
            filepath: Custom save path, default uses output_dir/llm_calls_{experiment_id}.{format}

        Returns:
            Actual file path saved
        """
        if not self.enabled:
            return Path()

        if filepath is None:
            suffix = "json" if self.format == "json" else "jsonl" if self.format == "jsonl" else "txt"
            filepath = self.output_dir / f"llm_calls_{self.experiment_id}.{suffix}"
        else:
            filepath = Path(filepath)

        with self._lock:
            records_copy = list(self._records)

        if self.format == "json":
            # Complete JSON format
            data = {
                "experiment_id": self.experiment_id,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "end_time": datetime.now().isoformat(),
                "total_calls": len(records_copy),
                "summary": self.get_summary(),
                "records": [r.to_dict() for r in records_copy],
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, ensure_ascii=False, fp=f, indent=2)

        elif self.format == "jsonl":
            # JSONL already appended during log_call, only update summary here
            summary_path = self.output_dir / f"llm_summary_{self.experiment_id}.json"
            summary = {
                "experiment_id": self.experiment_id,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "end_time": datetime.now().isoformat(),
                "total_calls": len(records_copy),
                "summary": self.get_summary(),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, ensure_ascii=False, fp=f, indent=2)

        elif self.format == "txt":
            # Human-readable text format
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"=" * 80 + "\n")
                f.write(f"LLM Call Log - {self.experiment_id}\n")
                f.write(f"=" * 80 + "\n\n")

                for i, record in enumerate(records_copy, 1):
                    f.write(f"[Call #{i}] {record.timestamp}\n")
                    f.write(f"Model: {record.model_name}\n")
                    if record.metadata:
                        f.write(f"Metadata: {json.dumps(record.metadata, ensure_ascii=False)}\n")
                    f.write(f"\n--- Input Prompt ---\n")
                    for msg in record.messages:
                        f.write(f"[{msg.get('role', 'unknown')}]:\n{msg.get('content', '')}\n\n")
                    f.write(f"--- Output Response ---\n")
                    f.write(f"{record.response}\n")
                    if record.error:
                        f.write(f"\n--- Error ---\n{record.error}\n")
                    f.write(f"\nLatency: {record.latency_ms:.2f}ms\n" if record.latency_ms else "")
                    f.write(f"Estimated Tokens: Input={record.prompt_tokens_est}, Output={record.completion_tokens_est}\n")
                    f.write("-" * 80 + "\n\n")

        print(f"[LLMLogger] Log saved: {filepath}")
        return filepath

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        with self._lock:
            records_copy = list(self._records)

        if not records_copy:
            return {
                "total_calls": 0,
                "error_count": 0,
                "total_prompt_tokens_est": 0,
                "total_completion_tokens_est": 0,
                "total_tokens_est": 0,
                "avg_latency_ms": 0.0,
                "by_model": {},
                "by_agent": {},
            }

        # Statistics by model
        model_stats: Dict[str, Dict[str, Any]] = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0
        error_count = 0

        for record in records_copy:
            model = record.model_name
            if model not in model_stats:
                model_stats[model] = {
                    "call_count": 0,
                    "prompt_tokens_est": 0,
                    "completion_tokens_est": 0,
                    "total_latency_ms": 0.0,
                    "error_count": 0,
                }

            model_stats[model]["call_count"] += 1
            model_stats[model]["prompt_tokens_est"] += record.prompt_tokens_est or 0
            model_stats[model]["completion_tokens_est"] += record.completion_tokens_est or 0
            if record.latency_ms:
                model_stats[model]["total_latency_ms"] += record.latency_ms
            if record.error:
                model_stats[model]["error_count"] += 1
                error_count += 1

            total_prompt_tokens += record.prompt_tokens_est or 0
            total_completion_tokens += record.completion_tokens_est or 0
            if record.latency_ms:
                total_latency += record.latency_ms

        # Statistics by Agent
        agent_stats: Dict[str, int] = {}
        for record in records_copy:
            agent = record.metadata.get("agent", "unknown")
            agent_stats[agent] = agent_stats.get(agent, 0) + 1

        return {
            "total_calls": len(records_copy),
            "error_count": error_count,
            "total_prompt_tokens_est": total_prompt_tokens,
            "total_completion_tokens_est": total_completion_tokens,
            "total_tokens_est": total_prompt_tokens + total_completion_tokens,
            "avg_latency_ms": total_latency / len(records_copy) if records_copy else 0,
            "by_model": model_stats,
            "by_agent": agent_stats,
        }

    def print_summary(self) -> None:
        """Print statistics summary"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("LLM Call Statistics Summary")
        print("=" * 60)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Total calls: {summary['total_calls']}")
        print(f"Error count: {summary['error_count']}")
        print(f"Estimated total tokens: {summary['total_tokens_est']}")
        print(f"  - Input: {summary['total_prompt_tokens_est']}")
        print(f"  - Output: {summary['total_completion_tokens_est']}")
        print(f"Average latency: {summary['avg_latency_ms']:.2f}ms")

        print("\n--- Statistics by Model ---")
        for model, stats in summary.get("by_model", {}).items():
            print(f"  {model}:")
            print(f"    Call count: {stats['call_count']}")
            print(f"    Estimated tokens: {stats['prompt_tokens_est'] + stats['completion_tokens_est']}")
            if stats['error_count'] > 0:
                print(f"    Error count: {stats['error_count']}")

        print("\n--- Statistics by Agent ---")
        for agent, count in summary.get("by_agent", {}).items():
            print(f"  {agent}: {count} calls")

        print("=" * 60 + "\n")

    def clear(self) -> None:
        """Clear all records"""
        with self._lock:
            self._records.clear()
            self._call_count = 0


# Global singleton
_global_logger: Optional[LLMLogger] = None
_global_lock = threading.Lock()


def init_global_logger(
    output_dir: str = "outputs/llm_logs",
    experiment_id: Optional[str] = None,
    enabled: bool = True,
    format: str = "jsonl",
) -> LLMLogger:
    """
    Initialize global LLM logger

    Args:
        output_dir: Output directory
        experiment_id: Experiment ID, default uses timestamp
        enabled: Whether to enable recording
        format: Output format ("json", "jsonl", "txt")

    Returns:
        LLMLogger instance
    """
    global _global_logger
    with _global_lock:
        _global_logger = LLMLogger(
            output_dir=Path(output_dir),
            experiment_id=experiment_id,
            enabled=enabled,
            format=format,
        )
    return _global_logger


def get_global_logger() -> Optional[LLMLogger]:
    """Get global LLM logger"""
    return _global_logger


def log_llm_call(
    model_name: str,
    messages: List[Dict[str, str]],
    response: str,
    **kwargs: Any,
) -> None:
    """
    Convenience function: Record an LLM call to global log

    If global log not initialized, silently ignore.
    """
    if _global_logger is not None:
        _global_logger.log_call(model_name=model_name, messages=messages, response=response, **kwargs)


__all__ = [
    "LLMLogger",
    "LLMCallRecord",
    "init_global_logger",
    "get_global_logger",
    "log_llm_call",
]
