from __future__ import annotations

"""
src/shared/prompt_logger.py

Unified logging for prompts and responses from Agent/Evaluation modules for post-analysis.
Log format: One JSON record per line, written to <log_dir>/<agent_name>.jsonl.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import datetime


@dataclass
class PromptLogRecord:
    agent_name: str
    stage: str
    prompt: str
    response: str
    model: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Avoid None affecting downstream processing
        d["metadata"] = d.get("metadata") or {}
        if not d.get("timestamp"):
            d["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        return d


class PromptLogger:
    """
    Non-intrusive lightweight logger:
    - Does not affect main flow; on file write failure, only prints warning
    - GenerationOrchestrator / EvaluationOrchestrator only need to pass log_dir
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save_agent_log(
        self,
        agent_name: str,
        stage: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Log a prompt/response call to <log_dir>/<agent_name>.jsonl.
        """
        record = PromptLogRecord(
            agent_name=agent_name,
            stage=stage,
            prompt=prompt,
            response=response,
            metadata=metadata or {},
            model=model,
        ).to_dict()

        file_path = self.log_dir / f"{agent_name}.jsonl"
        try:
            with file_path.open("a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:  # Log failure does not terminate main flow
            print(f"[PromptLogger] Failed to write log: {e} (path={file_path})")


__all__ = ["PromptLogger", "PromptLogRecord"]
