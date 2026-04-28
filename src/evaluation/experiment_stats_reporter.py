# src/evaluation/experiment_stats_reporter.py
# Experiment Statistics Report Generator

"""
Module Description:
ExperimentStatsReporter generates experiment statistics reports, including:
1. Detailed scores for each question (unit_id, question_type, material_type, AI score, pedagogical score)
2. Statistics grouped by question type (single-choice/essay)
3. Statistics grouped by material type (expository/argumentative)
4. Overall statistical summary

Output formats:
- Detailed report in JSON format
- Tabular data in CSV format (Excel-friendly)
- Console-friendly summary printout
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv


@dataclass
class QuestionResult:
    """
    Evaluation result for a single question
    """
    unit_id: str
    question_type: str  # "single-choice" | "essay"
    material_type: str  # "expository" | "argumentative" | "other"
    ai_score: Optional[float] = None
    pedagogical_score: Optional[float] = None
    final_decision: Optional[str] = None
    source: str = "generated"  # "generated" | "baseline"

    # Extended information
    dimension_count: int = 0
    stage1_success: bool = False
    stage2_success: bool = False
    error_info: Optional[str] = None


@dataclass
class GroupStats:
    """
    Grouped statistics result
    """
    group_name: str
    count: int = 0
    ai_scores: List[float] = field(default_factory=list)
    ped_scores: List[float] = field(default_factory=list)

    @property
    def avg_ai_score(self) -> float:
        return sum(self.ai_scores) / len(self.ai_scores) if self.ai_scores else 0.0

    @property
    def avg_ped_score(self) -> float:
        return sum(self.ped_scores) / len(self.ped_scores) if self.ped_scores else 0.0

    @property
    def min_ai_score(self) -> Optional[float]:
        return min(self.ai_scores) if self.ai_scores else None

    @property
    def max_ai_score(self) -> Optional[float]:
        return max(self.ai_scores) if self.ai_scores else None

    @property
    def min_ped_score(self) -> Optional[float]:
        return min(self.ped_scores) if self.ped_scores else None

    @property
    def max_ped_score(self) -> Optional[float]:
        return max(self.ped_scores) if self.ped_scores else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "count": self.count,
            "avg_ai_score": round(self.avg_ai_score, 2),
            "avg_ped_score": round(self.avg_ped_score, 2),
            "min_ai_score": round(self.min_ai_score, 2) if self.min_ai_score is not None else None,
            "max_ai_score": round(self.max_ai_score, 2) if self.max_ai_score is not None else None,
            "min_ped_score": round(self.min_ped_score, 2) if self.min_ped_score is not None else None,
            "max_ped_score": round(self.max_ped_score, 2) if self.max_ped_score is not None else None,
        }


@dataclass
class ExperimentStatsReport:
    """
    Complete experiment statistics report
    """
    experiment_id: str
    run_mode: str  # "generated" | "baseline" | "full"
    timestamp: str

    # Overall statistics
    total_questions: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0

    # Overall scores
    overall_ai_score: float = 0.0
    overall_ped_score: float = 0.0

    # Grouped by question type
    by_question_type: Dict[str, GroupStats] = field(default_factory=dict)

    # Grouped by material type
    by_material_type: Dict[str, GroupStats] = field(default_factory=dict)

    # Detailed results list
    question_results: List[QuestionResult] = field(default_factory=list)

    # Configuration info
    config_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_mode": self.run_mode,
            "timestamp": self.timestamp,
            "summary": {
                "total_questions": self.total_questions,
                "successful_evaluations": self.successful_evaluations,
                "failed_evaluations": self.failed_evaluations,
                "overall_ai_score": round(self.overall_ai_score, 2),
                "overall_ped_score": round(self.overall_ped_score, 2),
            },
            "by_question_type": {k: v.to_dict() for k, v in self.by_question_type.items()},
            "by_material_type": {k: v.to_dict() for k, v in self.by_material_type.items()},
            "question_results": [asdict(r) for r in self.question_results],
            "config_info": self.config_info,
        }


class ExperimentStatsReporter:
    """
    Experiment Statistics Report Generator

    Usage:
    ```python
    reporter = ExperimentStatsReporter(experiment_id="EXP_001")

    # Add result for each question
    reporter.add_result(
        unit_id="1",
        question_type="single-choice",
        material_type="expository",
        ai_score=85.0,
        ped_score=90.0,
    )

    # Generate report
    report = reporter.generate_report()
    reporter.save_json(output_path)
    reporter.save_csv(output_path)
    reporter.print_summary()
    ```
    """

    def __init__(
        self,
        experiment_id: str,
        run_mode: str = "generated",
        config_info: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_id = experiment_id
        self.run_mode = run_mode
        self.config_info = config_info or {}
        self.results: List[QuestionResult] = []

    def add_result(
        self,
        unit_id: str,
        question_type: str,
        material_type: str,
        ai_score: Optional[float] = None,
        pedagogical_score: Optional[float] = None,
        final_decision: Optional[str] = None,
        source: str = "generated",
        dimension_count: int = 0,
        stage1_success: bool = False,
        stage2_success: bool = False,
        error_info: Optional[str] = None,
    ) -> None:
        """
        Add evaluation result for a single question
        """
        # Normalize question type
        qt = self._normalize_question_type(question_type)
        # Normalize material type
        mt = self._normalize_material_type(material_type)

        result = QuestionResult(
            unit_id=str(unit_id),
            question_type=qt,
            material_type=mt,
            ai_score=ai_score,
            pedagogical_score=pedagogical_score,
            final_decision=final_decision,
            source=source,
            dimension_count=dimension_count,
            stage1_success=stage1_success,
            stage2_success=stage2_success,
            error_info=error_info,
        )
        self.results.append(result)

    def _normalize_question_type(self, qt: str) -> str:
        """Normalize question type"""
        if not qt:
            return "unknown"
        qt_lower = qt.lower()
        # Chinese keywords for choice questions
        if any(kw in qt_lower for kw in ["choice", "mcq"]) or any(kw in qt for kw in ["选择", "单选", "多选"]):
            return "single-choice"
        # Chinese keywords for essay questions
        if any(kw in qt_lower for kw in ["essay"]) or any(kw in qt for kw in ["简答", "主观", "论述"]):
            return "essay"
        return qt

    def _normalize_material_type(self, mt: str) -> str:
        """Normalize material type"""
        if not mt:
            return "other"
        # Chinese: expository text
        if "说明" in mt:
            return "expository"
        # Chinese: argumentative text
        if "议论" in mt:
            return "argumentative"
        return mt if mt else "other"

    def generate_report(self) -> ExperimentStatsReport:
        """Generate complete statistics report"""
        report = ExperimentStatsReport(
            experiment_id=self.experiment_id,
            run_mode=self.run_mode,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_info=self.config_info,
        )

        # Initialize groups
        qt_groups: Dict[str, GroupStats] = {}
        mt_groups: Dict[str, GroupStats] = {}

        all_ai_scores: List[float] = []
        all_ped_scores: List[float] = []

        for r in self.results:
            report.total_questions += 1

            # Check if evaluation succeeded
            if r.ai_score is not None or r.pedagogical_score is not None:
                report.successful_evaluations += 1
            else:
                report.failed_evaluations += 1

            # Group by question type
            qt = r.question_type
            if qt not in qt_groups:
                qt_groups[qt] = GroupStats(group_name=qt)
            qt_groups[qt].count += 1
            if r.ai_score is not None:
                qt_groups[qt].ai_scores.append(r.ai_score)
                all_ai_scores.append(r.ai_score)
            if r.pedagogical_score is not None:
                qt_groups[qt].ped_scores.append(r.pedagogical_score)
                all_ped_scores.append(r.pedagogical_score)

            # Group by material type
            mt = r.material_type
            if mt not in mt_groups:
                mt_groups[mt] = GroupStats(group_name=mt)
            mt_groups[mt].count += 1
            if r.ai_score is not None:
                mt_groups[mt].ai_scores.append(r.ai_score)
            if r.pedagogical_score is not None:
                mt_groups[mt].ped_scores.append(r.pedagogical_score)

        # Calculate overall average scores
        report.overall_ai_score = sum(all_ai_scores) / len(all_ai_scores) if all_ai_scores else 0.0
        report.overall_ped_score = sum(all_ped_scores) / len(all_ped_scores) if all_ped_scores else 0.0

        report.by_question_type = qt_groups
        report.by_material_type = mt_groups
        report.question_results = self.results

        return report

    def save_json(self, output_path: Path) -> None:
        """Save report in JSON format"""
        report = self.generate_report()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"[StatsReporter] JSON report saved: {output_path}")

    def save_csv(self, output_path: Path) -> None:
        """Save detailed data in CSV format (Excel-friendly)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "unit_id",
            "question_type",
            "material_type",
            "ai_score",
            "pedagogical_score",
            "final_decision",
            "source",
            "dimension_count",
            "stage1_success",
            "stage2_success",
            "error_info",
        ]

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for r in self.results:
                writer.writerow([
                    r.unit_id,
                    r.question_type,
                    r.material_type,
                    r.ai_score if r.ai_score is not None else "",
                    r.pedagogical_score if r.pedagogical_score is not None else "",
                    r.final_decision or "",
                    r.source,
                    r.dimension_count,
                    "Yes" if r.stage1_success else "No",
                    "Yes" if r.stage2_success else "No",
                    r.error_info or "",
                ])

        print(f"[StatsReporter] CSV report saved: {output_path}")

    def print_summary(self) -> None:
        """Print console-friendly summary"""
        report = self.generate_report()

        print("\n" + "=" * 70)
        print(f"Experiment Statistics Report - {report.experiment_id}")
        print("=" * 70)

        print(f"\n[Overall Statistics]")
        print(f"  - Total questions:        {report.total_questions}")
        print(f"  - Successful evaluations: {report.successful_evaluations}")
        print(f"  - Failed/Skipped:         {report.failed_evaluations}")
        print(f"  - AI dimension avg:       {report.overall_ai_score:.2f}")
        print(f"  - Pedagogical dim avg:    {report.overall_ped_score:.2f}")

        print(f"\n[By Question Type]")
        for qt, stats in sorted(report.by_question_type.items()):
            print(f"  {qt}:")
            print(f"    - Count:     {stats.count}")
            print(f"    - AI avg:    {stats.avg_ai_score:.2f} (range: {stats.min_ai_score:.2f} - {stats.max_ai_score:.2f})" if stats.ai_scores else f"    - AI avg:    N/A")
            print(f"    - Ped avg:   {stats.avg_ped_score:.2f} (range: {stats.min_ped_score:.2f} - {stats.max_ped_score:.2f})" if stats.ped_scores else f"    - Ped avg:   N/A")

        print(f"\n[By Material Type]")
        for mt, stats in sorted(report.by_material_type.items()):
            print(f"  {mt}:")
            print(f"    - Count:     {stats.count}")
            print(f"    - AI avg:    {stats.avg_ai_score:.2f}" if stats.ai_scores else f"    - AI avg:    N/A")
            print(f"    - Ped avg:   {stats.avg_ped_score:.2f}" if stats.ped_scores else f"    - Ped avg:   N/A")

        print("\n" + "=" * 70)


__all__ = [
    "QuestionResult",
    "GroupStats",
    "ExperimentStatsReport",
    "ExperimentStatsReporter",
]
