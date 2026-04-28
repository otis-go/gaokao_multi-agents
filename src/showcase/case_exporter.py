# -*- coding: utf-8 -*-
"""
[2026-01 New] Case Export Module

Generates structured folders and Markdown summary documents.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .case_collector import FullCaseRecord
from .prompt_highlighter import PromptHighlighter


class CaseExporter:
    """Export complete Case showcase documents"""

    def __init__(
        self,
        output_dir: Path,
        highlighter: Optional[PromptHighlighter] = None,
    ):
        self.output_dir = Path(output_dir)
        self.cases_dir = self.output_dir / "good_bad_cases"
        self.highlighter = highlighter or PromptHighlighter()

    def export_all_cases(
        self,
        good_cases: List[FullCaseRecord],
        bad_cases: List[FullCaseRecord],
        experiment_id: str = "",
        prompt_level: str = "C",
    ) -> Path:
        """
        Export all cases and generate summary document

        Args:
            good_cases: Good case list
            bad_cases: Bad case list
            experiment_id: Experiment ID
            prompt_level: Prompt level

        Returns:
            Path to index.md
        """
        # Create directories
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        (self.cases_dir / "good_cases").mkdir(exist_ok=True)
        (self.cases_dir / "bad_cases").mkdir(exist_ok=True)

        # Export each case
        exported_good = []
        for i, case in enumerate(good_cases, 1):
            case_dir = self._export_single_case(case, i, "good_cases")
            exported_good.append((case, case_dir))

        exported_bad = []
        for i, case in enumerate(bad_cases, 1):
            case_dir = self._export_single_case(case, i, "bad_cases")
            exported_bad.append((case, case_dir))

        # Generate summary document
        index_content = self._generate_index_md(
            good_cases=good_cases,
            bad_cases=bad_cases,
            experiment_id=experiment_id,
            prompt_level=prompt_level,
        )

        index_path = self.cases_dir / "index.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)

        return index_path

    def _export_single_case(
        self,
        case: FullCaseRecord,
        index: int,
        category: str,
    ) -> Path:
        """
        Export complete directory for a single Case

        Args:
            case: Case data
            index: Index number
            category: "good_cases" or "bad_cases"

        Returns:
            Case directory path
        """
        # Create case directory
        suffix = ""
        if case.case_type == "bad":
            if case.max_iteration_exceeded:
                suffix = "_iteration_exceeded"
            elif case.f1 == 0:
                suffix = "_f1_zero"
            else:
                suffix = "_low_f1"

        case_dir_name = f"case_{index:03d}_unit_{case.unit_id}{suffix}"
        case_dir = self.cases_dir / category / case_dir_name
        case_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export overview
        overview_content = self._generate_overview_md(case)
        with open(case_dir / "00_overview.md", "w", encoding="utf-8") as f:
            f.write(overview_content)

        # 2. Export material
        if case.material_text:
            material_content = self._generate_material_md(case)
            with open(case_dir / "01_material.md", "w", encoding="utf-8") as f:
                f.write(material_content)

        # 3. Export target dimensions and ABC prompts
        if case.target_dimensions or case.abc_prompts_used:
            dims_data = {
                "target_dimensions": case.target_dimensions,
                "predicted_dimensions": case.predicted_dimensions,
                "prompt_level": case.prompt_level,
                "abc_prompts": case.abc_prompts_used,
            }
            with open(case_dir / "02_target_dimensions.json", "w", encoding="utf-8") as f:
                json.dump(dims_data, f, ensure_ascii=False, indent=2)

        # 4. Export Stage1 iteration records
        if case.iterations:
            stage1_dir = case_dir / "stage1"
            stage1_dir.mkdir(exist_ok=True)

            for iteration in case.iterations:
                iter_dir = stage1_dir / f"iteration_{iteration.iteration}"
                iter_dir.mkdir(exist_ok=True)

                # Agent2 Anchor Discovery
                if iteration.agent2_interaction:
                    agent2_md = self.highlighter.format_agent_interaction_md(
                        agent_name=iteration.agent2_interaction.agent_name,
                        iteration=iteration.iteration,
                        prompt=iteration.agent2_interaction.prompt,
                        response=iteration.agent2_interaction.response,
                        model=iteration.agent2_interaction.model,
                        timestamp=iteration.agent2_interaction.timestamp,
                    )
                    with open(iter_dir / "agent2_anchor.md", "w", encoding="utf-8") as f:
                        f.write(agent2_md)

                # Agent3 Question Generation
                if iteration.agent3_interaction:
                    agent3_md = self.highlighter.format_agent_interaction_md(
                        agent_name=iteration.agent3_interaction.agent_name,
                        iteration=iteration.iteration,
                        prompt=iteration.agent3_interaction.prompt,
                        response=iteration.agent3_interaction.response,
                        model=iteration.agent3_interaction.model,
                        timestamp=iteration.agent3_interaction.timestamp,
                    )
                    with open(iter_dir / "agent3_generation.md", "w", encoding="utf-8") as f:
                        f.write(agent3_md)

                # Agent3 Repair (if any)
                if iteration.agent3_repair_interaction:
                    repair_md = self.highlighter.format_agent_interaction_md(
                        agent_name=iteration.agent3_repair_interaction.agent_name,
                        iteration=iteration.iteration,
                        prompt=iteration.agent3_repair_interaction.prompt,
                        response=iteration.agent3_repair_interaction.response,
                        model=iteration.agent3_repair_interaction.model,
                        timestamp=iteration.agent3_repair_interaction.timestamp,
                    )
                    with open(iter_dir / "agent3_repair.md", "w", encoding="utf-8") as f:
                        f.write(repair_md)

                # Agent4 Quality Check
                if iteration.agent4_interaction:
                    agent4_md = self.highlighter.format_agent_interaction_md(
                        agent_name=iteration.agent4_interaction.agent_name,
                        iteration=iteration.iteration,
                        prompt=iteration.agent4_interaction.prompt,
                        response=iteration.agent4_interaction.response,
                        model=iteration.agent4_interaction.model,
                        timestamp=iteration.agent4_interaction.timestamp,
                    )
                    with open(iter_dir / "agent4_verification.md", "w", encoding="utf-8") as f:
                        f.write(agent4_md)

                # Agent4 Decision Info
                if iteration.agent4_decision:
                    with open(iter_dir / "agent4_decision.json", "w", encoding="utf-8") as f:
                        json.dump(iteration.agent4_decision, f, ensure_ascii=False, indent=2)

        # 5. Export Stage2 evaluation
        if case.stage2_eval:
            stage2_dir = case_dir / "stage2"
            stage2_dir.mkdir(exist_ok=True)

            eval_md = self.highlighter.format_evaluation_md(
                eval_record=case.stage2_eval,
                gold_dimensions=case.target_dimensions,
                predicted_dimensions=case.predicted_dimensions,
            )
            with open(stage2_dir / "evaluation.md", "w", encoding="utf-8") as f:
                f.write(eval_md)

        # 6. Export final question
        if case.final_question:
            question_md = self.highlighter.format_question_md(
                question=case.final_question,
                question_type=case.question_type,
            )
            with open(case_dir / "final_question.md", "w", encoding="utf-8") as f:
                f.write(question_md)

            # Also save in JSON format
            with open(case_dir / "final_question.json", "w", encoding="utf-8") as f:
                json.dump(case.final_question, f, ensure_ascii=False, indent=2)

        return case_dir

    def _generate_overview_md(self, case: FullCaseRecord) -> str:
        """Generate Case overview Markdown"""
        lines = [
            f"# Case Overview - unit_{case.unit_id}",
            "",
            "## Basic Information",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Unit ID | {case.unit_id} |",
            f"| Case Type | {case.case_type.upper()} |",
            f"| Question Type | {case.question_type or 'N/A'} |",
            f"| Prompt Level | Level {case.prompt_level} |",
            f"| Iterations | {case.total_iterations} |",
            f"| Iteration Exceeded | {'Yes' if case.max_iteration_exceeded else 'No'} |",
            "",
        ]

        # Failure reason (Bad case)
        if case.fail_reason:
            lines.extend([
                "## Failure Reason",
                "",
                f"> {case.fail_reason}",
                "",
            ])

        # Evaluation metrics
        lines.extend([
            "## Evaluation Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| F1 | {case.f1:.4f} |",
            f"| Precision | {case.precision:.4f} |",
            f"| Recall | {case.recall:.4f} |",
            "",
        ])

        # Dimension information
        lines.extend([
            "## Dimension Information",
            "",
            f"**Target Dimensions (Gold)**: {', '.join(case.target_dimensions) if case.target_dimensions else 'None'}",
            "",
            f"**Predicted Dimensions (Pred)**: {', '.join(case.predicted_dimensions) if case.predicted_dimensions else 'None'}",
            "",
        ])

        # Directory structure
        lines.extend([
            "## File Structure",
            "",
            "```",
            f"case_xxx_unit_{case.unit_id}/",
            "├── 00_overview.md          # This file",
            "├── 01_material.md          # Reading material",
            "├── 02_target_dimensions.json  # Dimensions and prompts",
            "├── stage1/                 # Stage1 generation process",
        ])

        for i in range(1, case.total_iterations + 1):
            lines.append(f"│   └── iteration_{i}/")
            lines.append("│       ├── agent2_anchor.md")
            lines.append("│       ├── agent3_generation.md")
            lines.append("│       └── agent4_verification.md")

        lines.extend([
            "├── stage2/                 # Stage2 evaluation",
            "│   └── evaluation.md",
            "├── final_question.md       # Final question",
            "└── final_question.json",
            "```",
            "",
        ])

        return "\n".join(lines)

    def _generate_material_md(self, case: FullCaseRecord) -> str:
        """Generate material Markdown"""
        lines = [
            f"# Reading Material - unit_{case.unit_id}",
            "",
            "---",
            "",
            case.material_text,
            "",
            "---",
            "",
        ]
        return "\n".join(lines)

    def _generate_index_md(
        self,
        good_cases: List[FullCaseRecord],
        bad_cases: List[FullCaseRecord],
        experiment_id: str = "",
        prompt_level: str = "C",
    ) -> str:
        """Generate summary index document"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# Good/Bad Case Showcase Report",
            "",
            f"**Experiment ID**: {experiment_id or 'N/A'}",
            f"**Prompt Level**: Level {prompt_level}",
            f"**Generated Time**: {timestamp}",
            "",
        ]

        # Overview table
        lines.extend([
            "## Overview",
            "",
            "| Type | Count | Features |",
            "|------|-------|----------|",
        ])

        # Good cases statistics
        good_f1_avg = sum(c.f1 for c in good_cases) / len(good_cases) if good_cases else 0
        lines.append(f"| Good Cases | {len(good_cases)} | Average F1={good_f1_avg:.4f} |")

        # Bad cases statistics
        iteration_exceeded = sum(1 for c in bad_cases if c.max_iteration_exceeded)
        low_f1 = len(bad_cases) - iteration_exceeded
        bad_features = []
        if iteration_exceeded:
            bad_features.append(f"Iteration Exceeded({iteration_exceeded})")
        if low_f1:
            bad_features.append(f"Low F1({low_f1})")
        lines.append(f"| Bad Cases | {len(bad_cases)} | {', '.join(bad_features) or 'N/A'} |")
        lines.append("")

        # Good Cases list
        if good_cases:
            lines.extend([
                "## Good Cases",
                "",
            ])

            for i, case in enumerate(good_cases, 1):
                suffix = ""
                case_dir_name = f"case_{i:03d}_unit_{case.unit_id}{suffix}"

                lines.extend([
                    f"### Case {i}: unit_{case.unit_id} (F1={case.f1:.4f})",
                    "",
                    f"- **Question Type**: {case.question_type or 'N/A'}",
                    f"- **Target Dimensions**: {', '.join(case.target_dimensions[:5]) + ('...' if len(case.target_dimensions) > 5 else '') if case.target_dimensions else 'N/A'}",
                    f"- **Predicted Dimensions**: {', '.join(case.predicted_dimensions[:5]) + ('...' if len(case.predicted_dimensions) > 5 else '') if case.predicted_dimensions else 'N/A'}",
                    f"- **Iterations**: {case.total_iterations} rounds",
                    f"- [View Complete Record](./good_cases/{case_dir_name}/)",
                    "",
                ])

        # Bad Cases list
        if bad_cases:
            lines.extend([
                "## Bad Cases",
                "",
            ])

            for i, case in enumerate(bad_cases, 1):
                suffix = ""
                if case.max_iteration_exceeded:
                    suffix = "_iteration_exceeded"
                    fail_type = "Iteration Exceeded"
                elif case.f1 == 0:
                    suffix = "_f1_zero"
                    fail_type = "F1=0"
                else:
                    suffix = "_low_f1"
                    fail_type = f"Low F1 ({case.f1:.4f})"

                case_dir_name = f"case_{i:03d}_unit_{case.unit_id}{suffix}"

                lines.extend([
                    f"### Case {i}: unit_{case.unit_id} ({fail_type})",
                    "",
                    f"- **Failure Reason**: {case.fail_reason or fail_type}",
                    f"- **Question Type**: {case.question_type or 'N/A'}",
                    f"- **Iterations**: {case.total_iterations} rounds",
                    f"- [View Complete Record](./bad_cases/{case_dir_name}/)",
                    "",
                ])

        # ABC prompt level description
        lines.extend([
            "## Prompt Level Description",
            "",
            "| Level | Content |",
            "|-------|---------|",
            "| Level A | Dimension name + basic definition |",
            "| Level B | A + detailed dimension objectives/key points/acceptance criteria |",
            "| Level C | A + real question examples (recommended) |",
            "",
        ])

        return "\n".join(lines)
