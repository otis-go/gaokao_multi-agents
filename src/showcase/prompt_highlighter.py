# -*- coding: utf-8 -*-
"""
[2026-01 New] Prompt Highlighting Module

Identifies and marks dynamically filled content in prompts for paper presentation.
"""

import re
from typing import Dict, List, Optional, Tuple


class PromptHighlighter:
    """Mark filled content in prompts"""

    # Highlight markers
    HIGHLIGHT_START = "<!-- ========== Filled Content Start ========== -->"
    HIGHLIGHT_END = "<!-- ========== Filled Content End ========== -->"

    # Patterns for highlighting regions
    HIGHLIGHT_PATTERNS = [
        # Material content
        (r"(【阅读材料原文】|【材料】|【输入材料】|material:)\s*([\s\S]*?)(?=\n\n|\n【|\Z)", "Material"),
        # Fusion prompt
        (r"(【融合版命题提示词】|【维度提示词】|fusion_prompt:)\s*([\s\S]*?)(?=\n\n|\n【|\Z)", "FusionPrompt"),
        # Anchor info
        (r"(【锚点列表】|【锚点信息】|anchors:)\s*([\s\S]*?)(?=\n\n|\n【|\Z)", "Anchor"),
        # Question type
        (r"(【题型】|question_type:)\s*(\S+)", "QuestionType"),
        # Stem
        (r"(【题干】|stem:)\s*([\s\S]*?)(?=\n\n|\n【|\Z)", "Stem"),
        # Answer
        (r"(【答案】|【标准答案】|correct_answer:)\s*([\s\S]*?)(?=\n\n|\n【|\Z)", "Answer"),
        # Explanation
        (r"(【解析】|explanation:)\s*([\s\S]*?)(?=\n\n|\n【|\Z)", "Explanation"),
    ]

    def highlight_prompt(
        self,
        prompt: str,
        content_markers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Mark filled content in prompt

        Args:
            prompt: Complete prompt text
            content_markers: Known filled content markers {type: content}

        Returns:
            Prompt with highlight markers
        """
        if not prompt:
            return prompt

        highlighted = prompt

        # Use regex patterns to match and mark
        for pattern, label in self.HIGHLIGHT_PATTERNS:
            def replace_func(match):
                prefix = match.group(1)
                content = match.group(2).strip()
                if content and len(content) > 10:  # Only highlight parts with actual content
                    return f"{prefix}\n{self.HIGHLIGHT_START}\n{content}\n{self.HIGHLIGHT_END}"
                return match.group(0)

            highlighted = re.sub(pattern, replace_func, highlighted, flags=re.MULTILINE)

        return highlighted

    def highlight_with_sections(
        self,
        prompt: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> str:
        """
        Highlight based on known section positions

        Args:
            prompt: Complete prompt text
            sections: {section_name: (start_pos, end_pos)}

        Returns:
            Prompt with highlight markers
        """
        if not sections:
            return prompt

        # Replace from back to front to avoid position offset
        sorted_sections = sorted(sections.items(), key=lambda x: x[1][0], reverse=True)

        result = prompt
        for name, (start, end) in sorted_sections:
            content = prompt[start:end]
            marked = f"{self.HIGHLIGHT_START}\n{content}\n{self.HIGHLIGHT_END}"
            result = result[:start] + marked + result[end:]

        return result

    def to_markdown(self, text: str, code_block: bool = True) -> str:
        """
        Convert text to Markdown format

        Args:
            text: Input text (may contain highlight markers)
            code_block: Whether to wrap in code block

        Returns:
            Markdown formatted text
        """
        if code_block:
            # Use ~~~ instead of ``` to avoid conflicts with backticks in JSON
            return f"~~~\n{text}\n~~~"
        return text

    def format_agent_interaction_md(
        self,
        agent_name: str,
        iteration: int,
        prompt: str,
        response: str,
        model: str = "",
        timestamp: str = "",
        highlight: bool = True,
    ) -> str:
        """
        Format single Agent interaction as Markdown

        Args:
            agent_name: Agent name
            iteration: Iteration round
            prompt: Complete prompt
            response: LLM response
            model: Model name
            timestamp: Timestamp
            highlight: Whether to highlight filled content

        Returns:
            Markdown formatted document
        """
        # Agent name mapping
        agent_display_names = {
            "Agent2": "Agent2 Anchor Discovery",
            "Agent3_SingleChoice": "Agent3 Single Choice Generation",
            "Agent3_Essay": "Agent3 Essay Generation",
            "Agent3_Repair": "Agent3 Repair Retry",
            "Agent4_Solver": "Agent4 Consistency Check",
            "PedagogicalHitBasedEval": "Pedagogical Dimension Eval",
        }

        display_name = agent_display_names.get(agent_name, agent_name)

        lines = [
            f"# {display_name} - Iteration {iteration}",
            "",
        ]

        # Meta information
        if model or timestamp:
            meta_parts = []
            if model:
                meta_parts.append(f"**Model**: {model}")
            if timestamp:
                meta_parts.append(f"**Time**: {timestamp}")
            lines.append(" | ".join(meta_parts))
            lines.append("")

        # Prompt section
        lines.append("## Prompt")
        lines.append("")

        # Highlight processing
        if highlight:
            prompt = self.highlight_prompt(prompt)

        lines.append("~~~")
        lines.append(prompt)
        lines.append("~~~")
        lines.append("")

        # Response section
        lines.append("## Response")
        lines.append("")

        # Try to format as JSON
        try:
            import json
            resp_data = json.loads(response)
            formatted_resp = json.dumps(resp_data, ensure_ascii=False, indent=2)
            lines.append("~~~json")
            lines.append(formatted_resp)
            lines.append("~~~")
        except (json.JSONDecodeError, TypeError):
            lines.append("~~~")
            lines.append(response)
            lines.append("~~~")

        return "\n".join(lines)

    def format_question_md(
        self,
        question: Dict,
        question_type: str = "",
    ) -> str:
        """
        Format question as Markdown

        Args:
            question: Question data
            question_type: Question type

        Returns:
            Markdown formatted question display
        """
        lines = ["# Final Generated Question", ""]

        # Stem
        stem = question.get("stem", "")
        if stem:
            lines.append("## Stem")
            lines.append("")
            lines.append(f"> {stem}")
            lines.append("")

        # Options (for choice questions)
        options = question.get("options", [])
        if options:
            lines.append("## Options")
            lines.append("")
            lines.append("| Option | Content | Correct |")
            lines.append("|--------|---------|---------|")

            correct_answer = question.get("correct_answer", "").strip().upper()
            for opt in options:
                if isinstance(opt, dict):
                    label = opt.get("label", opt.get("key", ""))
                    content = opt.get("content", opt.get("text", opt.get("value", "")))
                else:
                    label = ""
                    content = str(opt)

                is_correct = "**√**" if label.upper() == correct_answer else ""
                # Bold if correct answer
                if is_correct:
                    lines.append(f"| **{label}** | **{content}** | {is_correct} |")
                else:
                    lines.append(f"| {label} | {content} | {is_correct} |")
            lines.append("")

        # Answer points (for essay questions)
        answer_points = question.get("answer_points", [])
        if answer_points:
            lines.append("## Answer Points")
            lines.append("")
            for i, point in enumerate(answer_points, 1):
                if isinstance(point, dict):
                    point_text = point.get("point", point.get("content", ""))
                    score = point.get("score", "")
                    lines.append(f"{i}. {point_text}" + (f" ({score} points)" if score else ""))
                else:
                    lines.append(f"{i}. {point}")
            lines.append("")

        # Explanation
        explanation = question.get("explanation", "")
        if explanation:
            lines.append("## Explanation")
            lines.append("")
            lines.append(explanation)
            lines.append("")

        return "\n".join(lines)

    def format_evaluation_md(
        self,
        eval_record,
        gold_dimensions: List[str],
        predicted_dimensions: List[str],
    ) -> str:
        """
        Format Stage2 evaluation as Markdown

        Args:
            eval_record: Stage2 evaluation record
            gold_dimensions: Gold (target) dimensions
            predicted_dimensions: Predicted dimensions

        Returns:
            Markdown formatted evaluation display
        """
        lines = ["# Stage2 Pedagogical Dimension Evaluation", ""]

        # Dimension comparison
        lines.append("## Dimension Comparison")
        lines.append("")
        lines.append(f"**Target Dimensions (Gold)**: {', '.join(gold_dimensions) if gold_dimensions else 'None'}")
        lines.append(f"**Predicted Dimensions (Pred)**: {', '.join(predicted_dimensions) if predicted_dimensions else 'None'}")
        lines.append("")

        # Metrics
        if eval_record and eval_record.metrics:
            lines.append("## Evaluation Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Precision | {eval_record.metrics.get('precision', 0):.4f} |")
            lines.append(f"| Recall | {eval_record.metrics.get('recall', 0):.4f} |")
            lines.append(f"| F1 | {eval_record.metrics.get('f1', 0):.4f} |")
            lines.append(f"| TP | {eval_record.metrics.get('tp', 0)} |")
            lines.append(f"| FP | {eval_record.metrics.get('fp', 0)} |")
            lines.append(f"| FN | {eval_record.metrics.get('fn', 0)} |")
            lines.append("")

        # Evaluation details by dimension
        if eval_record and eval_record.dimension_results:
            lines.append("## Evaluation Details by Dimension")
            lines.append("")

            for dim_code, result in sorted(eval_record.dimension_results.items()):
                hit = result.get("hit", False)
                reason = result.get("reason", "")
                hit_mark = "[OK]" if hit else "[X]"
                in_gold = dim_code in gold_dimensions
                gold_mark = " (Gold)" if in_gold else ""

                lines.append(f"### {dim_code}{gold_mark} - {hit_mark}")
                lines.append("")
                if reason:
                    lines.append(f"> {reason}")
                    lines.append("")

        # Independent scoring by model
        if eval_record and eval_record.model_results:
            lines.append("## Independent Scoring by Model")
            lines.append("")

            for model_name, model_result in eval_record.model_results.items():
                lines.append(f"### {model_name}")
                lines.append("")

                # Only show hit dimensions
                hit_dims = [d for d, r in model_result.items() if r.get("hit")]
                if hit_dims:
                    lines.append(f"Hit dimensions: {', '.join(sorted(hit_dims))}")
                else:
                    lines.append("Hit dimensions: None")
                lines.append("")

        # Evaluation Prompt and Response
        if eval_record and eval_record.eval_prompts:
            lines.append("## Evaluation Details (First Model)")
            lines.append("")

            first_prompt = eval_record.eval_prompts[0] if eval_record.eval_prompts else {}
            first_response = eval_record.eval_responses[0] if eval_record.eval_responses else {}

            if first_prompt:
                lines.append(f"### Prompt ({first_prompt.get('model', 'unknown')})")
                lines.append("")
                prompt_text = first_prompt.get("prompt", "")
                # Truncate to first 2000 characters
                if len(prompt_text) > 2000:
                    prompt_text = prompt_text[:2000] + "\n... [Truncated, see log file for full content]"
                lines.append("~~~")
                lines.append(prompt_text)
                lines.append("~~~")
                lines.append("")

            if first_response:
                lines.append(f"### Response ({first_response.get('model', 'unknown')})")
                lines.append("")
                response_text = first_response.get("response", "")
                # Try to format as JSON
                try:
                    import json
                    resp_data = json.loads(response_text)
                    response_text = json.dumps(resp_data, ensure_ascii=False, indent=2)
                    lines.append("~~~json")
                except (json.JSONDecodeError, TypeError):
                    lines.append("~~~")
                lines.append(response_text)
                lines.append("~~~")
                lines.append("")

        return "\n".join(lines)
