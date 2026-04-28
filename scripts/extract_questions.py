#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract generated questions from an experiment directory.

Data sources:
1. Prefer logs/prompts/Agent4_*.jsonl, which contains generated questions.
2. Supplement scoring information from stage2/unit_*/evaluation_state.json.

Updated 2025-12:
- Supports extraction from Agent4 logs even when evaluation_state.json is empty or incomplete.
- Outputs unit_id, question type, stem, options or answer points, answer, explanation, AI score, and pedagogical score.

Usage:
    python scripts/extract_questions.py <experiment_dir> [options]

Examples:
    python scripts/extract_questions.py outputs/EXP_BASELINE_gk+cs_C_20251209_213427
    python scripts/extract_questions.py outputs/ROUND_20251209_A_deepseek/subset40_stratified_seed42_gkcs_A_20251209_213438
    python scripts/extract_questions.py outputs/EXP_xxx --unit-id 10
    python scripts/extract_questions.py outputs/EXP_xxx --output questions.json
    python scripts/extract_questions.py outputs/EXP_xxx --format markdown --output questions.md
"""

import json
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from an LLM response.

    Supports fenced ```json ... ``` blocks and raw JSON. When standard JSON
    parsing fails, regex-based field extraction is used as a fallback.
    """
    if not response:
        return None

    json_str = None

    # Try fenced ```json ... ``` format.
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try direct parsing.
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to locate a JSON object in the response.
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback: extract key fields with regex when JSON is truncated.
    text_to_parse = json_str or response
    if text_to_parse:
        extracted = _extract_fields_via_regex(text_to_parse)
        if extracted:
            return extracted

    return None


def _extract_fields_via_regex(text: str) -> Optional[Dict]:
    """
    Extract key fields from malformed or truncated JSON with regex.

    Supports single-choice and essay formats.
    """
    result = {}

    # Extract stem with multiple supported field names.
    stem_patterns = [
        r'"stem"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)',  # Standard format; allows truncation.
        r'"question"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)',  # Alternate field name.
    ]
    for pattern in stem_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            stem = match.group(1)
            # Clean escaped characters.
            stem = stem.replace('\\n', '\n').replace('\\"', '"')
            result['stem'] = stem.strip()
            break

    # Extract correct_option for single-choice questions.
    correct_match = re.search(r'"correct_option"\s*:\s*"([A-D])"', text)
    if correct_match:
        result['correct_option'] = correct_match.group(1)
        result['_type'] = 'single-choice'

    # Extract options for single-choice questions.
    options_match = re.search(r'"options"\s*:\s*\{([^}]+)\}', text, re.DOTALL)
    if options_match:
        options_text = options_match.group(1)
        options = {}
        for opt in ['A', 'B', 'C', 'D']:
            opt_match = re.search(rf'"{opt}"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)', options_text, re.DOTALL)
            if opt_match:
                opt_content = opt_match.group(1)
                opt_content = opt_content.replace('\\n', '\n').replace('\\"', '"')
                options[opt] = opt_content.strip()
        if options:
            result['options'] = options

    # Extract anchor_ids_used.
    anchors_match = re.search(r'"anchor_ids_used"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if anchors_match:
        anchors_text = anchors_match.group(1)
        anchors = re.findall(r'"([^"]+)"', anchors_text)
        result['anchor_ids_used'] = anchors

    # Extract overall analysis.
    overall_match = re.search(r'"overall"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)', text, re.DOTALL)
    if overall_match:
        overall = overall_match.group(1)
        overall = overall.replace('\\n', '\n').replace('\\"', '"')
        result['analysis'] = {'overall': overall.strip()}

    # Extract answer_points or scoring_points for essays.
    points_match = re.search(r'"(?:answer_points|scoring_points)"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if points_match:
        result['_type'] = 'essay'
        points_text = points_match.group(1)
        # Extract each answer point.
        point_matches = re.findall(r'"point"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)', points_text, re.DOTALL)
        score_matches = re.findall(r'"score"\s*:\s*(\d+)', points_text)

        answer_points = []
        for i, pt in enumerate(point_matches):
            pt_clean = pt.replace('\\n', '\n').replace('\\"', '"').strip()
            score = int(score_matches[i]) if i < len(score_matches) else 0
            answer_points.append({'point': pt_clean, 'score': score})

        if answer_points:
            result['answer_points'] = answer_points

    # Extract total_score.
    total_score_match = re.search(r'"total_score"\s*:\s*(\d+)', text)
    if total_score_match:
        result['total_score'] = int(total_score_match.group(1))

    # Extract explanation for essays.
    explanation_match = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)', text, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1)
        explanation = explanation.replace('\\n', '\n').replace('\\"', '"')
        result['explanation'] = explanation.strip()

    # Return only when at least the stem was recovered.
    if result.get('stem'):
        result['_extracted_via_regex'] = True
        return {'questions': [result]}

    return None


def extract_questions_from_stage1(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract questions and anchor information from Stage1 pipeline_state files.

    This is the primary source for experiments that used Stage1. Anchor
    information is integrated into the extracted question object.

    Returns:
        Question list. Each item may include:
        - unit_id
        - question_type
        - stem
        - options for single-choice questions
        - correct_answer for single-choice questions
        - answer_points for essays
        - total_score for essays
        - explanation
        - anchors
        - anchor_count
    """
    stage1_dir = exp_dir / "stage1"
    if not stage1_dir.exists():
        return []

    questions = []

    for unit_dir in stage1_dir.iterdir():
        if not unit_dir.is_dir() or not unit_dir.name.startswith("unit_"):
            continue

        try:
            unit_id = int(unit_dir.name.replace("unit_", ""))
        except ValueError:
            continue

        # Find pipeline_state files, preferring successful files by modification time.
        pipeline_files = list(unit_dir.glob("pipeline_state_*.json"))
        if not pipeline_files:
            continue

        # Prefer files whose names do not contain "failed".
        success_files = [f for f in pipeline_files if "failed" not in f.name]
        if success_files:
            pipeline_file = max(success_files, key=lambda f: f.stat().st_mtime)
        else:
            # If no successful file exists, use the latest failed file, which may contain partial output.
            pipeline_file = max(pipeline_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(pipeline_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Extract agent4_output, which contains the generated question.
            agent4_output = state.get("agent4_output")
            if not agent4_output:
                continue

            # Extract agent3_output, which contains anchor information.
            agent3_output = state.get("agent3_output", {})
            raw_anchors = agent3_output.get("anchors", [])

            # Format anchor information.
            anchors = []
            for i, anchor in enumerate(raw_anchors, 1):
                anchors.append({
                    "id": f"A{i}",
                    "snippet": anchor.get("snippet", ""),
                    "reason_for_anchor": anchor.get("reason_for_anchor", ""),
                    "loc": anchor.get("loc", ""),
                    "paragraph_idx": anchor.get("paragraph_idx"),
                })

            # Extract question fields.
            question_type = agent4_output.get("question_type", "essay")
            stem = agent4_output.get("stem", "")
            explanation = agent4_output.get("explanation", "")

            # Build question object.
            question = {
                "unit_id": unit_id,
                "question_type": question_type,
                "stem": stem,
                "explanation": explanation,
                "source": "stage1_pipeline",
                "anchors": anchors,
                "anchor_count": len(anchors),
            }

            # Single-choice-specific fields.
            if question_type == "single-choice":
                options = agent4_output.get("options", {})
                if isinstance(options, dict):
                    options_list = [{"label": k, "content": v} for k, v in options.items()]
                else:
                    options_list = options or []
                question["options"] = options_list
                question["correct_answer"] = agent4_output.get("correct_option") or agent4_output.get("correct_answer", "")

                # Extract option-level analysis.
                analysis = agent4_output.get("analysis", {})
                if isinstance(analysis, dict):
                    overall = analysis.get("overall", "")
                    option_analysis = []
                    for opt_label in ["A", "B", "C", "D"]:
                        opt_info = analysis.get(opt_label, {})
                        if isinstance(opt_info, dict):
                            reason = opt_info.get("reason", "")
                            is_correct = opt_info.get("is_correct", False)
                            if reason:
                                marker = "[v]" if is_correct else "[x]"
                                option_analysis.append(f"{opt_label}. {marker} {reason}")
                    if option_analysis:
                        question["explanation"] = overall + "\n\n" + "\n\n".join(option_analysis)
                    else:
                        question["explanation"] = overall

            # Essay-specific fields.
            elif question_type == "essay":
                answer_points = agent4_output.get("answer_points", [])
                formatted_points = []
                total_score = 0
                for pt in answer_points:
                    if isinstance(pt, dict):
                        point_text = pt.get("point", pt.get("content", ""))
                        score = pt.get("score", pt.get("points", 0))
                        evidence_ref = pt.get("evidence_reference", pt.get("anchor_ids", []))
                        formatted_points.append({
                            "point": point_text,
                            "score": score,
                            "evidence_reference": evidence_ref,
                        })
                        total_score += score
                question["answer_points"] = formatted_points
                question["total_score"] = total_score or agent4_output.get("total_score")

            questions.append(question)

        except Exception as e:
            print(f"[WARN] Failed to parse {pipeline_file}: {e}", file=sys.stderr)

    return questions


def extract_questions_from_agent4_logs(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract generated questions from Agent4 log files as a fallback source.

    Returns:
        Question list. Each item may include:
        - unit_id (material_id)
        - question_type
        - stem
        - options for single-choice questions
        - correct_answer for single-choice questions
        - answer_points for essays
        - total_score for essays
        - explanation/analysis
    """
    logs_dir = exp_dir / "logs" / "prompts"
    if not logs_dir.exists():
        return []

    questions = []

    # Process single-choice logs.
    single_choice_log = logs_dir / "Agent4_SingleChoice.jsonl"
    if single_choice_log.exists():
        with open(single_choice_log, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    metadata = record.get("metadata", {})
                    response = record.get("response", "")

                    unit_id = metadata.get("material_id")
                    if not unit_id:
                        print(f"[WARN] Single-choice log line {line_num} is missing material_id", file=sys.stderr)
                        continue

                    # Parse question from the response.
                    resp_json = extract_json_from_response(response)
                    if not resp_json:
                        print(f"[WARN] Failed to parse single-choice line {line_num}, material_id={unit_id}", file=sys.stderr)
                        continue

                    # Report whether regex fallback was used.
                    extracted_via_regex = False
                    q_list = resp_json.get("questions", [resp_json])
                    for q in q_list:
                        if q.get("_extracted_via_regex"):
                            extracted_via_regex = True
                            print(f"[INFO] Single-choice material_id={unit_id} extracted through regex fallback", file=sys.stderr)

                        # Normalize option format.
                        options = q.get("options", {})
                        if isinstance(options, dict):
                            # Convert to list format.
                            options_list = [
                                {"label": k, "content": v}
                                for k, v in options.items()
                            ]
                        else:
                            options_list = options

                        # Extract explanation.
                        analysis = q.get("analysis", {})
                        if isinstance(analysis, dict):
                            explanation = analysis.get("overall", "")
                            # Add option-level analysis.
                            option_analysis = []
                            for opt_label in ["A", "B", "C", "D"]:
                                opt_info = analysis.get(opt_label, {})
                                if isinstance(opt_info, dict):
                                    reason = opt_info.get("reason", "")
                                    is_correct = opt_info.get("is_correct", False)
                                    if reason:
                                        marker = "[v]" if is_correct else "[x]"
                                        option_analysis.append(f"{opt_label}. {marker} {reason}")
                            if option_analysis:
                                explanation += "\n" + "\n".join(option_analysis)
                        else:
                            explanation = str(analysis) if analysis else ""

                        question = {
                            "unit_id": int(unit_id) if str(unit_id).isdigit() else unit_id,
                            "question_type": "single-choice",
                            "stem": q.get("stem", ""),
                            "options": options_list,
                            "correct_answer": q.get("correct_option", q.get("correct_answer", "")),
                            "explanation": explanation,
                            "anchor_ids_used": q.get("anchor_ids_used", []),
                            "source": "agent4_log" + ("_regex" if extracted_via_regex else ""),
                        }
                        questions.append(question)
                        break  # Only use the first question.

                except Exception as e:
                    print(f"[WARN] Failed to parse single-choice log line {line_num}: {e}", file=sys.stderr)

    # Process essay logs.
    essay_log = logs_dir / "Agent4_Essay.jsonl"
    if essay_log.exists():
        with open(essay_log, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    metadata = record.get("metadata", {})
                    response = record.get("response", "")

                    unit_id = metadata.get("material_id")
                    if not unit_id:
                        print(f"[WARN] Essay log line {line_num} is missing material_id", file=sys.stderr)
                        continue

                    resp_json = extract_json_from_response(response)
                    if not resp_json:
                        print(f"[WARN] Failed to parse essay line {line_num}, material_id={unit_id}", file=sys.stderr)
                        continue

                    # Report whether regex fallback was used.
                    extracted_via_regex = False
                    q_list = resp_json.get("questions", [resp_json])
                    for q in q_list:
                        if q.get("_extracted_via_regex"):
                            extracted_via_regex = True
                            print(f"[INFO] Essay material_id={unit_id} extracted through regex fallback", file=sys.stderr)

                        # Normalize answer points.
                        answer_points = q.get("answer_points", q.get("scoring_points", []))
                        if isinstance(answer_points, list):
                            formatted_points = []
                            total_score = 0
                            for pt in answer_points:
                                if isinstance(pt, dict):
                                    formatted_points.append({
                                        "point": pt.get("point", pt.get("content", "")),
                                        "score": pt.get("score", pt.get("points", 0)),
                                        "evidence_reference": pt.get("evidence_reference", pt.get("anchor_ids", [])),
                                    })
                                    total_score += pt.get("score", pt.get("points", 0))
                                else:
                                    formatted_points.append({"point": str(pt), "score": 0})
                        else:
                            formatted_points = []
                            total_score = 0

                        question = {
                            "unit_id": int(unit_id) if str(unit_id).isdigit() else unit_id,
                            "question_type": "essay",
                            "stem": q.get("stem", q.get("question", "")),
                            "answer_points": formatted_points,
                            "total_score": total_score or q.get("total_score"),
                            "explanation": q.get("explanation", q.get("analysis", "")),
                            "anchor_ids_used": q.get("anchor_ids_used", []),
                            "source": "agent4_log" + ("_regex" if extracted_via_regex else ""),
                        }
                        questions.append(question)
                        break

                except Exception as e:
                    print(f"[WARN] Failed to parse essay log line {line_num}: {e}", file=sys.stderr)

    return questions


def extract_anchors_from_stage1(exp_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    Extract anchor information from the Stage1 directory.

    Anchor information is present only for experiments that used Stage1 without Agent2 ablation.

    Returns:
        {unit_id: {
            "anchors": [...],
            "anchor_count": int,
            "anchor_discovery_reasoning": str,
        }}
    """
    stage1_dir = exp_dir / "stage1"
    if not stage1_dir.exists():
        return {}

    anchors_by_unit = {}

    for unit_dir in stage1_dir.iterdir():
        if not unit_dir.is_dir() or not unit_dir.name.startswith("unit_"):
            continue

        try:
            unit_id = int(unit_dir.name.replace("unit_", ""))
        except ValueError:
            continue

        # Find pipeline_state files, preferring the latest successful file.
        pipeline_files = list(unit_dir.glob("pipeline_state_*.json"))
        if not pipeline_files:
            continue

        # Prefer files whose names do not contain "failed", sorted by modification time.
        success_files = [f for f in pipeline_files if "failed" not in f.name]
        if success_files:
            pipeline_file = max(success_files, key=lambda f: f.stat().st_mtime)
        else:
            # If no successful file exists, use the latest failed file.
            pipeline_file = max(pipeline_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(pipeline_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Extract agent3_output from the anchor discovery agent.
            agent3_output = state.get("agent3_output")
            if agent3_output:
                raw_anchors = agent3_output.get("anchors", [])
                if raw_anchors:
                    anchors = []
                    for anchor in raw_anchors:
                        anchors.append({
                            "snippet": anchor.get("snippet", ""),
                            "reason_for_anchor": anchor.get("reason_for_anchor", ""),
                            "loc": anchor.get("loc", ""),
                            "paragraph_idx": anchor.get("paragraph_idx"),
                        })

                    anchors_by_unit[unit_id] = {
                        "anchors": anchors,
                        "anchor_count": len(anchors),
                        "anchor_discovery_reasoning": agent3_output.get("anchor_discovery_reasoning", ""),
                    }

        except Exception as e:
            print(f"[WARN] Failed to parse {pipeline_file}: {e}", file=sys.stderr)

    return anchors_by_unit


def extract_scores_from_stage2(exp_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    Extract scoring information from the Stage2 directory.

    Updated 2025-12: includes dimension-level scoring details.

    Returns:
        {unit_id: {
            "ai_score": float,
            "ped_score": float,
            "final_decision": str,
            "ai_dimensions": [...],
            "ped_dimensions": [...],
        }}
    """
    stage2_dir = exp_dir / "stage2"
    if not stage2_dir.exists():
        return {}

    scores = {}

    for unit_dir in stage2_dir.iterdir():
        if not unit_dir.is_dir() or not unit_dir.name.startswith("unit_"):
            continue

        try:
            unit_id = int(unit_dir.name.replace("unit_", ""))
        except ValueError:
            continue

        eval_file = unit_dir / "evaluation_state.json"

        if not eval_file.exists() or eval_file.stat().st_size < 10:
            continue

        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_state = json.load(f)

            score_info = {
                "final_decision": eval_state.get("final_decision", ""),
                "ai_score": None,
                "ped_score": None,
                "ai_dimensions": [],
                "ped_dimensions": [],
            }

            # Extract AI score and dimension details.
            ai_eval = eval_state.get("ai_eval", {})
            if ai_eval and ai_eval.get("success"):
                ai_result = ai_eval.get("result", {})
                score_info["ai_score"] = ai_result.get("overall_score")

                # Extract AI dimension details.
                ai_dims = ai_result.get("dimension_details", {})
                for dim_id, dim_data in ai_dims.items():
                    if isinstance(dim_data, dict):
                        dim_info = {
                            "dimension_id": dim_id,
                            "score": dim_data.get("aggregated_score"),
                            "level": dim_data.get("level", ""),
                            "weight": dim_data.get("weight", 1.0),
                            "per_model_scores": dim_data.get("per_model_scores", {}),
                        }
                        score_info["ai_dimensions"].append(dim_info)

            # Extract pedagogical score and dimension details.
            ped_eval = eval_state.get("pedagogical_eval", {})
            if ped_eval and ped_eval.get("success"):
                ped_result = ped_eval.get("result") or {}
                if isinstance(ped_result, dict):
                    # Pedagogical score uses f1 or overall_score.
                    score_info["ped_score"] = ped_result.get("overall_score") or ped_result.get("f1")

                    # Extract pedagogical dimension details.
                    ped_dims = ped_result.get("dimension_details", {})
                    for dim_name, dim_data in ped_dims.items():
                        if isinstance(dim_data, dict):
                            dim_info = {
                                "dimension_name": dim_name,
                                "dimension_id": dim_data.get("dimension_id", ""),
                                "score": dim_data.get("aggregated_score"),
                                "hit_level": dim_data.get("hit_level", ""),
                                "reasoning": dim_data.get("reasoning", "")[:200] if dim_data.get("reasoning") else "",
                                "per_model_scores": dim_data.get("per_model_scores", {}),
                            }
                            score_info["ped_dimensions"].append(dim_info)

            # Newer structures store GK/CS evaluation separately. If legacy pedagogical_eval
            # has no result, show the average GK/CS F1 when available.
            if score_info["ped_score"] is None:
                f1_scores = []
                for eval_key in ("gk_eval", "cs_eval"):
                    split_eval = eval_state.get(eval_key, {})
                    if split_eval and split_eval.get("success") and isinstance(split_eval.get("result"), dict):
                        f1 = split_eval["result"].get("f1")
                        if isinstance(f1, (int, float)):
                            f1_scores.append(float(f1))
                if f1_scores:
                    score_info["ped_score"] = sum(f1_scores) / len(f1_scores)

            # Also extract question information from input as a fallback.
            inp = eval_state.get("input", {})
            if inp:
                score_info["_input"] = inp

            scores[unit_id] = score_info

        except Exception as e:
            print(f"[WARN] Failed to parse {eval_file}: {e}", file=sys.stderr)

    return scores


def extract_questions(
    exp_dir: Path,
    unit_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract all questions from an experiment directory.

    Updated 2026-01 source priority:
    1. Stage1 pipeline_state, which includes questions and anchors.
    2. Agent4 logs.
    3. stage2 evaluation_state

    Args:
        exp_dir: Experiment directory path.
        unit_id: Optional unit ID filter.

    Returns:
        Question list sorted by unit_id.
    """
    # Prefer Stage1 extraction because it includes anchor information.
    questions = extract_questions_from_stage1(exp_dir)
    stage1_unit_ids = {q.get("unit_id") for q in questions}

    if questions:
        print(f"[INFO] Extracted {len(questions)} questions from Stage1, including anchors", file=sys.stderr)

    # 2. Add Agent4 log questions that are missing from Stage1.
    agent4_questions = extract_questions_from_agent4_logs(exp_dir)
    for q in agent4_questions:
        if q.get("unit_id") not in stage1_unit_ids:
            questions.append(q)

    # 3. Extract scoring information from Stage2.
    scores = extract_scores_from_stage2(exp_dir)

    # 4. Add Stage2 input questions that are missing from Stage1 and Agent4.
    existing_unit_ids = {q.get("unit_id") for q in questions}
    for uid, score_info in scores.items():
        if uid in existing_unit_ids:
            continue

        inp = score_info.get("_input", {})
        if not inp:
            continue

        question_type = inp.get("question_type")
        if not question_type:
            continue

        q = {
            "unit_id": uid,
            "question_type": question_type,
            "stem": inp.get("stem", ""),
            "explanation": inp.get("explanation", ""),
            "ai_score": score_info.get("ai_score"),
            "ped_score": score_info.get("ped_score"),
            "final_decision": score_info.get("final_decision", ""),
            "source": "stage2_input",
        }

        if question_type == "single-choice":
            q["options"] = inp.get("options", [])
            q["correct_answer"] = inp.get("correct_answer", "")
        elif question_type == "essay":
            q["answer_points"] = inp.get("answer_points", [])
            q["total_score"] = inp.get("total_score")

        questions.append(q)

    # 5. Merge scoring information.
    for q in questions:
        uid = q.get("unit_id")
        if uid in scores:
            score_info = scores[uid]
            q["ai_score"] = score_info.get("ai_score")
            q["ped_score"] = score_info.get("ped_score")
            q["final_decision"] = score_info.get("final_decision", "")
            # Add dimension-level scoring details.
            q["ai_dimensions"] = score_info.get("ai_dimensions", [])
            q["ped_dimensions"] = score_info.get("ped_dimensions", [])

    # 6. Filter the requested unit.
    if unit_id is not None:
        questions = [q for q in questions if q.get("unit_id") == unit_id]

    # 7. Sort by unit_id.
    questions.sort(key=lambda x: x.get("unit_id", 0) or 0)

    return questions


def format_question_text(q: Dict[str, Any], include_scores: bool = True) -> str:
    """Format a question as readable plain text."""
    lines = []

    unit_id = q.get("unit_id", "?")
    qtype = q.get("question_type", "unknown")
    qtype_label = "Single Choice" if qtype == "single-choice" else "Essay" if qtype == "essay" else qtype

    lines.append(f"{'='*70}")
    lines.append(f"Unit {unit_id} - {qtype_label}")

    if include_scores:
        ai_score = q.get("ai_score")
        ped_score = q.get("ped_score")
        decision = q.get("final_decision", "")
        score_info = []
        if ai_score is not None:
            score_info.append(f"AI score={ai_score:.1f}")
        if ped_score is not None:
            score_info.append(f"Pedagogical score={ped_score:.1f}")
        if decision:
            score_info.append(f"Decision={decision}")
        if score_info:
            lines.append(f"Scores: {', '.join(score_info)}")
        else:
            lines.append("Scores: (none)")

    lines.append(f"{'='*70}")

    # Stem.
    lines.append("\nStem")
    lines.append(q.get("stem", "(empty)"))

    # Single-choice-specific fields.
    if qtype == "single-choice":
        options = q.get("options", [])
        if options:
            lines.append("\nOptions")
            for opt in options:
                if isinstance(opt, dict):
                    label = opt.get("label", "?")
                    content = opt.get("content", "")
                    lines.append(f"  {label}. {content}")
                elif isinstance(opt, str):
                    lines.append(f"  {opt}")

        correct = q.get("correct_answer", "")
        if correct:
            lines.append(f"\nCorrect Answer: {correct}")

    # Essay-specific fields.
    elif qtype == "essay":
        answer_points = q.get("answer_points", [])
        total_score = q.get("total_score")

        if answer_points:
            lines.append("\nAnswer Points")
            for i, pt in enumerate(answer_points, 1):
                if isinstance(pt, dict):
                    point = pt.get("point", "")
                    score = pt.get("score", "")
                    evidence = pt.get("evidence_reference", [])
                    lines.append(f"  {i}. {point}")
                    if score:
                        lines.append(f"     (score: {score})")
                    if evidence:
                        lines.append(f"     (reference anchors: {', '.join(evidence)})")
                else:
                    lines.append(f"  {i}. {pt}")

        if total_score:
            lines.append(f"\nTotal Score: {total_score}")

    # Explanation.
    explanation = q.get("explanation", "")
    if explanation:
        lines.append("\nExplanation")
        lines.append(explanation)

    # Anchor information.
    anchors = q.get("anchor_ids_used", [])
    if anchors:
        lines.append(f"\nAnchors Used: {', '.join(anchors)}")

    lines.append("")
    return "\n".join(lines)


def format_question_markdown(q: Dict[str, Any], include_scores: bool = True) -> str:
    """Format a question as Markdown."""
    lines = []

    unit_id = q.get("unit_id", "?")
    qtype = q.get("question_type", "unknown")
    qtype_label = "Single Choice" if qtype == "single-choice" else "Essay" if qtype == "essay" else qtype

    lines.append(f"## Unit {unit_id} - {qtype_label}")

    if include_scores:
        ai_score = q.get("ai_score")
        ped_score = q.get("ped_score")
        decision = q.get("final_decision", "")
        score_parts = []
        if ai_score is not None:
            score_parts.append(f"AI: **{ai_score:.1f}**")
        if ped_score is not None:
            score_parts.append(f"Ped: **{ped_score:.1f}**")
        if decision:
            score_parts.append(f"Decision: {decision}")
        if score_parts:
            lines.append(f"> {' | '.join(score_parts)}")
        else:
            lines.append("> Scores: (none)")

    lines.append("")

    # Stem.
    lines.append("### Stem")
    lines.append(q.get("stem", "(empty)"))
    lines.append("")

    # Single-choice fields.
    if qtype == "single-choice":
        options = q.get("options", [])
        if options:
            lines.append("### Options")
            for opt in options:
                if isinstance(opt, dict):
                    label = opt.get("label", "?")
                    content = opt.get("content", "")
                    lines.append(f"- **{label}.** {content}")
                elif isinstance(opt, str):
                    lines.append(f"- {opt}")
            lines.append("")

        correct = q.get("correct_answer", "")
        if correct:
            lines.append("### Correct Answer")
            lines.append(f"**{correct}**")
            lines.append("")

    # Essay fields.
    elif qtype == "essay":
        answer_points = q.get("answer_points", [])
        total_score = q.get("total_score")

        if answer_points:
            lines.append("### Answer Points")
            for i, pt in enumerate(answer_points, 1):
                if isinstance(pt, dict):
                    point = pt.get("point", "")
                    score = pt.get("score", "")
                    evidence = pt.get("evidence_reference", [])
                    score_str = f" (score: {score})" if score else ""
                    evidence_str = f" [anchors: {', '.join(evidence)}]" if evidence else ""
                    lines.append(f"{i}. {point}{score_str}{evidence_str}")
                else:
                    lines.append(f"{i}. {pt}")
            lines.append("")

        if total_score:
            lines.append(f"**Total Score: {total_score}**")
            lines.append("")

    # Explanation.
    explanation = q.get("explanation", "")
    if explanation:
        lines.append("### Explanation")
        lines.append(explanation)
        lines.append("")

    # Dimension-level scoring details.
    if include_scores:
        ai_dims = q.get("ai_dimensions", [])
        ped_dims = q.get("ped_dimensions", [])

        if ai_dims or ped_dims:
            lines.append("### Dimension Score Details")
            lines.append("")

            if ai_dims:
                lines.append("**AI-Centric Dimensions:**")
                lines.append("")
                lines.append("| Dimension | Score | Level |")
                lines.append("|------|------|------|")
                for dim in ai_dims:
                    dim_id = dim.get("dimension_id", "")
                    score = dim.get("score")
                    level = dim.get("level", "")
                    score_str = f"{score:.1f}" if score is not None else "-"
                    lines.append(f"| {dim_id} | {score_str} | {level} |")
                lines.append("")

            if ped_dims:
                lines.append("**Pedagogical Dimensions:**")
                lines.append("")
                lines.append("| Dimension | Score | Hit Level |")
                lines.append("|------|------|----------|")
                for dim in ped_dims:
                    dim_name = dim.get("dimension_name", "")
                    score = dim.get("score")
                    hit_level = dim.get("hit_level", "")
                    score_str = f"{score:.1f}" if score is not None else "-"
                    lines.append(f"| {dim_name} | {score_str} | {hit_level} |")
                lines.append("")

    # Anchor information.
    anchors = q.get("anchors", [])
    anchor_count = q.get("anchor_count", 0)
    if anchors or anchor_count > 0:
        lines.append("### Evidence Anchors")
        lines.append("")
        lines.append(f"**Anchor count:** {anchor_count}")
        lines.append("")
        if anchors:
            for i, anchor in enumerate(anchors, 1):
                snippet = anchor.get("snippet", "")[:100]
                if len(anchor.get("snippet", "")) > 100:
                    snippet += "..."
                loc = anchor.get("loc", "")
                reason = anchor.get("reason_for_anchor", "")[:150]
                if len(anchor.get("reason_for_anchor", "")) > 150:
                    reason += "..."
                lines.append(f"**Anchor {i}** ({loc})")
                lines.append(f"> {snippet}")
                if reason:
                    lines.append(f"")
                    lines.append(f"*Reason: {reason}*")
                lines.append("")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def print_summary(questions: List[Dict[str, Any]]):
    """Print a summary."""
    total = len(questions)
    single_choice = sum(1 for q in questions if q.get("question_type") == "single-choice")
    essay = sum(1 for q in questions if q.get("question_type") == "essay")

    # Count questions with scores.
    with_ai_score = sum(1 for q in questions if q.get("ai_score") is not None)
    with_ped_score = sum(1 for q in questions if q.get("ped_score") is not None)

    pass_count = sum(1 for q in questions if q.get("final_decision") == "pass")
    reject_count = sum(1 for q in questions if q.get("final_decision") == "reject")
    pending_count = total - pass_count - reject_count

    print(f"\n{'='*60}")
    print("Question Extraction Summary")
    print(f"{'='*60}")
    print(f"  Total questions:             {total}")
    print(f"  - Single-choice questions:   {single_choice}")
    print(f"  - Essay questions:           {essay}")
    print(f"  With AI score:               {with_ai_score}")
    print(f"  With pedagogical score:      {with_ped_score}")
    print("  Evaluation results:")
    print(f"    - Pass:                    {pass_count}")
    print(f"    - Reject:                  {reject_count}")
    print(f"    - Pending:                 {pending_count}")
    print(f"{'='*60}\n")


def auto_extract_and_save(exp_dir: Path, experiment_id: str = "") -> Dict[str, Any]:
    """
    Automatically extract questions and save JSON/Markdown outputs.

    Intended for automatic CLI use at the end of an experiment.

    Args:
        exp_dir: Experiment directory path.
        experiment_id: Experiment ID used in generated metadata.

    Returns:
        {
            "success": bool,
            "total_questions": int,
            "json_path": str,
            "markdown_path": str,
            "error": str (if failed)
        }
    """
    result = {
        "success": False,
        "total_questions": 0,
        "json_path": None,
        "markdown_path": None,
        "error": None,
    }

    try:
        # Extract questions.
        questions = extract_questions(exp_dir)

        if not questions:
            result["error"] = "No questions found"
            return result

        result["total_questions"] = len(questions)

        # Generate filenames.
        base_name = "questions_extracted"

        # Save JSON output.
        json_path = exp_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(questions, ensure_ascii=False, indent=2, fp=f)
        result["json_path"] = str(json_path)

        # Generate Markdown output.
        md_lines = []
        md_lines.append("# Experiment Question Summary")
        md_lines.append("")
        if experiment_id:
            md_lines.append(f"**Experiment ID:** {experiment_id}")
            md_lines.append("")

        # Summary statistics.
        single_choice_count = sum(1 for q in questions if q.get("question_type") == "single-choice")
        essay_count = sum(1 for q in questions if q.get("question_type") == "essay")
        pass_count = sum(1 for q in questions if q.get("final_decision") == "pass")
        reject_count = sum(1 for q in questions if q.get("final_decision") == "reject")

        # Compute average scores.
        ai_scores = [q.get("ai_score") for q in questions if q.get("ai_score") is not None]
        ped_scores = [q.get("ped_score") for q in questions if q.get("ped_score") is not None]
        avg_ai = sum(ai_scores) / len(ai_scores) if ai_scores else 0
        avg_ped = sum(ped_scores) / len(ped_scores) if ped_scores else 0

        md_lines.append("## Summary")
        md_lines.append("")
        md_lines.append(f"- **Total questions:** {len(questions)}")
        md_lines.append(f"  - Single-choice: {single_choice_count}")
        md_lines.append(f"  - Essay: {essay_count}")
        md_lines.append("- **Evaluation results:**")
        md_lines.append(f"  - Pass: {pass_count}")
        md_lines.append(f"  - Reject: {reject_count}")
        md_lines.append(f"  - Pending: {len(questions) - pass_count - reject_count}")
        md_lines.append("- **Average scores:**")
        md_lines.append(f"  - AI score: {avg_ai:.1f}")
        md_lines.append(f"  - Pedagogical score: {avg_ped:.1f}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # Add each question.
        md_lines.append("## Questions")
        md_lines.append("")

        for q in questions:
            md_lines.append(format_question_markdown(q, include_scores=True))

        # Save Markdown output.
        md_path = exp_dir / f"{base_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        result["markdown_path"] = str(md_path)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract generated questions from an experiment directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and display all questions
  python scripts/extract_questions.py outputs/EXP_BASELINE_gk+cs_C_20251209_213427

  # Extract from Agent4 logs even when Stage2 evaluation is incomplete
  python scripts/extract_questions.py outputs/ROUND_20251209_A_deepseek/subset40_stratified_seed42_gkcs_A_20251209_213438

  # Extract only unit 10
  python scripts/extract_questions.py outputs/EXP_xxx --unit-id 10

  # Export JSON
  python scripts/extract_questions.py outputs/EXP_xxx --output questions.json

  # Export Markdown
  python scripts/extract_questions.py outputs/EXP_xxx --format markdown --output questions.md
        """
    )

    parser.add_argument("exp_dir", type=str, help="Experiment directory path")
    parser.add_argument("--unit-id", type=int, help="Extract only the specified unit")
    parser.add_argument("--output", "-o", type=str, help="Output file path; prints to terminal when omitted")
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format: text (default), markdown, or json"
    )
    parser.add_argument("--no-scores", action="store_true", help="Hide scoring information")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode; do not print summary statistics")

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: directory does not exist - {exp_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        questions = extract_questions(exp_dir, args.unit_id)

        if not questions:
            print("No questions found")
            print("Hint: check that logs/prompts/Agent4_*.jsonl or stage2/unit_*/evaluation_state.json exists")
            sys.exit(0)

        if not args.quiet:
            print_summary(questions)

        # Format output.
        include_scores = not args.no_scores

        if args.format == "json":
            output = json.dumps(questions, ensure_ascii=False, indent=2)
        elif args.format == "markdown":
            output = "\n".join(format_question_markdown(q, include_scores) for q in questions)
        else:  # text
            output = "\n".join(format_question_text(q, include_scores) for q in questions)

        # Emit output.
        if args.output:
            output_path = Path(args.output)
            # Ensure the output directory exists.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"[SUCCESS] Saved to: {output_path}")
        else:
            print(output)

    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
