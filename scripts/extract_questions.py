#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取实验中生成的题目

【数据来源】
1. 优先从 logs/prompts/Agent4_*.jsonl 提取（包含所有生成的题目）
2. 从 stage2/unit_*/evaluation_state.json 补充评分信息

【2025-12 更新】
- 支持从 Agent4 日志提取题目（即使 evaluation_state.json 为空或不完整）
- 输出包含：unit_id, 题型, 题干, 选项/答案要点, 正确答案, 解析, AI评分, 教育学评分

用法:
    python scripts/extract_questions.py <实验目录> [选项]

示例:
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
    从 LLM 响应中提取 JSON
    支持 ```json ... ``` 格式和直接 JSON

    【2025-12 增强】
    当标准JSON解析失败时（如LLM输出被截断导致字符串未闭合），
    使用正则表达式提取关键字段作为回退方案
    """
    if not response:
        return None

    json_str = None

    # 尝试提取 ```json ... ``` 格式
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试找到 JSON 对象
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 【回退方案】使用正则提取关键字段
    # 当JSON因截断而无法解析时，尝试提取必要字段
    text_to_parse = json_str or response
    if text_to_parse:
        extracted = _extract_fields_via_regex(text_to_parse)
        if extracted:
            return extracted

    return None


def _extract_fields_via_regex(text: str) -> Optional[Dict]:
    """
    使用正则表达式从损坏/截断的JSON中提取关键字段

    支持选择题和主观题两种格式
    """
    result = {}

    # 提取 stem（题干）- 支持多种格式
    stem_patterns = [
        r'"stem"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)',  # 标准格式，允许未闭合
        r'"question"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)',  # 备选字段名
    ]
    for pattern in stem_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            stem = match.group(1)
            # 清理转义字符
            stem = stem.replace('\\n', '\n').replace('\\"', '"')
            result['stem'] = stem.strip()
            break

    # 提取 correct_option（选择题正确答案）
    correct_match = re.search(r'"correct_option"\s*:\s*"([A-D])"', text)
    if correct_match:
        result['correct_option'] = correct_match.group(1)
        result['_type'] = 'single-choice'

    # 提取选项（选择题）
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

    # 提取 anchor_ids_used
    anchors_match = re.search(r'"anchor_ids_used"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if anchors_match:
        anchors_text = anchors_match.group(1)
        anchors = re.findall(r'"([^"]+)"', anchors_text)
        result['anchor_ids_used'] = anchors

    # 提取 overall analysis（解析）
    overall_match = re.search(r'"overall"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)', text, re.DOTALL)
    if overall_match:
        overall = overall_match.group(1)
        overall = overall.replace('\\n', '\n').replace('\\"', '"')
        result['analysis'] = {'overall': overall.strip()}

    # 提取主观题的 answer_points / scoring_points
    points_match = re.search(r'"(?:answer_points|scoring_points)"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if points_match:
        result['_type'] = 'essay'
        points_text = points_match.group(1)
        # 简单提取每个要点
        point_matches = re.findall(r'"point"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)', points_text, re.DOTALL)
        score_matches = re.findall(r'"score"\s*:\s*(\d+)', points_text)

        answer_points = []
        for i, pt in enumerate(point_matches):
            pt_clean = pt.replace('\\n', '\n').replace('\\"', '"').strip()
            score = int(score_matches[i]) if i < len(score_matches) else 0
            answer_points.append({'point': pt_clean, 'score': score})

        if answer_points:
            result['answer_points'] = answer_points

    # 提取 total_score
    total_score_match = re.search(r'"total_score"\s*:\s*(\d+)', text)
    if total_score_match:
        result['total_score'] = int(total_score_match.group(1))

    # 提取 explanation（主观题解析）
    explanation_match = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)', text, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1)
        explanation = explanation.replace('\\n', '\n').replace('\\"', '"')
        result['explanation'] = explanation.strip()

    # 如果提取到了题干，返回结果；否则返回 None
    if result.get('stem'):
        result['_extracted_via_regex'] = True
        return {'questions': [result]}

    return None


def extract_questions_from_stage1(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    【2026-01 新增】从 stage1 的 pipeline_state 中提取题目和锚点信息

    这是主要的题目提取来源，适用于使用了 Stage1 的实验。
    锚点信息会直接整合到题目的解析中。

    Returns:
        题目列表，每个题目包含:
        - unit_id
        - question_type
        - stem
        - options (选择题)
        - correct_answer (选择题)
        - answer_points (主观题)
        - total_score (主观题)
        - explanation
        - anchors (锚点列表)
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

        # 查找 pipeline_state 文件（优先选择成功的，按修改时间排序）
        pipeline_files = list(unit_dir.glob("pipeline_state_*.json"))
        if not pipeline_files:
            continue

        # 优先选择不包含 "failed" 的文件
        success_files = [f for f in pipeline_files if "failed" not in f.name]
        if success_files:
            pipeline_file = max(success_files, key=lambda f: f.stat().st_mtime)
        else:
            # 如果没有成功的，取最新的失败文件（可能包含部分生成的题目）
            pipeline_file = max(pipeline_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(pipeline_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 提取 agent4_output（生成的题目）
            agent4_output = state.get("agent4_output")
            if not agent4_output:
                continue

            # 提取 agent3_output（锚点信息）
            agent3_output = state.get("agent3_output", {})
            raw_anchors = agent3_output.get("anchors", [])

            # 格式化锚点信息
            anchors = []
            for i, anchor in enumerate(raw_anchors, 1):
                anchors.append({
                    "id": f"A{i}",
                    "snippet": anchor.get("snippet", ""),
                    "reason_for_anchor": anchor.get("reason_for_anchor", ""),
                    "loc": anchor.get("loc", ""),
                    "paragraph_idx": anchor.get("paragraph_idx"),
                })

            # 提取题目信息
            question_type = agent4_output.get("question_type", "essay")
            stem = agent4_output.get("stem", "")
            explanation = agent4_output.get("explanation", "")

            # 构建题目对象
            question = {
                "unit_id": unit_id,
                "question_type": question_type,
                "stem": stem,
                "explanation": explanation,
                "source": "stage1_pipeline",
                "anchors": anchors,
                "anchor_count": len(anchors),
            }

            # 选择题特有字段
            if question_type == "single-choice":
                options = agent4_output.get("options", {})
                if isinstance(options, dict):
                    options_list = [{"label": k, "content": v} for k, v in options.items()]
                else:
                    options_list = options or []
                question["options"] = options_list
                question["correct_answer"] = agent4_output.get("correct_option") or agent4_output.get("correct_answer", "")

                # 提取选项解析
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
                                marker = "✓" if is_correct else "✗"
                                option_analysis.append(f"【{opt_label}】{marker} {reason}")
                    if option_analysis:
                        question["explanation"] = overall + "\n\n" + "\n\n".join(option_analysis)
                    else:
                        question["explanation"] = overall

            # 主观题特有字段
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
            print(f"[警告] 解析 {pipeline_file} 失败: {e}", file=sys.stderr)

    return questions


def extract_questions_from_agent4_logs(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    从 Agent4 日志文件提取生成的题目（备选来源）

    Returns:
        题目列表，每个题目包含:
        - unit_id (material_id)
        - question_type
        - stem
        - options (选择题)
        - correct_answer (选择题)
        - answer_points (主观题)
        - total_score (主观题)
        - explanation/analysis
    """
    logs_dir = exp_dir / "logs" / "prompts"
    if not logs_dir.exists():
        return []

    questions = []

    # 处理选择题日志
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
                        print(f"[警告] 选择题第 {line_num} 行缺少 material_id", file=sys.stderr)
                        continue

                    # 解析响应中的题目
                    resp_json = extract_json_from_response(response)
                    if not resp_json:
                        print(f"[警告] 选择题第 {line_num} 行解析失败, material_id={unit_id}", file=sys.stderr)
                        continue

                    # 检查是否通过正则提取（用于调试）
                    extracted_via_regex = False
                    q_list = resp_json.get("questions", [resp_json])
                    for q in q_list:
                        if q.get("_extracted_via_regex"):
                            extracted_via_regex = True
                            print(f"[信息] 选择题 material_id={unit_id} 通过正则回退提取", file=sys.stderr)

                        # 处理选项格式
                        options = q.get("options", {})
                        if isinstance(options, dict):
                            # 转换为列表格式
                            options_list = [
                                {"label": k, "content": v}
                                for k, v in options.items()
                            ]
                        else:
                            options_list = options

                        # 提取解析
                        analysis = q.get("analysis", {})
                        if isinstance(analysis, dict):
                            explanation = analysis.get("overall", "")
                            # 添加各选项的解析
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
                        break  # 只取第一道题

                except Exception as e:
                    print(f"[警告] 解析选择题日志失败 (第 {line_num} 行): {e}", file=sys.stderr)

    # 处理主观题日志
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
                        print(f"[警告] 主观题第 {line_num} 行缺少 material_id", file=sys.stderr)
                        continue

                    resp_json = extract_json_from_response(response)
                    if not resp_json:
                        print(f"[警告] 主观题第 {line_num} 行解析失败, material_id={unit_id}", file=sys.stderr)
                        continue

                    # 检查是否通过正则提取（用于调试）
                    extracted_via_regex = False
                    q_list = resp_json.get("questions", [resp_json])
                    for q in q_list:
                        if q.get("_extracted_via_regex"):
                            extracted_via_regex = True
                            print(f"[信息] 主观题 material_id={unit_id} 通过正则回退提取", file=sys.stderr)

                        # 处理答案要点
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
                    print(f"[警告] 解析主观题日志失败 (第 {line_num} 行): {e}", file=sys.stderr)

    return questions


def extract_anchors_from_stage1(exp_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    【2026-01 新增】从 stage1 目录提取锚点信息

    只有使用了 Stage1 且没有做 agent2 消融的实验才会有锚点信息

    Returns:
        {unit_id: {
            "anchors": [...],  # 锚点列表
            "anchor_count": int,  # 锚点数量
            "anchor_discovery_reasoning": str,  # 锚点发现理由
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

        # 查找 pipeline_state 文件（可能有多个，取最新的成功的）
        pipeline_files = list(unit_dir.glob("pipeline_state_*.json"))
        if not pipeline_files:
            continue

        # 优先选择不包含 "failed" 的文件，按修改时间排序
        success_files = [f for f in pipeline_files if "failed" not in f.name]
        if success_files:
            pipeline_file = max(success_files, key=lambda f: f.stat().st_mtime)
        else:
            # 如果没有成功的，取最新的失败文件
            pipeline_file = max(pipeline_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(pipeline_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 提取 agent3_output（锚点发现器输出）
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
            print(f"[警告] 解析 {pipeline_file} 失败: {e}", file=sys.stderr)

    return anchors_by_unit


def extract_scores_from_stage2(exp_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    从 stage2 目录提取评分信息

    【2025-12 增强】添加维度评分明细提取

    Returns:
        {unit_id: {
            "ai_score": float,
            "ped_score": float,
            "final_decision": str,
            "ai_dimensions": [...],  # AI维度评分明细
            "ped_dimensions": [...], # 教育学维度评分明细
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

            # 提取 AI 评分及维度明细
            ai_eval = eval_state.get("ai_eval", {})
            if ai_eval and ai_eval.get("success"):
                ai_result = ai_eval.get("result", {})
                score_info["ai_score"] = ai_result.get("overall_score")

                # 提取AI维度明细
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

            # 提取教育学评分及维度明细
            ped_eval = eval_state.get("pedagogical_eval", {})
            if ped_eval and ped_eval.get("success"):
                ped_result = ped_eval.get("result", {})
                # 教育学评分使用 f1 或 overall_score
                score_info["ped_score"] = ped_result.get("overall_score") or ped_result.get("f1")

                # 提取教育学维度明细
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

            # 也尝试从 input 中提取题目信息（作为备选）
            inp = eval_state.get("input", {})
            if inp:
                score_info["_input"] = inp

            scores[unit_id] = score_info

        except Exception as e:
            print(f"[警告] 解析 {eval_file} 失败: {e}", file=sys.stderr)

    return scores


def extract_questions(
    exp_dir: Path,
    unit_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    从实验目录提取所有题目

    【2026-01 更新】优先级：
    1. stage1 pipeline_state（包含题目和锚点，最完整）
    2. Agent4 日志
    3. stage2 evaluation_state

    Args:
        exp_dir: 实验目录路径
        unit_id: 可选，只提取指定 unit 的题目

    Returns:
        题目列表，按 unit_id 排序
    """
    # 【2026-01 更新】优先从 stage1 提取题目（包含锚点信息）
    questions = extract_questions_from_stage1(exp_dir)
    stage1_unit_ids = {q.get("unit_id") for q in questions}

    if questions:
        print(f"[信息] 从 stage1 提取了 {len(questions)} 道题目（包含锚点）", file=sys.stderr)

    # 2. 从 Agent4 日志补充（stage1 中没有的题目）
    agent4_questions = extract_questions_from_agent4_logs(exp_dir)
    for q in agent4_questions:
        if q.get("unit_id") not in stage1_unit_ids:
            questions.append(q)

    # 3. 从 stage2 提取评分信息
    scores = extract_scores_from_stage2(exp_dir)

    # 4. 补充从 stage2 的 input 中提取（stage1 和 agent4 都没有的题目）
    existing_unit_ids = {q.get("unit_id") for q in questions}
    for uid, score_info in scores.items():
        if uid in existing_unit_ids:
            continue  # 已有该 unit 的题目，跳过

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

    # 5. 合并评分信息
    for q in questions:
        uid = q.get("unit_id")
        if uid in scores:
            score_info = scores[uid]
            q["ai_score"] = score_info.get("ai_score")
            q["ped_score"] = score_info.get("ped_score")
            q["final_decision"] = score_info.get("final_decision", "")
            # 【2025-12 增强】添加维度评分明细
            q["ai_dimensions"] = score_info.get("ai_dimensions", [])
            q["ped_dimensions"] = score_info.get("ped_dimensions", [])

    # 6. 过滤指定 unit
    if unit_id is not None:
        questions = [q for q in questions if q.get("unit_id") == unit_id]

    # 7. 按 unit_id 排序
    questions.sort(key=lambda x: x.get("unit_id", 0) or 0)

    return questions


def format_question_text(q: Dict[str, Any], include_scores: bool = True) -> str:
    """将题目格式化为可读文本"""
    lines = []

    unit_id = q.get("unit_id", "?")
    qtype = q.get("question_type", "unknown")
    qtype_cn = "选择题" if qtype == "single-choice" else "主观题" if qtype == "essay" else qtype

    lines.append(f"{'='*70}")
    lines.append(f"【Unit {unit_id}】{qtype_cn}")

    if include_scores:
        ai_score = q.get("ai_score")
        ped_score = q.get("ped_score")
        decision = q.get("final_decision", "")
        score_info = []
        if ai_score is not None:
            score_info.append(f"AI评分={ai_score:.1f}")
        if ped_score is not None:
            score_info.append(f"教育学评分={ped_score:.1f}")
        if decision:
            score_info.append(f"决策={decision}")
        if score_info:
            lines.append(f"【评分】{', '.join(score_info)}")
        else:
            lines.append(f"【评分】(暂无)")

    lines.append(f"{'='*70}")

    # 题干
    lines.append(f"\n【题干】")
    lines.append(q.get("stem", "(无)"))

    # 选择题特有
    if qtype == "single-choice":
        options = q.get("options", [])
        if options:
            lines.append(f"\n【选项】")
            for opt in options:
                if isinstance(opt, dict):
                    label = opt.get("label", "?")
                    content = opt.get("content", "")
                    lines.append(f"  {label}. {content}")
                elif isinstance(opt, str):
                    lines.append(f"  {opt}")

        correct = q.get("correct_answer", "")
        if correct:
            lines.append(f"\n【正确答案】{correct}")

    # 主观题特有
    elif qtype == "essay":
        answer_points = q.get("answer_points", [])
        total_score = q.get("total_score")

        if answer_points:
            lines.append(f"\n【答案要点】")
            for i, pt in enumerate(answer_points, 1):
                if isinstance(pt, dict):
                    point = pt.get("point", "")
                    score = pt.get("score", "")
                    evidence = pt.get("evidence_reference", [])
                    lines.append(f"  {i}. {point}")
                    if score:
                        lines.append(f"     (分值: {score}分)")
                    if evidence:
                        lines.append(f"     (参考锚点: {', '.join(evidence)})")
                else:
                    lines.append(f"  {i}. {pt}")

        if total_score:
            lines.append(f"\n【总分】{total_score}分")

    # 解析
    explanation = q.get("explanation", "")
    if explanation:
        lines.append(f"\n【解析】")
        lines.append(explanation)

    # 锚点信息
    anchors = q.get("anchor_ids_used", [])
    if anchors:
        lines.append(f"\n【使用锚点】{', '.join(anchors)}")

    lines.append("")
    return "\n".join(lines)


def format_question_markdown(q: Dict[str, Any], include_scores: bool = True) -> str:
    """将题目格式化为 Markdown"""
    lines = []

    unit_id = q.get("unit_id", "?")
    qtype = q.get("question_type", "unknown")
    qtype_cn = "选择题" if qtype == "single-choice" else "主观题" if qtype == "essay" else qtype

    lines.append(f"## Unit {unit_id} - {qtype_cn}")

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
            emoji = "✅" if decision == "pass" else "❌" if decision == "reject" else "⏳"
            score_parts.append(f"Decision: {emoji} {decision}")
        if score_parts:
            lines.append(f"> {' | '.join(score_parts)}")
        else:
            lines.append("> 评分: (暂无)")

    lines.append("")

    # 题干
    lines.append("### 题干")
    lines.append(q.get("stem", "(无)"))
    lines.append("")

    # 选择题
    if qtype == "single-choice":
        options = q.get("options", [])
        if options:
            lines.append("### 选项")
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
            lines.append(f"### 正确答案")
            lines.append(f"**{correct}**")
            lines.append("")

    # 主观题
    elif qtype == "essay":
        answer_points = q.get("answer_points", [])
        total_score = q.get("total_score")

        if answer_points:
            lines.append("### 答案要点")
            for i, pt in enumerate(answer_points, 1):
                if isinstance(pt, dict):
                    point = pt.get("point", "")
                    score = pt.get("score", "")
                    evidence = pt.get("evidence_reference", [])
                    score_str = f" ({score}分)" if score else ""
                    evidence_str = f" [锚点: {', '.join(evidence)}]" if evidence else ""
                    lines.append(f"{i}. {point}{score_str}{evidence_str}")
                else:
                    lines.append(f"{i}. {pt}")
            lines.append("")

        if total_score:
            lines.append(f"**总分: {total_score}分**")
            lines.append("")

    # 解析
    explanation = q.get("explanation", "")
    if explanation:
        lines.append("### 解析")
        lines.append(explanation)
        lines.append("")

    # 【2025-12 增强】维度评分明细
    if include_scores:
        ai_dims = q.get("ai_dimensions", [])
        ped_dims = q.get("ped_dimensions", [])

        if ai_dims or ped_dims:
            lines.append("### 维度评分明细")
            lines.append("")

            if ai_dims:
                lines.append("**AI角度评估维度:**")
                lines.append("")
                lines.append("| 维度 | 得分 | 等级 |")
                lines.append("|------|------|------|")
                for dim in ai_dims:
                    dim_id = dim.get("dimension_id", "")
                    score = dim.get("score")
                    level = dim.get("level", "")
                    score_str = f"{score:.1f}" if score is not None else "-"
                    lines.append(f"| {dim_id} | {score_str} | {level} |")
                lines.append("")

            if ped_dims:
                lines.append("**教育学角度评估维度:**")
                lines.append("")
                lines.append("| 维度 | 得分 | 命中等级 |")
                lines.append("|------|------|----------|")
                for dim in ped_dims:
                    dim_name = dim.get("dimension_name", "")
                    score = dim.get("score")
                    hit_level = dim.get("hit_level", "")
                    score_str = f"{score:.1f}" if score is not None else "-"
                    lines.append(f"| {dim_name} | {score_str} | {hit_level} |")
                lines.append("")

    # 【2026-01 新增】锚点信息
    anchors = q.get("anchors", [])
    anchor_count = q.get("anchor_count", 0)
    if anchors or anchor_count > 0:
        lines.append("### 证据锚点")
        lines.append("")
        lines.append(f"**锚点数量:** {anchor_count}")
        lines.append("")
        if anchors:
            for i, anchor in enumerate(anchors, 1):
                snippet = anchor.get("snippet", "")[:100]  # 截断显示
                if len(anchor.get("snippet", "")) > 100:
                    snippet += "..."
                loc = anchor.get("loc", "")
                reason = anchor.get("reason_for_anchor", "")[:150]
                if len(anchor.get("reason_for_anchor", "")) > 150:
                    reason += "..."
                lines.append(f"**锚点 {i}** ({loc})")
                lines.append(f"> {snippet}")
                if reason:
                    lines.append(f"")
                    lines.append(f"*原因: {reason}*")
                lines.append("")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def print_summary(questions: List[Dict[str, Any]]):
    """打印统计摘要"""
    total = len(questions)
    single_choice = sum(1 for q in questions if q.get("question_type") == "single-choice")
    essay = sum(1 for q in questions if q.get("question_type") == "essay")

    # 统计有评分的题目
    with_ai_score = sum(1 for q in questions if q.get("ai_score") is not None)
    with_ped_score = sum(1 for q in questions if q.get("ped_score") is not None)

    pass_count = sum(1 for q in questions if q.get("final_decision") == "pass")
    reject_count = sum(1 for q in questions if q.get("final_decision") == "reject")
    pending_count = total - pass_count - reject_count

    print(f"\n{'='*60}")
    print(f"题目提取统计")
    print(f"{'='*60}")
    print(f"  总题数:       {total}")
    print(f"  - 选择题:     {single_choice}")
    print(f"  - 主观题:     {essay}")
    print(f"  有AI评分:     {with_ai_score}")
    print(f"  有教育学评分: {with_ped_score}")
    print(f"  评估结果:")
    print(f"    - 通过:     {pass_count}")
    print(f"    - 拒绝:     {reject_count}")
    print(f"    - 待评估:   {pending_count}")
    print(f"{'='*60}\n")


def auto_extract_and_save(exp_dir: Path, experiment_id: str = "") -> Dict[str, Any]:
    """
    【2025-12 新增】自动提取题目并保存为JSON和Markdown格式

    供CLI在实验结束时自动调用

    Args:
        exp_dir: 实验目录路径
        experiment_id: 实验ID（用于文件命名）

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
        # 提取题目
        questions = extract_questions(exp_dir)

        if not questions:
            result["error"] = "未找到任何题目"
            return result

        result["total_questions"] = len(questions)

        # 生成文件名
        base_name = "questions_extracted"

        # 保存JSON格式
        json_path = exp_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(questions, ensure_ascii=False, indent=2, fp=f)
        result["json_path"] = str(json_path)

        # 生成Markdown内容
        md_lines = []
        md_lines.append(f"# 实验题目汇总")
        md_lines.append("")
        if experiment_id:
            md_lines.append(f"**实验ID:** {experiment_id}")
            md_lines.append("")

        # 统计信息
        single_choice_count = sum(1 for q in questions if q.get("question_type") == "single-choice")
        essay_count = sum(1 for q in questions if q.get("question_type") == "essay")
        pass_count = sum(1 for q in questions if q.get("final_decision") == "pass")
        reject_count = sum(1 for q in questions if q.get("final_decision") == "reject")

        # 计算平均分
        ai_scores = [q.get("ai_score") for q in questions if q.get("ai_score") is not None]
        ped_scores = [q.get("ped_score") for q in questions if q.get("ped_score") is not None]
        avg_ai = sum(ai_scores) / len(ai_scores) if ai_scores else 0
        avg_ped = sum(ped_scores) / len(ped_scores) if ped_scores else 0

        md_lines.append("## 统计概览")
        md_lines.append("")
        md_lines.append(f"- **总题数:** {len(questions)}")
        md_lines.append(f"  - 选择题: {single_choice_count}")
        md_lines.append(f"  - 主观题: {essay_count}")
        md_lines.append(f"- **评估结果:**")
        md_lines.append(f"  - 通过: {pass_count}")
        md_lines.append(f"  - 拒绝: {reject_count}")
        md_lines.append(f"  - 待评估: {len(questions) - pass_count - reject_count}")
        md_lines.append(f"- **平均分:**")
        md_lines.append(f"  - AI评分: {avg_ai:.1f}")
        md_lines.append(f"  - 教育学评分: {avg_ped:.1f}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # 添加每道题目
        md_lines.append("## 题目列表")
        md_lines.append("")

        for q in questions:
            md_lines.append(format_question_markdown(q, include_scores=True))

        # 保存Markdown格式
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
        description="从实验目录提取生成的题目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取所有题目并显示
  python scripts/extract_questions.py outputs/EXP_BASELINE_gk+cs_C_20251209_213427

  # 从含 Agent4 日志的实验提取（即使 stage2 评估未完成）
  python scripts/extract_questions.py outputs/ROUND_20251209_A_deepseek/subset40_stratified_seed42_gkcs_A_20251209_213438

  # 只提取 unit 10 的题目
  python scripts/extract_questions.py outputs/EXP_xxx --unit-id 10

  # 导出为 JSON（方便后续处理）
  python scripts/extract_questions.py outputs/EXP_xxx --output questions.json

  # 导出为 Markdown
  python scripts/extract_questions.py outputs/EXP_xxx --format markdown --output questions.md
        """
    )

    parser.add_argument("exp_dir", type=str, help="实验目录路径")
    parser.add_argument("--unit-id", type=int, help="只提取指定 unit 的题目")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径（不指定则打印到终端）")
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "markdown", "json"],
        default="text",
        help="输出格式: text(默认), markdown, json"
    )
    parser.add_argument("--no-scores", action="store_true", help="不显示评分信息")
    parser.add_argument("--quiet", "-q", action="store_true", help="安静模式，不打印统计信息")

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"错误: 目录不存在 - {exp_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        questions = extract_questions(exp_dir, args.unit_id)

        if not questions:
            print("未找到任何题目")
            print("提示: 请确认实验目录下存在 logs/prompts/Agent4_*.jsonl 或 stage2/unit_*/evaluation_state.json")
            sys.exit(0)

        if not args.quiet:
            print_summary(questions)

        # 格式化输出
        include_scores = not args.no_scores

        if args.format == "json":
            output = json.dumps(questions, ensure_ascii=False, indent=2)
        elif args.format == "markdown":
            output = "\n".join(format_question_markdown(q, include_scores) for q in questions)
        else:  # text
            output = "\n".join(format_question_text(q, include_scores) for q in questions)

        # 输出
        if args.output:
            output_path = Path(args.output)
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"[SUCCESS] 已保存到: {output_path}")
        else:
            print(output)

    except Exception as e:
        print(f"[ERROR] 提取失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
