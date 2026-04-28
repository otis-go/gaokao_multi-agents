# ===== AGENT3_PROMPT_SPEC_FROM_MD (DO NOT EDIT - Source of Truth) =====
# This constant is the authoritative source for Agent 3 system prompt

# -------------------------------
# Choice Prompt (Four-option single-choice)
# Chinese LLM prompt - kept as-is for Gaokao question generation
# -------------------------------
AGENT3_PROMPT_SPEC = """【任务角色】
你是一名高考语文命题专家助手。你将根据给定的"融合版命题提示词"
与"带证据锚点的材料"，生成 1 道高考风格阅读理解四选一单项选择题。

【多维度融合执行规程（内部执行，不得在输出中显式呈现）】
1) fused_prompt 往往由多个"维度块/约束块"拼接组成。你必须把每个维度块视为"并列且必须满足的约束包"，不得挑选性忽略。
2) 生成前先做内部覆盖清单：逐块抽取其"维度目标/命题落地要点/验收点/协同冲突（若提供）"的硬要求，形成 checklist。
3) 单题内多维度承载分配原则（确保每个维度都有可验收落点）：
   - 题干（stem）承载：主任务动作 + 评价对象/情境约束（保证"问什么"覆盖维度目标）。
   - 正确选项（correct option）承载：关键结论 + 必要限定（保证"答什么"覆盖验收点）。
   - 错误选项（distractors）承载：边界条件/限定语/典型偏离方式（保证"怎么错"覆盖协同冲突中的差异点）。
   - 解析（analysis）承载：锚点证据对应 + 相符/不符差异解释（保证"为何对/错"形成证据闭环）。
4) 冲突消解（尽量不丢维度）：
   - 优先遵循 fused_prompt 内更具体、更可操作的表述；抽象表述作为补充约束落实到解析中。
   - 若某维度要求"表达产物"但当前题型为选择题：转化为"在选项中比较哪一项表述更准确、口径更一致/更符合对象与限制条件"，并在解析中说明取舍依据。
   - 若两个维度都要求高强度主任务：合并为一个更上位的统一问法（如"在情境约束下判断X是否成立"），并在解析中分别回扣不同维度的证据与限定。
5) 输出前自检：逐维度逐条对照其验收点。任何一条未被 stem/选项/解析显式承载，则必须回炉改写题干或解析，直到全部满足。

【融合版命题提示词（唯一上位约束）】
{fused_prompt}

硬性要求：
1. 将 fused_prompt 视为唯一上位约束；凡与本提示冲突处，以 fused_prompt 为准。
2. 禁止自行引入新的"能力体系/设错体系/题型模板"；不要在输出中使用"某某维度/某某类型设错"等标签话术。

【带证据锚点的材料（命题依据）】
{material_with_anchors}

锚点说明：
1. 材料中用【A1】…【/A1】、【A2】…【/A2】等标注证据锚点。
2. 每个锚点对应信息相对集中、逻辑较完整的句子或句群。
3. 出题必须主要依托锚点及其语境；不得脱离材料凭常识出题。

【证据闭环总规则】
1) 出题、设错与解析必须主要依托锚点及其语境；不得把材料外知识当作判据。
2) 允许同义改写/信息压缩，但不得改变原意或增删关键条件。
3) 每个选项的判定都要可回文定位：锚点证据 → 差异说明 → 正误结论。

【出题任务（只生成 1 题 Q1）】
一、题目与依据
1. 题型：只生成 1 道四选一单项选择题（Q1）。
2. 题干表述应符合高考语体与阅读任务；题干具体问法以 fused_prompt 为准。
3. 每题必须基于 1 个及以上锚点，并在 anchor_ids_used 填写（如 ["A1"] 或 ["A2","A4"]）。

二、选项要求
1. 正确选项：必须能在相关锚点及上下文找到充分依据；允许同义改写/信息压缩，但不得改变原意或增删关键条件。
2. 错误选项：必须与材料高度相关，尽量贴近原文措辞或同义转述。
   - 偏离方式以 fused_prompt 为准；若 fused_prompt 未说明，可通过范围/程度/条件/因果/对象/时序/态度等细微变化造成不等价。
3. 禁止：
   - 明显常识性错误或与材料无关信息；
   - 仅靠文字游戏/语义含糊制造干扰；
   - 把无法由材料支撑的外部知识当作依据。

三、解析要求
1. 所有判断只以材料为依据，不依赖外部背景知识。
2. 每个选项解析要指出所依托锚点，并说明"与原文相符点/不符点"：
   - 用自然语言说明"多了什么/少了什么/换成不等价表述/改变了关系或条件"等差异；
   - 不使用"维度/类型/设错术语"的标签化表述。
3. 解析必须体现覆盖性：若 fused_prompt 含多维度要求，overall 与四个选项解析中必须能观察到各维度约束的落地痕迹（体现在任务动作、对象限定、条件边界或证据链差异上）。

【输出格式要求（必须严格遵守）】
1. 最终输出必须是一个合法 JSON 对象，仅包含指定字段，不添加多余字段或注释。
2. 严格按以下结构输出（字段名与层级必须一致）：

{{
  "questions": [
    {{
      "id": "Q1",
      "stem": "……题干……",
      "options": {{
        "A": "……选项 A ……",
        "B": "……选项 B ……",
        "C": "……选项 C ……",
        "D": "……选项 D ……"
      }},
      "correct_option": "C",
      "anchor_ids_used": ["A1", "A3"],
      "analysis": {{
        "overall": "……用自然语言概述本题考查意图（不要提"维度/类型/设错术语"）……",
        "A": {{ "is_correct": false, "reason": "……指出依据锚点与不符点……" }},
        "B": {{ "is_correct": false, "reason": "……指出依据锚点与不符点……" }},
        "C": {{ "is_correct": true,  "reason": "……指出依据锚点与相符点……" }},
        "D": {{ "is_correct": false, "reason": "……指出依据锚点与不符点……" }}
      }}
    }}
  ]
}}

【严格格式约束 - 必须遵守】
- 严禁输出 <think>、<analysis>、<reasoning> 或任何 XML/HTML 风格标签
- 严禁在 JSON 前后输出解释性文字或 markdown/code fence
- 你的回复必须以 {{ 开头，以 }} 结尾，中间只包含合法 JSON
- 如果因任何原因无法生成完整题目，请输出最小合法 JSON：{{"questions": []}}
"""
# -------------------------------
# Essay Prompt (Short-answer) - Completely separated from choice to avoid template interference
# Chinese LLM prompt - kept as-is for Gaokao question generation
# -------------------------------
AGENT3_ESSAY_PROMPT_SPEC = """【任务角色】
你是一名高考语文命题专家助手。你将根据给定的"融合版命题提示词"
与"带证据锚点的材料"，生成 1 道高考风格阅读理解简答题。

【多维度融合执行规程（内部执行，不得在输出中显式呈现）】
1) fused_prompt 往往由多个"维度块/约束块"拼接组成。你必须把每个维度块视为"并列且必须满足的约束包"，不得挑选性忽略。
2) 生成前先做内部覆盖清单：逐块抽取其"维度目标/命题落地要点/验收点/协同冲突（若提供）"的硬要求，形成 checklist。
3) 单题内多维度承载分配原则（确保每个维度都有可验收落点）：
   - 题干（stem）承载：主任务动作 + 评价对象/情境约束（保证"问什么"覆盖维度目标）。
   - 答案要点（answer_points）承载：关键结论 + 结构化分点（保证"答什么"覆盖验收点；每个维度至少对应到一个要点或要点中的一个侧面）。
   - evidence_reference / anchor_ids_used 承载：证据锚点映射（保证"依据在哪里"可回文定位）。
   - explanation 承载：限定条件、证据对应关系、要点口径与得分理由（保证"为何得分"覆盖协同冲突里的差异点与边界）。
4) 冲突消解（尽量不丢维度）：
   - 优先遵循 fused_prompt 内更具体、更可操作的表述；抽象表述作为补充约束落实到 explanation 中。
   - 若两个维度都要求高强度主任务：合并为一个更上位的统一问法（如"在情境约束下分析X为何成立/如何成立"），并用不同要点分别回扣不同维度的证据与限定。
5) 输出前自检：逐维度逐条对照其验收点。任何一条未被 stem/答案要点/解释显式承载，则必须回炉改写题干或要点，直到全部满足。

【融合版命题提示词（唯一上位约束）】
{fused_prompt}

硬性要求：
1. 将 fused_prompt 视为唯一上位约束；凡与本提示冲突处，以 fused_prompt 为准。
2. 禁止自行引入新的"能力体系/题型模板"；不要在输出中使用"某某维度"等标签话术。

【带证据锚点的材料（命题依据）】
{material_with_anchors}

锚点说明：
1. 材料中用【A1】…【/A1】、【A2】…【/A2】等标注证据锚点。
2. 出题与答案要点必须主要依托锚点及其语境；不得脱离材料凭常识作答。

【证据闭环总规则】
1) 题干、答案要点与解释必须主要依托锚点及其语境；不得把材料外知识当作判据。
2) 允许同义改写/信息压缩，但不得改变原意或增删关键条件。
3) 每个要点都要可回文定位：锚点证据 → 推断/概括说明 → 得分结论（要点成立）。

【出题任务（只生成 1 题）】
1. 题型：只生成 1 道简答题。
2. 题干表述风格与任务要求以 fused_prompt 为准。
3. answer_points 必须给出若干要点：
   - point：用自己的话概括（尽量不照抄原句）；
   - score：正整数；
   - evidence_reference：该要点所依托的锚点列表（可为空列表，但建议填写）。
4. anchor_ids_used：填写本题主要依托的锚点 ID 列表。
5. explanation 必须体现覆盖性：若 fused_prompt 含多维度要求，explanation 与答案要点中必须能观察到各维度约束的落地痕迹（体现在任务动作、对象限定、条件边界或证据链对应上）。

【重要约束 - 禁止分小问】
1. **严禁**将题目拆分为多个小问（如"(1)…(2)…"或"第一问…第二问…"）
2. **严禁**在题干中使用"概括以下两点/三点"、"分别说明"、"请回答以下问题"等暗示多问的表述
3. 题干必须是**一个完整、统一的问题**，答案要点是对这一个问题的多角度回答，而非多个独立问题的答案
4. 正确示例：
   - "请结合材料，分析作者对中国建筑未来发展持乐观态度的原因。"（一个问题，多角度回答）
   - "作者认为中国传统建筑具有哪些独特价值？请概括说明。"（一个问题，分点作答）
5. 错误示例（绝对禁止）：
   - "请根据材料概括：(1)近代以来中国建筑出现的问题；(2)作者认为的有利条件。"
   - "第一，说明XX的原因；第二，分析XX的影响。"
   - "请分别从A和B两个角度回答以下问题：……"
6. 如果 fused_prompt 中有暗示多问的表述，请将其**整合为一个统一问题**。

【输出格式要求（必须严格遵守）】
最终输出必须是一个合法 JSON（仅 1 题），结构如下：

{{
  "questions": [
    {{
      "stem": "题干（只包含一道简答题的问题表述，禁止分小问）",
      "answer_points": [
        {{ "point": "答案要点1（自己的话概括）", "score": 2, "evidence_reference": ["A1"] }},
        {{ "point": "答案要点2……", "score": 2, "evidence_reference": ["A2","A3"] }}
      ],
      "total_score": 6,
      "explanation": "对答案要点与得分依据的简要说明（可结合锚点解释）。",
      "anchor_ids_used": ["A1", "A2", "A3"],
      "generation_reasoning": "（可选）简要说明生成依据，不要包含长篇推理。"
    }}
  ]
}}

【严格格式约束 - 必须遵守】
- 严禁输出 <think>、<analysis>、<reasoning> 或任何 XML/HTML 风格标签
- 严禁在 JSON 前后输出解释性文字或 markdown/code fence
- 你的回复必须以 {{ 开头，以 }} 结尾，中间只包含合法 JSON
- 如果因任何原因无法生成完整题目，请输出最小合法 JSON：{{"questions": []}}
"""

# ====================================================
# src/generation/agents/agent3_qg_solver.py
# Agent 3: Question Generator + Solver (Enhanced Robust Version)

from typing import Dict, Any, Optional, Tuple, List
import json
import re
import os
from pathlib import Path

from src.shared.schemas import (
    MaterialDimSelection,
    SynthesizedPrompt,
    AnchorSet,
    GeneratedQuestion,
    OptionItem,
    AnswerPoint,
)
from src.shared.config import Agent3Config
from src.shared.llm_interface import LLMClient
from src.shared.prompt_logger import PromptLogger
from src.shared.api_config import is_no_temperature_model


# =============================================================================
# Ablation Prompt Loading
# =============================================================================
def load_ablation_prompts() -> Dict[str, str]:
    """
    Load ablation experiment prompts (used when Agent2 is skipped).

    Returns:
        Dict with keys: 'choice_prompt', 'essay_prompt'
    """
    base_path = Path(__file__).resolve().parent.parent.parent.parent
    ablation_file = base_path / "data" / "agent3_ablation_prompts.json"

    if not ablation_file.exists():
        print(f"[Agent 3] [WARNING] Ablation prompt file not found: {ablation_file}")
        return {}

    try:
        with open(ablation_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        print(f"[Agent 3] Successfully loaded ablation prompts: {list(prompts.keys())}")
        return prompts
    except Exception as e:
        print(f"[Agent 3] [ERROR] Failed to load ablation prompts: {e}")
        return {}


# =============================================================================
# Parsing & Repair Tools
# =============================================================================

def _coerce_text(resp: Any) -> str:
    """
    LLMClient.generate may return str or dict; unify to str.
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("content", "text", "message", "output"):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(resp, ensure_ascii=False)
    return str(resp)


def _extract_outer_json(text: str) -> Optional[str]:
    """
    Extract outermost JSON object {...} (prioritize object).
    - Support prefix text/code blocks
    - Ignore braces inside strings
    """
    if not text:
        return None
    t = text.strip()

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.S)
    if m:
        t = m.group(1).strip()

    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if esc:
            esc = False
            continue
        if in_str and ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1].strip()
    return None


def _repair_common_json(s: str) -> str:
    """
    Fix trailing comma, Chinese quotes, etc., minimize destructive changes.

    [2025-12 Enhancement] Added fix for unterminated strings and missing brackets
    """
    if not s:
        return s
    s2 = s.strip()

    # 1. Fix trailing comma
    s2 = re.sub(r",\s*([}\]])", r"\1", s2)

    # 2. Fix Chinese quotes
    s2 = s2.replace(""", '"').replace(""", '"')

    # 3. Remove illegal control characters (keep newline, tab)
    s2 = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s2)

    # 4. [2025-12 Added] Fix unterminated strings
    s2 = _repair_unterminated_strings(s2)

    # 5. [2025-12 Added] Add missing brackets
    s2 = _repair_missing_brackets(s2)

    return s2


def _repair_unterminated_strings(s: str) -> str:
    """
    [2025-12 Added] Fix unterminated strings.

    When LLM output is truncated, strings may be cut in the middle.
    """
    if not s:
        return s

    # Count non-escaped double quotes
    quote_count = 0
    i = 0
    while i < len(s):
        if s[i] == '"':
            num_backslashes = 0
            j = i - 1
            while j >= 0 and s[j] == '\\':
                num_backslashes += 1
                j -= 1
            if num_backslashes % 2 == 0:
                quote_count += 1
        i += 1

    # If quote count is odd, there's an unterminated string
    if quote_count % 2 == 1:
        s_stripped = s.rstrip()
        if not s_stripped.endswith('"'):
            s = s_stripped + '"'
            print("[Agent 3] [Repair] Added closing quote to fix truncated string")

    return s


def _repair_missing_brackets(s: str) -> str:
    """
    [2025-12 Added] Add missing brackets.
    """
    if not s:
        return s

    # Count brackets (brackets inside strings don't count)
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # Add missing brackets
    repairs = []
    if open_brackets > 0:
        repairs.append(']' * open_brackets)
    if open_braces > 0:
        repairs.append('}' * open_braces)

    if repairs:
        s = s.rstrip()
        suffix = ''.join(repairs)
        s = s + suffix
        print(f"[Agent 3] [Repair] Added missing brackets: {suffix}")

    return s


def _json_loads_best_effort(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Never-throw json.loads."""
    try:
        return json.loads(text), None
    except Exception as e:
        return None, str(e)


def _extract_json_string_field(text: str, field: str) -> Optional[str]:
    """
    Regex extract JSON style string field: "<field>": "..."
    Supports escape characters.
    """
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return bytes(m.group(1), "utf-8").decode("unicode_escape")
    except Exception:
        return m.group(1)


def _extract_correct_option_from_text(text: str) -> Optional[str]:
    for key in ("correct_option", "correct_answer", "answer", "correct", "gold"):
        m = re.search(rf'"{key}"\s*:\s*"([ABCD])"', text)
        if m:
            return m.group(1)
    return None


def _salvage_choice_dict_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    When JSON is truncated/mixed with chain-of-thought, try to salvage key choice fields:
    - stem, options(A-D), correct_option, anchor_ids_used, analysis.overall (optional)
    """
    if not text:
        return None

    stem = _extract_json_string_field(text, "stem") or ""
    correct = _extract_correct_option_from_text(text)

    options: Dict[str, str] = {}
    m = re.search(r'"options"\s*:\s*\{([\s\S]*?)\}\s*,', text)
    if m:
        options_block = "{" + m.group(1) + "}"
        options_block = _repair_common_json(options_block)
        obj, _ = _json_loads_best_effort(options_block)
        if isinstance(obj, dict):
            for k in ("A", "B", "C", "D"):
                if k in obj and isinstance(obj[k], str):
                    options[k] = obj[k]

    if len(options) < 4:
        for k in ("A", "B", "C", "D"):
            v = _extract_json_string_field(text, k)
            if v:
                options[k] = v

    anchor_ids_used: List[str] = []
    anchors = re.findall(r'"anchor_ids_used"\s*:\s*\[([^\]]*)\]', text)
    if anchors:
        inner = anchors[0]
        anchor_ids_used = re.findall(r'"(A\d+)"', inner)

    overall = ""
    m2 = re.search(r'"analysis"\s*:\s*\{[\s\S]*?"overall"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if m2:
        overall = m2.group(1)

    if not stem or len(options) < 4:
        return None

    q = {
        "id": "Q1",
        "stem": stem,
        "options": {k: options.get(k, "") for k in ("A", "B", "C", "D")},
        "correct_option": correct or "",
        "anchor_ids_used": anchor_ids_used,
        "analysis": {"overall": overall},
    }
    return {"questions": [q]}


# =============================================================================
# Agent 3 Main Class
# =============================================================================

class Agent3QuestionGeneratorSolver:
    """
    Agent 3: Question Generator + Solver (Enhanced Robust Version)

    Enhancements:
    - Parsing: Supports <think>, prefix text, code fence, trailing comma; regex salvage for truncated JSON
    - Error correction: Triggers "JSON repair" retry when key fields (especially correct_option) are missing
    - Fallback: Returns minimal GeneratedQuestion on failure (triggers Agent4 need_revision instead of crash)
    """

    def __init__(self, config: Agent3Config, llm_client: LLMClient, prompt_logger: PromptLogger):
        self.config = config
        self.llm_client = llm_client
        self.prompt_logger = prompt_logger

        # Load ablation prompts (for when Agent2 is skipped)
        self.ablation_prompts = load_ablation_prompts()

    # ------------------------------------------------------------------ #
    # Ablation Mode Detection
    # ------------------------------------------------------------------ #
    def _is_ablation_mode(self, agent2_output: AnchorSet) -> bool:
        """
        Detect if in ablation mode (Agent2 skipped).

        Check by following conditions:
        1. anchors list is empty
        2. anchor_discovery_reasoning contains "[ABLATION_SKIP_AGENT2]" marker

        Args:
            agent2_output: Agent2 output

        Returns:
            bool: True means Agent2 is ablated, need to use ablation prompts
        """
        if not agent2_output:
            return True

        # Check if anchors list is empty
        anchors_empty = not agent2_output.anchors

        # Check for ablation marker
        reasoning = getattr(agent2_output, 'anchor_discovery_reasoning', '')
        has_ablation_marker = '[ABLATION_SKIP_AGENT2]' in reasoning

        return anchors_empty and has_ablation_marker

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        agent1_output: MaterialDimSelection,
        agent2_output: AnchorSet,
        synthesized_prompt: SynthesizedPrompt,
    ) -> GeneratedQuestion:
        print("[Agent 3] Starting question generation + solving...")
        print(f"[Agent 3] Question type: {agent1_output.question_type}")

        # generation_strategy not effective in current implementation; not printing/depending on it

        if agent1_output.question_type == "single-choice":
            return self._generate_single_choice(agent1_output, synthesized_prompt, agent2_output)
        if agent1_output.question_type == "essay":
            return self._generate_essay(agent1_output, synthesized_prompt, agent2_output)
        raise ValueError(f"[Agent 3] Unsupported question type: {agent1_output.question_type}")

    # ------------------------------------------------------------------ #
    # Single-Choice Generation (Enhanced: Validation + Repair Retry)
    # ------------------------------------------------------------------ #
    def _generate_single_choice(
        self,
        agent1_output: MaterialDimSelection,
        synthesized_prompt: SynthesizedPrompt,
        agent2_output: AnchorSet,
    ) -> GeneratedQuestion:
        prompt = self._build_single_choice_prompt(agent1_output, synthesized_prompt, agent2_output)

        raw_resp = self._call_llm(prompt, mode="SingleChoice", agent1_output=agent1_output, synthesized_prompt=synthesized_prompt)
        data, note = self._parse_or_salvage_json(raw_resp)

        if not self._is_valid_choice_payload(data):
            repaired = self._try_repair_with_llm(raw_resp, is_essay=False, agent1_output=agent1_output)
            if repaired is not None:
                data = repaired

        if not self._is_valid_choice_payload(data):
            print("[Agent 3] [WARN] Single-choice JSON still incomplete, using minimal fallback (Agent4 will trigger need_revision)")
            return self._build_minimal_choice(agent1_output, agent2_output, generation_reason=note)

        result = self._build_generated_question_from_choice_data(data, agent1_output, agent2_output)

        if not result.correct_answer:
            inferred = self._infer_correct_answer_from_generated(result)
            if inferred:
                result.correct_answer = inferred

        print("[Agent 3] [OK] Single-choice generation complete")
        print(f"  - Stem: {result.stem[:50]}...")
        print(f"  - Correct answer: {result.correct_answer}")
        return result

    # ------------------------------------------------------------------ #
    # Essay Generation (Enhanced: Validation + Repair Retry)
    # ------------------------------------------------------------------ #
    def _generate_essay(
        self,
        agent1_output: MaterialDimSelection,
        synthesized_prompt: SynthesizedPrompt,
        agent2_output: AnchorSet,
    ) -> GeneratedQuestion:
        prompt = self._build_essay_prompt(agent1_output, synthesized_prompt, agent2_output)

        raw_resp = self._call_llm(prompt, mode="Essay", agent1_output=agent1_output, synthesized_prompt=synthesized_prompt)
        data, note = self._parse_or_salvage_json(raw_resp)

        if not self._is_valid_essay_payload(data):
            repaired = self._try_repair_with_llm(raw_resp, is_essay=True, agent1_output=agent1_output)
            if repaired is not None:
                data = repaired

        if not self._is_valid_essay_payload(data):
            print("[Agent 3] [WARN] Essay JSON still incomplete, using minimal fallback (Agent4 will trigger need_revision)")
            return self._build_minimal_essay(agent1_output, agent2_output, generation_reason=note)

        result = self._build_generated_question_from_essay_data(data, agent1_output, agent2_output)

        print("[Agent 3] [OK] Essay generation complete")
        print(f"  - Stem: {result.stem[:50]}...")
        print(f"  - Answer points count: {len(result.answer_points) if result.answer_points else 0}")
        return result

    # ------------------------------------------------------------------ #
    # LLM Call (Unified: add system constraint, reduce <think>)
    # ------------------------------------------------------------------ #
    def _call_llm(self, user_prompt: str, mode: str, agent1_output: MaterialDimSelection, synthesized_prompt: SynthesizedPrompt) -> str:
        # Chinese system prompt for LLM - kept for domain-specific Gaokao question generation interaction
        system = (
            "你必须严格只输出一个合法 JSON（不要输出<think>、不要输出解释、不要输出markdown或代码块）。"
            "如果做不到，请输出最小合法 JSON：{\"questions\": []}。"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]

        kwargs: Dict[str, Any] = {}

        # [2025-12-26 Fix] No longer hard-coding temperature at Agent level
        # Let LLMClient use its instance-level temperature (from api_config.py)
        # Only pass temperature if explicitly configured, otherwise use LLMClient default
        model_name = self.llm_client.model_name or ""
        if not is_no_temperature_model(model_name):
            # Only override when config explicitly sets temperature
            temp_val = getattr(self.config, "temperature", None)
            if temp_val is not None:
                kwargs["temperature"] = temp_val
            # Otherwise don't pass temperature, use LLMClient instance default (from api_config.py)
        # For no-temperature models, don't pass temperature, let LLMClient use its default behavior

        # [2025-12-26 Fix] Same for max_tokens, use LLMClient default if not set
        max_tokens_val = getattr(self.config, "max_tokens", None)
        if max_tokens_val is not None:
            kwargs["max_tokens"] = max_tokens_val

        response = self.llm_client.generate(messages, **kwargs)
        text = _coerce_text(response)

        self.prompt_logger.save_agent_log(
            agent_name=f"Agent3_{mode}",
            stage="generation",
            prompt=user_prompt,
            response=text,
            metadata={
                "material_id": agent1_output.material_id,
                "question_type": agent1_output.question_type,
                "ability_point": synthesized_prompt.ability_point,
            },
            model=self.llm_client.model_name,
        )
        return text

    def _try_repair_with_llm(self, raw_response: str, is_essay: bool, agent1_output: MaterialDimSelection) -> Optional[Dict[str, Any]]:
        """
        Only trigger repair call when first result is incomplete:
        - Use raw_response as input, ask model to "only output pure JSON"
        - Cost: at most +1 call (rarely triggered)
        """
        # Chinese repair prompt - kept for LLM interaction (domain-specific Gaokao question generation)
        fix_prompt = f"""你刚才的输出包含非JSON文字、或JSON不完整/字段缺失。请你执行"JSON修复"：

【任务】
- 只输出一个合法 JSON。
- 顶层必须是：{{"questions":[...]}}, 且 questions 至少包含 1 道题（Q1）。
- 必须包含关键字段：
  - 选择题：stem/options(A-D)/correct_option/anchor_ids_used/analysis(含 overall 且 A-D 均有 is_correct+reason)
  - 简答题：stem/answer_points(point+score)/total_score/anchor_ids_used/explanation
- 严禁输出 <think>、解释文字、markdown/code fence。

【你上一条输出（原样）】
{raw_response}

现在只输出修复后的 JSON：
"""
        try:
            repaired_text = self._call_llm(
                fix_prompt,
                mode="Repair",
                agent1_output=agent1_output,
                synthesized_prompt=SynthesizedPrompt(synthesized_instruction="", ability_point=""),
            )
            data, _ = self._parse_or_salvage_json(repaired_text)
            if is_essay:
                return data if self._is_valid_essay_payload(data) else None
            return data if self._is_valid_choice_payload(data) else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Material + Anchor Marking (Fix: avoid strip/filter empty lines causing paragraph_idx offset)
    # ------------------------------------------------------------------ #
    def _format_material_with_anchors(self, material_text: str, anchors: list) -> str:
        from collections import defaultdict

        # Preserve original newline structure: don't strip, don't filter empty lines
        raw_lines: List[str] = material_text.split("\n")
        marked_lines: List[str] = list(raw_lines)

        # Compatibility: if material has empty lines and anchor paragraph_idx looks like "non-empty line count", map it
        nonempty_map: List[int] = [i for i, ln in enumerate(raw_lines) if ln.strip() != ""]
        has_empty_lines = len(nonempty_map) != len(raw_lines)
        max_pidx = max((getattr(a, "paragraph_idx", -1) for a in anchors), default=-1)

        # Default: assume paragraph_idx is "non-empty line index" (consistent with old implementation), enable mapping when empty lines exist
        use_nonempty_indexing = bool(has_empty_lines and max_pidx >= 0 and max_pidx < len(nonempty_map))

        if use_nonempty_indexing:
            work_paras = [raw_lines[i] for i in nonempty_map]
            work_to_raw = nonempty_map
        else:
            work_paras = raw_lines
            work_to_raw = list(range(len(raw_lines)))

        # Prepare anchor markers (anchor_id order depends on input anchors order)
        anchor_markers = []
        for i, anchor in enumerate(anchors):
            anchor_markers.append(
                {
                    "anchor_id": f"A{i+1}",
                    "paragraph_idx": getattr(anchor, "paragraph_idx", -1),
                    "start_char": getattr(anchor, "start_char", -1),
                    "end_char": getattr(anchor, "end_char", -1),
                    "snippet": getattr(anchor, "snippet", "") or "",
                }
            )

        para_anchors = defaultdict(list)
        for marker in anchor_markers:
            pidx = marker["paragraph_idx"]
            if isinstance(pidx, int) and 0 <= pidx < len(work_paras):
                para_anchors[pidx].append(marker)

        # Multiple anchors in same paragraph: insert in reverse start_char order to avoid offset
        for para_idx in para_anchors:
            para_anchors[para_idx].sort(key=lambda x: int(x["start_char"] or 0), reverse=True)

        marked_work_paras: List[str] = list(work_paras)
        for para_idx, para in enumerate(work_paras):
            current_para = para
            if para_idx in para_anchors and current_para:
                for marker in para_anchors[para_idx]:
                    start = int(marker.get("start_char", -1))
                    end = int(marker.get("end_char", -1))
                    anchor_id = marker["anchor_id"]
                    snip = marker.get("snippet", "") or ""

                    L = len(current_para)
                    invalid = (start < 0 or end < 0 or start >= end or end > L)

                    # If position is unreliable, try to find snippet in text
                    if invalid and snip and snip in current_para:
                        start = current_para.find(snip)
                        end = start + len(snip)
                        invalid = (start < 0 or start >= end or end > len(current_para))

                    if invalid:
                        continue

                    current_para = current_para[:end] + f"【/{anchor_id}】" + current_para[end:]
                    current_para = current_para[:start] + f"【{anchor_id}】" + current_para[start:]

            marked_work_paras[para_idx] = current_para

        # Write back to original line structure, preserving empty lines and newline positions
        for work_idx, raw_idx in enumerate(work_to_raw):
            marked_lines[raw_idx] = marked_work_paras[work_idx]

        return "\n".join(marked_lines)

    # ------------------------------------------------------------------ #
    # Prompt Construction (choice/essay completely separated, supports ablation mode)
    # ------------------------------------------------------------------ #
    def _build_single_choice_prompt(self, agent1_output: MaterialDimSelection, synthesized_prompt: SynthesizedPrompt, agent2_output: AnchorSet) -> str:
        # Check if in ablation mode
        is_ablation = self._is_ablation_mode(agent2_output)

        if is_ablation:
            # Ablation mode: use ablation prompts (no anchors)
            print("[Agent 3] [ABLATION] Detected Agent2 ablation, using ablation prompts (no anchor version)")

            ablation_choice_prompt = self.ablation_prompts.get('choice_prompt')
            if not ablation_choice_prompt:
                print("[Agent 3] [ERROR] Ablation prompts not loaded, falling back to normal prompts")
                # Fall back to normal flow
                material_with_anchors = self._format_material_with_anchors(agent1_output.material_text, agent2_output.anchors)
                anchor_analysis = self._build_anchor_analysis_section(agent2_output.anchors)
                base_prompt = AGENT3_PROMPT_SPEC.format(
                    fused_prompt=synthesized_prompt.synthesized_instruction,
                    material_with_anchors=material_with_anchors,
                )
                return base_prompt + anchor_analysis

            # Use ablation prompt (only need to replace material and fused_prompt)
            return ablation_choice_prompt.format(
                material=agent1_output.material_text,
                fused_prompt=synthesized_prompt.synthesized_instruction,
            )
        else:
            # Normal mode: use prompts with anchors
            material_with_anchors = self._format_material_with_anchors(agent1_output.material_text, agent2_output.anchors)

            # [2025-12 Enhancement] Build anchor analysis info, including Agent2's semantic analysis
            anchor_analysis = self._build_anchor_analysis_section(agent2_output.anchors)

            # Use AGENT3_PROMPT_SPEC (single-choice)
            base_prompt = AGENT3_PROMPT_SPEC.format(
                fused_prompt=synthesized_prompt.synthesized_instruction,
                material_with_anchors=material_with_anchors,
            )

            # [2025-12 Enhancement] Insert anchor analysis info in prompt
            # This ensures Agent3 can utilize Agent2's analysis for question design
            return base_prompt + anchor_analysis

    def _build_essay_prompt(self, agent1_output: MaterialDimSelection, synthesized_prompt: SynthesizedPrompt, agent2_output: AnchorSet) -> str:
        # Check if in ablation mode
        is_ablation = self._is_ablation_mode(agent2_output)

        if is_ablation:
            # Ablation mode: use ablation prompts (no anchors)
            print("[Agent 3] [ABLATION] Detected Agent2 ablation, using ablation prompts (no anchor version)")

            ablation_essay_prompt = self.ablation_prompts.get('essay_prompt')
            if not ablation_essay_prompt:
                print("[Agent 3] [ERROR] Ablation prompts not loaded, falling back to normal prompts")
                # Fall back to normal flow
                material_with_anchors = self._format_material_with_anchors(agent1_output.material_text, agent2_output.anchors)
                anchor_analysis = self._build_anchor_analysis_section(agent2_output.anchors)
                base_prompt = AGENT3_ESSAY_PROMPT_SPEC.format(
                    fused_prompt=synthesized_prompt.synthesized_instruction,
                    material_with_anchors=material_with_anchors,
                )
                return base_prompt + anchor_analysis

            # Use ablation prompt (only need to replace material and fused_prompt)
            return ablation_essay_prompt.format(
                material=agent1_output.material_text,
                fused_prompt=synthesized_prompt.synthesized_instruction,
            )
        else:
            # Normal mode: use prompts with anchors
            material_with_anchors = self._format_material_with_anchors(agent1_output.material_text, agent2_output.anchors)

            # [2025-12 Enhancement] Build anchor analysis info, including Agent2's semantic analysis
            anchor_analysis = self._build_anchor_analysis_section(agent2_output.anchors)

            # Use AGENT3_ESSAY_PROMPT_SPEC (essay)
            base_prompt = AGENT3_ESSAY_PROMPT_SPEC.format(
                fused_prompt=synthesized_prompt.synthesized_instruction,
                material_with_anchors=material_with_anchors,
            )

            # [2025-12 Enhancement] Insert anchor analysis info in prompt
            return base_prompt + anchor_analysis

    def _build_anchor_analysis_section(self, anchors: list) -> str:
        """
        [2025-12 Added] Build anchor analysis info section.

        Organize Agent3's anchor semantic analysis into structured text
        for Agent4 to reference during question design.

        Args:
            anchors: Anchor list from Agent3 output

        Returns:
            str: Formatted text of anchor analysis info
        """
        if not anchors:
            return ""

        # Check if any anchor contains semantic info
        has_semantic_info = any(
            getattr(a, "reason_for_anchor", "") or getattr(a, "loc", "")
            for a in anchors
        )

        if not has_semantic_info:
            return ""

        lines = ["\n\n【锚点分析信息（来自证据锚点标注器的命题建议）】"]
        lines.append("以下是每个证据锚点的命题价值分析，请在出题时重点参考：\n")

        for i, anchor in enumerate(anchors):
            anchor_id = f"A{i+1}"
            snippet = getattr(anchor, "snippet", "")[:50] + "..." if len(getattr(anchor, "snippet", "")) > 50 else getattr(anchor, "snippet", "")
            reason = getattr(anchor, "reason_for_anchor", "")
            loc = getattr(anchor, "loc", "")

            lines.append(f"【{anchor_id}】")
            if loc:
                lines.append(f"  - 位置：{loc}")
            if snippet:
                lines.append(f"  - 内容片段：{snippet}")
            if reason:
                lines.append(f"  - 命题价值：{reason}")
            lines.append("")

        lines.append("请基于上述锚点分析，结合融合版命题提示词的要求进行命题。")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # JSON Parsing (Enhanced: outer-json extraction + repair + salvage)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_xml_tags(text: str) -> str:
        paired_tags = ["think", "analysis", "reasoning", "reflection", "thought"]
        t = text or ""
        for tag in paired_tags:
            t = re.sub(rf"<{tag}[^>]*>[\s\S]*?</{tag}>", "", t, flags=re.IGNORECASE)

        for tag in paired_tags:
            if re.search(rf"<{tag}[^>]*>", t, flags=re.IGNORECASE) and not re.search(rf"</{tag}>", t, flags=re.IGNORECASE):
                t = re.sub(rf"<{tag}[^>]*>[\s\S]*?(?=\{{)", "", t, flags=re.IGNORECASE)

        return t.strip()

    def _parse_or_salvage_json(self, response_text: str) -> Tuple[Dict[str, Any], str]:
        raw = _coerce_text(response_text)
        cleaned = self._strip_xml_tags(raw)

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        cand = _extract_outer_json(cleaned)
        if cand:
            # [2025-12 Fix] Try direct parsing first, avoid _repair_common_json breaking valid JSON
            obj, err = _json_loads_best_effort(cand)
            if isinstance(obj, dict):
                return obj, "parsed_outer_json_ok"
            # Direct parsing failed, try repair then parse
            cand2 = _repair_common_json(cand)
            obj, err = _json_loads_best_effort(cand2)
            if isinstance(obj, dict):
                return obj, "parsed_outer_json_repaired"
            return {}, f"outer_json_not_dict: {type(obj).__name__}" if obj is not None else f"outer_json_load_failed: {err}"

        decoder = json.JSONDecoder()
        brace_positions = [i for i, c in enumerate(cleaned) if c == "{"]

        last_error: Optional[Exception] = None
        for pos in brace_positions:
            try:
                obj, _ = decoder.raw_decode(cleaned, pos)
                if isinstance(obj, dict):
                    print(f"[Agent 3] JSON parsing successful (raw_decode from position {pos})")
                    return obj, "parsed_raw_decode_ok"
            except Exception as e:
                last_error = e

        salvaged = _salvage_choice_dict_from_text(cleaned)
        if isinstance(salvaged, dict):
            print("[Agent 3] [WARN] JSON parsing failed, enabled salvage (regex recovered key fields)")
            return salvaged, "salvaged_from_text"

        preview = cleaned[:260].replace("\n", " ")
        note = f"json_parse_failed: {last_error}; preview={preview}"
        print(f"[Agent 3] JSON parsing failed: {note}")
        return {}, note

    # ------------------------------------------------------------------ #
    # Payload Validation (Critical: ensure correct_option is not empty)
    # ------------------------------------------------------------------ #
    def _is_valid_choice_payload(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        qs = data.get("questions")
        if not isinstance(qs, list) or not qs:
            return False
        q = qs[0]
        if not isinstance(q, dict):
            return False
        if not isinstance(q.get("stem"), str) or not q["stem"].strip():
            return False
        options = q.get("options")
        if not isinstance(options, dict):
            return False
        for k in ("A", "B", "C", "D"):
            if k not in options or not isinstance(options[k], str) or not options[k].strip():
                return False

        co = q.get("correct_option") or q.get("correct_answer") or q.get("answer")
        if not isinstance(co, str) or co.strip() not in ("A", "B", "C", "D"):
            analysis = q.get("analysis")
            if isinstance(analysis, dict):
                for k in ("A", "B", "C", "D"):
                    item = analysis.get(k)
                    if isinstance(item, dict) and item.get("is_correct") is True:
                        q["correct_option"] = k
                        return True
            return False

        q["correct_option"] = co.strip()
        return True

    def _is_valid_essay_payload(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        qs = data.get("questions")
        if not isinstance(qs, list) or not qs:
            return False
        q = qs[0]
        if not isinstance(q, dict):
            return False
        if not isinstance(q.get("stem"), str) or not q["stem"].strip():
            return False
        ap = q.get("answer_points")
        if not isinstance(ap, list) or not ap:
            return False
        for p in ap:
            if not isinstance(p, dict):
                return False
            if not p.get("point") or "score" not in p:
                return False
        return True

    def _infer_correct_answer_from_generated(self, gq: GeneratedQuestion) -> Optional[str]:
        if not gq.options:
            return None
        correct = [o.label for o in gq.options if getattr(o, "is_correct", False)]
        if len(correct) == 1 and correct[0] in ("A", "B", "C", "D"):
            return correct[0]
        return None

    # ------------------------------------------------------------------ #
    # Minimal Fallback Questions (avoid hard crash)
    # ------------------------------------------------------------------ #
    def _build_minimal_choice(self, agent1_output: MaterialDimSelection, agent2_output: AnchorSet, generation_reason: str) -> GeneratedQuestion:
        options = [
            OptionItem(label="A", content="(generation failed)", is_correct=False, error_type=None, reasoning=""),
            OptionItem(label="B", content="(generation failed)", is_correct=False, error_type=None, reasoning=""),
            OptionItem(label="C", content="(generation failed)", is_correct=False, error_type=None, reasoning=""),
            OptionItem(label="D", content="(generation failed)", is_correct=False, error_type=None, reasoning=""),
        ]
        return GeneratedQuestion(
            stem="(Question generation failed, please retry)",
            question_type="single-choice",
            options=options,
            correct_answer=None,
            distractor_blueprints=None,
            answer_points=None,
            total_score=None,
            material_text=agent1_output.material_text,
            explanation=generation_reason,
            evidence_anchors=list(range(len(agent2_output.anchors))) if agent2_output.anchors else [],
            generation_reasoning=generation_reason,
        )

    def _build_minimal_essay(self, agent1_output: MaterialDimSelection, agent2_output: AnchorSet, generation_reason: str) -> GeneratedQuestion:
        return GeneratedQuestion(
            stem="(Question generation failed, please retry)",
            question_type="essay",
            options=None,
            correct_answer=None,
            distractor_blueprints=None,
            answer_points=[AnswerPoint(point="(generation failed)", score=0, evidence_reference=[])],
            total_score=0,
            material_text=agent1_output.material_text,
            explanation=generation_reason,
            evidence_anchors=list(range(len(agent2_output.anchors))) if agent2_output.anchors else [],
            generation_reasoning=generation_reason,
                )

    # ------------------------------------------------------------------ #
    # GeneratedQuestion Construction (single-choice): Enhanced correct_* field compatibility
    # ------------------------------------------------------------------ #
    def _build_generated_question_from_choice_data(
        self,
        data: Dict[str, Any],
        agent1_output: MaterialDimSelection,
        agent2_output: AnchorSet,
    ) -> GeneratedQuestion:
        if isinstance(data, dict) and isinstance(data.get("questions"), list) and data["questions"]:
            question_data = data["questions"][0]
        else:
            question_data = data

        correct_option = (
            (question_data.get("correct_option") or question_data.get("correct_answer") or question_data.get("answer") or "")
            .strip()
        )
        if correct_option not in ("A", "B", "C", "D"):
            correct_option = ""

        options: List[OptionItem] = []
        options_data = question_data.get("options", {})
        analysis = question_data.get("analysis", {})

        if isinstance(options_data, dict):
            for label in ("A", "B", "C", "D"):
                if label not in options_data:
                    continue
                opt_analysis = analysis.get(label, {}) if isinstance(analysis, dict) else {}
                is_correct = opt_analysis.get("is_correct", label == correct_option)
                options.append(
                    OptionItem(
                        label=label,
                        content=options_data[label],
                        is_correct=bool(is_correct),
                        error_type=None,
                        reasoning=str(opt_analysis.get("reason", "") or ""),
                    )
                )
        else:
            for opt_data in (options_data or []):
                options.append(
                    OptionItem(
                        label=opt_data["label"],
                        content=opt_data["content"],
                        is_correct=opt_data.get("is_correct", False),
                        error_type=opt_data.get("error_type"),
                        reasoning=opt_data.get("reasoning"),
                    )
                )

        anchor_ids_used = question_data.get("anchor_ids_used", []) or []
        if anchor_ids_used:
            evidence_anchors: List[int] = []
            for anchor_id in anchor_ids_used:
                if isinstance(anchor_id, str) and anchor_id.startswith("A"):
                    try:
                        idx = int(anchor_id[1:]) - 1
                        if 0 <= idx < len(agent2_output.anchors):
                            evidence_anchors.append(idx)
                    except ValueError:
                        continue
        else:
            evidence_anchors = list(range(len(agent2_output.anchors)))

        if isinstance(analysis, dict) and "overall" in analysis:
            parts = [f"【整体分析】{analysis['overall']}"]
            for label in ("A", "B", "C", "D"):
                if label in analysis and isinstance(analysis[label], dict):
                    opt_analysis = analysis[label]
                    status = "正确" if opt_analysis.get("is_correct") else "错误"
                    parts.append(f"\n【{label}项】{status}。{opt_analysis.get('reason', '')}")
            explanation = "\n".join(parts)
        else:
            explanation = question_data.get("explanation", "")

        return GeneratedQuestion(
            stem=question_data.get("stem", ""),
            question_type="single-choice",
            options=options,
            correct_answer=correct_option or None,
            distractor_blueprints=None,
            answer_points=None,
            total_score=None,
            material_text=agent1_output.material_text,
            explanation=explanation,
            evidence_anchors=evidence_anchors,
                        generation_reasoning=(analysis.get("overall", "") if isinstance(analysis, dict) else question_data.get("generation_reasoning")),
        )

    # ------------------------------------------------------------------ #
    # GeneratedQuestion Construction (essay)
    # ------------------------------------------------------------------ #
    def _build_generated_question_from_essay_data(
        self,
        data: Dict[str, Any],
        agent1_output: MaterialDimSelection,
        agent2_output: AnchorSet,
    ) -> GeneratedQuestion:
        if isinstance(data, dict) and isinstance(data.get("questions"), list) and data["questions"]:
            question_data = data["questions"][0]
        else:
            question_data = data

        answer_points: List[AnswerPoint] = []
        for point_data in question_data.get("answer_points", []) or []:
            answer_points.append(
                AnswerPoint(
                    point=point_data.get("point", ""),
                    score=point_data.get("score", 0),
                    evidence_reference=point_data.get("evidence_reference"),
                )
            )

        anchor_ids_used = question_data.get("anchor_ids_used", []) or []
        if anchor_ids_used:
            evidence_anchors: List[int] = []
            for anchor_id in anchor_ids_used:
                if isinstance(anchor_id, str) and anchor_id.startswith("A"):
                    try:
                        idx = int(anchor_id[1:]) - 1
                        if 0 <= idx < len(agent2_output.anchors):
                            evidence_anchors.append(idx)
                    except ValueError:
                        continue
        else:
            evidence_anchors = []
            for p in answer_points:
                if p.evidence_reference:
                    evidence_anchors.extend(p.evidence_reference)
            evidence_anchors = list(set(evidence_anchors))

        total_score = question_data.get("total_score")
        if total_score is None:
            total_score = sum((p.score or 0) for p in answer_points)

        return GeneratedQuestion(
            stem=question_data.get("stem", ""),
            question_type="essay",
            options=None,
            correct_answer=None,
            distractor_blueprints=None,
            answer_points=answer_points,
            total_score=total_score,
            material_text=agent1_output.material_text,
            explanation=question_data.get("explanation", ""),
            evidence_anchors=evidence_anchors,
            generation_reasoning=question_data.get("generation_reasoning"),
        )


__all__ = ["Agent3QuestionGeneratorSolver"]
