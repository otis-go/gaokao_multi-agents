# ===== AGENT2_PROMPT_SPEC_FROM_MD (DO NOT EDIT - Source of Truth) =====
# This constant comes from data/agent2.md, the only authoritative source for Agent 2 system prompt
# If modification needed, modify here first and sync back to MD as backup

# Chinese LLM prompt - kept as-is for LLM interaction (domain-specific Gaokao question generation prompt)
# The prompt instructs the model to act as an "evidence anchor annotator" for exam question design
AGENT2_PROMPT_SPEC = """【任务角色】
你是一名"证据锚点标注器"，需要在给定的论述类、说明类等非虚构文本中，找出若干适合命题的"证据锚点"。这些锚点将作为下游自动出题模型设计试题（如选择题、主观题等）的依据。

【输入信息】
你将接收到以下信息：

1. 阅读材料原文
   - material: {material}

2.题型信息

- question_type: {question_type}
  （例如："单项选择题"或"主观题" ）

3.融合版命题提示词（来自子任务一）

- fusion_prompt: {fusion_prompt}
这是上一道子任务中，你的"搭档模型"已经生成的、整合所有维度的命题总说明。

【总体要求】
1. 你需要在充分理解 material、dimensions 和 fusion_prompt 的基础上，识别若干"可命题证据锚点"。
2. 锚点的选择要**服务于 fusion_prompt 所描述的命题重点**，并兼顾本次输入中出现的各维度要求；不要凭空引入维度信息中没有的考查方向或设错方式。
3. 不同材料长度和复杂度可能不同，一般建议给出 **3～8 个锚点**，但可根据文本实际情况略微增减，只要每个锚点都确有命题价值即可。

【证据锚点的基本标准】
在阅读材料时，请优先选择符合以下特点的语句或句群作为锚点：

1. 信息相对完整、逻辑关系清晰
   - 例如：
     - "对象—特征—原因—结果"的链条；
     - 明确的条件—结论关系；
     - 完整的操作步骤及目的；
     - 结构层次分明的小段落（如"先提出问题，再解释原因，最后给出对策"的段落）。

2. 能支撑多种命题角度
   - 既可以用来考查信息提取与概括、结构与逻辑梳理，也可以支持根据本次维度信息要求的命题方向（如科学思维、知识获取、思维发展、错误选项设计方式等）。
   - 如果 fusion_prompt 中对"选项设置""设错方式""论证分析""结构梳理"等有特殊要求，请优先选择有利于这些要求发挥的段落作为锚点。

3. 便于构造干扰项或高区分度设问
   - 锚点内部信息要足够丰富、关系清楚，这样下游模型在命题时可以：
     - 略作删减、扩张或重组信息，形成错误理解或不充分概括；
     - 或改变逻辑关系（如因果、并列、递进、对比等）来设置干扰项；
     - 或在多材料情境下，与其他段落信息拼接或对照，形成综合判断型设问。
   - 具体采用哪种设错方式，由下游命题模型根据 fusion_prompt 和维度信息决定；你只需保证锚点本身适合被"利用"。

【锚点输出格式与内容要求】
你的输出分为两部分：

---

一、锚点列表（anchors）

请按顺序给出每一个锚点的信息，使用统一的字段说明。锚点编号用 "A1"、"A2"、"A3"…… 表示。每个锚点须包含：

- id：锚点编号，如 "A1"、"A2"。
- span：从原文中直接截取的句子或句群，尽量保持原文顺序和措辞，可适当略去与命题无关的枝节，但不得改写原意。
- loc：用简短自然语言说明该锚点在原文中的大致位置，例如：
  - "关于浇灌干湿的问答部分"；
  - "材料开头对问题背景的总起段"；
  - "第二则材料中分析原因的段落"。
- reason_for_anchor：说明该处为何适合作为命题依据，重点包括：
  - 该处包含了哪些较完整的知识点或逻辑关系；
  - 可以结合哪些维度标签或 fusion_prompt 中的哪些考查方向来出题（例如信息整合、逻辑推断、结构梳理、错误选项设计等）；
  - 下游模型可以怎样利用该锚点来构造区分度（例如通过信息增删、逻辑变形、观点对比、多材料综合等方式），但不要强行指定某一种具体设错模式，除非该模式已在本次 levelC_prompts 中被明确提出。

建议 anchors 部分的组织形式类似下面这种结构示意（仅为格式示例，不要抄其中内容）：

- A1
  - span: "……"
  - loc: "……"
  - reason_for_anchor: "……"

- A2
  - span: "……"
  - loc: "……"
  - reason_for_anchor: "……"

……

---

二、带锚点标记的材料（material_with_anchors）

在原文 material 的基础上，将你选定的每一个锚点用成对的标记括起来：

- 形式为：【A1】……【/A1】、【A2】……【/A2】等；
- 标记应准确包裹在 span 所对应的原文位置上，保持原文其他部分的文字、顺序和标点不变；
- 若不同锚点存在交叠，优先以较大的逻辑单元为主：
  - 即可以只标注外层锚点（例如只保留【A1】…【/A1】），避免复杂嵌套影响后续解析。
- 请确保 anchors 部分的 span 与 material_with_anchors 中被【Ai】…【/Ai】包裹的文本一致，避免内容不对应。

---

【输出格式约定】
你的最终输出应按以下顺序组织：

1. 先给出"锚点列表"，按 A1、A2……编号，包含 id / span / loc / reason_for_anchor 四个字段的自然语言说明。
2. 然后给出一段完整的"带锚点标记的材料"（material_with_anchors），只在相应位置增加【Ai】和【/Ai】标记，不做其他改写。

不要输出任何示例题目，也不要重复粘贴命题维度原文或 fusion_prompt 原文，只需在 reason_for_anchor 中用简明语言引用、概括与之相关的考查方向即可。
"""

# ====================================================
# src/generation/agents/agent2_anchor_finder.py
# Agent 2: Anchor Finder

"""
Agent 2: Anchor Finder (Stage 1 Middle Phase)

Features:
- Based on material text from Agent 1 + fusion prompts from prompt extraction module,
  use LLM to find suitable "evidence anchors" (AnchorSpan) for question design.

Position:
- Second step in Stage 1 question-solving-QC pipeline (material+dimensions -> fusion prompt -> anchors).

Input:
- MaterialDimSelection (from Agent 1)
- SynthesizedPrompt (from prompt extraction module)
- Agent2Config (anchor count range, etc.)

Output:
- AnchorSet: Anchor collection (for Agent 3 to generate questions and answers)

Legacy Mapping:
- Corresponds to evidence_spans generation logic in old stage_1_planner.py,
  upgraded version of PlanJSON.evidence_spans field.
"""

from typing import List
import json

from src.shared.schemas import (
    MaterialDimSelection,
    SynthesizedPrompt,
    AnchorSet,
    AnchorSpan,
)
from src.shared.config import Agent2Config
from src.shared.llm_interface import LLMClient
from src.shared.prompt_logger import PromptLogger


# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)
class Agent2AnchorFinder:
    """
    Agent 2: Anchor Finder.

    Functions:
    - Construct anchor discovery prompt
    - Call LLM to analyze material
    - Parse results into AnchorSet
    """

    def __init__(
        self,
        config: Agent2Config,
        llm_client: LLMClient,
        prompt_logger: PromptLogger,
    ) -> None:
        """
        Initialize Agent 2.

        Args:
            config: Agent 2 configuration.
            llm_client: LLM client instance.
            prompt_logger: Prompt logger.
        """
        self.config = config
        self.llm_client = llm_client
        self.prompt_logger = prompt_logger

    # --------------------------------------------------------------------- #
    # Public APIs
    # --------------------------------------------------------------------- #

    def run(
        self,
        agent1_output: MaterialDimSelection,
        agent2_output: SynthesizedPrompt,
    ) -> AnchorSet:
        """
        Execute anchor discovery once, return AnchorSet.

        Args:
            agent1_output: Agent 1 output (material, question type, etc.).
            agent2_output: Synthesized prompt output.

        Returns:
            AnchorSet: Discovered anchor set.
        """
        print("[Agent 2] Starting anchor discovery...")
        print(
            f"[Agent 2] Anchor count range: "
            f"{self.config.min_anchors}-{self.config.max_anchors}"
        )

        # Step 1: Split material into paragraphs
        paragraphs = self._split_material_to_paragraphs(
            agent1_output.material_text
        )

        # Step 2: Build prompt (material + question_type + fusion_prompt, with JSON output constraint)
        prompt = self._build_prompt(
            material_text=agent1_output.material_text,
            paragraphs=paragraphs,
            question_type=agent1_output.question_type,
            ability_point=agent2_output.ability_point,
            synthesized_instruction=agent2_output.synthesized_instruction,
        )

        # Step 3: Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(messages)

        # Step 4: Log for experiment reproduction and error analysis
        self.prompt_logger.save_agent_log(
            agent_name="Agent2",
            stage="generation",
            prompt=prompt,
            response=response,
            metadata={
                "material_id": agent1_output.material_id,
                "ability_point": agent2_output.ability_point,
            },
            model=self.llm_client.model_name,
        )

        # Step 5: Parse JSON response into AnchorSpan list
        anchors = self._parse_response(
            response, agent1_output.material_text, paragraphs
        )

        # Step 6: Trim count by config range
        if len(anchors) < self.config.min_anchors:
            print(
                f"[Agent 2] [Warning] Insufficient anchors "
                f"({len(anchors)} < {self.config.min_anchors})"
            )
        if len(anchors) > self.config.max_anchors:
            print(
                f"[Agent 2] [Warning] Too many anchors, trimming to first "
                f"{self.config.max_anchors}"
            )
            anchors = anchors[: self.config.max_anchors]

        # Step 7: Build AnchorSet output
        result = AnchorSet(
            anchors=anchors,
            anchor_discovery_reasoning=(
                f"Found {len(anchors)} key evidence spans via LLM analysis"
            ),
        )

        print(f"[Agent 2] [OK] Found {len(anchors)} anchors")
        for i, anchor in enumerate(anchors, 1):
            print(
                f"  {i}. Paragraph {anchor.paragraph_idx}: "
                f"{anchor.snippet[:30]}..."
            )

        return result

    # --------------------------------------------------------------------- #
    # Prompt Construction & Material Preprocessing
    # --------------------------------------------------------------------- #

    def _split_material_to_paragraphs(self, material_text: str) -> List[str]:
        """
        Split material by paragraphs.

        Current strategy:
        - Split by newline
        - Remove empty lines and trim whitespace
        """
        paragraphs = [
            p.strip() for p in material_text.split("\n") if p.strip()
        ]
        return paragraphs

    def _build_prompt(
        self,
        material_text: str,
        paragraphs: List[str],
        question_type: str,
        ability_point: str,
        synthesized_instruction: str,
    ) -> str:
        """
        Build anchor discovery prompt using AGENT2_PROMPT_SPEC with JSON output format requirement.

        Args:
            material_text: Material text (for logging and validation).
            paragraphs: Material split by paragraphs.
            question_type: Internal question type marker (e.g., "single-choice" / "essay").
            ability_point: Ability test point (for logging).
            synthesized_instruction: Fusion prompt.

        Returns:
            str: Constructed full prompt.
        """
        # 1) Build numbered paragraph text
        # Paragraph numbering starts from 0, aligned with paragraph_idx
        numbered_paragraphs = "\n\n".join(
            [f"【段落{i}】\n{para}" for i, para in enumerate(paragraphs)]
        )

        # 2) Internal question type -> Chinese question type name (for AGENT2_PROMPT_SPEC)
        question_type_map = {
            "single-choice": "单项选择题",
            "essay": "主观题",
        }
        question_type_human = question_type_map.get(
            question_type, question_type
        )

        # 3) Prepare placeholder data, inject into MD spec prompt
        placeholders = {
            "material": numbered_paragraphs,
            "question_type": question_type_human,
            "fusion_prompt": synthesized_instruction,
        }

        base_prompt = AGENT2_PROMPT_SPEC.format(**placeholders)

        # 4) [2025-12 Fix] Replace output format requirement with pure JSON
        # Original AGENT2_PROMPT_SPEC required natural language format, conflicting with JSON
        # [2025-12 Enhancement] Added reason_for_anchor and loc fields for semantic info
        # Chinese JSON format instruction - kept for LLM interaction
        json_format_instruction = f"""

【重要】输出格式（请严格遵守，只输出 JSON）：
忽略上文中关于"自然语言格式"或"material_with_anchors"的要求。
你只需要输出一个严格的 JSON 对象，格式如下：

{{
  "anchors": [
    {{
      "paragraph_idx": <整数，段落编号，从0开始，对应【段落N】中的N>,
      "start_char": <整数，该段落内起始字符位置，从0开始>,
      "end_char": <整数，该段落内结束字符位置>,
      "snippet": "<字符串，从原文中截取的锚点文本>",
      "loc": "<字符串，简短描述该锚点在原文中的位置，如'关于浇灌干湿的问答部分'>",
      "reason_for_anchor": "<字符串，说明该处为何适合作为命题依据：包含哪些知识点/逻辑关系，可结合哪些考查方向出题，下游模型可如何利用该锚点构造区分度>"
    }}
  ],
  "reasoning": "<字符串，简要说明你选择这些锚点的整体理由>"
}}

关键要求：
1. 只输出 JSON，不要输出任何其他文字、解释或 markdown 代码块。
2. paragraph_idx 对应【段落N】中的N（从0开始）。
3. snippet 必须是原文中的准确片段。
4. 【重要】reason_for_anchor 必须填写，说明该锚点的命题价值，这是下游出题模型的重要参考！
5. 锚点数量控制在 {self.config.min_anchors}-{self.config.max_anchors} 个。
6. 确保 JSON 语法正确：字符串用双引号，最后一项后不要加逗号。
"""

        full_prompt = base_prompt + json_format_instruction
        return full_prompt

    # --------------------------------------------------------------------- #
    # LLM Response Parsing & Fallback
    # --------------------------------------------------------------------- #

    def _parse_response(
        self,
        response: str,
        material_text: str,
        paragraphs: List[str],
    ) -> List[AnchorSpan]:
        """
        Parse LLM response text, extract anchor list.

        Priority strategy:
        0. Detect empty/non-JSON response, use fallback
        1. Try cleaning markdown wrappers then parse JSON
        2. If failed, try lightweight JSON repair
        3. If still failed, use simple heuristic fallback
        """
        anchors: List[AnchorSpan] = []

        # [2025-12 Enhancement] Detect empty response
        if not response or not response.strip():
            print("[Agent 2] [Warning] LLM returned empty response, using fallback...")
            return self._fallback_anchor_extraction(paragraphs)

        try:
            # Clean possible markdown code block markers
            json_text = response.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            # First attempt: Direct JSON parsing
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # Second attempt: Lightweight JSON repair then parse
                print("[Agent 2] First JSON parse failed, trying lightweight repair...")
                json_text = self._lightweight_json_repair(json_text)
                data = json.loads(json_text)
            anchor_list = data.get("anchors", [])

            for anchor_data in anchor_list:
                try:
                    para_idx = anchor_data["paragraph_idx"]
                    start_char = anchor_data["start_char"]
                    end_char = anchor_data["end_char"]

                    # Validate paragraph index
                    if para_idx < 0 or para_idx >= len(paragraphs):
                        print(
                            f"[Agent 2] [Warning] Paragraph index out of range: {para_idx}, skipping anchor"
                        )
                        continue

                    para_text = paragraphs[para_idx]

                    # snippet: Prefer model-provided field, otherwise extract from paragraph
                    if "snippet" in anchor_data and anchor_data["snippet"]:
                        snippet = anchor_data["snippet"]
                    else:
                        if (
                            start_char >= 0
                            and end_char <= len(para_text)
                            and start_char < end_char
                        ):
                            snippet = para_text[start_char:end_char]
                        else:
                            snippet = para_text[:100]  # fallback: extract paragraph start

                    # [2025-12 Enhancement] Extract semantic info fields
                    reason_for_anchor = anchor_data.get("reason_for_anchor", "")
                    loc = anchor_data.get("loc", "")

                    anchor = AnchorSpan(
                        paragraph_idx=para_idx,
                        start_char=start_char,
                        end_char=end_char,
                        snippet=snippet,
                        reason_for_anchor=reason_for_anchor,
                        loc=loc,
                    )
                    anchors.append(anchor)

                except (KeyError, ValueError, TypeError) as e:
                    print(
                        f"[Agent 2] [Warning] Failed to parse single anchor data, skipping: {e}"
                    )
                    continue

        except json.JSONDecodeError as e:
            print(f"[Agent 2] JSON parse failed: {e}")
            print(f"[Agent 2] Response snippet: {response[:200]}...")
            anchors = self._fallback_anchor_extraction(paragraphs)

        return anchors
# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)

    def _lightweight_json_repair(self, json_text: str) -> str:
        """
        Lightweight JSON repair: Try to fix common LLM output JSON syntax errors.

        Repair strategies:
        1. Remove trailing comma: e.g., [1, 2, 3,] -> [1, 2, 3]
        2. Remove illegal control characters
        3. [2025-12] Fix unterminated strings
        4. [2025-12] Truncate to last complete array element
        5. Add missing closing braces/brackets
        6. Fix single quotes to double quotes

        Args:
            json_text: Original JSON text

        Returns:
            str: Repaired JSON text
        """
        import re

        repaired = json_text

        # 1. Remove trailing comma (before ] or })
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # 2. Remove illegal control characters (keep newline and tab)
        repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)

        # 3. [2025-12] Fix unterminated strings
        repaired = self._repair_unterminated_strings(repaired)

        # 4. [2025-12] Try to truncate to last complete array element
        # When Gemini output is truncated, last anchor object may be incomplete
        repaired = self._truncate_to_last_complete_object(repaired)

        # 5. Count brackets (considering brackets inside strings), add missing closing brackets
        open_braces = 0
        close_braces = 0
        open_brackets = 0
        close_brackets = 0
        in_string = False
        escape_next = False

        for ch in repaired:
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
                close_braces += 1
            elif ch == '[':
                open_brackets += 1
            elif ch == ']':
                close_brackets += 1

        # Add missing ]
        if open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            repaired = repaired.rstrip() + ']' * missing_brackets
            print(f"[Agent 2] [Repair] Added {missing_brackets} missing ']'")

        # Add missing }
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            repaired = repaired.rstrip() + '}' * missing_braces
            print(f"[Agent 2] [Repair] Added {missing_braces} missing '}}'")

        # 6. Try to fix common quote issues (single quote to double quote)
        if "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')
            print("[Agent 2] [Repair] Single quotes replaced with double quotes")

        return repaired

    def _truncate_to_last_complete_object(self, json_text: str) -> str:
        """
        [2025-12] Truncate to last complete array element.

        When LLM output is truncated, the last object in anchors array may be incomplete, e.g.:
        {"anchors": [{...complete...}, {...complete...}, {"paragraph_idx": 4, "snippet": "truncated

        In this case, simply adding closing brackets will cause JSON syntax errors.
        Strategy: Find last complete object (ends with "}", followed by "," or "]"),
        then truncate the incomplete part.

        Args:
            json_text: JSON text

        Returns:
            Truncated JSON text
        """
        # Find position of "anchors" array
        anchors_match = json_text.find('"anchors"')
        if anchors_match == -1:
            return json_text

        # Find start position of "anchors": [
        bracket_start = json_text.find('[', anchors_match)
        if bracket_start == -1:
            return json_text

        # From [, find all complete object positions
        # Complete object: ends with "}", followed by "," or whitespace+"]"
        complete_object_ends = []
        i = bracket_start + 1
        brace_depth = 0
        in_string = False
        escape_next = False
        object_start = -1

        while i < len(json_text):
            ch = json_text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if ch == '\\' and in_string:
                escape_next = True
                i += 1
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                i += 1
                continue

            if in_string:
                i += 1
                continue

            if ch == '{':
                if brace_depth == 0:
                    object_start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and object_start != -1:
                    # Found a complete object, record its end position
                    complete_object_ends.append(i)
                    object_start = -1
            elif ch == ']' and brace_depth == 0:
                # Array ended
                break

            i += 1

        # Check if there's incomplete object (brace_depth > 0)
        if brace_depth > 0 and len(complete_object_ends) > 0:
            # Find position of last complete object
            last_complete_end = complete_object_ends[-1]

            # Check if there's incomplete content after last complete object
            remaining = json_text[last_complete_end + 1:].strip()
            if remaining.startswith(','):
                # Has comma, means there's incomplete object after, need to truncate
                truncated = json_text[:last_complete_end + 1]

                # Need to add ] and }
                print(f"[Agent 2] [Repair] Detected truncated anchor object, truncating to last complete object")

                # Check original JSON structure, add necessary closing symbols
                # Assume structure is {"anchors": [...], "reasoning": "..."}
                truncated = truncated.rstrip() + ']}'

                return truncated

        return json_text

    def _repair_unterminated_strings(self, s: str) -> str:
        """
        [2025-12] Fix unterminated strings.

        When LLM output is truncated, strings may be cut in the middle, causing "Unterminated string" error.

        Args:
            s: JSON string

        Returns:
            Repaired string
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
                # Add closing quote at the end
                s = s_stripped + '"'
                print("[Agent 2] [Repair] Added closing quote to fix truncated string")

        return s

    def _fallback_anchor_extraction(
        self, paragraphs: List[str]
    ) -> List[AnchorSpan]:
        """
        Fallback strategy when JSON parsing fails:
        - Use simple heuristic to select anchors from longer paragraphs.
        - Extract at least 3 anchors (improve fallback quality)
        """
        print("[Agent 2] Using fallback method to extract anchors from paragraph length heuristic...")
        anchors: List[AnchorSpan] = []

        # Fallback extracts at least 3 anchors for quality assurance
        fallback_min_anchors = max(3, self.config.min_anchors)

        for i, para in enumerate(paragraphs):
            if len(para) > 50:  # Only select paragraphs with > 50 chars
                anchor = AnchorSpan(
                    paragraph_idx=i,
                    start_char=0,
                    end_char=min(100, len(para)),
                    snippet=para[:100],
                )
                anchors.append(anchor)

                if len(anchors) >= fallback_min_anchors:
                    break

        # If all paragraphs are short, at least return first 3 non-empty paragraphs
        if len(anchors) < fallback_min_anchors:
            for i, para in enumerate(paragraphs):
                if para.strip() and i not in [a.paragraph_idx for a in anchors]:
                    anchor = AnchorSpan(
                        paragraph_idx=i,
                        start_char=0,
                        end_char=len(para),
                        snippet=para,
                    )
                    anchors.append(anchor)
                    if len(anchors) >= fallback_min_anchors:
                        break

        return anchors


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Agent2AnchorFinder",
]
