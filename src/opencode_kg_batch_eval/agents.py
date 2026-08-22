import json
from typing import Any, Dict, List

from .io_utils import safe_json_dumps
from .labels import LABEL_OPTION_MAP, label_to_option
from .opencode_runner import OpenCodeRunner, OpenCodeServerManager
from .parsing import extract_first_json_object, load_json_object, parse_audit_result


VALID_AUDIT_LABELS = {"B", "C", "D", "E", "F", "G"}
RELATION_LABELS = {"E", "F"}

AUDIT_SYSTEM_BASE = """
你是一名临床药师。请根据患者信息、诊断和处方药品，对电子处方进行审核。

你必须使用 neo4j MCP 工具检索知识图谱证据并进行最终判断（除重复用药外，知识图谱当中不包含重复用药的判断信息)。

MCP 工具使用要求：
1. 只能执行只读 Cypher 查询。
2. 不允许编造药学知识。
3. 如果知识图谱未查到证据，直接输出“未查到证据”。
4. 不要描述工具调用过程。
5. 不要输出多余解释。

选项含义如下：
A: 合理
定义：处方无任何适应症、用法用量、配伍等问题。
B: 用法、用量不适宜
定义：处方存在用法、用量不合理的情况，如剂量过大、过小，给药频次不合理等。
C: 适应症不适宜
定义：处方中药品的适应症不合理，如处方药品的适应症与患者的诊断不符，或处方药品没有适应症。
D: 药品剂型或给药途径不适宜
定义：处方中药品的剂型或给药途径不合理，如口服药物处方为注射剂，或注射药物处方为口服剂等。
E: 有配伍禁忌或不良相互作用
定义：处方中存在配伍禁忌或不良相互作用的药品组合，如两种药物之间存在严重的相互作用，可能导致患者出现不良反应或降低疗效。
F: 重复给药
定义：处方中存在重复给药的情况，如同一药品的不同剂型或不同药品的相同成分等(包括成分相同、作用机制相同)。
G: 遴选的药品不适宜
定义：所选药物的适应证与患者的临床诊断相符，但在药物的选择上存在一些不适合患者个体情况的因素，从而可能影响治疗效果或增加不良反应的风险；或是药物不适用于患者的特定生理状态、特殊人群类别或基础健康状况，例如儿童、老人、孕妇、哺乳期妇女、肝肾功能不全者；或是患者的诊断或身体状况属于药品说明书中明确列出的禁忌症、注意事项。例如：为消化性溃疡活动期患者开具阿司匹林。

重要补充：
1. 若患者诊断或已知病情与某药的 CONTRAINDICATION、PRECAUTION、SpecialPopulation当中限制直接匹配，通常应标记 G，不要轻易输出 A。
2. 若剂量判断依赖体重且处方未提供体重，不要仅凭猜测标记 B。
3. 若未在图谱中检索到能与目前处方诊断匹配的适应症，则判定为 C。
4. 若图谱中没有直接的相互作用或重复给药证据，不要猜测 E/F; 当确认为F时，不需要在标记为E。
5. 若存在多个问题，final_decision 选最主要问题，final_labels 保留全部问题标签。
6. 判断遴选的药品不适宜时，遵守知识图谱当中检索到的节点信息，不要自行推测患者可能的并发症进行匹配。
""".strip()

FINAL_AUDIT_RULES = """
你只能输出最终审核结果，不要输出推理过程，不要解释，不要输出多余文字。
请严格使用以下固定 JSON 格式输出：
{
  "prescription_id": "",
  "final_decision": "",
  "final_labels": [],
  "final_pairs": [],
  "evidence_summary": []
}

判定规则：
1. 如果处方完全合理：
   - final_decision = "A"
   - final_labels = []
   - final_pairs = []
2. 如果处方存在问题：
   - final_decision 必须是 B/C/D/E/F/G 中最主要的 1 个选项
   - final_labels 只能从 B/C/D/E/F/G 中选择，可多选，且必须包含 final_decision
3. final_pairs 仅在 E 或 F 时填写，格式为：
   [["药品A", "药品B", "E"]]
   或
   [["药品A", "药品B", "F"]]
4. evidence_summary 只保留简短证据短句，不要解释原因，不要编号。
5. 最终只能输出一个 JSON 对象，不允许输出任何其他内容。
""".strip()

SUBAGENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "agent_1": {
        "name": "剂量/途径审核子智能体",
        "labels": ["B", "D"],
        "role": "审核处方药品用法用量问题、剂型/给药途径问题。",
    },
    "agent_2": {
        "name": "适应症/遴选审核子智能体",
        "labels": ["C", "G"],
        "role": "审核处方药品适应症匹配问题、药品遴选适宜性问题。",
    },
    "agent_3": {
        "name": "相互作用/重复给药审核子智能体",
        "labels": ["E", "F"],
        "role": "审核处方内药物相互作用/配伍禁忌问题、重复给药问题。",
    },
}

DEFAULT_SUBAGENT_ORDER = ["agent_1", "agent_2", "agent_3"]


def _skill_instruction(agent_cfg: Dict[str, Any], subagent_id: str = "") -> str:
    subagent_skill_names = agent_cfg.get("subagent_skill_names", {})
    if isinstance(subagent_skill_names, dict) and subagent_id:
        skill_name = str(subagent_skill_names.get(subagent_id, "") or "").strip()
        if skill_name:
            return f"Use the {skill_name} skill.\n\n"
    skill_name = str(agent_cfg.get("skill_name", "prescription-audit") or "").strip()
    if not skill_name:
        return ""
    return f"Use the {skill_name} skill.\n\n"


def _empty_tool_summary() -> Dict[str, Any]:
    return {
        "total": 0,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "unknown": 0,
        "tools": [],
    }


def _merge_tool_summaries(*summaries: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = _empty_tool_summary()
    seen = set()
    for summary in summaries:
        tools = summary.get("tools", []) if isinstance(summary, dict) else []
        for item in tools:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool", "")).strip() or "unknown"
            status = str(item.get("status", "unknown")).strip() or "unknown"
            output_hint = str(item.get("output_hint", "")).strip()
            call_id = str(item.get("call_id", "")).strip()
            key = call_id or (tool_name, status, output_hint)
            if key in seen:
                continue
            seen.add(key)
            merged_item = {
                "tool": tool_name,
                "status": status,
            }
            if output_hint:
                merged_item["output_hint"] = output_hint
            if call_id:
                merged_item["call_id"] = call_id
            merged["tools"].append(merged_item)
            merged["total"] += 1
            if status in {"completed", "failed", "running"}:
                merged[status] += 1
            else:
                merged["unknown"] += 1
    return merged


def _empty_kg_interaction_summary() -> Dict[str, Any]:
    return {
        "kg_tool": "neo4j_read-cypher",
        "kg_call_count": 0,
        "kg_event_count": 0,
        "count_method": "none",
        "completed": 0,
        "failed": 0,
        "running": 0,
        "unknown": 0,
        "calls": [],
    }


def _merge_kg_interaction_summaries(*summaries: Dict[str, Any]) -> Dict[str, Any]:
    merged = _empty_kg_interaction_summary()
    calls_by_key: Dict[str, Dict[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        merged["kg_event_count"] += int(summary.get("kg_event_count", 0) or 0)
        for index, call in enumerate(summary.get("calls", []) if isinstance(summary.get("calls", []), list) else []):
            if not isinstance(call, dict):
                continue
            key = str(call.get("call_id", "")).strip() or f"{call.get('source', '')}:{index}:{call.get('tool', '')}:{call.get('status', '')}"
            current = calls_by_key.get(key)
            if current:
                current["event_count"] = int(current.get("event_count", 0) or 0) + int(call.get("event_count", 0) or 0)
                if current.get("status") == "unknown" and call.get("status") != "unknown":
                    current["status"] = call.get("status")
            else:
                calls_by_key[key] = dict(call)
    merged["calls"] = list(calls_by_key.values())
    merged["kg_call_count"] = len(merged["calls"])
    merged["count_method"] = "merged_call_id"
    for call in merged["calls"]:
        status = str(call.get("status", "unknown") or "unknown")
        merged[status if status in {"completed", "failed", "running"} else "unknown"] += 1
    return merged


def _tool_summary_has_problem(tool_summary: Dict[str, Any]) -> bool:
    for item in tool_summary.get("tools", []) if isinstance(tool_summary, dict) else []:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "failed":
            return True
        if item.get("output_hint"):
            return True
    return False


def _judge_called_tool(judge_result: Dict[str, Any]) -> bool:
    tool_summary = judge_result.get("tool_summary", {})
    if isinstance(tool_summary, dict) and int(tool_summary.get("total", 0) or 0) > 0:
        return True
    return "tool_use" in str(judge_result.get("event_summary", ""))


def _is_clean_json_only_output(raw_output: str) -> bool:
    value = (raw_output or "").strip()
    return value.startswith("{") and value.endswith("}")


def _judge_should_fallback(
    judge_result: Dict[str, Any],
    final_raw_output: str,
    final_parsed: Dict[str, Any],
    rule_result: Dict[str, Any],
    agent_cfg: Dict[str, Any],
) -> str:
    if judge_result.get("returncode") != 0:
        return "judge_nonzero_returncode"
    if final_parsed.get("status") != "ok":
        return "judge_parse_failed"
    if not _is_clean_json_only_output(final_raw_output):
        return "judge_output_not_json_only"
    if _judge_called_tool(judge_result):
        return "judge_called_tool"
    if _tool_summary_has_problem(judge_result.get("tool_summary", {})):
        return "judge_tool_error"

    allow_override = bool(agent_cfg.get("judge_allow_override", False))
    rule_decision = str(rule_result.get("final_decision", "")).strip().upper()
    if not allow_override and rule_decision and rule_decision != "A":
        judge_labels = {
            label_to_option(label) or str(label).strip().upper()
            for label in final_parsed.get("labels", [])
        }
        if rule_decision not in judge_labels:
            return "judge_overrode_positive_rule_result"
    return ""


def _format_medications(medications: List[Dict[str, Any]]) -> str:
    lines = []
    for index, med in enumerate(medications, start=1):
        lines.append(
            "\n".join(
                [
                    f"{index}. drug_name: {med.get('drug_name', '')}",
                    f"   specification: {med.get('specification', '')}",
                    f"   manufacturer: {med.get('manufacturer', '')}",
                    f"   usage_dosage: {med.get('usage_dosage', '')}",
                    f"   administration_route: {med.get('administration_route', '')}",
                ]
            )
        )
    return "\n".join(lines)


def _build_context_block(item: Dict[str, Any]) -> str:
    patient = item.get("patient_info", {})
    medications = item.get("medications", [])
    option_block = "\n".join([f"{key}: {value}" for key, value in LABEL_OPTION_MAP.items()])
    return (
        f"处方ID: {item.get('prescription_id', '')}\n"
        f"科室: {patient.get('dept', '')}\n"
        f"年龄: {patient.get('age', '')}\n"
        f"性别: {patient.get('gender', '')}\n"
        f"诊断: {patient.get('diagnosis', '')}\n\n"
        f"药品列表:\n{_format_medications(medications)}\n\n"
        f"标签定义:\n{option_block}"
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y"}


def _as_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _normalize_string_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _normalize_related_pairs(value: Any, label: str) -> List[List[str]]:
    if label not in RELATION_LABELS or not isinstance(value, list):
        return []
    pairs = []
    seen = set()
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            continue
        drug_a = str(item[0]).strip()
        drug_b = str(item[1]).strip()
        pair_label = str(item[2]).strip().upper()
        if not drug_a or not drug_b or pair_label != label:
            continue
        normalized = tuple(sorted([drug_a, drug_b]) + [label])
        if normalized in seen:
            continue
        seen.add(normalized)
        pairs.append([normalized[0], normalized[1], normalized[2]])
    return pairs


def _normalize_subagent_id(value: Any) -> str:
    subagent_id = str(value or "").strip()
    return subagent_id if subagent_id in SUBAGENT_DEFINITIONS else ""


def _configured_subagent_ids(agent_cfg: Dict[str, Any]) -> List[str]:
    configured = agent_cfg.get("sub_agents")
    if not isinstance(configured, list):
        return DEFAULT_SUBAGENT_ORDER[:]

    ids = []
    seen = set()
    for item in configured:
        if not isinstance(item, dict):
            continue
        subagent_id = _normalize_subagent_id(item.get("id"))
        if subagent_id and subagent_id not in seen:
            ids.append(subagent_id)
            seen.add(subagent_id)
    return ids or DEFAULT_SUBAGENT_ORDER[:]


def _labels_for_subagents(subagent_ids: List[str]) -> List[str]:
    labels = []
    seen = set()
    for subagent_id in subagent_ids:
        for label in SUBAGENT_DEFINITIONS[subagent_id]["labels"]:
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def _agent_model(agent_cfg: Dict[str, Any], subagent_id: str) -> str:
    models = agent_cfg.get("agent_models", {})
    if not isinstance(models, dict):
        return ""
    definition = SUBAGENT_DEFINITIONS[subagent_id]
    for key in [subagent_id, definition.get("name", "")]:
        model = str(models.get(key, "") or "").strip()
        if model:
            return model
    return ""


def _subagent_schema(item: Dict[str, Any], subagent_id: str) -> Dict[str, Any]:
    labels = SUBAGENT_DEFINITIONS[subagent_id]["labels"]
    return {
        "prescription_id": str(item.get("prescription_id", "")),
        "subagent_id": subagent_id,
        "results": [
            {
                "label": label,
                "issue_present": False,
                "confidence": 0.0,
                "evidence_summary": [],
                "related_pairs": [],
            }
            for label in labels
        ],
    }


def build_subagent_prompt(
    item: Dict[str, Any],
    subagent_id: str,
    agent_cfg: Dict[str, Any] | None = None,
) -> str:
    agent_cfg = agent_cfg or {}
    definition = SUBAGENT_DEFINITIONS[subagent_id]
    labels = definition["labels"]
    max_evidence = int(agent_cfg.get("max_agent_evidence_bullets", 5))
    return (
        f"{AUDIT_SYSTEM_BASE}\n\n"
        f"{_skill_instruction(agent_cfg, subagent_id)}"
        "你现在是负责特定处方审核任务的临床药师。\n"
        "上文“最终判断”仅指你分配的标签：不要判断其他标签，不要输出 final_decision；处方整体的 final_decision 由后续 chief judge 汇总。\n\n"
        f"子智能体ID: {subagent_id}\n"
        f"子智能体名称: {definition['name']}\n"
        f"任务定位: {definition['role']}\n"
        f"只审核并输出这些标签: {', '.join(labels)}\n"
        "证据检索和具体知识图谱查询方式以当前加载的 skill 为准。\n\n"
        "输出契约:\n"
        "1. 最终回复只能是一个完整的 JSON 对象，不得包含分析文字、重复 JSON、Markdown 代码块或工具调用说明。\n"
        "2. JSON 的最外层以 { 开始、以 } 结束；输出前检查 { } 和 [ ] 是否完整匹配。\n"
        "3. 最外层字段必须且仅为 prescription_id, subagent_id, results；subagent_id 必须与当前分配一致。\n"
        "4. results 必须且只能包含每个分配标签各 1 项；每项仅使用 label, issue_present, confidence, evidence_summary, related_pairs。\n"
        "5. result.label 只能使用已分配标签；只有处方内容或检索证据支持时 issue_present 才能为 true；confidence 必须在 0 到 1 之间。\n"
        f"6. evidence_summary 最多 {max_evidence} 条短句，不要解释。\n"
        "7. related_pairs 仅在 E 或 F 时填写，格式为 [[\"药品A\", \"药品B\", \"E\"]] 或 [[\"药品A\", \"药品B\", \"F\"]]；其他标签必须为 []。\n\n"
        f"处方上下文:\n{_build_context_block(item)}\n\n"
        f"输出 JSON 示例:\n{json.dumps(_subagent_schema(item, subagent_id), ensure_ascii=False)}"
    )


def build_subagent_finalize_prompt(
    item: Dict[str, Any],
    subagent_id: str,
    raw_output: str,
    max_evidence: int,
) -> str:
    prescription_id = str(item.get("prescription_id", ""))
    return (
        "你刚才没有输出合格的特定问题审核的 JSON。现在不要再调用任何工具，不要继续分析。\n"
        "请只根据已有处方上下文、已检索证据和上一轮输出，生成一个完整 JSON 对象。最外层必须以 { 开始、以 } 结束，不得输出其他内容。\n\n"
        f"prescription_id 必须是 \"{prescription_id}\"。\n"
        f"subagent_id 必须是 \"{subagent_id}\"。\n"
        f"results 必须且只能包含这些标签: {json.dumps(SUBAGENT_DEFINITIONS[subagent_id]['labels'], ensure_ascii=False)}。\n"
        "每个 results 项的键必须是: label, issue_present, confidence, evidence_summary, related_pairs。\n"
        f"每个 evidence_summary 最多 {max_evidence} 条简短证据短句；related_pairs 仅在 E 或 F 时填写，其他标签必须为 []。\n\n"
        f"处方上下文:\n{_build_context_block(item)}\n\n"
        f"上一轮助手输出:\n{raw_output}\n\n"
        f"输出 JSON 示例:\n{json.dumps(_subagent_schema(item, subagent_id), ensure_ascii=False)}"
    )


def _parse_label_payload(
    payload: Dict[str, Any],
    raw_output: str,
    prescription_id: str,
    label: str,
    max_evidence: int,
) -> Dict[str, Any]:
    payload_label = str(payload.get("label", label)).strip().upper()
    status = "ok"
    if payload_label != label or "issue_present" not in payload:
        status = "error"

    issue_present = _as_bool(payload.get("issue_present", False)) if status == "ok" else False
    related_pairs = _normalize_related_pairs(payload.get("related_pairs", []), label)
    if label in RELATION_LABELS and related_pairs:
        issue_present = True

    return {
        "status": status,
        "prescription_id": prescription_id,
        "label": label,
        "subagent_id": "",
        "issue_present": issue_present,
        "confidence": _as_confidence(payload.get("confidence", 0.0)),
        "evidence_summary": _normalize_string_list(payload.get("evidence_summary", []), max_evidence),
        "related_pairs": related_pairs,
        "raw_output": raw_output,
    }


def parse_subagent_result(
    raw_output: str,
    prescription_id: str,
    subagent_id: str,
    max_evidence: int,
) -> Dict[str, Any]:
    labels = SUBAGENT_DEFINITIONS[subagent_id]["labels"]
    candidate = extract_first_json_object(raw_output)
    if not candidate:
        status = "unknown"
        payload = {}
    else:
        loaded = load_json_object(candidate)
        if loaded is None:
            payload = {}
            status = "error"
        else:
            payload = loaded
            status = "ok"

    if status == "ok" and str(payload.get("subagent_id", "")).strip() != subagent_id:
        status = "error"

    results = payload.get("results", []) if status == "ok" else []
    result_by_label: Dict[str, Dict[str, Any]] = {}
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            label = str(result.get("label", "")).strip().upper()
            if label in labels and label not in result_by_label:
                result_by_label[label] = result

    parsed_by_label: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        if status != "ok" or label not in result_by_label:
            parsed_by_label[label] = {
                "status": status if status != "ok" else "error",
                "prescription_id": prescription_id,
                "label": label,
                "subagent_id": subagent_id,
                "issue_present": False,
                "confidence": 0.0,
                "evidence_summary": [],
                "related_pairs": [],
                "raw_output": raw_output,
            }
            continue
        parsed = _parse_label_payload(result_by_label[label], raw_output, prescription_id, label, max_evidence)
        parsed["subagent_id"] = subagent_id
        parsed_by_label[label] = parsed

    subagent_status = "ok" if all(item.get("status") == "ok" for item in parsed_by_label.values()) else "error"
    if status in {"unknown", "error"}:
        subagent_status = status
    return {
        "status": subagent_status,
        "prescription_id": prescription_id,
        "subagent_id": subagent_id,
        "parsed_by_label": parsed_by_label,
        "raw_output": raw_output,
    }


def _single_medication_negative_subagent(item: Dict[str, Any], subagent_id: str, max_evidence: int) -> Dict[str, Any] | None:
    medications = item.get("medications", [])
    prescription_id = str(item.get("prescription_id", ""))
    if subagent_id != "agent_3" or len(medications) >= 2:
        return None

    evidence = "处方药品少于两种，无法形成处方内药物相互作用或重复给药药品对。"
    parsed_by_label = {}
    for label in SUBAGENT_DEFINITIONS[subagent_id]["labels"]:
        parsed_by_label[label] = {
            "status": "ok",
            "prescription_id": prescription_id,
            "label": label,
            "subagent_id": subagent_id,
            "issue_present": False,
            "confidence": 1.0,
            "evidence_summary": [evidence][:max_evidence],
            "related_pairs": [],
            "raw_output": "",
            "returncode": 0,
            "stderr": "",
            "short_circuit": True,
        }
    return {
        "status": "ok",
        "prescription_id": prescription_id,
        "subagent_id": subagent_id,
        "parsed_by_label": parsed_by_label,
        "raw_output": "",
        "returncode": 0,
        "stderr": "",
        "short_circuit": True,
    }


def aggregate_agent_results(
    prescription_id: str,
    agent_results: List[Dict[str, Any]],
    agent_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    agent_cfg = agent_cfg or {}
    threshold = float(agent_cfg.get("confidence_threshold", 0.5))
    priority = [str(item).strip().upper() for item in agent_cfg.get("decision_priority", ["E", "F", "B", "C", "D", "G"])]
    max_evidence = int(agent_cfg.get("max_final_evidence_bullets", agent_cfg.get("max_agent_evidence_bullets", 8)))

    positive = [
        item for item in agent_results
        if item.get("issue_present") and float(item.get("confidence", 0.0)) >= threshold
    ]
    labels = sorted({
        str(item.get("label", "")).upper()
        for item in positive
        if str(item.get("label", "")).upper() in VALID_AUDIT_LABELS
    })

    pairs: List[List[str]] = []
    seen_pairs = set()
    evidence: List[str] = []
    for item in positive:
        label = str(item.get("label", "")).upper()
        for pair in item.get("related_pairs", []):
            key = tuple(pair)
            if key not in seen_pairs:
                seen_pairs.add(key)
                pairs.append(pair)
        for text in item.get("evidence_summary", []):
            line = f"{label}: {text}"
            if line not in evidence:
                evidence.append(line)
            if len(evidence) >= max_evidence:
                break
        if len(evidence) >= max_evidence:
            break

    if not labels:
        return {
            "prescription_id": prescription_id,
            "final_decision": "A",
            "final_labels": [],
            "final_pairs": [],
            "evidence_summary": evidence[:max_evidence] or ["No B-G issue found by the three specialist sub-agents."],
        }

    decision = labels[0]
    for candidate in priority:
        if candidate in labels:
            decision = candidate
            break

    return {
        "prescription_id": prescription_id,
        "final_decision": decision,
        "final_labels": labels,
        "final_pairs": pairs,
        "evidence_summary": evidence[:max_evidence],
    }


def build_judge_prompt(
    item: Dict[str, Any],
    agent_results: List[Dict[str, Any]],
    rule_result: Dict[str, Any],
    agent_cfg: Dict[str, Any] | None = None,
) -> str:
    agent_cfg = agent_cfg or {}
    prescription_id = str(item.get("prescription_id", ""))
    max_evidence = int(agent_cfg.get("max_final_evidence_bullets", 8))
    compact_results = []
    for result in agent_results:
        compact_results.append({
            "subagent_id": result.get("subagent_id"),
            "label": result.get("label"),
            "status": result.get("status"),
            "issue_present": result.get("issue_present"),
            "confidence": result.get("confidence"),
            "evidence_summary": result.get("evidence_summary", []),
            "related_pairs": result.get("related_pairs", []),
        })
    return (
        f"{AUDIT_SYSTEM_BASE}\n\n"
        f"{FINAL_AUDIT_RULES}\n\n"
        "你是一名高级临床药学专家，负责汇总最终处方审核结论。\n"
        "三个专职的临床药师已经分别完成 B/D、C/G、E/F 审核。\n"
        "除非专职临床药师审核结果存在无法根据现有证据解决的内部矛盾，否则不要调用任何工具。\n"
        "你的任务是根据专职临床药师审核结果进行汇总分析，输出最终审核 JSON。\n"
        f"evidence_summary 最多 {max_evidence} 条短句，不要解释。\n"
        "输出前自检：删除所有分析文字、重复 JSON、Markdown 代码块，只保留最终一个 JSON 对象。\n\n"
        f"处方上下文:\n{_build_context_block(item)}\n\n"
        f"专职临床药师审核结果:\n{safe_json_dumps(compact_results)}\n\n"
        f"规则聚合候选结果:\n{safe_json_dumps(rule_result)}\n\n"
        f"prescription_id 必须是 \"{prescription_id}\"。"
    )


def _runner_for_subagent(base_runner: OpenCodeRunner) -> OpenCodeRunner:
    return OpenCodeRunner(config=base_runner.config)


def _run_subagent_finalize(
    runner: OpenCodeRunner,
    item: Dict[str, Any],
    subagent_id: str,
    raw_output: str,
    max_evidence: int,
) -> Dict[str, Any] | None:
    finalize_prompt = build_subagent_finalize_prompt(item, subagent_id, raw_output, max_evidence)
    finalize_result = runner.run_prompt(finalize_prompt)
    finalize_raw_output = finalize_result.get("assistant_output", "")
    finalize_parsed = parse_subagent_result(
        finalize_raw_output,
        str(item.get("prescription_id", "")),
        subagent_id,
        max_evidence,
    )
    if finalize_result.get("returncode") == 0 and finalize_parsed.get("status") == "ok":
        for parsed in finalize_parsed["parsed_by_label"].values():
            parsed["returncode"] = finalize_result.get("returncode")
            parsed["stderr"] = finalize_result.get("stderr", "")
        return {
            "result": finalize_result,
            "raw_output": finalize_raw_output,
            "parsed": finalize_parsed,
        }
    return None


def _run_subagent(
    runner: OpenCodeRunner,
    server: OpenCodeServerManager,
    item: Dict[str, Any],
    config: Dict[str, Any],
    subagent_id: str,
    max_evidence: int,
    prescription_id: str,
) -> Dict[str, Any]:
    agent_cfg = config.get("multi_agent", {})
    short_circuit_result = _single_medication_negative_subagent(item, subagent_id, max_evidence)
    if short_circuit_result is not None:
        return {
            "subagent_id": subagent_id,
            "parsed": short_circuit_result,
            "raw_agent_output": {
                "parsed": short_circuit_result,
                "command": [],
                "returncode": 0,
                "stderr": "",
                "assistant_output": "",
                "tool_summary": _empty_tool_summary(),
                "event_summary": "short_circuit_single_medication",
                "finalize": None,
            },
            "tool_summary": _empty_tool_summary(),
            "event_summaries": [f"{subagent_id}=short_circuit_single_medication"],
        }

    subagent_runner = _runner_for_subagent(runner)
    subagent_runner.session_id = config["opencode"].get("session_id") or server.create_session()
    prompt = build_subagent_prompt(item, subagent_id, agent_cfg)
    model_override = _agent_model(agent_cfg, subagent_id) or None
    subagent_runner.model_override = model_override
    try:
        result = subagent_runner.run_prompt(prompt)
    finally:
        subagent_runner.model_override = None

    raw_output = result.get("assistant_output", "")
    initial_result = result
    combined_tool_summary = initial_result.get("tool_summary", {})
    combined_kg_interaction_summary = initial_result.get("kg_interaction_summary", {})
    initial_event_summary = initial_result.get("event_summary", "")
    parsed = parse_subagent_result(raw_output, prescription_id, subagent_id, max_evidence)
    for label_result in parsed["parsed_by_label"].values():
        label_result["returncode"] = result.get("returncode")
        label_result["stderr"] = result.get("stderr", "")
    finalize_debug = None

    if result.get("returncode") == 0 and parsed.get("status") != "ok":
        subagent_runner.model_override = model_override
        try:
            finalized = _run_subagent_finalize(subagent_runner, item, subagent_id, raw_output, max_evidence)
        finally:
            subagent_runner.model_override = None
        if finalized is not None:
            finalize_debug = {
                "command": finalized["result"].get("command", []),
                "returncode": finalized["result"].get("returncode"),
                "stderr": finalized["result"].get("stderr", ""),
                "assistant_output": finalized["raw_output"],
                "tool_summary": finalized["result"].get("tool_summary", {}),
                "event_summary": finalized["result"].get("event_summary", ""),
                "session_message_debug": finalized["result"].get("session_message_debug", {}),
                "kg_interaction_summary": finalized["result"].get("kg_interaction_summary", {}),
            }
            combined_tool_summary = _merge_tool_summaries(
                combined_tool_summary,
                finalized["result"].get("tool_summary", {}),
            )
            combined_kg_interaction_summary = _merge_kg_interaction_summaries(
                combined_kg_interaction_summary,
                finalized["result"].get("kg_interaction_summary", {}),
            )
            result = dict(finalized["result"])
            result["tool_summary"] = combined_tool_summary
            result["kg_interaction_summary"] = combined_kg_interaction_summary
            result["event_summary"] = "; ".join(
                item for item in [initial_event_summary, finalized["result"].get("event_summary", "")] if item
            )
            raw_output = finalized["raw_output"]
            parsed = finalized["parsed"]

    event_summaries = []
    if finalize_debug:
        if initial_event_summary:
            event_summaries.append(f"{subagent_id}={initial_event_summary}")
    elif result.get("event_summary"):
        event_summaries.append(f"{subagent_id}={result.get('event_summary')}")
    if finalize_debug and finalize_debug.get("event_summary"):
        event_summaries.append(f"{subagent_id}_finalize={finalize_debug.get('event_summary')}")

    return {
        "subagent_id": subagent_id,
        "parsed": parsed,
        "raw_agent_output": {
            "parsed": parsed,
            "command": result.get("command", []),
            "returncode": result.get("returncode"),
            "stderr": result.get("stderr", ""),
            "assistant_output": raw_output,
            "tool_summary": result.get("tool_summary", {}),
            "event_summary": result.get("event_summary", ""),
            "session_message_debug": result.get("session_message_debug", {}),
            "kg_interaction_summary": result.get("kg_interaction_summary", {}),
            "finalize": finalize_debug,
        },
        "tool_summary": result.get("tool_summary", {}),
        "kg_interaction_summary": result.get("kg_interaction_summary", {}),
        "event_summaries": event_summaries,
    }


def run_multi_agent_audit(
    runner: OpenCodeRunner,
    server: OpenCodeServerManager,
    item: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    agent_cfg = config.get("multi_agent", {})
    subagent_ids = _configured_subagent_ids(agent_cfg)
    label_order = _labels_for_subagents(subagent_ids)
    max_evidence = int(agent_cfg.get("max_agent_evidence_bullets", 5))
    prescription_id = str(item.get("prescription_id", ""))

    merged_tool_summary: Dict[str, Any] = _empty_tool_summary()
    merged_kg_interaction_summary: Dict[str, Any] = _empty_kg_interaction_summary()
    event_summaries: List[str] = []
    agent_results_by_label: Dict[str, Dict[str, Any]] = {}
    raw_agent_outputs: Dict[str, Any] = {}

    subagent_results = []
    for subagent_id in subagent_ids:
        subagent_results.append(
            _run_subagent(runner, server, item, config, subagent_id, max_evidence, prescription_id)
        )

    for subagent_result in subagent_results:
        subagent_id = subagent_result["subagent_id"]
        parsed = subagent_result["parsed"]
        raw_agent_outputs[subagent_id] = subagent_result["raw_agent_output"]
        merged_tool_summary = _merge_tool_summaries(merged_tool_summary, subagent_result.get("tool_summary", {}))
        merged_kg_interaction_summary = _merge_kg_interaction_summaries(
            merged_kg_interaction_summary,
            subagent_result.get("kg_interaction_summary", {}),
        )
        event_summaries.extend(subagent_result.get("event_summaries", []))
        for label, label_result in parsed.get("parsed_by_label", {}).items():
            agent_results_by_label[label] = label_result

    agent_results: List[Dict[str, Any]] = [
        agent_results_by_label[label]
        for label in label_order
        if label in agent_results_by_label
    ]

    failed_results = [
        item for item in agent_results
        if item.get("returncode") not in (0, None) or item.get("status") != "ok"
    ]
    if failed_results:
        return {
            "raw_output": "",
            "agent_results": agent_results,
            "raw_agent_outputs": raw_agent_outputs,
            "rule_result": None,
            "judge_result": None,
            "tool_summary": merged_tool_summary,
            "kg_interaction_summary": merged_kg_interaction_summary,
            "event_summary": "; ".join(event_summaries),
            "returncode": 1,
            "stderr": "three_subagent_audit_failed",
        }

    rule_result = aggregate_agent_results(prescription_id, agent_results, agent_cfg)
    judge_enabled = bool(agent_cfg.get("judge_enabled", True))
    if judge_enabled:
        runner.session_id = config["opencode"].get("session_id") or server.create_session()
        judge_prompt = build_judge_prompt(item, agent_results, rule_result, agent_cfg)
        judge_result = runner.run_prompt(judge_prompt)
        final_raw_output = judge_result.get("assistant_output", "")
        final_parsed = parse_audit_result(final_raw_output, prescription_id)
        fallback_reason = _judge_should_fallback(judge_result, final_raw_output, final_parsed, rule_result, agent_cfg)
        if fallback_reason:
            final_raw_output = json.dumps(rule_result, ensure_ascii=False)
            final_parsed = parse_audit_result(final_raw_output, prescription_id)
            event_summaries.append(f"judge_fallback={fallback_reason}")
        else:
            merged_tool_summary = _merge_tool_summaries(merged_tool_summary, judge_result.get("tool_summary", {}))
            merged_kg_interaction_summary = _merge_kg_interaction_summaries(
                merged_kg_interaction_summary,
                judge_result.get("kg_interaction_summary", {}),
            )
        if judge_result.get("event_summary"):
            event_summaries.append(f"judge={judge_result.get('event_summary')}")
        judge_debug = {
            "command": judge_result.get("command", []),
            "returncode": judge_result.get("returncode"),
            "stderr": judge_result.get("stderr", ""),
            "assistant_output": final_raw_output,
            "tool_summary": judge_result.get("tool_summary", {}),
            "event_summary": judge_result.get("event_summary", ""),
            "session_message_debug": judge_result.get("session_message_debug", {}),
            "rule_result": rule_result,
            "fallback_reason": fallback_reason,
        }
        return {
            "raw_output": final_raw_output,
            "agent_results": agent_results,
            "raw_agent_outputs": raw_agent_outputs,
            "rule_result": rule_result,
            "judge_result": judge_debug,
            "tool_summary": merged_tool_summary,
            "kg_interaction_summary": merged_kg_interaction_summary,
            "event_summary": "; ".join(event_summaries),
            "returncode": 0 if final_parsed.get("status") == "ok" else judge_result.get("returncode"),
            "stderr": "" if final_parsed.get("status") == "ok" else judge_result.get("stderr", ""),
        }

    final_raw_output = json.dumps(rule_result, ensure_ascii=False)
    return {
        "raw_output": final_raw_output,
        "agent_results": agent_results,
        "raw_agent_outputs": raw_agent_outputs,
        "rule_result": rule_result,
        "judge_result": None,
        "tool_summary": merged_tool_summary,
        "kg_interaction_summary": merged_kg_interaction_summary,
        "event_summary": "; ".join(event_summaries),
        "returncode": 0,
        "stderr": "",
    }
