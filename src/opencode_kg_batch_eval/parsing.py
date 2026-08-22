import itertools
import json
import re
from typing import Any, Dict, List

from .io_utils import safe_json_dumps
from .labels import ALL_ERROR_LABELS, RELATION_LABELS, canonicalize_label


ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text or "")


def extract_first_json_object(text: str) -> str:
    value = strip_ansi(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", value, re.S | re.I)
    if fenced:
        value = fenced.group(1).strip()
    start = value.find("{")
    if start < 0:
        return value
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(value)):
        ch = value[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return value[start: idx + 1]
    return value


def _close_truncated_json(candidate: str) -> str:
    value = (candidate or "").strip()
    if not value.startswith("{"):
        return value

    stack = []
    in_string = False
    escaped = False
    for ch in value:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return value

    if in_string:
        return value
    return value + "".join(reversed(stack))


def load_json_object(candidate: str) -> Dict[str, Any] | None:
    if not candidate:
        return None
    try:
        loaded = json.loads(candidate)
    except Exception:
        repaired = _close_truncated_json(candidate)
        if repaired == candidate:
            return None
        try:
            loaded = json.loads(repaired)
        except Exception:
            return None
    return loaded if isinstance(loaded, dict) else None


def get_gold_prescription_reasonable(item: Dict[str, Any]) -> bool:
    medications = item.get("medications", [])
    if not medications:
        return True
    return all(bool(med.get("is_reasonable", True)) for med in medications)


def get_gold_prescription_labels(item: Dict[str, Any]) -> List[str]:
    labels = set()
    for med in item.get("medications", []):
        for label in med.get("gold_standard_labels", []) or []:
            canonical = canonicalize_label(label)
            if canonical and canonical != "合理":
                labels.add(canonical)
    return sorted(labels)


def build_gold_relation_edges(item: Dict[str, Any]) -> List[List[str]]:
    medications = item.get("medications", [])
    edges = set()
    for relation_label in RELATION_LABELS:
        idxs = []
        for idx, med in enumerate(medications):
            med_labels = {canonicalize_label(x) for x in (med.get("gold_standard_labels", []) or [])}
            if relation_label in med_labels:
                idxs.append(idx)
        for i, j in itertools.combinations(idxs, 2):
            drug_a = str(medications[i].get("drug_name", "")).strip()
            drug_b = str(medications[j].get("drug_name", "")).strip()
            if drug_a and drug_b:
                pair = tuple(sorted([drug_a, drug_b]))
                edges.add((pair[0], pair[1], relation_label))
    return [list(edge) for edge in sorted(edges)]


def format_medications_for_export(medications: List[Dict[str, Any]]) -> str:
    simplified = []
    for med in medications:
        simplified.append({
            "drug_name": med.get("drug_name", ""),
            "specification": med.get("specification", ""),
            "manufacturer": med.get("manufacturer", ""),
            "usage_dosage": med.get("usage_dosage", ""),
            "administration_route": med.get("administration_route", ""),
        })
    return safe_json_dumps(simplified)


def _normalize_relation_edges(items: Any) -> List[List[str]]:
    if not isinstance(items, list):
        return []
    normalized = set()
    for item in items:
        if not isinstance(item, list) or len(item) != 3:
            continue
        drug_a = str(item[0]).strip()
        drug_b = str(item[1]).strip()
        label = canonicalize_label(item[2])
        if not drug_a or not drug_b or label not in RELATION_LABELS:
            continue
        pair = sorted([drug_a, drug_b])
        normalized.add((pair[0], pair[1], label))
    return [list(edge) for edge in sorted(normalized)]


def _normalize_labels(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    labels = []
    for item in items:
        canonical = canonicalize_label(item)
        if canonical in ALL_ERROR_LABELS:
            labels.append(canonical)
    return sorted(set(labels))


def _decision_to_reasonable(decision: str) -> Any:
    value = str(decision or "").strip().upper()
    if value == "A":
        return True
    if value in {"B", "C", "D", "E", "F", "G"}:
        return False
    return None


def _normalize_pairs_from_final_pairs(items: Any) -> List[List[str]]:
    if not isinstance(items, list):
        return []
    normalized = set()
    for item in items:
        if not isinstance(item, list) or len(item) != 3:
            continue
        drug_a = str(item[0]).strip()
        drug_b = str(item[1]).strip()
        label_option = str(item[2]).strip().upper()
        label = canonicalize_label(label_option)
        if not drug_a or not drug_b or label not in RELATION_LABELS:
            continue
        pair = sorted([drug_a, drug_b])
        normalized.add((pair[0], pair[1], label))
    return [list(edge) for edge in sorted(normalized)]


def parse_audit_result(raw_text: str, prescription_id: str) -> Dict[str, Any]:
    candidate = extract_first_json_object(raw_text)
    if not candidate:
        return {
            "status": "unknown",
            "prescription_id": prescription_id,
            "is_reasonable": None,
            "labels": [],
            "relation_edges": [],
            "evidence_summary": [],
        }
    payload = load_json_object(candidate)
    if payload is None:
        return {
            "status": "error",
            "prescription_id": prescription_id,
            "is_reasonable": None,
            "labels": [],
            "relation_edges": [],
            "evidence_summary": [],
        }

    if "final_decision" in payload or "final_labels" in payload or "final_pairs" in payload:
        decision = str(payload.get("final_decision", "")).strip().upper()
        labels = _normalize_labels(payload.get("final_labels", []))
        if decision == "A":
            labels = []
        elif decision in {"B", "C", "D", "E", "F", "G", "H"}:
            decision_label = canonicalize_label(decision)
            if decision_label in ALL_ERROR_LABELS:
                labels = sorted(set(labels + [decision_label]))
        return {
            "status": "ok",
            "prescription_id": str(payload.get("prescription_id", prescription_id)),
            "is_reasonable": _decision_to_reasonable(decision),
            "labels": labels,
            "relation_edges": _normalize_pairs_from_final_pairs(payload.get("final_pairs", [])),
            "evidence_summary": payload.get("evidence_summary", []) if isinstance(payload.get("evidence_summary", []), list) else [],
        }

    return {
        "status": "ok",
        "prescription_id": str(payload.get("prescription_id", prescription_id)),
        "is_reasonable": bool(payload.get("is_reasonable", False)) if payload.get("is_reasonable") is not None else None,
        "labels": _normalize_labels(payload.get("labels", [])),
        "relation_edges": _normalize_relation_edges(payload.get("relation_edges", [])),
        "evidence_summary": payload.get("evidence_summary", []) if isinstance(payload.get("evidence_summary", []), list) else [],
    }
