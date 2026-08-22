import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from .io_utils import append_csv_row, append_jsonl, load_progress, loads_list, safe_json_dumps
from .labels import labels_to_option_string
from .metrics import compute_metrics
from .agents import run_multi_agent_audit
from .opencode_runner import OpenCodeRunner, OpenCodeServerManager
from .parsing import (
    build_gold_relation_edges,
    extract_first_json_object,
    format_medications_for_export,
    get_gold_prescription_labels,
    get_gold_prescription_reasonable,
    parse_audit_result,
)


def _resolve_limit(config: Dict[str, Any], cli_limit: int | None) -> int | None:
    if cli_limit is not None:
        return cli_limit
    return config.get("evaluation", {}).get("limit")


def _requires_neo4j_tool_call(config: Dict[str, Any]) -> bool:
    return bool(config.get("evaluation", {}).get("require_neo4j_tool_call", True))


def _multi_agent_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("multi_agent", {}).get("enabled", False))


def _ensure_output_dir(output_root: str, run_name: str) -> str:
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _result_paths(run_dir: str) -> Dict[str, str]:
    return {
        "progress_csv": os.path.join(run_dir, "progress.csv"),
        "raw_jsonl": os.path.join(run_dir, "raw_outputs.jsonl"),
        "debug_jsonl": os.path.join(run_dir, "opencode_debug.jsonl"),
        "result_csv": os.path.join(run_dir, "prescription_results.csv"),
        "metrics_binary_csv": os.path.join(run_dir, "metrics_binary.csv"),
        "metrics_multilabel_csv": os.path.join(run_dir, "metrics_multilabel.csv"),
        "metrics_edge_csv": os.path.join(run_dir, "metrics_edge.csv"),
        "leaderboard_csv": os.path.join(run_dir, "leaderboard.csv"),
        "summary_json": os.path.join(run_dir, "summary.json"),
    }


def _has_tool_error(tool_summary: Dict[str, Any]) -> bool:
    has_clean_neo4j_result = False
    has_hint_error = False
    for item in tool_summary.get("tools", []) if isinstance(tool_summary, dict) else []:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool", "")).strip().replace("_", "-")
        if item.get("status") == "failed":
            return True
        if tool_name == "neo4j-read-cypher" and item.get("status") == "completed" and not item.get("output_hint"):
            has_clean_neo4j_result = True
        if item.get("output_hint"):
            has_hint_error = True
    if has_hint_error and not has_clean_neo4j_result:
        return True
    return False


def _has_required_neo4j_call(tool_summary: Dict[str, Any]) -> bool:
    tools = tool_summary.get("tools", []) if isinstance(tool_summary, dict) else []
    for item in tools:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool", "")).strip().replace("_", "-")
        if tool_name == "neo4j-read-cypher" and item.get("status") == "completed":
            return True
    return False


def _merge_tool_summaries(*summaries: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "total": 0,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "unknown": 0,
        "tools": [],
    }
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


def _merge_kg_interaction_summaries(*summaries: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "kg_tool": "neo4j_read-cypher",
        "kg_call_count": 0,
        "kg_event_count": 0,
        "count_method": "merged_call_id",
        "completed": 0,
        "failed": 0,
        "running": 0,
        "unknown": 0,
        "calls": [],
    }
    calls_by_key: Dict[str, Dict[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        merged["kg_event_count"] += int(summary.get("kg_event_count", 0) or 0)
        calls = summary.get("calls", []) if isinstance(summary.get("calls", []), list) else []
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            key = str(call.get("call_id", "")).strip() or f"{call.get('source', '')}:{index}:{call.get('tool', '')}:{call.get('status', '')}"
            current = calls_by_key.get(key)
            if current:
                current["event_count"] = int(current.get("event_count", 0) or 0) + int(call.get("event_count", 0) or 0)
            else:
                calls_by_key[key] = dict(call)
    merged["calls"] = list(calls_by_key.values())
    merged["kg_call_count"] = len(merged["calls"])
    for call in merged["calls"]:
        status = str(call.get("status", "unknown") or "unknown")
        merged[status if status in {"completed", "failed", "running"} else "unknown"] += 1
    return merged


def _json_object_payload(raw_output: str) -> str:
    candidate = extract_first_json_object(raw_output or "").strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate
    return ""


def _is_clean_json_only_output(raw_output: str) -> bool:
    value = (raw_output or "").strip()
    if not value:
        return False
    return value.startswith("{") and value.endswith("}")


def _quality_warnings(parsed: Dict[str, Any]) -> List[str]:
    if parsed.get("status") != "ok":
        return []
    evidence = parsed.get("evidence_summary", [])
    if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
        return ["empty_evidence_summary"]
    if parsed.get("is_reasonable") is True and parsed.get("labels"):
        return ["reasonable_decision_with_labels"]
    if parsed.get("is_reasonable") is False and not parsed.get("labels"):
        return ["unreasonable_decision_without_labels"]
    return []


def _write_outputs(config: Dict[str, Any], paths: Dict[str, str], result_df: pd.DataFrame, metric_sheets, summary, run_name: str):
    output_cfg = config.get("output", {})
    result_columns = output_cfg.get("result_columns", [])
    if output_cfg.get("write_result_csv", True):
        export_columns = [col for col in result_columns if col in result_df.columns] or [
            "prescription_id",
            "dept",
            "age",
            "gender",
            "diagnosis",
            "medications",
            "gold_standard_labels",
            "audit_result",
            "predicted_labels",
            "predicted_relation_edges",
            "_output_status",
        ]
        result_df[export_columns].to_csv(paths["result_csv"], index=False, encoding="utf-8-sig")
    if output_cfg.get("write_metrics_binary_csv", True):
        metric_sheets["binary"].to_csv(paths["metrics_binary_csv"], index=False, encoding="utf-8-sig")
    if output_cfg.get("write_metrics_multilabel_csv", True):
        metric_sheets["multilabel"].to_csv(paths["metrics_multilabel_csv"], index=False, encoding="utf-8-sig")
    if output_cfg.get("write_metrics_edge_csv", True):
        metric_sheets["edge"].to_csv(paths["metrics_edge_csv"], index=False, encoding="utf-8-sig")
    leaderboard = pd.DataFrame([{"run_name": run_name, **summary}])
    if output_cfg.get("write_leaderboard_csv", True):
        leaderboard.to_csv(paths["leaderboard_csv"], index=False, encoding="utf-8-sig")
    if output_cfg.get("write_summary_json", True):
        with open(paths["summary_json"], "w", encoding="utf-8") as f:
            json.dump({"run_name": run_name, **summary}, f, ensure_ascii=False, indent=2)


def run_batch_evaluation(
    config: Dict[str, Any],
    input_json: str,
    output_root: str,
    cli_limit: int | None = None,
    cli_force_rerun: bool = False,
) -> str:
    with open(input_json, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    limit = _resolve_limit(config, cli_limit)
    if limit:
        data = data[: int(limit)]

    run_name = config.get("run_name", "opencode_kg_batch_eval")
    run_dir = _ensure_output_dir(output_root, run_name)
    paths = _result_paths(run_dir)

    force_rerun = bool(cli_force_rerun or config.get("evaluation", {}).get("force_rerun", False))
    if force_rerun:
        for output_path in paths.values():
            if os.path.exists(output_path):
                os.remove(output_path)

    progress_df = load_progress(paths["progress_csv"])
    processed_ids = set(progress_df["prescription_id"].astype(str).tolist()) if not progress_df.empty else set()

    server = OpenCodeServerManager(config["opencode"])
    runner = OpenCodeRunner(config["opencode"])
    require_neo4j_tool_call = _requires_neo4j_tool_call(config)
    server.ensure_started()

    try:
        for item in tqdm(data, desc=f"[{run_name}] eval"):
            prescription_id = str(item.get("prescription_id", ""))
            if prescription_id in processed_ids:
                continue

            patient = item.get("patient_info", {})
            medications = item.get("medications", [])
            gold_labels = get_gold_prescription_labels(item)
            gold_is_reasonable = get_gold_prescription_reasonable(item)
            gold_relation_edges = build_gold_relation_edges(item)

            raw_output = ""
            parsed = {
                "status": "unknown",
                "prescription_id": prescription_id,
                "is_reasonable": None,
                "labels": [],
                "relation_edges": [],
                "evidence_summary": [],
            }
            error_msg = ""
            accepted = False
            agent_trace: Dict[str, Any] | None = None
            merged_tool_summary = {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "running": 0,
                "unknown": 0,
                "tools": [],
            }
            merged_kg_interaction_summary = _merge_kg_interaction_summaries()
            merged_event_summary = ""

            max_retry = int(config.get("evaluation", {}).get("max_retry", 1))
            request_interval = float(config.get("evaluation", {}).get("request_interval", 1.0))

            if not _multi_agent_enabled(config):
                raise RuntimeError("This 3MA project requires multi_agent.enabled=true; the legacy single-prompt path has been removed.")


            result = {
                "returncode": 1,
                "stderr": "",
                "event_summary": "",
                "tool_summary": {},
            }
            for attempt in range(1, max_retry + 2):
                multi_agent_result = run_multi_agent_audit(runner, server, item, config)
                agent_trace = {
                    "agent_results": multi_agent_result.get("agent_results", []),
                    "raw_agent_outputs": multi_agent_result.get("raw_agent_outputs", {}),
                    "rule_result": multi_agent_result.get("rule_result"),
                    "judge_result": multi_agent_result.get("judge_result"),
                    "kg_interaction_summary": multi_agent_result.get("kg_interaction_summary", {}),
                }
                raw_output = multi_agent_result.get("raw_output", "")
                merged_tool_summary = _merge_tool_summaries(
                    merged_tool_summary,
                    multi_agent_result.get("tool_summary", {}),
                )
                merged_kg_interaction_summary = _merge_kg_interaction_summaries(
                    merged_kg_interaction_summary,
                    multi_agent_result.get("kg_interaction_summary", {}),
                )
                merged_event_summary = (
                    f"{merged_event_summary}; attempt_{attempt}={multi_agent_result.get('event_summary', '')}"
                    if merged_event_summary
                    else f"attempt_{attempt}={multi_agent_result.get('event_summary', '')}"
                )
                result = {
                    "returncode": multi_agent_result.get("returncode", 0),
                    "stderr": multi_agent_result.get("stderr", ""),
                    "event_summary": multi_agent_result.get("event_summary", ""),
                    "tool_summary": multi_agent_result.get("tool_summary", {}),
                    "kg_interaction_summary": multi_agent_result.get("kg_interaction_summary", {}),
                }
                has_tool_error = _has_tool_error(merged_tool_summary)
                has_required_neo4j_call = (
                    _has_required_neo4j_call(merged_tool_summary)
                    or int(merged_kg_interaction_summary.get("completed", 0) or 0) > 0
                )
                json_payload = _json_object_payload(raw_output)
                parsed = parse_audit_result(raw_output, prescription_id)
                quality_warnings = _quality_warnings(parsed)
                if (
                    parsed["status"] == "ok"
                    and json_payload
                    and not has_tool_error
                    and (has_required_neo4j_call or not require_neo4j_tool_call)
                    and not quality_warnings
                ):
                    raw_output = json_payload
                    accepted = True
                    error_msg = ""
                    break
                warnings = []
                if parsed["status"] != "ok":
                    warnings.append(f"parse_failed_attempt_{attempt}: no_final_json_in_event_stream")
                elif not json_payload:
                    warnings.append(f"format_violation_attempt_{attempt}: no_extractable_json_object")
                if has_tool_error:
                    warnings.append(f"tool_error_attempt_{attempt}: tool_output_contains_error")
                if require_neo4j_tool_call and not has_required_neo4j_call:
                    warnings.append(f"quality_warning_attempt_{attempt}: missing_required_neo4j_call")
                warnings.extend(f"quality_warning_attempt_{attempt}: {warning}" for warning in quality_warnings)
                error_msg = "; ".join(warnings) if warnings else f"unexpected_output_attempt_{attempt}"
                if result.get("stderr"):
                    error_msg = f"{error_msg}; stderr={result.get('stderr')}"
                if merged_event_summary:
                    error_msg = f"{error_msg}; {merged_event_summary}"
                if merged_tool_summary:
                    error_msg = f"{error_msg}; tools={safe_json_dumps(merged_tool_summary)}"
                if attempt < max_retry + 1:
                    time.sleep(request_interval)

            if not accepted:
                parsed = {
                    "status": "invalid",
                    "prescription_id": prescription_id,
                    "is_reasonable": None,
                    "labels": [],
                    "relation_edges": [],
                    "evidence_summary": parsed.get("evidence_summary", []),
                }

            pred_is_reasonable = parsed["is_reasonable"]
            pred_labels = parsed["labels"]
            pred_relation_edges = parsed["relation_edges"]
            if pred_is_reasonable is None:
                pred_is_reasonable = False
                pred_labels = []
                pred_relation_edges = []

            ground_truth_option = labels_to_option_string(gold_is_reasonable, gold_labels)
            predicted_option = labels_to_option_string(pred_is_reasonable, pred_labels)
            is_correct = ground_truth_option == predicted_option

            progress_row = {
                "prescription_id": prescription_id,
                "dept": patient.get("dept", ""),
                "age": patient.get("age", ""),
                "gender": patient.get("gender", ""),
                "diagnosis": patient.get("diagnosis", ""),
                "medications": format_medications_for_export(medications),
                "gold_standard_labels": safe_json_dumps(gold_labels),
                "audit_result": extract_first_json_object(raw_output) if raw_output else f"call_failed_or_empty: {error_msg}",
                "predicted_labels": safe_json_dumps(pred_labels),
                "predicted_relation_edges": safe_json_dumps(pred_relation_edges),
                "_gold_is_reasonable": gold_is_reasonable,
                "_pred_is_reasonable": pred_is_reasonable,
                "_gold_labels": safe_json_dumps(gold_labels),
                "_pred_labels": safe_json_dumps(pred_labels),
                "_gold_relation_edges": safe_json_dumps(gold_relation_edges),
                "_pred_relation_edges": safe_json_dumps(pred_relation_edges),
                "_output_status": parsed["status"],
            }
            append_csv_row(paths["progress_csv"], progress_row)
            if config.get("output", {}).get("write_raw_outputs_jsonl", True):
                append_jsonl(paths["raw_jsonl"], {
                    "prescription_id": prescription_id,
                    "gold_standard_labels": gold_labels,
                    "predicted_labels": pred_labels,
                    "ground_truth": ground_truth_option,
                    "predicted": predicted_option,
                    "is_correct": is_correct,
                    "error": error_msg,
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "tool_summary": merged_tool_summary,
                    "kg_interaction_summary": merged_kg_interaction_summary,
                    "event_summary": merged_event_summary or result.get("event_summary", ""),
                    "format_clean": _is_clean_json_only_output(raw_output),
                    "tool_error_detected": _has_tool_error(merged_tool_summary),
                    "agent_trace": agent_trace,
                })
            if config.get("output", {}).get("write_opencode_debug_jsonl", False):
                append_jsonl(paths["debug_jsonl"], {
                    "prescription_id": prescription_id,
                    "accepted": accepted,
                    "command": result.get("command", []),
                    "returncode": result.get("returncode"),
                    "stderr": result.get("stderr", ""),
                    "stdout": result.get("stdout", ""),
                    "events": result.get("events", []),
                    "assistant_output": raw_output,
                    "tool_summary": merged_tool_summary,
                    "kg_interaction_summary": merged_kg_interaction_summary,
                    "event_summary": merged_event_summary or result.get("event_summary", ""),
                    "agent_trace": agent_trace,
                })
            processed_ids.add(prescription_id)
            time.sleep(request_interval)

    finally:
        server.stop()

    result_df = pd.read_csv(paths["progress_csv"], encoding="utf-8-sig")
    result_df["_gold_labels_obj"] = result_df["_gold_labels"].apply(loads_list)
    result_df["_pred_labels_obj"] = result_df["_pred_labels"].apply(loads_list)
    result_df["_gold_relation_edges_obj"] = result_df["_gold_relation_edges"].apply(loads_list)
    result_df["_pred_relation_edges_obj"] = result_df["_pred_relation_edges"].apply(loads_list)

    metric_sheets, summary = compute_metrics(result_df)
    _write_outputs(config, paths, result_df, metric_sheets, summary, run_name)
    return paths["leaderboard_csv"]
