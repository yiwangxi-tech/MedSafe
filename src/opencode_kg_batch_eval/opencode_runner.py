import os
import socket
import subprocess
import time
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _can_connect(url: str) -> bool:
    raw = url.replace("http://", "").replace("https://", "")
    host_port = raw.split("/", 1)[0]
    if ":" not in host_port:
        return False
    host, port_text = host_port.rsplit(":", 1)
    try:
        port = int(port_text)
    except ValueError:
        return False
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@dataclass
class OpenCodeServerManager:
    config: Dict[str, Any]
    process: Optional[subprocess.Popen] = None

    def ensure_started(self):
        if _can_connect(self.config["attach_url"]):
            return
        if not self.config.get("auto_start_server", False):
            raise RuntimeError(
                f"OpenCode server is not reachable at {self.config['attach_url']}. "
                "Start `opencode serve` manually or enable auto_start_server in config."
            )
        command = self.config.get("serve_command") or []
        if not command:
            raise RuntimeError("auto_start_server is enabled but serve_command is empty.")
        env = os.environ.copy()
        config_path = self.config.get("config_path")
        if config_path:
            env["OPENCODE_CONFIG"] = str(config_path)
        self.process = subprocess.Popen(
            command,
            cwd=self.config.get("serve_cwd") or self.config.get("project_dir"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        deadline = time.time() + float(self.config.get("startup_timeout", 30))
        interval = float(self.config.get("startup_poll_interval", 1.0))
        while time.time() < deadline:
            if _can_connect(self.config["attach_url"]):
                return
            time.sleep(interval)
        raise TimeoutError(f"Timed out waiting for OpenCode server at {self.config['attach_url']}")

    def create_session(self) -> str:
        title = self.config.get("session_title", "batch-eval")
        url = self.config["attach_url"].rstrip("/") + "/session"
        payload = json.dumps({"title": title}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            raise RuntimeError(f"Failed to create OpenCode session at {url}: {e}") from e
        try:
            payload_obj = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"Invalid OpenCode session response: {raw}") from e
        session_id = payload_obj.get("id") or payload_obj.get("session_id")
        if not session_id:
            raise RuntimeError(f"OpenCode session response missing id: {raw}")
        return str(session_id)

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


@dataclass
class OpenCodeRunner:
    config: Dict[str, Any]
    session_id: Optional[str] = None
    model_override: Optional[str] = None

    def build_command(self, prompt: str) -> List[str]:
        command = [
            self.config.get("binary", "opencode"),
            "run",
            "--attach",
            self.config["attach_url"],
            "--dir",
            self.config["project_dir"],
        ]
        model = self.model_override or self.config.get("model")
        if model:
            command.extend(["--model", model])
        if self.config.get("agent"):
            command.extend(["--agent", self.config["agent"]])
        if self.session_id:
            command.extend(["--session", self.session_id])
        command.extend(self.config.get("extra_run_args", []))
        command.extend(["--format", "json"])
        command.append(prompt)
        return command

    def run_prompt(self, prompt: str) -> Dict[str, Any]:
        command = self.build_command(prompt)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=int(self.config.get("request_timeout", 600)),
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as e:
            stdout = _decode_timeout_stream(e.stdout)
            stderr = _decode_timeout_stream(e.stderr)
            events = _parse_jsonl_events(stdout)
            assistant_output = _extract_assistant_output(events, stdout)
            tool_summary = summarize_tool_calls(events)
            session_message_debug: Dict[str, Any] = {}
            session_payload = None
            if not assistant_output:
                session_payload, selected_message, session_message_debug = self._fetch_stable_session_payload()
                assistant_output = _extract_assistant_output_from_messages(selected_message)
                if tool_summary.get("total", 0) == 0:
                    session_tool_summary = summarize_tools_from_messages(session_payload)
                    if session_tool_summary.get("total", 0) > 0:
                        tool_summary = session_tool_summary
            return {
                "returncode": 124,
                "command": command,
                "stdout": stdout,
                "stderr": stderr or f"timeout_after_{int(self.config.get('request_timeout', 600))}_seconds",
                "events": events,
                "assistant_output": assistant_output,
                "event_summary": summarize_events(events),
                "tool_summary": tool_summary,
                "session_message_debug": session_message_debug,
                "kg_interaction_summary": summarize_kg_interactions(events, tool_summary, session_payload),
            }
        stdout = completed.stdout or ""
        events = _parse_jsonl_events(stdout)
        assistant_output = _extract_assistant_output(events, stdout)
        tool_summary = summarize_tool_calls(events)
        session_message_debug: Dict[str, Any] = {}
        session_payload = None
        if not assistant_output or tool_summary.get("total", 0) == 0:
            session_payload, selected_message, session_message_debug = self._fetch_stable_session_payload()
            if not assistant_output:
                assistant_output = _extract_assistant_output_from_messages(selected_message)
            if tool_summary.get("total", 0) == 0:
                session_tool_summary = summarize_tools_from_messages(session_payload)
                if session_tool_summary.get("total", 0) > 0:
                    tool_summary = session_tool_summary
        return {
            "returncode": completed.returncode,
            "command": command,
            "stdout": stdout,
            "stderr": completed.stderr or "",
            "events": events,
            "assistant_output": assistant_output,
            "event_summary": summarize_events(events),
            "tool_summary": tool_summary,
            "session_message_debug": session_message_debug,
            "kg_interaction_summary": summarize_kg_interactions(events, tool_summary, session_payload),
        }

    def _fetch_session_assistant_output(self) -> str:
        _, selected_message, _ = self._fetch_stable_session_payload()
        return _extract_assistant_output_from_messages(selected_message)

    def _fetch_session_payload_once(self) -> Any:
        if not self.session_id:
            return None
        base_url = self.config["attach_url"].rstrip("/")
        session_id = urllib.parse.quote(self.session_id, safe="")
        url = f"{base_url}/session/{session_id}/message?limit=20"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _fetch_stable_session_payload(self) -> tuple[Any, Any, Dict[str, Any]]:
        timeout_seconds = max(0.0, float(self.config.get("session_message_poll_timeout", 5.0)))
        interval_seconds = max(0.05, float(self.config.get("session_message_poll_interval", 0.25)))
        required_stable_reads = max(1, int(self.config.get("session_message_stable_reads", 2)))
        deadline = time.monotonic() + timeout_seconds
        previous_fingerprint = ""
        stable_reads = 0
        attempts = 0
        last_debug: Dict[str, Any] = {
            "source": "session_fallback",
            "status": "no_completed_assistant_message",
            "poll_attempts": 0,
            "stable_reads": 0,
        }

        while True:
            attempts += 1
            payload = self._fetch_session_payload_once()
            message = _latest_completed_assistant_message(payload)
            if message is not None:
                fingerprint = _assistant_message_fingerprint(message)
                stable_reads = stable_reads + 1 if fingerprint == previous_fingerprint else 1
                previous_fingerprint = fingerprint
                debug = _assistant_message_debug(message)
                debug.update({
                    "source": "session_fallback",
                    "poll_attempts": attempts,
                    "stable_reads": stable_reads,
                })
                if stable_reads >= required_stable_reads:
                    debug["status"] = "completed_stable"
                    return payload, [message], debug
                last_debug = debug
            else:
                last_debug.update({
                    "poll_attempts": attempts,
                    "stable_reads": 0,
                })

            if time.monotonic() >= deadline:
                last_debug["status"] = "poll_timeout"
                return None, None, last_debug
            time.sleep(interval_seconds)


def _decode_timeout_stream(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _parse_jsonl_events(stdout: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _extract_assistant_output(events: List[Dict[str, Any]], stdout: str) -> str:
    direct_candidates: List[str] = []
    fallback_candidates: List[str] = []

    for event in events:
        event_type = str(event.get("type", "")).lower()
        candidate = _find_final_json_candidate(event)
        if candidate:
            direct_candidates.append(candidate)
            continue
        if event_type in {"message", "message_delta", "text", "text_delta", "response", "final"}:
            fallback_candidates.extend(_collect_text_fragments(event))

    if direct_candidates:
        return direct_candidates[-1]
    if fallback_candidates:
        return "\n".join(fragment for fragment in fallback_candidates if fragment).strip()

    stripped = (stdout or "").strip()
    candidate = _find_final_json_candidate(stripped)
    if candidate:
        return candidate
    return ""


def _extract_assistant_output_from_messages(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            payload = payload.get("messages")
        elif isinstance(payload.get("data"), list):
            payload = payload.get("data")
    if not isinstance(payload, list):
        return ""
    fallback_candidates: List[str] = []
    for item in reversed(payload):
        if not isinstance(item, dict):
            continue
        info = item.get("info", {}) if isinstance(item.get("info", {}), dict) else {}
        role = str(info.get("role", item.get("role", ""))).strip().lower()
        if role and role != "assistant":
            continue
        structured = _find_final_json_candidate(info.get("structured_output"))
        if structured:
            return structured
        parts = item.get("parts", [])
        candidate = _find_final_json_candidate(parts)
        if candidate:
            return candidate
        fallback_candidates.extend(_collect_text_fragments(parts))
    return "\n".join(_dedupe_fragments(fallback_candidates)).strip()


def _session_messages(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            payload = payload.get("messages")
        elif isinstance(payload.get("data"), list):
            payload = payload.get("data")
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _latest_completed_assistant_message(payload: Any) -> Dict[str, Any] | None:
    for item in reversed(_session_messages(payload)):
        info = item.get("info", {}) if isinstance(item.get("info", {}), dict) else {}
        role = str(info.get("role", item.get("role", ""))).strip().lower()
        if role != "assistant":
            continue
        completed = info.get("time", {}).get("completed") if isinstance(info.get("time", {}), dict) else None
        # Do not reuse an older answer while the newest assistant message is still streaming.
        return item if completed is not None else None
    return None


def _assistant_message_fingerprint(message: Dict[str, Any]) -> str:
    info = message.get("info", {}) if isinstance(message.get("info", {}), dict) else {}
    parts = message.get("parts", [])
    try:
        parts_value = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        parts_value = repr(parts)
    return "|".join([
        str(info.get("id", "")),
        str(info.get("finish", "")),
        str(info.get("time", {}).get("completed", "") if isinstance(info.get("time", {}), dict) else ""),
        parts_value,
    ])


def _assistant_message_debug(message: Dict[str, Any]) -> Dict[str, Any]:
    info = message.get("info", {}) if isinstance(message.get("info", {}), dict) else {}
    time_info = info.get("time", {}) if isinstance(info.get("time", {}), dict) else {}
    error = info.get("error")
    return {
        "selected_message_id": str(info.get("id", "")),
        "role": str(info.get("role", "")),
        "completed": time_info.get("completed") is not None,
        "completed_at": time_info.get("completed"),
        "finish_reason": info.get("finish", ""),
        "error": error,
        "tokens": info.get("tokens", {}),
    }


def _find_final_json_candidate(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if _looks_like_final_json(text):
            return _extract_json_block(text)
        return ""
    if isinstance(value, dict):
        if any(key in value for key in ("final_decision", "final_labels", "final_pairs")):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return ""
        for item in value.values():
            candidate = _find_final_json_candidate(item)
            if candidate:
                return candidate
        return ""
    if isinstance(value, list):
        for item in value:
            candidate = _find_final_json_candidate(item)
            if candidate:
                return candidate
    return ""


def _collect_text_fragments(value: Any) -> List[str]:
    fragments: List[str] = []
    if isinstance(value, dict):
        part_type = str(value.get("type", "")).strip().lower()
        if part_type in {"tool", "tool_use", "tool_result", "tool-call", "tool-result", "skill"}:
            return fragments
        for key in ("text", "delta", "content"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                fragments.append(item.strip())
        for key, item in value.items():
            if key in {"output", "state", "tool", "skill_content"}:
                continue
            fragments.extend(_collect_text_fragments(item))
    elif isinstance(value, list):
        for item in value:
            fragments.extend(_collect_text_fragments(item))
    return _dedupe_fragments(fragments)


def _dedupe_fragments(fragments: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for fragment in fragments:
        value = str(fragment or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _looks_like_final_json(text: str) -> bool:
    return text.startswith("{") and any(token in text for token in ('"final_decision"', '"final_labels"', '"final_pairs"'))


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1].strip()
    return text[start:].strip()


def summarize_events(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "no_events"
    type_counts: Dict[str, int] = {}
    step_reasons: List[str] = []
    text_parts = 0
    error_messages: List[str] = []

    for event in events:
        event_type = str(event.get("type", "")).strip() or "unknown"
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        if event_type == "text":
            text_parts += 1
        if event_type == "step_finish":
            reason = str(event.get("part", {}).get("reason", "")).strip()
            if reason:
                step_reasons.append(reason)
        if event_type == "error":
            message = str(event.get("error", {}).get("data", {}).get("message", "")).strip()
            if message:
                error_messages.append(message)

    counts = ",".join(f"{key}:{value}" for key, value in sorted(type_counts.items()))
    pieces = [f"events[{counts}]"]
    if step_reasons:
        pieces.append(f"step_finish={','.join(step_reasons)}")
    pieces.append(f"text_parts={text_parts}")
    if error_messages:
        pieces.append(f"errors={' | '.join(error_messages)}")
    return "; ".join(pieces)


def _normalize_tool_name(tool_name: Any) -> str:
    return str(tool_name or "").strip().replace("_", "-").lower()


def _is_kg_tool(tool_name: Any) -> bool:
    return _normalize_tool_name(tool_name) == "neo4j-read-cypher"


def _extract_tool_call_id(event: Dict[str, Any], part: Dict[str, Any], state: Dict[str, Any]) -> str:
    for container in (part, state, event):
        for key in ("id", "call_id", "tool_call_id", "invocation_id"):
            value = container.get(key) if isinstance(container, dict) else None
            if value:
                return str(value)
    return ""


def summarize_kg_interactions(
    events: List[Dict[str, Any]],
    tool_summary: Optional[Dict[str, Any]] = None,
    session_payload: Any = None,
) -> Dict[str, Any]:
    calls_by_id: Dict[str, Dict[str, Any]] = {}
    fallback_calls: List[Dict[str, Any]] = []
    kg_event_count = 0

    for index, event in enumerate(events or []):
        if str(event.get("type", "")).strip() != "tool_use":
            continue
        part = event.get("part", {}) if isinstance(event.get("part", {}), dict) else {}
        state = part.get("state", {}) if isinstance(part.get("state", {}), dict) else {}
        tool_name = str(part.get("tool", "")).strip() or "unknown"
        if not _is_kg_tool(tool_name):
            continue
        kg_event_count += 1
        status = str(state.get("status", "unknown")).strip() or "unknown"
        status = status if status in {"completed", "failed", "running"} else "unknown"
        call_id = _extract_tool_call_id(event, part, state)
        call = {
            "call_id": call_id or f"event:{index}",
            "tool": tool_name,
            "status": status,
            "source": "events",
            "event_count": 1,
        }
        if call_id:
            current = calls_by_id.get(call_id)
            if current:
                current["event_count"] += 1
                current["status"] = status if status != "unknown" else current["status"]
            else:
                calls_by_id[call_id] = call
        else:
            fallback_calls.append(call)

    calls = list(calls_by_id.values())
    count_method = "event_call_id"
    if not calls and fallback_calls:
        calls = fallback_calls
        count_method = "event_without_call_id"
    if not calls:
        for index, item in enumerate((tool_summary or {}).get("tools", [])):
            if not isinstance(item, dict) or not _is_kg_tool(item.get("tool")):
                continue
            calls.append({
                "call_id": str(item.get("call_id", "") or f"tool_summary:{index}"),
                "tool": item.get("tool", "neo4j_read-cypher"),
                "status": str(item.get("status", "unknown") or "unknown"),
                "source": "tool_summary",
                "event_count": int(item.get("event_count", 0) or 0),
            })
        count_method = "tool_summary"
    if not calls and session_payload is not None:
        session_summary = summarize_tools_from_messages(session_payload)
        for index, item in enumerate(session_summary.get("tools", [])):
            if not isinstance(item, dict) or not _is_kg_tool(item.get("tool")):
                continue
            calls.append({
                "call_id": str(item.get("call_id", "") or f"session:{index}"),
                "tool": item.get("tool", "neo4j_read-cypher"),
                "status": str(item.get("status", "unknown") or "unknown"),
                "source": "session_messages",
                "event_count": 0,
            })
        count_method = "session_messages"

    status_counts = {"completed": 0, "failed": 0, "running": 0, "unknown": 0}
    for call in calls:
        status = str(call.get("status", "unknown") or "unknown")
        status_counts[status if status in status_counts else "unknown"] += 1
    return {
        "kg_tool": "neo4j_read-cypher",
        "kg_call_count": len(calls),
        "kg_event_count": kg_event_count,
        "count_method": count_method,
        **status_counts,
        "calls": calls,
    }


def summarize_tool_calls(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": 0,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "unknown": 0,
        "tools": [],
    }
    for event in events:
        if str(event.get("type", "")).strip() != "tool_use":
            continue
        part = event.get("part", {}) if isinstance(event.get("part", {}), dict) else {}
        state = part.get("state", {}) if isinstance(part.get("state", {}), dict) else {}
        status = str(state.get("status", "unknown")).strip() or "unknown"
        tool_name = str(part.get("tool", "")).strip() or "unknown"
        output_text = str(state.get("output", "")).strip()
        item = {
            "tool": tool_name,
            "status": status,
        }
        call_id = _extract_tool_call_id(event, part, state)
        if call_id:
            item["call_id"] = call_id
        if output_text and ("failed" in output_text.lower() or "error" in output_text.lower()):
            item["output_hint"] = output_text[:300]
        summary["tools"].append(item)
        summary["total"] += 1
        if status in {"completed", "failed", "running"}:
            summary[status] += 1
        else:
            summary["unknown"] += 1
    return summary


def summarize_tools_from_messages(payload: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": 0,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "unknown": 0,
        "tools": [],
    }
    seen = set()
    for item in _collect_tool_items(payload):
        key = item.get("call_id") or (
            item.get("tool", ""), item.get("status", ""), item.get("output_hint", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        summary["tools"].append(item)
        summary["total"] += 1
        status = item.get("status", "unknown")
        if status in {"completed", "failed", "running"}:
            summary[status] += 1
        else:
            summary["unknown"] += 1
    if summary["total"] == 0:
        tool_name = _find_tool_name_text(payload)
        if tool_name:
            summary["tools"].append({
                "tool": tool_name,
                "status": "completed",
            })
            summary["total"] = 1
            summary["completed"] = 1
    return summary


def _collect_tool_items(value: Any) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if isinstance(value, dict):
        item = _tool_item_from_message_part(value)
        if item:
            items.append(item)
        for child in value.values():
            items.extend(_collect_tool_items(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_collect_tool_items(child))
    return items


def _tool_item_from_message_part(value: Dict[str, Any]) -> Optional[Dict[str, str]]:
    part_type = str(value.get("type", "")).strip().lower()
    state = value.get("state", {}) if isinstance(value.get("state", {}), dict) else {}
    tool_name = ""
    if isinstance(value.get("tool"), dict):
        tool_name = str(value["tool"].get("name", "")).strip()
    if not tool_name:
        tool_name = str(value.get("tool", "") or value.get("name", "")).strip()
    if not tool_name and isinstance(state.get("tool"), str):
        tool_name = str(state.get("tool", "")).strip()

    is_tool_part = part_type in {
        "tool",
        "tool_use",
        "tool_result",
        "tool-call",
        "tool-result",
        "tool_call",
        "tool_call_delta",
    }
    if not is_tool_part and not tool_name.startswith("neo4j_"):
        return None

    if not tool_name:
        tool_name = "unknown"

    status = str(state.get("status", "") or value.get("status", "")).strip().lower()
    output_text = str(
        state.get("output", "")
        or value.get("output", "")
        or value.get("result", "")
        or value.get("content", "")
    ).strip()
    if status not in {"completed", "failed", "running"}:
        if output_text or part_type in {"tool_result", "tool-result"}:
            status = "completed"
        else:
            status = "unknown"

    item = {
        "tool": tool_name,
        "status": status,
    }
    call_id = str(value.get("call_id", "") or value.get("tool_call_id", "") or value.get("id", "")).strip()
    if call_id:
        item["call_id"] = call_id
    if output_text and ("failed" in output_text.lower() or "error" in output_text.lower()):
        item["output_hint"] = output_text[:300]
    return item


def _find_tool_name_text(value: Any) -> str:
    if isinstance(value, str):
        if "neo4j_read-cypher" in value:
            return "neo4j_read-cypher"
        if "neo4j_read_cypher" in value:
            return "neo4j_read-cypher"
        return ""
    if isinstance(value, dict):
        for child in value.values():
            tool_name = _find_tool_name_text(child)
            if tool_name:
                return tool_name
    elif isinstance(value, list):
        for child in value:
            tool_name = _find_tool_name_text(child)
            if tool_name:
                return tool_name
    return ""
