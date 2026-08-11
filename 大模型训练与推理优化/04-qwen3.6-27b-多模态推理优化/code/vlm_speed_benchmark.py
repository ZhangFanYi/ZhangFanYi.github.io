#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import resource
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any


PORTABLE_VERSION = "hcc_vlm_speed_portable_v1"
AUTHORITATIVE_RUNNER_SHA256 = "91584e23889abc31c760e72ac0b5e381180db1939bc2bc57817b814a8fe06f6b"
DEFAULT_MODEL = "Qwen3.6-27B-FP8-vlm-judge"
MAX_JUDGE_INPUT_TOKENS = 262144
QWEN_REQUEST_IMAGE_PIXELS = {"min_pixels": 65536, "max_pixels": 1048576}
SCORE_DIMENSIONS = (
    "faithfulness", "completeness", "correctness", "target_entity_grounding",
    "answerability", "spatial_validity", "temporal_validity", "action_relevance",
)
TARGET_ENTITY_GROUNDING = "target_entity_grounding"
GENERAL_DIMENSIONS = ("faithfulness", "completeness", "correctness", TARGET_ENTITY_GROUNDING)
SPATIAL_TYPES = {"bbox_grounding", "pointing_coordinate", "spatial_relation", "depth_distance_viewpoint", "affordance_where_to_place"}
TEMPORAL_TYPES = {"temporal_order"}
ACTION_TYPES = {"task_progress", "next_subtask_or_action", "failure_anomaly"}
ROBOT_SOURCES = {"robovqa", "cosmos_reason1_sft", "eo_data15m", "robo2vlm1", "cosmos_reason1_benchmark", "cosmos_reason1_rl", "eo_bench"}
VALID_REASON_CODE_ORDER = (
    "clear_grounding", "ambiguous_question", "object_or_answer_not_visible",
    "media_corrupt_or_low_quality", "question_answer_mismatch", "text_media_mismatch",
    "unverifiable_measurement_or_depth", "hallucinated_object_or_relation",
    "unsupported_temporal_or_motion", "required_target_absent_or_ungrounded",
    "unsafe_or_sensitive", "format_or_ocr_noise", "other",
)
VALID_REASON_CODES = set(VALID_REASON_CODE_ORDER)

_TOKENIZER: Any | None = None
_TOKENIZER_PATH: Path | None = None
_TOKENIZER_LOCK = threading.Lock()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def clean_content(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def score_track_and_dimensions(record: dict[str, Any]) -> tuple[str, list[str]]:
    question_type = str(record.get("question_type") or "")
    embodied = bool(record.get("is_robot_or_trajectory_derived")) or str(record.get("source_key") or "") in ROBOT_SOURCES or question_type in (SPATIAL_TYPES | TEMPORAL_TYPES | ACTION_TYPES)
    if not embodied:
        return "general", list(GENERAL_DIMENSIONS)
    dimensions = list(GENERAL_DIMENSIONS) + ["answerability"]
    if question_type in SPATIAL_TYPES:
        dimensions.append("spatial_validity")
    if question_type in TEMPORAL_TYPES:
        dimensions.append("temporal_validity")
    if bool(record.get("is_robot_or_trajectory_derived")) or str(record.get("source_key") or "") in ROBOT_SOURCES or question_type in ACTION_TYPES:
        dimensions.append("action_relevance")
    return "embodied", dimensions


def response_schema() -> dict[str, Any]:
    dimensions = list(SCORE_DIMENSIONS)
    score = {"type": ["integer", "null"], "minimum": 1, "maximum": 5}
    return {
        "name": "cosmos3_vqa_multidim_score_v3_target_entity_grounding",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [*dimensions, "reason_codes", "rationale"],
            "properties": {
                **{name: score for name in dimensions},
                "reason_codes": {"type": "array", "maxItems": 4, "items": {"type": "string", "enum": list(VALID_REASON_CODE_ORDER)}},
                "rationale": {"type": "string", "maxLength": 160},
            },
        },
    }


def prompt_text(record: dict[str, Any], turns: list[dict[str, Any]], notes: list[str], track: str, applicable: list[str]) -> str:
    inapplicable = [axis for axis in SCORE_DIMENSIONS if axis not in applicable]
    lines = [
        "You are a VQA data-quality scorer. You do not decide keep/reject, data routing, or a final threshold.",
        f"Evaluation track: {track}.",
        "Return strict JSON only. Do not emit applicable_dimensions or any extra key: applicability is supplied by the caller, not inferred by you.",
        f"Applicable score fields (all required as integer 1-5): {', '.join(applicable)}.",
        f"Inapplicable score fields (all required as literal null): {', '.join(inapplicable) or 'none'}.",
        "Use integer scores 1-5: 1=clearly unsupported/incorrect, 3=mixed or partly supported, 5=fully supported/correct.",
        "General dimensions: faithfulness=answer claims supported by media; completeness=answer addresses the question; correctness=factual/logical/task correctness.",
        "Target-entity grounding is mandatory and independent: identify the entities and states required to evaluate the question and candidate answer. For action, task-progress, satisfied, possible/feasible, relation, trajectory, location, or next-step questions, every required target entity/state must be visibly grounded. If any required target is absent, unreadable, or cannot be localized in the supplied media, set target_entity_grounding=1, even if a literal answer no would otherwise sound plausible. Score 3 only for partial or ambiguous grounding and 5 only for clear grounding. reason_codes are audit-only and must not contradict the target_entity_grounding score.",
        "The only absence exception is a direct presence/visibility/existence question whose candidate answer is negative, for example Is there a X in the video?, Can you see X?, or Is X visible in the image?. In that narrow case, clear visual absence supports target_entity_grounding=5. Questions about possible right now, satisfied, next action, location, relation, task progress, or affordance are not presence questions and do not receive this exception.",
        "Embodied dimensions when applicable: answerability=answerable from given media/context; spatial_validity=locations/relations/coordinates/scale supported; temporal_validity=video or trajectory time window supports claim; action_relevance=affordance, task progress, interaction, next action, or failure signal relevant to VLA.",
        "Do not infer metric depth or distance from an uncalibrated image. Do not infer motion or temporal order from a static image. Do not use unsafe_or_sensitive except for genuine safety-sensitive content.",
        "Use at most four distinct reason_codes. The rationale must be one evidence-based sentence of at most 160 characters; do not include quotation marks or line breaks.",
        "", f"source_key: {record.get('source_key')}", f"question_type: {record.get('question_type') or 'unknown'}", f"is_robot_or_trajectory_derived: {bool(record.get('is_robot_or_trajectory_derived'))}", f"notes: {notes[:5]}", "", "conversation:",
    ]
    if turns:
        for idx, turn in enumerate(turns):
            content = clean_content(turn.get("content"))
            structured = turn.get("structured_items")
            if structured:
                content += " structured_items=" + json.dumps(structured, ensure_ascii=False, sort_keys=True)
            lines.append(f"{idx}. {turn.get('role', 'unknown')}: {content}")
    else:
        lines.append(clean_content(record.get("text_preview"))[:2000])
    return "\n".join(lines)


def request_mm_processor_kwargs(media_items: list[dict[str, Any]]) -> dict[str, Any]:
    if any(item["type"] == "video" for item in media_items):
        return {"min_pixels": 65536, "max_pixels": 1048576, "fps": 2, "do_sample_frames": False}
    return dict(QWEN_REQUEST_IMAGE_PIXELS)


def data_uri_for_media(media: dict[str, Any], mime: str) -> tuple[str, str]:
    path = Path(str(media.get("path") or ""))
    data = path.read_bytes()
    return str(media["type"]), f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def build_judge_payload(model: str, record: dict[str, Any], turns: list[dict[str, Any]], notes: list[str], media_items: list[dict[str, Any]], media_mimes: list[str], track: str, applicable: list[str]) -> dict[str, Any]:
    if not media_items or len(media_items) != len(media_mimes):
        raise ValueError("media_items_mimes_mismatch")
    content: list[dict[str, Any]] = []
    for item, mime in zip(media_items, media_mimes):
        media_type, uri = data_uri_for_media(item, mime)
        if media_type == "image":
            content.append({"type": "image_url", "image_url": {"url": uri}})
        elif media_type == "video":
            content.append({"type": "video_url", "video_url": {"url": uri}})
        else:
            raise ValueError(f"unsupported_qwen_media_type:{media_type}")
    content.append({"type": "text", "text": prompt_text(record, turns, notes, track, applicable)})
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": response_schema()},
        "messages": [{"role": "user", "content": content}],
        "mm_processor_kwargs": request_mm_processor_kwargs(media_items),
    }


def set_tokenizer_path(path: str | Path) -> None:
    global _TOKENIZER_PATH
    _TOKENIZER_PATH = Path(path).resolve()


def ensure_judge_tokenizer_loaded() -> Any:
    global _TOKENIZER
    if _TOKENIZER_PATH is None:
        raise RuntimeError("tokenizer_path_not_set")
    if _TOKENIZER is None:
        with _TOKENIZER_LOCK:
            if _TOKENIZER is None:
                from transformers import AutoTokenizer
                _TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, local_files_only=True)
    return _TOKENIZER


def parse_judge_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge response is not a JSON object")
    return value


def validate_judgement(value: dict[str, Any], applicable: list[str]) -> tuple[dict[str, Any], list[str]]:
    required = {*SCORE_DIMENSIONS, "reason_codes", "rationale"}
    if set(value) != required:
        raise ValueError(f"judge_schema_keys_mismatch:missing={sorted(required-set(value))} extra={sorted(set(value)-required)}")
    normalizations: list[str] = []
    for key in SCORE_DIMENSIONS:
        score = value[key]
        if key in applicable:
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
                raise ValueError(f"judge_schema_bad_score:{key}={score!r}")
        elif score is not None:
            value[key] = None
            normalizations.append(f"nonapplicable_forced_null:{key}")
    codes = value["reason_codes"]
    if not isinstance(codes, list) or len(codes) > 4 or any(code not in VALID_REASON_CODES for code in codes):
        raise ValueError(f"judge_schema_bad_reason_codes:{codes!r}")
    if not isinstance(value["rationale"], str) or len(value["rationale"]) > 400:
        raise ValueError("judge_schema_bad_rationale")
    return value, normalizations


def exact_input_token_count(endpoint: str, payload: dict[str, Any], timeout: int) -> int:
    root = endpoint.rsplit("/v1/chat/completions", 1)[0]
    body = {key: payload[key] for key in ("model", "messages", "chat_template_kwargs", "mm_processor_kwargs")}
    req = urllib.request.Request(root + "/tokenize", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        tokenized = json.loads(response.read().decode("utf-8"))
    count = tokenized.get("count")
    if not isinstance(count, int) or count < 1:
        raise ValueError(f"invalid_vllm_tokenize_response:{tokenized!r}")
    return count


def completion_token_count(text: str) -> int:
    return len(ensure_judge_tokenizer_loaded()(str(text or ""), add_special_tokens=False)["input_ids"])


def resolve_rows(input_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    for row in rows:
        for item in row.get("media_items") or []:
            path = Path(str(item.get("path") or ""))
            if not path.is_absolute():
                path = (input_path.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"missing media: {path}")
            item["path"] = str(path)
            item["mime"] = str(item.get("mime") or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            if isinstance(item.get("decoded"), dict):
                item["decoded"]["path"] = str(path)
    return rows


def call_once(row: dict[str, Any], endpoint: str, model: str, timeout: int, retries: int) -> dict[str, Any]:
    request_started_wall = time.time()
    stage_started = time.monotonic()
    record = row["record"]
    turns = row["turns"]
    notes = row.get("notes") or ["tp_token_throughput_benchmark"]
    media_items = [{"type": item["type"], "path": item["path"], "decoded": item.get("decoded") or {}} for item in row["media_items"]]
    media_mimes = [item["mime"] for item in row["media_items"]]
    track, applicable = score_track_and_dimensions(record)
    base = {
        "benchmark_instance_id": row["benchmark_instance_id"], "record_uid": record.get("record_uid"),
        "source_key": record.get("source_key"), "endpoint": endpoint, "media_count": len(media_items),
        "media_types": [item["type"] for item in media_items], "judge_text_tokens": row.get("judge_text_tokens"),
    }
    payload = build_judge_payload(model, record, turns, notes, media_items, media_mimes, track, applicable)
    payload_build_ms = round((time.monotonic() - stage_started) * 1000, 3)
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            tokenize_started = time.monotonic()
            prompt_tokens = exact_input_token_count(endpoint, payload, timeout)
            tokenize_ms = round((time.monotonic() - tokenize_started) * 1000, 3)
            if prompt_tokens > MAX_JUDGE_INPUT_TOKENS:
                return {**base, "judge_status": "excluded_complete_input_over_8192_tokens", "attempt": attempt, "prompt_tokens": prompt_tokens, "error": f"prompt_tokens>{MAX_JUDGE_INPUT_TOKENS}"}
            started = time.monotonic()
            request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                obj = json.loads(response.read().decode("utf-8"))
            latency_ms = round((time.monotonic() - started) * 1000, 3)
            parse_started = time.monotonic()
            message = obj["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content")
            parsed = parse_judge_json(content)
            judgement, normalizations = validate_judgement(parsed, applicable)
            parse_validate_ms = round((time.monotonic() - parse_started) * 1000, 3)
            usage = obj.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            if not isinstance(completion_tokens, int):
                completion_tokens = completion_token_count(content)
            return {
                **base, "judge_status": "pass", "attempt": attempt, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens,
                "usage": usage, "latency_ms": latency_ms, "request_started_wall": request_started_wall,
                "stage_ms": {"payload_build_media_io_and_base64": payload_build_ms, "exact_tokenize_precheck": tokenize_ms, "chat_completion_wait": latency_ms, "response_parse_and_validate": parse_validate_ms},
                "finish_reason": obj["choices"][0].get("finish_reason"),
                "qwen_request_mm_processor_kwargs": request_mm_processor_kwargs(media_items),
                "schema_normalizations": normalizations,
                **{key: judgement.get(key) for key in SCORE_DIMENSIONS},
            }
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
    return {**base, "judge_status": "fail", "attempt": retries + 1, "error": last_error}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return round(values[idx], 3)


def summarize(rows: list[dict[str, Any]], started: float, finished: float, args: argparse.Namespace, endpoints: list[str], input_sha: str, input_rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration = finished - started
    passed = [row for row in rows if row.get("judge_status") == "pass"]
    lats = [float(row["latency_ms"]) for row in passed]
    prompt = sum(int(row.get("prompt_tokens") or 0) for row in passed)
    completion = sum(int(row.get("completion_tokens") or 0) for row in passed)
    total = prompt + completion
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("endpoint"))].append(row)
    by_endpoint = {}
    for endpoint, items in grouped.items():
        pitems = [row for row in items if row.get("judge_status") == "pass"]
        by_endpoint[endpoint] = {
            "records": len(items), "pass": len(pitems),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in pitems),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in pitems),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in pitems),
            "latency_ms_p50": percentile([float(row["latency_ms"]) for row in pitems], 0.5),
            "latency_ms_p95": percentile([float(row["latency_ms"]) for row in pitems], 0.95),
        }
    return {
        "portable_version": PORTABLE_VERSION, "authoritative_runner_sha256": AUTHORITATIVE_RUNNER_SHA256,
        "records": len(rows), "pass_records": len(passed), "nonpass_records": len(rows) - len(passed),
        "status_counts": dict(Counter(str(row.get("judge_status")) for row in rows)),
        "duration_sec": round(duration, 3), "records_per_sec": round(len(passed) / duration, 6) if duration > 0 else None,
        "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
        "prompt_tokens_per_sec": round(prompt / duration, 3) if duration > 0 else None,
        "completion_tokens_per_sec": round(completion / duration, 3) if duration > 0 else None,
        "total_tokens_per_sec": round(total / duration, 3) if duration > 0 else None,
        "latency_ms": {"min": percentile(lats, 0), "p50": percentile(lats, 0.5), "p90": percentile(lats, 0.9), "p95": percentile(lats, 0.95), "p99": percentile(lats, 0.99), "max": percentile(lats, 1)},
        "input": str(args.input.resolve()), "input_sha256": input_sha, "output": str(args.output.resolve()),
        "workers": args.workers, "timeout": args.timeout, "retries": args.retries,
        "repair_rounds": 0, "repair_retries": 0, "model": args.model,
        "endpoints": endpoints, "per_endpoint": by_endpoint,
        "input_stable_sha256": stable_sha256(input_rows),
        "record_uid_sequence_sha256": stable_sha256([row.get("benchmark_instance_id") for row in input_rows]),
    }


def payload_hash(row: dict[str, Any], model: str) -> str:
    record = row["record"]
    media_items = [{"type": item["type"], "path": item["path"], "decoded": item.get("decoded") or {}} for item in row["media_items"]]
    media_mimes = [item["mime"] for item in row["media_items"]]
    track, applicable = score_track_and_dimensions(record)
    payload = build_judge_payload(model, record, row["turns"], row.get("notes") or ["tp_token_throughput_benchmark"], media_items, media_mimes, track, applicable)
    return stable_sha256(payload)


def run_self_test(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    selected: list[int] = []
    wanted = {"image", "video", "image+image"}
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        kinds = "+".join(item["type"] for item in row["media_items"])
        by_kind[kinds] = by_kind.get(kinds, 0) + 1
        if kinds in wanted and kinds not in seen:
            selected.append(idx)
            seen.add(kinds)
    hashes = {str(idx): payload_hash(rows[idx], args.model) for idx in selected}
    result = {"passed": len(rows) > 0 and all(Path(item["path"]).is_file() for row in rows for item in row["media_items"]), "records": len(rows), "media_type_counts": by_kind, "selected_payload_hashes": hashes}
    if args.golden_payload_hashes:
        golden = json.loads(args.golden_payload_hashes.read_text(encoding="utf-8"))
        result["golden_match"] = hashes == golden["selected_payload_hashes"]
        result["passed"] = bool(result["passed"] and result["golden_match"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared portable JSONL; relative media paths are resolved beside this file")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.jsonl"))
    parser.add_argument("--endpoints", default="", help="Comma-separated /v1/chat/completions URLs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-path", type=Path, help="Local Qwen model/tokenizer directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=480)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="Build representative payloads without contacting a server")
    parser.add_argument("--golden-payload-hashes", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    rows = resolve_rows(args.input, args.limit)
    if args.self_test:
        result = run_self_test(args, rows)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 4
    if not args.endpoints or args.tokenizer_path is None:
        parser.error("benchmark mode requires --endpoints and --tokenizer-path")
    set_tokenizer_path(args.tokenizer_path)
    ensure_judge_tokenizer_loaded()
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    nofile_required = min(nofile_hard, max(8192, args.workers * 8))
    if nofile_soft < nofile_required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile_required, nofile_hard))
    endpoints = [endpoint for endpoint in args.endpoints.split(",") if endpoint]
    if not endpoints:
        parser.error("no valid endpoints")
    out: list[dict[str, Any] | None] = [None] * len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress = args.output.with_suffix(args.output.suffix + ".progress.jsonl")
    started = time.monotonic()
    with progress.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(call_once, row, endpoints[idx % len(endpoints)], args.model, args.timeout, args.retries): idx for idx, row in enumerate(rows)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                out[idx] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    finished = time.monotonic()
    result_rows = [row for row in out if row is not None]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result_rows), encoding="utf-8")
    summary = summarize(result_rows, started, finished, args, endpoints, sha256_file(args.input), rows)
    summary["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["nonpass_records"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

