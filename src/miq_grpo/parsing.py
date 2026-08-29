"""Deterministic, adversarially tested parsing of new model completions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ParseStatus(StrEnum):
    OK = "OK"
    EMPTY = "EMPTY"
    MALFORMED = "MALFORMED"
    MULTIPLE_CONFLICTING_ANSWERS = "MULTIPLE_CONFLICTING_ANSWERS"
    INVALID_JSON = "INVALID_JSON"
    INVALID_SMILES = "INVALID_SMILES"
    SANITIZATION_FAILURE = "SANITIZATION_FAILURE"
    VERIFIER_ERROR = "VERIFIER_ERROR"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"


FAILURE_STATUSES = {
    ParseStatus.EMPTY,
    ParseStatus.MALFORMED,
    ParseStatus.MULTIPLE_CONFLICTING_ANSWERS,
    ParseStatus.INVALID_JSON,
    ParseStatus.INVALID_SMILES,
    ParseStatus.SANITIZATION_FAILURE,
    ParseStatus.VERIFIER_ERROR,
    ParseStatus.UNSUPPORTED_TASK,
}

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_NUMBER = re.compile(r"^[+-]?(?:\d+|\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


_STRICT_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_pairs,
    parse_float=_finite_json_float,
    parse_constant=_reject_json_constant,
)


@dataclass(frozen=True)
class ParsedCompletion:
    status: ParseStatus
    answer: Any = None
    raw_text: str = ""
    extracted_text: str = ""
    exact_answer_tags: bool = False
    strict_json: bool = False

    @property
    def format_ok(self) -> bool:
        return self.status == ParseStatus.OK and self.exact_answer_tags and self.strict_json


def completion_text(completion: Any) -> str:
    """Normalize TRL standard or conversational completion structures."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) == 1 and isinstance(completion[0], dict):
        content = completion[0].get("content")
        return content if isinstance(content, str) else ""
    if isinstance(completion, dict):
        content = completion.get("content")
        return content if isinstance(content, str) else ""
    return ""


def _json_candidates(text: str) -> list[tuple[int, int, Any, str]]:
    candidates: list[tuple[int, int, Any, str]] = []
    for index, character in enumerate(text):
        if character not in '{["-0123456789':
            continue
        try:
            value, consumed = _STRICT_DECODER.raw_decode(text[index:])
        except (json.JSONDecodeError, ValueError):
            continue
        end = index + consumed
        candidates.append((index, end, value, text[index:end]))
    # Eliminate candidates nested inside an already decoded outer object.
    outer: list[tuple[int, int, Any, str]] = []
    for candidate in candidates:
        if any(start <= candidate[0] and candidate[1] <= end for start, end, *_ in outer):
            continue
        outer.append(candidate)
    return outer


def _parse_scalar(text: str) -> int | float | None:
    stripped = text.strip()
    if not _NUMBER.fullmatch(stripped):
        return None
    value = float(stripped)
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _extract_payload(text: str) -> tuple[Any, bool, ParseStatus | None]:
    stripped = text.strip().strip("`").strip()
    if not stripped:
        return None, False, ParseStatus.EMPTY
    try:
        return _STRICT_DECODER.decode(stripped), True, None
    except (json.JSONDecodeError, ValueError):
        # If the response claims to be a complete JSON container, never salvage
        # one of its nested strings/numbers after the container itself failed.
        if stripped.startswith(("{", "[")):
            return None, False, ParseStatus.INVALID_JSON
        candidates = _json_candidates(stripped)
        if candidates:
            distinct = {_canonical(candidate[2]) for candidate in candidates}
            if len(distinct) > 1:
                return None, False, ParseStatus.MULTIPLE_CONFLICTING_ANSWERS
            return candidates[-1][2], False, None
        scalar = _parse_scalar(stripped)
        if scalar is not None:
            return scalar, False, None
    return stripped, False, None


def parse_completion(completion: Any, task_family: str) -> ParsedCompletion:
    raw = completion_text(completion)
    if not raw.strip():
        return ParsedCompletion(ParseStatus.EMPTY, raw_text=raw)

    blocks = _ANSWER_BLOCK.findall(raw)
    has_unbalanced_tags = ("<answer>" in raw or "</answer>" in raw) and not blocks
    if has_unbalanced_tags:
        return ParsedCompletion(ParseStatus.MALFORMED, raw_text=raw)

    exact_tags = len(blocks) == 1
    if blocks:
        parsed_blocks = [_extract_payload(block) for block in blocks]
        if any(status is not None for _, _, status in parsed_blocks):
            status = next(status for _, _, status in parsed_blocks if status is not None)
            return ParsedCompletion(status, raw_text=raw, exact_answer_tags=exact_tags)
        distinct = {_canonical(value) for value, _, _ in parsed_blocks}
        if len(distinct) > 1:
            return ParsedCompletion(
                ParseStatus.MULTIPLE_CONFLICTING_ANSWERS,
                raw_text=raw,
                exact_answer_tags=False,
            )
        answer, strict_json, _ = parsed_blocks[-1]
        extracted = blocks[-1].strip()
    else:
        answer, strict_json, status = _extract_payload(raw)
        if status is not None:
            return ParsedCompletion(status, raw_text=raw)
        extracted = raw.strip()

    if task_family == "count":
        if isinstance(answer, bool) or not isinstance(answer, (dict, int, float, str)):
            return ParsedCompletion(ParseStatus.INVALID_JSON, raw_text=raw, extracted_text=extracted)
    elif task_family == "index":
        if not isinstance(answer, (dict, list, str)):
            return ParsedCompletion(ParseStatus.INVALID_JSON, raw_text=raw, extracted_text=extracted)
    elif task_family == "constraint":
        if isinstance(answer, dict):
            smiles_keys = [key for key in answer if isinstance(key, str) and key.lower() == "smiles"]
            if len(smiles_keys) != 1 or not isinstance(answer[smiles_keys[0]], str):
                return ParsedCompletion(ParseStatus.INVALID_JSON, raw_text=raw, extracted_text=extracted)
            smiles_value = answer[smiles_keys[0]].strip()
            if not smiles_value or any(char.isspace() for char in smiles_value):
                return ParsedCompletion(ParseStatus.INVALID_JSON, raw_text=raw, extracted_text=extracted)
        elif (
            not isinstance(answer, str)
            or not answer.strip()
            or any(char.isspace() for char in answer.strip())
        ):
            return ParsedCompletion(ParseStatus.INVALID_JSON, raw_text=raw, extracted_text=extracted)
    else:
        return ParsedCompletion(ParseStatus.UNSUPPORTED_TASK, raw_text=raw, extracted_text=extracted)

    # The official prompt requires a JSON object inside exactly one answer block.
    strict_format_json = strict_json and isinstance(answer, dict)
    return ParsedCompletion(
        ParseStatus.OK,
        answer=answer,
        raw_text=raw,
        extracted_text=extracted,
        exact_answer_tags=exact_tags,
        strict_json=strict_format_json,
    )


def extract_smiles(answer: Any) -> str:
    if isinstance(answer, dict):
        for key, value in answer.items():
            if isinstance(key, str) and key.lower() == "smiles" and isinstance(value, str):
                return value.strip()
        return ""
    return answer.strip() if isinstance(answer, str) else ""
