"""Normalize supported raw preference formats into the internal schema."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

from llm_posttraining_ops.data.schemas import PreferenceRecord, SplitName

PreferenceFormat: TypeAlias = Literal["direct", "messages"]
SUPPORTED_PREFERENCE_FORMATS: tuple[PreferenceFormat, ...] = ("direct", "messages")


class PreferenceNormalizationError(ValueError):
    """Raised when a raw preference record cannot be normalized."""


def _string_field(
    record: Mapping[str, Any],
    field: str,
    *,
    default: str | None = None,
) -> str:
    value = record.get(field, default)
    if not isinstance(value, str):
        raise PreferenceNormalizationError(f"field '{field}' must be a string")
    return value


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("metadata", {})
    if not isinstance(value, Mapping):
        raise PreferenceNormalizationError("field 'metadata' must be an object")
    return dict(value)


def _generated_id(source: str, format_name: PreferenceFormat, index: int) -> str:
    source_slug = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-") or "local"
    return f"{source_slug}-{format_name}-{index:06d}"


def _common_fields(
    record: Mapping[str, Any],
    *,
    format_name: PreferenceFormat,
    source: str,
    index: int,
) -> tuple[str, SplitName, str, dict[str, Any]]:
    raw_id = record.get("id")
    if raw_id is None or raw_id == "":
        record_id = _generated_id(source, format_name, index)
    elif isinstance(raw_id, str):
        record_id = raw_id
    else:
        raise PreferenceNormalizationError("field 'id' must be a string")

    split = _string_field(record, "split", default="train")
    record_source = _string_field(record, "source", default=source)
    return record_id, cast(SplitName, split), record_source, _metadata(record)


def normalize_direct_preference(
    record: Mapping[str, Any],
    *,
    source: str,
    index: int,
) -> PreferenceRecord:
    """Normalize instruction/input/chosen/rejected fields."""

    record_id, split, record_source, metadata = _common_fields(
        record,
        format_name="direct",
        source=source,
        index=index,
    )
    return PreferenceRecord(
        id=record_id,
        split=split,
        instruction=_string_field(record, "instruction", default=""),
        input=_string_field(record, "input", default=""),
        chosen=_string_field(record, "chosen", default=""),
        rejected=_string_field(record, "rejected", default=""),
        source=record_source,
        metadata=metadata,
    )


def _assistant_content(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        raise PreferenceNormalizationError(
            f"field '{field}' must be a string or assistant message"
        )
    role = value.get("role")
    content = value.get("content")
    if role != "assistant" or not isinstance(content, str):
        raise PreferenceNormalizationError(
            f"field '{field}' must contain an assistant message"
        )
    return content


def normalize_messages_preference(
    record: Mapping[str, Any],
    *,
    source: str,
    index: int,
) -> PreferenceRecord:
    """Normalize prompt messages plus chosen/rejected assistant responses."""

    prompt = record.get("prompt")
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise PreferenceNormalizationError("field 'prompt' must be a list of messages")

    user_messages: list[str] = []
    system_messages: list[str] = []
    for message in prompt:
        if not isinstance(message, Mapping):
            raise PreferenceNormalizationError("each prompt message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise PreferenceNormalizationError("message role and content must be strings")
        if role == "user":
            user_messages.append(content)
        elif role == "system":
            system_messages.append(content)
    if not user_messages:
        raise PreferenceNormalizationError("prompt must contain at least one user message")

    record_id, split, record_source, metadata = _common_fields(
        record,
        format_name="messages",
        source=source,
        index=index,
    )
    metadata["prompt_message_count"] = len(prompt)
    if system_messages:
        metadata["system_prompt"] = "\n".join(system_messages)
    return PreferenceRecord(
        id=record_id,
        split=split,
        instruction=user_messages[-1],
        input=_string_field(record, "input", default=""),
        chosen=_assistant_content(record.get("chosen"), "chosen"),
        rejected=_assistant_content(record.get("rejected"), "rejected"),
        source=record_source,
        metadata=metadata,
    )


def normalize_preference_records(
    records: Sequence[Mapping[str, Any]],
    *,
    format_name: PreferenceFormat,
    source: str,
) -> list[PreferenceRecord]:
    """Normalize raw preference records in a supported local format."""

    if format_name not in SUPPORTED_PREFERENCE_FORMATS:
        raise PreferenceNormalizationError(f"unsupported preference format: {format_name}")
    normalizer = {
        "direct": normalize_direct_preference,
        "messages": normalize_messages_preference,
    }[format_name]
    normalized: list[PreferenceRecord] = []
    for index, record in enumerate(records):
        try:
            normalized.append(normalizer(record, source=source, index=index))
        except PreferenceNormalizationError as exc:
            raise PreferenceNormalizationError(f"record {index + 1}: {exc}") from exc
    return normalized
