"""Normalize supported raw instruction formats into the internal SFT schema."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

from llm_posttraining_ops.data.schemas import SFTRecord, SplitName

SFTFormat: TypeAlias = Literal["alpaca", "messages"]
SUPPORTED_SFT_FORMATS: tuple[SFTFormat, ...] = ("alpaca", "messages")


class NormalizationError(ValueError):
    """Raised when a raw record cannot be mapped to normalized SFT fields."""


def _string_field(
    record: Mapping[str, Any],
    field: str,
    *,
    default: str | None = None,
) -> str:
    value = record.get(field, default)
    if not isinstance(value, str):
        raise NormalizationError(f"field '{field}' must be a string")
    return value


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("metadata", {})
    if not isinstance(value, Mapping):
        raise NormalizationError("field 'metadata' must be an object")
    return dict(value)


def _generated_id(source: str, format_name: SFTFormat, index: int) -> str:
    source_slug = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-") or "local"
    return f"{source_slug}-{format_name}-{index:06d}"


def _common_fields(
    record: Mapping[str, Any],
    *,
    format_name: SFTFormat,
    source: str,
    index: int,
) -> tuple[str, SplitName, str, dict[str, Any]]:
    raw_id = record.get("id")
    if raw_id is None or raw_id == "":
        record_id = _generated_id(source, format_name, index)
    elif isinstance(raw_id, str):
        record_id = raw_id
    else:
        raise NormalizationError("field 'id' must be a string")

    split = _string_field(record, "split", default="train")
    record_source = _string_field(record, "source", default=source)
    return record_id, cast(SplitName, split), record_source, _metadata(record)


def normalize_alpaca_record(
    record: Mapping[str, Any],
    *,
    source: str,
    index: int,
) -> SFTRecord:
    """Normalize one instruction/input/output record."""

    record_id, split, record_source, metadata = _common_fields(
        record,
        format_name="alpaca",
        source=source,
        index=index,
    )
    return SFTRecord(
        id=record_id,
        split=split,
        instruction=_string_field(record, "instruction", default=""),
        input=_string_field(record, "input", default=""),
        output=_string_field(record, "output", default=""),
        source=record_source,
        metadata=metadata,
    )


def normalize_messages_record(
    record: Mapping[str, Any],
    *,
    source: str,
    index: int,
) -> SFTRecord:
    """Normalize the first user/assistant pair from a messages record."""

    messages = record.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise NormalizationError("field 'messages' must be a list")

    user_content: str | None = None
    assistant_content: str | None = None
    system_messages: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise NormalizationError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise NormalizationError("message role and content must be strings")
        if role == "system":
            system_messages.append(content)
        elif role == "user" and user_content is None:
            user_content = content
        elif role == "assistant" and user_content is not None:
            assistant_content = content
            break

    if user_content is None or assistant_content is None:
        raise NormalizationError("messages must contain a user turn followed by an assistant turn")

    record_id, split, record_source, metadata = _common_fields(
        record,
        format_name="messages",
        source=source,
        index=index,
    )
    metadata["message_count"] = len(messages)
    if system_messages:
        metadata["system_prompt"] = "\n".join(system_messages)

    return SFTRecord(
        id=record_id,
        split=split,
        instruction=user_content,
        input=_string_field(record, "input", default=""),
        output=assistant_content,
        source=record_source,
        metadata=metadata,
    )


def normalize_sft_records(
    records: Sequence[Mapping[str, Any]],
    *,
    format_name: SFTFormat,
    source: str,
) -> list[SFTRecord]:
    """Normalize raw records in a supported local format."""

    if format_name not in SUPPORTED_SFT_FORMATS:
        raise NormalizationError(f"unsupported SFT format: {format_name}")
    normalizer = {
        "alpaca": normalize_alpaca_record,
        "messages": normalize_messages_record,
    }[format_name]
    normalized: list[SFTRecord] = []
    for index, record in enumerate(records):
        try:
            normalized.append(normalizer(record, source=source, index=index))
        except NormalizationError as exc:
            raise NormalizationError(f"record {index + 1}: {exc}") from exc
    return normalized
