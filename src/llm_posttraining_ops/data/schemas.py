"""Typed records used by the post-training datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

SplitName: TypeAlias = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SFTRecord:
    """A supervised fine-tuning instruction/response example."""

    id: str
    split: SplitName
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    """An instruction paired with preferred and rejected responses."""

    id: str
    split: SplitName
    instruction: str
    input: str
    chosen: str
    rejected: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return asdict(self)
