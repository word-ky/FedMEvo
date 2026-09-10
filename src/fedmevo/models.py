from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

LEVELS: tuple[str, ...] = ("compact", "standard", "detailed")
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}

def _require_levels(values: Mapping[str, object], field_name: str) -> None:
    missing = set(LEVELS) - set(values)
    extra = set(values) - set(LEVELS)
    if missing or extra:
        raise ValueError(
            f"{field_name} must contain exactly {LEVELS}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

@dataclass(frozen=True)
class MemoryDescriptor:

    memory_id: str
    client_id: str
    tag: str
    embedding_q: tuple[int, ...]
    embedding_scale: float
    q_loc: Mapping[str, float]
    payload_bytes: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.memory_id or not self.client_id or not self.tag:
            raise ValueError("memory_id, client_id, and tag must be non-empty")
        if not self.embedding_q:
            raise ValueError("embedding_q must be non-empty")
        if any(value < -127 or value > 127 for value in self.embedding_q):
            raise ValueError("embedding_q values must be signed int8 values")
        if self.embedding_scale <= 0:
            raise ValueError("embedding_scale must be positive")
        _require_levels(self.q_loc, "q_loc")
        _require_levels(self.payload_bytes, "payload_bytes")
        if any(size <= 0 for size in self.payload_bytes.values()):
            raise ValueError("all payload sizes must be positive")

    def embedding(self) -> tuple[float, ...]:
        return tuple(value * self.embedding_scale for value in self.embedding_q)

@dataclass(frozen=True)
class MenuOption:

    client_id: str
    k: int
    level: str | None
    memory_ids: tuple[str, ...]
    predicted_gain: float
    payload_bytes: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k must be non-negative")
        if self.k == 0:
            if self.level is not None or self.memory_ids or self.payload_bytes != 0:
                raise ValueError("inactive option must have no level, memories, or payload")
            if self.predicted_gain != 0:
                raise ValueError("inactive option must have zero gain")
        else:
            if self.level not in LEVELS:
                raise ValueError(f"unknown semantic level: {self.level}")
            if len(self.memory_ids) != self.k:
                raise ValueError("k must equal the selected memory count")
            if self.payload_bytes <= 0:
                raise ValueError("active options need a positive payload size")

    @property
    def key(self) -> tuple[int, str | None]:
        return self.k, self.level

@dataclass(frozen=True)
class ClientMenu:

    client_id: str
    spectral_efficiency: float
    options: Mapping[tuple[int, str | None], MenuOption]

    def __post_init__(self) -> None:
        if self.spectral_efficiency <= 0:
            raise ValueError("spectral_efficiency must be positive")
        if self.options.get((0, None)) is None:
            raise ValueError("menu must include the inactive option")
        if any(option.client_id != self.client_id for option in self.options.values()):
            raise ValueError("all options must belong to this menu's client")

    @property
    def max_k(self) -> int:
        return max(k for k, _ in self.options)

    def get(self, k: int, level: str | None) -> MenuOption:
        return self.options[(k, level)]

@dataclass(frozen=True)
class ScheduleResult:

    selections: Mapping[str, MenuOption]
    bandwidth_hz: Mapping[str, float]
    objective: float
    predicted_gain: float
    aggregate_uplink_seconds: float
    makespan_seconds: float
    payload_bytes: int
    objective_trace: tuple[float, ...] = field(default_factory=tuple)

    @property
    def active_clients(self) -> tuple[str, ...]:
        return tuple(
            client_id
            for client_id, option in self.selections.items()
            if option.k > 0
        )
