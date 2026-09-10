from __future__ import annotations

import base64
import json
import math
from typing import Iterable, Mapping, Sequence

from .models import LEVELS, ClientMenu, MemoryDescriptor, MenuOption

def quantize_embedding(values: Sequence[float]) -> tuple[tuple[int, ...], float]:

    if not values:
        raise ValueError("embedding must be non-empty")
    peak = max(abs(float(value)) for value in values)
    scale = peak / 127.0 if peak > 0 else 1.0
    quantized = tuple(
        max(-127, min(127, int(round(float(value) / scale)))) for value in values
    )
    return quantized, scale

def serialize_descriptor(descriptor: MemoryDescriptor) -> bytes:

    packed_embedding = bytes(value % 256 for value in descriptor.embedding_q)
    record = {
        "v": 1,
        "m": descriptor.memory_id,
        "c": descriptor.client_id,
        "t": descriptor.tag,
        "e": base64.b64encode(packed_embedding).decode("ascii"),
        "s": descriptor.embedding_scale,
        "q": [float(descriptor.q_loc[level]) for level in LEVELS],
        "z": [int(descriptor.payload_bytes[level]) for level in LEVELS],
    }
    return json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def descriptor_bytes(descriptor: MemoryDescriptor) -> int:

    return len(serialize_descriptor(descriptor))

def normalized_cosine(a: Sequence[float], b: Sequence[float]) -> float:

    if len(a) != len(b):
        raise ValueError("embedding dimensions must match")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return 0.5 * (cosine + 1.0)

def novelty_score(
    embedding: Sequence[float], global_embeddings: Iterable[Sequence[float]]
) -> float:

    similarities = [
        normalized_cosine(embedding, global_embedding)
        for global_embedding in global_embeddings
    ]
    return 1.0 if not similarities else 1.0 - max(similarities)

def estimated_utilities(
    descriptor: MemoryDescriptor,
    global_embeddings: Iterable[Sequence[float]],
    novelty_floor: float = 0.1,
) -> dict[str, float]:

    if not 0 <= novelty_floor <= 1:
        raise ValueError("novelty_floor must be in [0, 1]")
    novelty = novelty_score(descriptor.embedding(), global_embeddings)
    soft_novelty = novelty_floor + (1.0 - novelty_floor) * novelty
    return {
        level: max(0.0, float(descriptor.q_loc[level])) * soft_novelty
        for level in LEVELS
    }

def estimated_utilities_v6(
    descriptor: MemoryDescriptor,
    global_embeddings: Iterable[Sequence[float]],
    demand_match: float,
    novelty_floor: float = 0.1,
    demand_weight: float = 0.2,
) -> dict[str, float]:

    if not 0 <= novelty_floor <= 1:
        raise ValueError("novelty_floor must be in [0, 1]")
    if not 0 <= demand_match <= 1:
        raise ValueError("demand_match must be in [0, 1]")
    if demand_weight < 0:
        raise ValueError("demand_weight must be non-negative")
    novelty = novelty_score(descriptor.embedding(), global_embeddings)
    soft_novelty = novelty_floor + (1.0 - novelty_floor) * novelty
    demand_term = demand_weight * float(demand_match)
    return {
        level: soft_novelty
        * (max(0.0, float(descriptor.q_loc[level])) + demand_term)
        for level in LEVELS
    }

def build_client_menu(
    client_id: str,
    descriptors: Sequence[MemoryDescriptor],
    spectral_efficiency: float,
    global_embeddings: Iterable[Sequence[float]] = (),
    novelty_floor: float = 0.1,
    max_memories: int | None = None,
    demand_match_by_memory: Mapping[str, float] | None = None,
    demand_weight: float = 0.2,
) -> ClientMenu:

    if any(descriptor.client_id != client_id for descriptor in descriptors):
        raise ValueError("descriptor client mismatch")
    if demand_match_by_memory is not None:
        missing = [
            descriptor.memory_id
            for descriptor in descriptors
            if descriptor.memory_id not in demand_match_by_memory
        ]
        if missing:
            raise ValueError(
                "demand_match_by_memory missing memories: " + ", ".join(missing)
            )
    global_vectors = tuple(tuple(vector) for vector in global_embeddings)
    if demand_match_by_memory is None:
        scored = {
            descriptor.memory_id: estimated_utilities(
                descriptor, global_vectors, novelty_floor
            )
            for descriptor in descriptors
        }
    else:
        scored = {
            descriptor.memory_id: estimated_utilities_v6(
                descriptor,
                global_vectors,
                float(demand_match_by_memory[descriptor.memory_id]),
                novelty_floor,
                demand_weight=demand_weight,
            )
            for descriptor in descriptors
        }
    limit = len(descriptors) if max_memories is None else min(max_memories, len(descriptors))
    if limit < 0:
        raise ValueError("max_memories must be non-negative")

    options: dict[tuple[int, str | None], MenuOption] = {}
    inactive = MenuOption(client_id, 0, None, (), 0.0, 0)
    options[inactive.key] = inactive

    for level in LEVELS:
        ranked = sorted(
            descriptors,
            key=lambda descriptor: (
                -scored[descriptor.memory_id][level], descriptor.memory_id
            ),
        )
        selected_ids: list[str] = []
        cumulative_gain = 0.0
        cumulative_bytes = 0
        for descriptor in ranked[:limit]:
            selected_ids.append(descriptor.memory_id)
            cumulative_gain += scored[descriptor.memory_id][level]
            cumulative_bytes += int(descriptor.payload_bytes[level])
            option = MenuOption(
                client_id=client_id,
                k=len(selected_ids),
                level=level,
                memory_ids=tuple(selected_ids),
                predicted_gain=cumulative_gain,
                payload_bytes=cumulative_bytes,
            )
            options[option.key] = option

    return ClientMenu(
        client_id=client_id,
        spectral_efficiency=float(spectral_efficiency),
        options=options,
    )
