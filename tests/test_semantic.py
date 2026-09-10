from __future__ import annotations

import json

import pytest

from fedmevo.models import LEVELS, MemoryDescriptor
from fedmevo.semantic import (
    build_client_menu,
    descriptor_bytes,
    estimated_utilities,
    estimated_utilities_v6,
    normalized_cosine,
    quantize_embedding,
)

def descriptor(
    memory_id: str,
    embedding: tuple[float, ...],
    gains: tuple[float, float, float],
    sizes: tuple[int, int, int],
) -> MemoryDescriptor:
    embedding_q, scale = quantize_embedding(embedding)
    return MemoryDescriptor(
        memory_id=memory_id,
        client_id="c0",
        tag="wireless",
        embedding_q=embedding_q,
        embedding_scale=scale,
        q_loc=dict(zip(LEVELS, gains)),
        payload_bytes=dict(zip(LEVELS, sizes)),
    )

def test_descriptor_is_payload_free_and_has_actual_wire_size() -> None:
    item = descriptor("m0", (1.0, 0.0), (0.2, 0.4, 0.5), (640, 960, 1440))
    encoded_size = descriptor_bytes(item)
    assert 0 < encoded_size < min(item.payload_bytes.values())
    payload = json.dumps(item.__dict__, default=list)
    assert "raw_task" not in payload
    assert "trajectory" not in payload
    assert "validation" not in payload
    assert "semantic payload" not in payload

def test_similarity_and_soft_novelty_bounds() -> None:
    assert normalized_cosine((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert normalized_cosine((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(0.0)
    item = descriptor("m0", (1.0, 0.0), (-1.0, 0.5, 1.0), (640, 960, 1440))
    utility = estimated_utilities(item, [(1.0, 0.0)], novelty_floor=0.1)
    assert utility["compact"] == 0.0
    assert utility["standard"] == pytest.approx(0.05, abs=1e-3)
    assert utility["detailed"] == pytest.approx(0.1, abs=1e-3)

def test_menu_ranks_each_level_independently() -> None:
    first = descriptor("a", (1.0, 0.0), (0.9, 0.2, 0.1), (30, 60, 90))
    second = descriptor("b", (0.0, 1.0), (0.1, 0.8, 0.7), (20, 50, 80))
    menu = build_client_menu("c0", [first, second], 2.0)
    assert menu.get(0, None).predicted_gain == 0
    assert menu.get(1, "compact").memory_ids == ("a",)
    assert menu.get(1, "standard").memory_ids == ("b",)
    assert menu.get(1, "detailed").memory_ids == ("b",)
    for level in LEVELS:
        assert menu.get(2, level).payload_bytes >= menu.get(1, level).payload_bytes

def test_build_client_menu_passes_explicit_demand_weight() -> None:
    item = descriptor("m0", (1.0, 0.0), (0.0, 0.0, 0.0), (30, 60, 90))
    defaulted = build_client_menu(
        "c0", [item], 2.0, demand_match_by_memory={"m0": 1.0}, demand_weight=0.2
    )
    stronger = build_client_menu(
        "c0", [item], 2.0, demand_match_by_memory={"m0": 1.0}, demand_weight=0.4
    )
    assert defaulted.get(1, "compact").predicted_gain == pytest.approx(0.2)
    assert stronger.get(1, "compact").predicted_gain == pytest.approx(0.4)
    assert estimated_utilities_v6(item, (), demand_match=1.0, demand_weight=0.4)[
        "compact"
    ] == pytest.approx(0.4)
