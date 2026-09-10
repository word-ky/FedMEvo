from __future__ import annotations

import pytest

from fedmevo.models import ClientMenu, MemoryDescriptor, MenuOption
from fedmevo.scheduler import (
    exact_schedule,
    marginal_greedy,
    singleton_option_diagnostics,
)
from fedmevo.semantic import build_client_menu, quantize_embedding
from fedmevo.wireless import allocate_bandwidth, uplink_times

def make_descriptor(client_id: str, memory_id: str, base_gain: float) -> MemoryDescriptor:
    embedding_q, scale = quantize_embedding((base_gain, 1.0 - base_gain, 0.5))
    return MemoryDescriptor(
        memory_id=memory_id,
        client_id=client_id,
        tag="wireless",
        embedding_q=embedding_q,
        embedding_scale=scale,
        q_loc={
            "compact": base_gain,
            "standard": base_gain * 1.25,
            "detailed": base_gain * 1.35,
        },
        payload_bytes={"compact": 100, "standard": 220, "detailed": 500},
    )

def menus():
    return {
        "fast": build_client_menu(
            "fast",
            [make_descriptor("fast", "f0", 0.8), make_descriptor("fast", "f1", 0.5)],
            spectral_efficiency=4.0,
        ),
        "slow": build_client_menu(
            "slow",
            [make_descriptor("slow", "s0", 0.7), make_descriptor("slow", "s1", 0.3)],
            spectral_efficiency=1.0,
        ),
    }

def test_closed_form_bandwidth_is_feasible_and_beats_equal() -> None:
    client_menus = menus()
    selected = {
        client_id: menu.get(1, "standard")
        for client_id, menu in client_menus.items()
    }
    optimized = allocate_bandwidth(selected, client_menus, 1_000.0)
    assert sum(optimized.values()) == pytest.approx(1_000.0)
    assert all(value > 0 for value in optimized.values())
    optimized_sum = sum(uplink_times(selected, client_menus, optimized).values())
    equal = {client_id: 500.0 for client_id in selected}
    equal_sum = sum(uplink_times(selected, client_menus, equal).values())
    assert optimized_sum <= equal_sum + 1e-12

def test_greedy_is_monotone_and_gap_is_measurable() -> None:
    client_menus = menus()
    greedy = marginal_greedy(client_menus, 20_000.0, lambda_latency=0.4)
    exact = exact_schedule(client_menus, 20_000.0, lambda_latency=0.4)
    assert all(
        after >= before - 1e-12
        for before, after in zip(greedy.objective_trace, greedy.objective_trace[1:])
    )
    assert exact.objective >= greedy.objective - 1e-12
    assert greedy.objective >= 0
    for client_id, option in greedy.selections.items():
        assert option.key in client_menus[client_id].options

def test_high_latency_penalty_can_leave_clients_inactive() -> None:
    result = marginal_greedy(menus(), 100.0, lambda_latency=1_000.0)
    assert not result.active_clients
    assert all(bandwidth == 0 for bandwidth in result.bandwidth_hz.values())

def test_singleton_diagnostics_are_exact_finite_and_decision_inert() -> None:
    client_menus = menus()
    before = marginal_greedy(client_menus, 1_000.0, lambda_latency=0.001)

    diagnostics = singleton_option_diagnostics(
        client_menus, 1_000.0, lambda_latency=0.001
    )
    after = marginal_greedy(client_menus, 1_000.0, lambda_latency=0.001)

    assert before == after
    assert len(diagnostics) == 6
    assert all(
        set(item)
        == {
            "client_id",
            "level",
            "predicted_gain",
            "payload_bytes",
            "aggregate_uplink_seconds",
            "objective",
            "break_even_lambda",
            "passes_scheduler_margin",
        }
        for item in diagnostics
    )
    for item in diagnostics:
        seconds = float(item["aggregate_uplink_seconds"])
        gain = float(item["predicted_gain"])
        objective = float(item["objective"])
        assert objective == pytest.approx(gain - 0.001 * seconds)
        assert item["break_even_lambda"] == pytest.approx(gain / seconds)
        assert item["passes_scheduler_margin"] is (objective > 1e-12)

def test_greedy_can_change_level_while_adding_quantity() -> None:
    client_id = "client"
    options = {
        (0, None): MenuOption(client_id, 0, None, (), 0.0, 0),
        (1, "compact"): MenuOption(
            client_id, 1, "compact", ("m0",), 0.2, 100
        ),
        (1, "standard"): MenuOption(
            client_id, 1, "standard", ("m0",), 0.5, 150
        ),
        (1, "detailed"): MenuOption(
            client_id, 1, "detailed", ("m0",), 0.4, 300
        ),
        (2, "compact"): MenuOption(
            client_id, 2, "compact", ("m0", "m1"), 0.9, 220
        ),
        (2, "standard"): MenuOption(
            client_id, 2, "standard", ("m0", "m1"), 0.55, 300
        ),
        (2, "detailed"): MenuOption(
            client_id, 2, "detailed", ("m0", "m1"), 0.45, 600
        ),
    }
    client_menus = {
        client_id: ClientMenu(client_id, spectral_efficiency=1.0, options=options)
    }
    greedy = marginal_greedy(client_menus, 1_000.0, lambda_latency=0.1)
    exact = exact_schedule(client_menus, 1_000.0, lambda_latency=0.1)
    assert greedy.selections[client_id].key == (2, "compact")
    assert greedy.objective == pytest.approx(exact.objective)
