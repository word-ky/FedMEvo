from __future__ import annotations

import itertools
import math
from typing import Iterable, Mapping

from .models import LEVELS, ClientMenu, MenuOption, ScheduleResult
from .wireless import (
    allocate_bandwidth,
    optimized_aggregate_uplink_seconds,
    uplink_times,
)

def schedule_objective(
    selections: Mapping[str, MenuOption],
    menus: Mapping[str, ClientMenu],
    total_bandwidth_hz: float,
    lambda_latency: float,
) -> tuple[float, float, float]:

    gain = sum(option.predicted_gain for option in selections.values())
    cost = optimized_aggregate_uplink_seconds(
        selections, menus, total_bandwidth_hz
    )
    return gain - lambda_latency * cost, gain, cost

def singleton_option_diagnostics(
    menus: Mapping[str, ClientMenu],
    total_bandwidth_hz: float,
    lambda_latency: float,
) -> tuple[dict[str, float | int | str | bool | None], ...]:

    if lambda_latency < 0:
        raise ValueError("lambda_latency must be non-negative")
    inactive = {
        client_id: menu.get(0, None) for client_id, menu in menus.items()
    }
    diagnostics: list[dict[str, float | int | str | bool | None]] = []
    for client_id in sorted(menus):
        menu = menus[client_id]
        for level in LEVELS:
            option = menu.options.get((1, level))
            if option is None:
                continue
            proposed = dict(inactive)
            proposed[client_id] = option
            objective, gain, seconds = schedule_objective(
                proposed, menus, total_bandwidth_hz, lambda_latency
            )
            break_even = gain / seconds if seconds > 0 else None
            numeric = (objective, gain, seconds)
            if any(not math.isfinite(value) for value in numeric) or (
                break_even is not None and not math.isfinite(break_even)
            ):
                raise ValueError("singleton scheduler diagnostic is non-finite")
            diagnostics.append(
                {
                    "client_id": client_id,
                    "level": level,
                    "predicted_gain": gain,
                    "payload_bytes": int(option.payload_bytes),
                    "aggregate_uplink_seconds": seconds,
                    "objective": objective,
                    "break_even_lambda": break_even,
                    "passes_scheduler_margin": objective > 1e-12,
                }
            )
    return tuple(diagnostics)

def _neighbors(menu: ClientMenu, current: MenuOption) -> Iterable[MenuOption]:
    if current.k == 0:
        for level in LEVELS:
            option = menu.options.get((1, level))
            if option is not None:
                yield option
        return

    for quantity in (current.k, current.k + 1):
        for level in LEVELS:
            option = menu.options.get((quantity, level))
            if option is not None and option.key != current.key:
                yield option

def _finalize(
    selections: Mapping[str, MenuOption],
    menus: Mapping[str, ClientMenu],
    total_bandwidth_hz: float,
    lambda_latency: float,
    objective_trace: tuple[float, ...],
) -> ScheduleResult:
    objective, gain, aggregate_seconds = schedule_objective(
        selections, menus, total_bandwidth_hz, lambda_latency
    )
    bandwidth = allocate_bandwidth(selections, menus, total_bandwidth_hz)
    times = uplink_times(selections, menus, bandwidth)
    return ScheduleResult(
        selections=dict(selections),
        bandwidth_hz=bandwidth,
        objective=objective,
        predicted_gain=gain,
        aggregate_uplink_seconds=aggregate_seconds,
        makespan_seconds=max(times.values(), default=0.0),
        payload_bytes=sum(option.payload_bytes for option in selections.values()),
        objective_trace=objective_trace,
    )

def marginal_greedy(
    menus: Mapping[str, ClientMenu],
    total_bandwidth_hz: float,
    lambda_latency: float,
) -> ScheduleResult:

    if lambda_latency < 0:
        raise ValueError("lambda_latency must be non-negative")
    selections = {
        client_id: menu.get(0, None) for client_id, menu in menus.items()
    }
    current_objective = 0.0
    trace = [current_objective]

    while True:
        best: tuple[tuple[float, str, int, int], str, MenuOption, float] | None = None
        for client_id in sorted(menus):
            for candidate in _neighbors(menus[client_id], selections[client_id]):
                proposed = dict(selections)
                proposed[client_id] = candidate
                proposed_objective, _, _ = schedule_objective(
                    proposed, menus, total_bandwidth_hz, lambda_latency
                )
                marginal = proposed_objective - current_objective
                level_index = LEVELS.index(candidate.level) if candidate.level else -1

                key = (-marginal, client_id, candidate.k, level_index)
                proposal = (key, client_id, candidate, proposed_objective)
                if best is None or key < best[0]:
                    best = proposal
        if best is None or -best[0][0] <= 1e-12:
            break
        _, client_id, candidate, current_objective = best
        selections[client_id] = candidate
        trace.append(current_objective)

    return _finalize(
        selections,
        menus,
        total_bandwidth_hz,
        lambda_latency,
        tuple(trace),
    )

def exact_schedule(
    menus: Mapping[str, ClientMenu],
    total_bandwidth_hz: float,
    lambda_latency: float,
    max_configurations: int = 250_000,
) -> ScheduleResult:

    client_ids = sorted(menus)
    choices = [
        sorted(
            menus[client_id].options.values(),
            key=lambda option: (
                option.k,
                -1 if option.level is None else LEVELS.index(option.level),
            ),
        )
        for client_id in client_ids
    ]
    count = math.prod(len(options) for options in choices)
    if count > max_configurations:
        raise ValueError(
            f"exact enumeration requires {count} configurations; "
            f"limit is {max_configurations}"
        )
    best_selections: dict[str, MenuOption] | None = None
    best_objective = -math.inf
    best_signature: tuple[tuple[int, str], ...] | None = None
    for combination in itertools.product(*choices):
        selections = dict(zip(client_ids, combination))
        objective, _, _ = schedule_objective(
            selections, menus, total_bandwidth_hz, lambda_latency
        )
        signature = tuple((option.k, option.level or "") for option in combination)
        if (
            best_selections is None
            or objective > best_objective + 1e-12
            or (
                abs(objective - best_objective) <= 1e-12
                and best_signature is not None
                and signature < best_signature
            )
        ):
            best_selections = selections
            best_objective = objective
            best_signature = signature
    assert best_selections is not None
    return _finalize(
        best_selections,
        menus,
        total_bandwidth_hz,
        lambda_latency,
        (best_objective,),
    )
