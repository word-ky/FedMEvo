from __future__ import annotations

import math
from typing import Mapping

from .models import ClientMenu, MenuOption

BITS_PER_BYTE = 8.0

def spectral_efficiency_from_snr_db(snr_db: float) -> float:

    return math.log2(1.0 + 10.0 ** (float(snr_db) / 10.0))

def communication_root(option: MenuOption, spectral_efficiency: float) -> float:
    if option.k == 0:
        return 0.0
    if spectral_efficiency <= 0:
        raise ValueError("spectral_efficiency must be positive")
    payload_bits = BITS_PER_BYTE * option.payload_bytes
    return math.sqrt(payload_bits / spectral_efficiency)

def allocate_bandwidth(
    selections: Mapping[str, MenuOption],
    menus: Mapping[str, ClientMenu],
    total_bandwidth_hz: float,
) -> dict[str, float]:

    if total_bandwidth_hz <= 0:
        raise ValueError("total_bandwidth_hz must be positive")
    roots = {
        client_id: communication_root(option, menus[client_id].spectral_efficiency)
        for client_id, option in selections.items()
    }
    denominator = sum(roots.values())
    if denominator == 0:
        return {client_id: 0.0 for client_id in selections}
    return {
        client_id: total_bandwidth_hz * root / denominator
        for client_id, root in roots.items()
    }

def uplink_times(
    selections: Mapping[str, MenuOption],
    menus: Mapping[str, ClientMenu],
    bandwidth_hz: Mapping[str, float],
) -> dict[str, float]:
    times: dict[str, float] = {}
    for client_id, option in selections.items():
        if option.k == 0:
            times[client_id] = 0.0
            continue
        bandwidth = bandwidth_hz[client_id]
        if bandwidth <= 0:
            raise ValueError("active client received non-positive bandwidth")
        rate = bandwidth * menus[client_id].spectral_efficiency
        times[client_id] = BITS_PER_BYTE * option.payload_bytes / rate
    return times

def optimized_aggregate_uplink_seconds(
    selections: Mapping[str, MenuOption],
    menus: Mapping[str, ClientMenu],
    total_bandwidth_hz: float,
) -> float:

    if total_bandwidth_hz <= 0:
        raise ValueError("total_bandwidth_hz must be positive")
    z = sum(
        communication_root(option, menus[client_id].spectral_efficiency)
        for client_id, option in selections.items()
    )
    return z * z / total_bandwidth_hz
