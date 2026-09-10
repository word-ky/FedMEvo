from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from .llm import TextBackend
from .memory import (FederatedClient, FusionEngine, GlobalMemoryPool, MemoryCandidate,
                     PersonalizedUpdateGenerator, serialize_demand_broadcast, text_embedding)
from .models import LEVELS
from .scheduler import marginal_greedy
from .semantic import build_client_menu, descriptor_bytes, normalized_cosine, novelty_score

@dataclass(frozen=True)
class Task:

    id: str
    client_id: str
    tag: str
    question: str

@dataclass(frozen=True)
class Experience:

    task: Task
    feedback: str

def choice_label(text: str) -> str:
    match = re.match(r"^\s*([A-E])(?:\b|[.)])", text)
    return match.group(1) if match else ""

def predict(backend: TextBackend, task: Task, memories: Sequence[str]) -> str:
    return backend.generate(
        "Answer the task using relevant memories only. For multiple choice, return only the option letter.",
        json.dumps({"question": task.question, "memories": list(memories)}, ensure_ascii=False),
    ).strip()

def retrieve(client: FederatedClient, question: str, limit: int = 3) -> list[str]:
    query = text_embedding(question)
    records = [(m.variants["standard"], m.embedding) for m in client.candidates.values()]
    records += [(m.content, m.embedding) for m in client.received_library.values()]
    ranked = sorted(records, key=lambda item: -normalized_cosine(query, item[1]))
    return list(dict.fromkeys(text for text, _ in ranked))[:limit]

def construct_memory(backend: TextBackend, experience: Experience, memory_id: str,
                     context: Sequence[str], replay: Sequence[Experience],
                     scorer: Callable[[str, str], float]) -> MemoryCandidate:
    task = experience.task
    response = backend.generate(
        "Distill a reusable memory from the completed task and feedback. Return only a JSON object "
        "with nonempty string fields core, conditions, procedure. Do not include sample IDs. "
        "Use prior memories to refine the reusable lesson, not to copy an answer table.",
        json.dumps({"tag": task.tag, "task": task.question, "feedback": experience.feedback,
                    "prior_memories": list(context)}, ensure_ascii=False),
    )
    note = json.loads(response)
    core, conditions, procedure = (note[k].strip() for k in ("core", "conditions", "procedure"))
    if not all((core, conditions, procedure)):
        raise ValueError("Distillation requires nonempty core, conditions, and procedure")
    variants = {"compact": core, "standard": core + "\nConditions: " + conditions,
                "detailed": core + "\nConditions: " + conditions + "\nProcedure: " + procedure}

    baseline = [scorer(predict(backend, item.task, context), item.feedback) for item in replay]
    quality = {}
    for level in LEVELS:
        gains = [scorer(predict(backend, item.task, [*context, variants[level]]), item.feedback) - before
                 for item, before in zip(replay, baseline)]
        quality[level] = sum(gains) / len(gains) if gains else 0.0
    return MemoryCandidate(memory_id, task.client_id, task.tag, variants, quality, text_embedding(core))

def run_federation(backend: TextBackend, experiences: dict[str, list[Experience]],
                   replay: dict[str, list[Experience]], targets: Sequence[Task], *,
                   rounds: int = 2, bandwidth_hz: float = 500.0,
                   lambda_latency: float = 0.005, demand_weight: float = 0.4,
                   max_memories: int = 2, novelty_stop: float = 0.005,
                   spectral_efficiencies: dict[str, float] | None = None,
                   scorer: Callable[[str, str], float] | None = None) -> dict:

    if rounds < 1 or max_memories < 1:
        raise ValueError("rounds and max_memories must be positive")
    if not 0 <= novelty_stop <= 1:
        raise ValueError("novelty_stop must be in [0, 1]")
    scorer = scorer or (lambda prediction, gold: float(choice_label(prediction) == gold))
    ids = sorted(experiences)
    clients = {cid: FederatedClient(cid, (spectral_efficiencies or {}).get(cid, 2.0 + i), {})
               for i, cid in enumerate(ids)}
    needed = {cid: {task.tag for task in targets if task.client_id == cid} for cid in ids}
    pool, fusion, delivery = GlobalMemoryPool(), FusionEngine(), PersonalizedUpdateGenerator()
    history = []
    stop_reason = "round_limit"
    for round_id in range(1, rounds + 1):
        fresh = {}
        for cid, client in clients.items():
            if round_id > len(experiences[cid]):
                fresh[cid] = []
                continue
            item = experiences[cid][round_id - 1]
            context = retrieve(client, item.task.question)
            candidate = construct_memory(backend, item, f"{cid}:r{round_id}", context,
                                         replay.get(cid, []), scorer)
            client.candidates[candidate.memory_id] = candidate
            fresh[cid] = [candidate.descriptor()]
        if not any(fresh.values()):
            stop_reason = "experience_exhausted"
            break
        demands = {cid: client.demand_tags(needed[cid]) for cid, client in clients.items()}
        control = sum(descriptor_bytes(d) for values in fresh.values() for d in values)
        control += sum(len(serialize_demand_broadcast(cid, tags)) for cid, tags in demands.items())
        newness = max(novelty_score(d.embedding(), pool.embeddings)
                      for values in fresh.values() for d in values)
        if round_id > 1 and newness <= novelty_stop:
            control += len(json.dumps({"stop": "novelty_exhausted"}).encode("utf-8"))
            history.append({"round": round_id, "uplink_bytes": 0, "downlink_bytes": 0,
                            "control_bytes": control, "total_bytes": control, "selections": {}})
            stop_reason = "novelty_exhausted"
            break
        menus = {}
        for cid, client in clients.items():
            peer_demand = set().union(*(tags for other, tags in demands.items() if other != cid))
            menus[cid] = build_client_menu(
                cid, fresh[cid], client.spectral_efficiency, pool.embeddings,
                max_memories=max_memories, demand_weight=demand_weight,
                demand_match_by_memory={d.memory_id: float(d.tag in peer_demand) for d in fresh[cid]},
            )
        schedule = marginal_greedy(menus, bandwidth_hz, lambda_latency)
        grants = {cid: {"memory_ids": list(option.memory_ids), "level": option.level,
                        "bandwidth_hz": schedule.bandwidth_hz[cid]}
                  for cid, option in schedule.selections.items()}
        control += len(json.dumps(grants, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        payloads = [p for cid, client in clients.items() for p in client.upload(schedule.selections[cid])]
        uplink = sum(len(p.serialize()) for p in payloads)
        descriptors = {(d.client_id, d.memory_id): d for values in fresh.values() for d in values}
        fusion.fuse(pool, payloads, descriptors, round_id)
        downlink, updates = 0, {}
        for cid, client in clients.items():
            operations = tuple(op for op in delivery.generate(pool, client) if op.memory.tag in needed[cid])
            downlink += sum(len(op.serialize()) for op in operations)
            client.apply_update(operations)
            updates[cid] = len(operations)
        history.append({"round": round_id, "selections": grants, "objective": schedule.objective,
                        "objective_trace": list(schedule.objective_trace),
                        "uplink_seconds_model": schedule.aggregate_uplink_seconds,
                        "uplink_bytes": uplink, "downlink_bytes": downlink,
                        "control_bytes": control, "total_bytes": uplink + downlink + control,
                        "updates_per_client": updates, "pool_size": len(pool.records)})
    predictions = [{"id": task.id, "client_id": task.client_id,
                    "prediction": predict(backend, task, retrieve(clients[task.client_id], task.question))}
                   for task in targets]
    return {"framework": "FedMEvo", "backend": type(backend).__name__,
            "mode": "wiring_demo" if type(backend).__name__ == "DemoBackend" else "reference_inference",
            "stop_reason": stop_reason, "rounds": history,
            "communication": {key: sum(r[key] for r in history)
                              for key in ("uplink_bytes", "downlink_bytes", "control_bytes", "total_bytes")},
            "predictions": predictions,
            "libraries": {cid: [asdict(m) for m in c.received_library.values()] for cid, c in clients.items()}}
