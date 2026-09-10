from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .models import LEVELS, MemoryDescriptor, MenuOption
from .semantic import (
    normalized_cosine,
    quantize_embedding,
)

def serialize_demand_broadcast(client_id: str, tags: Iterable[str]) -> bytes:

    return json.dumps(
        {
            "v": 1,
            "c": client_id,
            "d": sorted(str(tag) for tag in tags),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

def text_embedding(text: str, dimension: int = 128) -> tuple[float, ...]:

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    vector = [0.0] * dimension
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\W\s]", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)

@dataclass(frozen=True)
class MemoryPayload:
    memory_id: str
    client_id: str
    tag: str
    level: str
    content: str

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown semantic level: {self.level}")
        if not self.content:
            raise ValueError("payload content must be non-empty")

    def serialize(self) -> bytes:
        return json.dumps(
            {
                "v": 1,
                "m": self.memory_id,
                "c": self.client_id,
                "t": self.tag,
                "l": self.level,
                "p": self.content,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    client_id: str
    tag: str
    variants: Mapping[str, str]
    q_loc: Mapping[str, float]
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if set(self.variants) != set(LEVELS) or set(self.q_loc) != set(LEVELS):
            raise ValueError("variants and q_loc must cover all semantic levels")
        if not self.embedding:
            raise ValueError("candidate embedding must be non-empty")
        previous_size = 0
        for level in LEVELS:
            size = len(self.payload(level).serialize())
            if size <= previous_size:
                raise ValueError("payload wire sizes must strictly increase by level")
            previous_size = size

    def payload(self, level: str) -> MemoryPayload:
        return MemoryPayload(
            memory_id=self.memory_id,
            client_id=self.client_id,
            tag=self.tag,
            level=level,
            content=self.variants[level],
        )

    def descriptor(self) -> MemoryDescriptor:
        embedding_q, scale = quantize_embedding(self.embedding)
        return MemoryDescriptor(
            memory_id=self.memory_id,
            client_id=self.client_id,
            tag=self.tag,
            embedding_q=embedding_q,
            embedding_scale=scale,
            q_loc={level: float(self.q_loc[level]) for level in LEVELS},
            payload_bytes={
                level: len(self.payload(level).serialize()) for level in LEVELS
            },
        )

@dataclass
class CanonicalMemory:
    canonical_id: str
    tag: str
    content: str
    embedding: tuple[float, ...]
    quality: float
    level: str
    source_client: str
    source_memory: str
    round_id: int
    provenance: list[tuple[str, str, int, str]] = field(default_factory=list)

@dataclass
class GlobalMemoryPool:
    records: dict[str, CanonicalMemory] = field(default_factory=dict)

    @property
    def embeddings(self) -> tuple[tuple[float, ...], ...]:
        return tuple(record.embedding for record in self.records.values())

@dataclass(frozen=True)
class UpdateOperation:
    operation: str
    memory: CanonicalMemory

    def __post_init__(self) -> None:
        if self.operation not in {"ADD", "REPLACE", "MERGE"}:
            raise ValueError(f"unsupported update operation: {self.operation}")

    def serialize(self) -> bytes:
        record = {
            "v": 2,
            "op": self.operation,
            "id": self.memory.canonical_id,
            "tag": self.memory.tag,
            "level": self.memory.level,
            "content": self.memory.content,
            "embedding": list(self.memory.embedding),
            "quality": self.memory.quality,
            "source_client": self.memory.source_client,
            "source_memory": self.memory.source_memory,
            "round_id": self.memory.round_id,
            "provenance": [list(item) for item in self.memory.provenance],
        }
        return json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "UpdateOperation":
        record = json.loads(payload.decode("utf-8"))
        if record.get("v") != 2:
            raise ValueError("unsupported update wire version")
        memory = CanonicalMemory(
            canonical_id=str(record["id"]),
            tag=str(record["tag"]),
            content=str(record["content"]),
            embedding=tuple(float(value) for value in record["embedding"]),
            quality=float(record["quality"]),
            level=str(record["level"]),
            source_client=str(record["source_client"]),
            source_memory=str(record["source_memory"]),
            round_id=int(record["round_id"]),
            provenance=[
                (str(item[0]), str(item[1]), int(item[2]), str(item[3]))
                for item in record["provenance"]
            ],
        )
        return cls(str(record["op"]), memory)

@dataclass
class FederatedClient:
    client_id: str
    spectral_efficiency: float
    candidates: dict[str, MemoryCandidate]
    received_library: dict[str, CanonicalMemory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spectral_efficiency <= 0:
            raise ValueError("spectral_efficiency must be positive")
        if any(candidate.client_id != self.client_id for candidate in self.candidates.values()):
            raise ValueError("candidate client mismatch")

    def build_descriptors(self) -> tuple[MemoryDescriptor, ...]:
        return tuple(
            self.candidates[memory_id].descriptor()
            for memory_id in sorted(self.candidates)
        )

    def upload(self, grant: MenuOption) -> tuple[MemoryPayload, ...]:
        if grant.client_id != self.client_id:
            raise ValueError("grant addressed to a different client")
        if grant.k == 0:
            return ()
        assert grant.level is not None
        payloads: list[MemoryPayload] = []
        for memory_id in grant.memory_ids:
            if memory_id not in self.candidates:
                raise ValueError(f"grant names unknown memory: {memory_id}")
            payloads.append(self.candidates[memory_id].payload(grant.level))
        actual_bytes = sum(len(payload.serialize()) for payload in payloads)
        if actual_bytes != grant.payload_bytes:
            raise ValueError(
                f"grant payload accounting mismatch: expected {grant.payload_bytes}, "
                f"got {actual_bytes}"
            )
        return tuple(payloads)

    def coverage_embeddings(self) -> tuple[tuple[float, ...], ...]:
        local = tuple(candidate.embedding for candidate in self.candidates.values())
        received = tuple(memory.embedding for memory in self.received_library.values())
        return local + received

    def demand_tags(self, topic_universe: Iterable[str]) -> frozenset[str]:

        covered = {candidate.tag for candidate in self.candidates.values()}
        covered.update(memory.tag for memory in self.received_library.values())
        return frozenset(
            str(tag) for tag in topic_universe if str(tag) not in covered
        )

    def apply_update(self, operations: Sequence[UpdateOperation]) -> None:

        received_operations = tuple(
            UpdateOperation.deserialize(operation.serialize())
            for operation in operations
        )
        for operation in received_operations:
            canonical_id = operation.memory.canonical_id
            if operation.operation == "ADD":
                self.received_library.setdefault(canonical_id, operation.memory)
            elif operation.operation == "REPLACE":
                self.received_library[canonical_id] = operation.memory
            else:
                current = self.received_library.get(canonical_id)
                if current is None:
                    self.received_library[canonical_id] = operation.memory
                elif operation.memory.content not in current.content:
                    current.content = current.content + "\n\n" + operation.memory.content

class FusionEngine:

    def __init__(self, similarity_threshold: float = 0.9) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self.similarity_threshold = similarity_threshold

    def fuse(
        self,
        pool: GlobalMemoryPool,
        payloads: Sequence[MemoryPayload],
        descriptors: Mapping[tuple[str, str], MemoryDescriptor],
        round_id: int,
    ) -> GlobalMemoryPool:
        for payload in payloads:
            descriptor = descriptors[(payload.client_id, payload.memory_id)]
            embedding = descriptor.embedding()
            quality = max(0.0, float(descriptor.q_loc[payload.level]))
            candidates = [
                record
                for record in pool.records.values()
                if record.tag == payload.tag
                and len(record.embedding) == len(embedding)
            ]
            nearest = max(
                candidates,
                key=lambda record: normalized_cosine(embedding, record.embedding),
                default=None,
            )
            similarity = (
                normalized_cosine(embedding, nearest.embedding)
                if nearest is not None
                else -1.0
            )
            provenance = (payload.client_id, payload.memory_id, round_id, payload.level)
            if nearest is not None and similarity >= self.similarity_threshold:
                nearest.provenance.append(provenance)
                if quality > nearest.quality:
                    nearest.content = payload.content
                    nearest.embedding = embedding
                    nearest.quality = quality
                    nearest.level = payload.level
                    nearest.source_client = payload.client_id
                    nearest.source_memory = payload.memory_id
                    nearest.round_id = round_id
                continue
            canonical_id = f"g:{payload.client_id}:{payload.memory_id}"
            pool.records[canonical_id] = CanonicalMemory(
                canonical_id=canonical_id,
                tag=payload.tag,
                content=payload.content,
                embedding=embedding,
                quality=quality,
                level=payload.level,
                source_client=payload.client_id,
                source_memory=payload.memory_id,
                round_id=round_id,
                provenance=[provenance],
            )
        return pool

class PersonalizedUpdateGenerator:
    def __init__(self, coverage_threshold: float = 0.9) -> None:
        self.coverage_threshold = coverage_threshold

    def generate(
        self, pool: GlobalMemoryPool, client: FederatedClient
    ) -> tuple[UpdateOperation, ...]:
        coverage = [
            (candidate.tag, candidate.embedding)
            for candidate in client.candidates.values()
        ]
        coverage.extend(
            (memory.tag, memory.embedding)
            for memory in client.received_library.values()
        )
        updates: list[UpdateOperation] = []
        for canonical_id in sorted(pool.records):
            memory = pool.records[canonical_id]
            existing = client.received_library.get(canonical_id)
            if existing is not None:
                if existing != memory:
                    updates.append(UpdateOperation("REPLACE", memory))
                continue
            already_covered = any(
                covering_tag == memory.tag
                and len(existing) == len(memory.embedding)
                and normalized_cosine(existing, memory.embedding)
                >= self.coverage_threshold
                for covering_tag, existing in coverage
            )
            if already_covered:
                continue
            updates.append(UpdateOperation("ADD", memory))
            coverage.append((memory.tag, memory.embedding))
        return tuple(updates)
