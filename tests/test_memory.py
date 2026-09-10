from fedmevo.memory import (FederatedClient, FusionEngine, GlobalMemoryPool,
                            MemoryCandidate, PersonalizedUpdateGenerator, UpdateOperation)

def candidate(cid, mid, gain):
    return MemoryCandidate(mid, cid, "radio", {"compact": "Rule", "standard": "Rule and conditions",
                           "detailed": "Rule and conditions with a detailed procedure"},
                           {level: gain for level in ("compact", "standard", "detailed")}, (1.0, 0.0))

def test_fusion_update_roundtrip_and_object_isolation():
    low, high = candidate("a", "m1", 0.2), candidate("a", "m2", 0.8)
    pool, engine = GlobalMemoryPool(), FusionEngine()
    descriptions = {("a", "m1"): low.descriptor(), ("a", "m2"): high.descriptor()}
    engine.fuse(pool, [low.payload("compact")], descriptions, 1)
    client = FederatedClient("b", 2.0, {})
    generator = PersonalizedUpdateGenerator()
    operations = generator.generate(pool, client)
    assert UpdateOperation.deserialize(operations[0].serialize()) == operations[0]
    client.apply_update(operations)
    key = next(iter(pool.records))
    assert client.received_library[key] is not pool.records[key]
    engine.fuse(pool, [high.payload("detailed")], descriptions, 2)
    assert len(pool.records) == 1
    assert client.received_library[key].quality == 0.2
    operations = generator.generate(pool, client)
    assert operations[0].operation == "REPLACE"
    client.apply_update(operations)
    assert client.received_library[key].quality == 0.8
    assert client.received_library[key] is not pool.records[key]
