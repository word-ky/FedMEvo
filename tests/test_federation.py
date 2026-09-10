import json

from fedmevo.federation import Experience, Task, construct_memory, run_federation
from fedmevo.llm import DemoBackend
from fedmevo.models import LEVELS

def case():
    experiences = {cid: [Experience(Task(f"{cid}-{r}", cid, tag, "A or B?"), "A") for r in range(2)]
                   for cid, tag in [("a", "radio"), ("b", "airflow")]}
    targets = [Task("ta", "a", "airflow", "A or B?"), Task("tb", "b", "radio", "A or B?")]
    return experiences, targets

def test_rounds_exchange_memories_and_account_bytes():
    experiences, targets = case()
    result = run_federation(DemoBackend(), experiences, {}, targets, novelty_stop=0)
    assert len(result["rounds"]) == 2
    assert all(result["libraries"].values())
    for row in result["rounds"]:
        assert row["total_bytes"] == row["uplink_bytes"] + row["downlink_bytes"] + row["control_bytes"]
        assert sum(g["bandwidth_hz"] for g in row["selections"].values()) <= 500.000001
    assert result["rounds"][0]["uplink_bytes"] > 0
    assert len(result["predictions"]) == 2

def test_second_round_distillation_consumes_received_context():
    class Recording(DemoBackend):
        prompts = []

        def generate(self, system, prompt):
            if system.startswith("Distill"):
                self.prompts.append(json.loads(prompt))
            return super().generate(system, prompt)

    backend = Recording()
    experiences, targets = case()
    run_federation(backend, experiences, {}, targets)
    assert not backend.prompts[0]["prior_memories"]
    assert any("airflow" in text for text in backend.prompts[2]["prior_memories"])

def test_representation_sizes_and_local_replay_utility():
    item = Experience(Task("h", "a", "radio", "A or B?"), "A")
    candidate = construct_memory(DemoBackend(), item, "m", [], [item], lambda pred, gold: float(pred == gold))
    sizes = [len(candidate.payload(level).serialize()) for level in LEVELS]
    assert sizes == sorted(set(sizes))
    assert all(v == 0 for v in candidate.q_loc.values())

def test_no_budget_benefit_does_not_upload_payload():
    experiences, targets = case()
    result = run_federation(DemoBackend(), experiences, {}, targets, lambda_latency=1e6)
    assert result["communication"]["uplink_bytes"] == 0
    assert result["communication"]["control_bytes"] > 0

def test_redundancy_stops_and_still_counts_control():
    experiences, targets = case()
    result = run_federation(DemoBackend(), experiences, {}, targets)
    assert result["stop_reason"] == "novelty_exhausted"
    assert result["rounds"][-1]["uplink_bytes"] == 0
    assert result["rounds"][-1]["control_bytes"] > 0
