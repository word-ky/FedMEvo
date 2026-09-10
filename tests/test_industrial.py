import json
from collections import Counter
from dataclasses import asdict

from industrial.adapter import DEFAULT_DATASET, load_experiment, model_task, score_predictions
from fedmevo.federation import run_federation
from fedmevo.llm import DemoBackend

def test_dataset_splits_roles_and_evaluation_projection():
    data = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    assert Counter(row["split"] for row in data["records"]) == {
        "construction": 60, "validation": 40, "test": 150, "canary": 1}
    assert len({row["id"] for row in data["records"]}) == 251
    assert set(row["client_id"] for row in data["records"]) == set(range(5))
    for row in data["records"]:
        visible = asdict(model_task(row))
        assert "answer" not in visible
        assert "Confirmed historical outcome:" not in visible["question"]
    for split, clients in data["evaluation_manifests"].items():
        wanted = {row["id"] for row in data["records"] if row["split"] == split}
        ids = [item for values in clients.values() for item in values]
        assert len(ids) == len(set(ids))
        assert set(ids) == wanted

def test_industrial_end_to_end():
    experiences, replay, targets, gold = load_experiment(targets_per_client=1)
    result = run_federation(DemoBackend(), experiences, replay, targets)
    assert len(result["predictions"]) == 5
    assert result["communication"]["control_bytes"] > 0
    assert result["communication"]["uplink_bytes"] > 0
    assert result["communication"]["downlink_bytes"] > 0
    assert len(result["rounds"]) == 2
    assert len(gold) == 5

def test_evaluation_labels_do_not_change_runtime_inputs(tmp_path):
    data = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    before = load_experiment(DEFAULT_DATASET)
    for row in data["records"]:
        if row["split"] != "construction":
            row["answer"] = "B" if row["answer"] == "A" else "A"
    altered = tmp_path / "changed_gold.json"
    altered.write_text(json.dumps(data), encoding="utf-8")
    after = load_experiment(altered)
    assert before[:3] == after[:3]
    assert before[3] != after[3]

def test_known_predictions_use_exact_labels():
    score = score_predictions([{"id": "a", "prediction": "A"}, {"id": "b", "prediction": "bad"}],
                              {"a": "A", "b": "B"})
    assert score == {"correct": 1, "count": 2, "accuracy": 0.5}
