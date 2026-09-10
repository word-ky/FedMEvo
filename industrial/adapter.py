import json
from pathlib import Path

from fedmevo.federation import Experience, Task, choice_label

DEFAULT_DATASET = Path(__file__).parent / "data" / "coalmine_methane.json"

def model_task(record: dict) -> Task:

    question = record["question"].split("\n\nConfirmed historical outcome:", 1)[0]
    return Task(record["id"], f"c{record['client_id']}", record["semantic_tag"], question)

def load_experiment(path: Path = DEFAULT_DATASET, split: str = "validation",
                    targets_per_client: int = 2):
    if split not in {"validation", "test", "canary"}:
        raise ValueError("Choose validation, test, or canary for prediction")
    if targets_per_client < 1:
        raise ValueError("targets_per_client must be positive")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {row["id"]: row for row in data["records"]}
    experiences, replay = {}, {}
    for cid, roles in sorted(data["construction_roles"].items()):
        ordered = [item for pair in zip(roles["E0"], roles["E1"]) for item in pair]
        experiences[cid] = [Experience(model_task(records[item]), records[item]["answer"])
                            for item in ordered]
        item = roles["replay"][0]
        replay[cid] = [Experience(model_task(records[item]), records[item]["answer"])]
    targets = []
    for _, ids in sorted(data["evaluation_manifests"][split].items()):
        targets.extend(model_task(records[item]) for item in ids[:targets_per_client])

    gold = {task.id: records[task.id]["answer"] for task in targets}
    return experiences, replay, targets, gold

def score_predictions(predictions: list[dict], gold: dict[str, str]) -> dict:
    correct = sum(choice_label(item["prediction"]) == gold[item["id"]] for item in predictions)
    return {"correct": correct, "count": len(predictions), "accuracy": correct / len(predictions)}
