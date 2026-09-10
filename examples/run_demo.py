import argparse
import json
from pathlib import Path

from fedmevo.federation import Experience, Task, run_federation
from fedmevo.llm import add_backend_arguments, make_backend

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo.json"))
    args = parser.parse_args()
    topics = ["radio", "ventilation", "maintenance"]
    questions = ["Which action addresses interference? A. Inspect channel use. B. Ignore it.",
                 "Which action checks ventilation? A. Inspect airflow. B. Ignore airflow.",
                 "Which action supports maintenance? A. Inspect equipment. B. Ignore equipment."]
    experiences, replay, targets = {}, {}, []
    for i, topic in enumerate(topics):
        cid = f"c{i}"
        experiences[cid] = [Experience(Task(f"{cid}-history-{r}", cid, topic, questions[i]), "A")
                            for r in range(args.rounds)]
        replay[cid] = [Experience(Task(f"{cid}-replay", cid, topic, questions[i]), "A")]
        targets.append(Task(f"{cid}-target", cid, topics[(i + 1) % 3], questions[(i + 1) % 3]))
    result = run_federation(make_backend(args), experiences, replay, targets, rounds=args.rounds)
    result["data"] = "Synthetic wiring example, not benchmark evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "rounds": len(result["rounds"]),
                      "communication": result["communication"], "output": str(args.output)}, indent=2))

if __name__ == "__main__":
    main()
