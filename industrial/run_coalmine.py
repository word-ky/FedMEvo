import argparse
import json
from pathlib import Path

from fedmevo.federation import run_federation
from fedmevo.llm import add_backend_arguments, make_backend
from .adapter import DEFAULT_DATASET, load_experiment, score_predictions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["validation", "test", "canary"], default="validation")
    parser.add_argument("--targets-per-client", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/coalmine.json"))
    args = parser.parse_args()
    experiences, replay, targets, gold = load_experiment(args.dataset, args.split, args.targets_per_client)
    result = run_federation(make_backend(args), experiences, replay, targets, rounds=args.rounds)
    result["dataset"] = "CoalMine Methane-Risk Identification"
    result["split"] = args.split
    result["sample_count"] = len(targets)
    if args.backend != "demo":
        result["evaluation"] = score_predictions(result["predictions"], gold)
    else:
        result["evaluation"] = {"status": "not_scored", "reason": "DemoBackend is a wiring fixture"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "sample_count": result["sample_count"],
                      "communication": result["communication"],
                      "evaluation": result["evaluation"], "output": str(args.output)}, indent=2))

if __name__ == "__main__":
    main()
