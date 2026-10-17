"""Run a controlled BIRD ablation with eight independent 7B trajectories."""

import argparse
import json
from pathlib import Path

from src.bird_evaluator import run_bird_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-json", required=True)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--selection", choices=("self_consistency", "dspy_verifier"),
                        default="self_consistency")
    parser.add_argument("--prompt-mode", choices=("compiled_dspy", "uncompiled_dspy", "baseline"),
                        default="compiled_dspy")
    parser.add_argument("--compiled-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verification-samples", type=int, default=1,
                        help="DSPy rating samples per candidate (1 is the low-cost ablation; use 8 to match sample count)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    output = args.output or str(Path("artifacts") / "ablation" / args.selection / "predict.json")
    run_bird_evaluation(
        args.dev_json, args.db_dir, output, args.limit,
        num_candidates=args.candidates, selection_mode=args.selection,
        max_turns=args.max_turns, max_workers=args.workers,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        prompt_mode=args.prompt_mode, compiled_model_path=args.compiled_model,
        verifier_path=args.verifier_model, verification_samples=args.verification_samples,
        execution_timeout=30.0,
    )
    report = json.loads(Path(output).with_suffix(".metrics.json").read_text(encoding="utf-8"))
    print(f"{args.selection}: EX={report['execution_accuracy']:.2f}% ({report['correct']}/{report['total']})")
    if "self_consistency" in report:
        baseline = report["self_consistency"]
        print(f"self_consistency (same candidates): EX={baseline['execution_accuracy']:.2f}% "
              f"({baseline['correct']}/{baseline['total']})")
    totals = report["totals"]
    print(f"All-candidate + selector tokens: {totals['total_tokens']}; cost USD: {totals['cost_usd']}")
    print(f"Predictions: {output}")
    print(f"Trajectories, selection, configuration and exact metrics: {Path(output).with_suffix('.metrics.json')}")


if __name__ == "__main__":
    main()
