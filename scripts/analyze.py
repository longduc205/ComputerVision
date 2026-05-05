"""Analyze evaluation results and generate visualizations."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results-dir", required=True, help="Directory with eval_results.json")
    parser.add_argument("--output-dir", default="results/analysis", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    from src.evaluation.analyzers import (
        analyze_per_class_metrics,
        analyze_domain_performance,
        save_results_summary,
    )

    results_path = Path(args.results_dir) / "eval_results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return

    with open(results_path) as f:
        results = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "per_class" in results:
        analyze_per_class_metrics(
            results["per_class"],
            class_names={i: f"class_{i}" for i in results["per_class"].keys()},
            output_dir=output_dir,
        )

    domain_results = {k: v for k, v in results.items() if k not in ("per_class", "summary")}
    if len(domain_results) > 1:
        analyze_domain_performance(domain_results, output_dir)

    save_results_summary(results, output_dir)
    print(f"Analysis complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
