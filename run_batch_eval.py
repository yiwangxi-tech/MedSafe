import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from opencode_kg_batch_eval.config import load_config
from opencode_kg_batch_eval.pipeline import run_batch_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to evaluation config JSON")
    parser.add_argument("--input-json", required=True, help="Path to prescription benchmark JSON")
    parser.add_argument("--output-root", required=True, help="Directory for outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing progress and rerun")
    args = parser.parse_args()

    config = load_config(args.config)
    leaderboard_path = run_batch_evaluation(
        config=config,
        input_json=args.input_json,
        output_root=args.output_root,
        cli_limit=args.limit,
        cli_force_rerun=args.force_rerun,
    )
    print(f"leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
