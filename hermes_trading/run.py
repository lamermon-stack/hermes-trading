#!/usr/bin/env python3
"""
Entry point for the Hermes trading worker.
Parses --asset from goal.yaml (override with --asset flag). Starts the loop.
"""
import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from hermes_trading.loop import run_loop


def load_goal(goal_path: Path) -> dict:
    with open(goal_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Worker")
    parser.add_argument(
        "--asset", type=str, help="Override asset from goal.yaml (e.g., BTC/USDT)"
    )
    parser.add_argument(
        "--goal-path",
        type=Path,
        default=Path("/app/state/goal.yaml"),
        help="Path to goal.yaml",
    )
    args = parser.parse_args()

    goal = load_goal(args.goal_path)
    asset = args.asset or goal.get("asset", "BTC/USDT")

    print(f"[hermes-trading] Starting worker for {asset}")
    print(f"[hermes-trading] Goal: {goal}")

    try:
        asyncio.run(run_loop(asset, goal, args.goal_path))
    except KeyboardInterrupt:
        print("\n[hermes-trading] Shutdown requested")
    except Exception as e:
        print(f"[hermes-trading] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()