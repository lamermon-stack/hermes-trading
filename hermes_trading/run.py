#!/usr/bin/env python3
"""
Entry point for the Hermes trading worker.
Parses --asset from goal.yaml (override with --asset flag). Starts the loop.
"""
import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import yaml

from hermes_trading.loop import run_loop


DEFAULT_STATE_FILES = {
    "goal.yaml": """asset: "BTC/USDT"
target_return_30d: 0.05
max_drawdown: 0.08
min_sharpe: 1.2
failure_below: -0.04
reflection_every: 6
one_variable_only: true
position_size_r: 0.5
""",
    "strategy.yaml": """entry:
  direction: long
  indicator: rsi
  threshold: 80
position_size_r: 0.5
stop_loss_pct: 2.0
version: '03'
""",
    "hypotheses.jsonl": "",
    "trades.jsonl": "",
    "heartbeat.json": "{}",
}


def seed_state_dir(state_dir: Path):
    """Copy default state files to volume if they don't exist."""
    state_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in DEFAULT_STATE_FILES.items():
        filepath = state_dir / filename
        # Always overwrite strategy.yaml so version changes take effect
        if filename == "strategy.yaml" or not filepath.exists():
            filepath.write_text(content)
            print(f"[hermes-trading] Seeded {filename}")


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

    # Seed state directory on first run
    seed_state_dir(args.goal_path.parent)

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