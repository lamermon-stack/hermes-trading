#!/usr/bin/env python3
"""
Reflection cycle with TWO modes:
- --fallback: deterministic rule, used before Hermes is installed.
              If realised return < target → loosen entry.threshold by 2.
              If drawdown > max → tighten stop_loss_pct by 0.2.
              Always changes exactly ONE variable.
              Bumps version, saves prior to state/history/v{NNNN}.yaml,
              appends to state/hypotheses.jsonl.
- --hermes: production mode. Reads latest 25 trades and current strategy,
            formats as prompt, calls `hermes` as subprocess, parses hypothesis,
            applies it.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from hermes_trading.score import load_goal, load_trades, score_trades


def load_strategy(strategy_path: Path) -> dict:
    with open(strategy_path, "r") as f:
        return yaml.safe_load(f)


def save_strategy(strategy_path: Path, strategy: dict):
    with open(strategy_path, "w") as f:
        yaml.dump(strategy, f)


def save_history(strategy: dict, history_dir: Path):
    history_dir.mkdir(parents=True, exist_ok=True)
    version = strategy.get("version", "00")
    history_file = history_dir / f"v{version}.yaml"
    with open(history_file, "w") as f:
        yaml.dump(strategy, f)


def append_hypothesis(hypotheses_file: Path, hypothesis: dict):
    with open(hypotheses_file, "a") as f:
        f.write(json.dumps(hypothesis) + "\n")


def calculate_metrics(trades: List[dict]) -> dict:
    closed = [t for t in trades if t.get("status") == "closed"]
    if not closed:
        return {"total_return": 0, "max_drawdown": 0, "sharpe": 0, "trade_count": 0}

    returns = [t.get("pnl_pct", 0) / 100 for t in closed]
    total_return = sum(returns)

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))

    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    import numpy as np
    if len(returns) >= 2:
        excess = np.array(returns) - 0.02 / 252
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0
    else:
        sharpe = 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trade_count": len(closed),
    }


def fallback_reflection(
    strategy: dict, metrics: dict, goal: dict
) -> Tuple[dict, str, dict]:
    """
    Deterministic fallback reflection.
    Returns (new_strategy, hypothesis_text, hypothesis_dict)
    """
    target_return = goal.get("target_return_30d", 0.05)
    max_drawdown_allowed = goal.get("max_drawdown", 0.08)

    total_return = metrics["total_return"]
    max_dd = metrics["max_drawdown"]

    new_strategy = dict(strategy)
    version_num = int(new_strategy.get("version", "01"))
    new_strategy["version"] = f"{version_num + 1:02d}"

    if total_return < target_return:
        threshold = new_strategy.get("entry", {}).get("threshold", 30)
        new_threshold = max(threshold - 2, 10)
        new_strategy["entry"]["threshold"] = new_threshold
        hypothesis_text = f"Total return {total_return:.2%} below target {target_return:.2%}. Loosening entry threshold from {threshold} to {new_threshold} to capture more trades."
        hypothesis = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "fallback",
            "variable_changed": "entry.threshold",
            "old_value": threshold,
            "new_value": new_threshold,
            "reasoning": hypothesis_text,
            "metrics": metrics,
            "strategy_version": new_strategy["version"],
        }
    elif max_dd > max_drawdown_allowed:
        stop_loss = new_strategy.get("stop_loss_pct", 2.0)
        new_stop_loss = round(stop_loss + 0.2, 1)
        new_strategy["stop_loss_pct"] = new_stop_loss
        hypothesis_text = f"Max drawdown {max_dd:.2%} exceeds limit {max_drawdown_allowed:.2%}. Tightening stop loss from {stop_loss}% to {new_stop_loss}%."
        hypothesis = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "fallback",
            "variable_changed": "stop_loss_pct",
            "old_value": stop_loss,
            "new_value": new_stop_loss,
            "reasoning": hypothesis_text,
            "metrics": metrics,
            "strategy_version": new_strategy["version"],
        }
    else:
        position_size = new_strategy.get("position_size_r", 0.5)
        new_position_size = round(min(position_size + 0.1, 1.0), 1)
        new_strategy["position_size_r"] = new_position_size
        hypothesis_text = f"Performance within bounds. Slightly increasing position size from {position_size} to {new_position_size} to compound gains."
        hypothesis = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "fallback",
            "variable_changed": "position_size_r",
            "old_value": position_size,
            "new_value": new_position_size,
            "reasoning": hypothesis_text,
            "metrics": metrics,
            "strategy_version": new_strategy["version"],
        }

    return new_strategy, hypothesis_text, hypothesis


def format_hermes_prompt(trades: List[dict], strategy: dict, goal: dict, metrics: dict) -> str:
    recent_trades = trades[-25:] if len(trades) > 25 else trades
    closed_trades = [t for t in recent_trades if t.get("status") == "closed"]

    trades_summary = []
    for t in closed_trades:
        trades_summary.append(
            f"  {t['timestamp'][:19]} | {t['side']} | entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} | pnl={t['pnl_pct']:.2f}% | entry_sig={t['entry_signal']} exit_sig={t['exit_signal']}"
        )

    trades_text = "\n".join(trades_summary) if trades_summary else "  (no closed trades yet)"

    return f"""You are the brain of a self-improving trading agent. Analyze the recent trades and current strategy, then propose ONE variable change.

CURRENT STRATEGY (v{strategy.get("version", "00")}):
  entry: {strategy.get("entry", {})}
  stop_loss_pct: {strategy.get("stop_loss_pct", 0)}%
  position_size_r: {strategy.get("position_size_r", 0)}

GOAL (locked):
  asset: {goal.get("asset")}
  target_return_30d: {goal.get("target_return_30d"):.1%}
  max_drawdown: {goal.get("max_drawdown"):.1%}
  min_sharpe: {goal.get("min_sharpe")}
  reflection_every: {goal.get("reflection_every")}

RECENT METRICS (last {metrics["trade_count"]} trades):
  total_return: {metrics["total_return"]:.2%}
  max_drawdown: {metrics["max_drawdown"]:.2%}
  sharpe: {metrics["sharpe"]:.2f}

LAST 25 CLOSED TRADES:
{trades_text}

OUTPUT FORMAT (JSON only, no markdown):
{{
  "variable": "entry.threshold | stop_loss_pct | position_size_r | entry.direction",
  "old_value": <current>,
  "new_value": <proposed>,
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explaining why this change>"
}}

Constraints:
- Change EXACTLY ONE variable
- Variable must exist in current strategy
- Confidence must be 0.0-1.0
- Reasoning must reference specific metrics or trade patterns
"""


def parse_hermes_output(output: str) -> Optional[dict]:
    try:
        lines = output.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
    except Exception:
        pass
    return None


def apply_hermes_hypothesis(strategy: dict, hypothesis: dict) -> Tuple[dict, str]:
    new_strategy = dict(strategy)
    version_num = int(new_strategy.get("version", "00"))
    new_strategy["version"] = f"{version_num + 1:02d}"

    var = hypothesis["variable"]
    old_value = hypothesis["old_value"]
    new_value = hypothesis["new_value"]

    if var == "entry.threshold":
        new_strategy.setdefault("entry", {})["threshold"] = new_value
    elif var == "stop_loss_pct":
        new_strategy["stop_loss_pct"] = new_value
    elif var == "position_size_r":
        new_strategy["position_size_r"] = new_value
    elif var == "entry.direction":
        new_strategy.setdefault("entry", {})["direction"] = new_value
    else:
        raise ValueError(f"Unknown variable: {var}")

    hypothesis_text = f"Hermes: {hypothesis['reasoning']} (confidence: {hypothesis['confidence']:.0%})"

    return new_strategy, hypothesis_text


async def run_reflection(
    state_dir: Path,
    goal: dict,
    trades_file: Path,
    strategy_file: Path,
    hypotheses_file: Path,
    fallback: bool = True,
):
    trades = load_trades(trades_file)
    strategy = load_strategy(strategy_file)
    metrics = calculate_metrics(trades)

    save_history(strategy, state_dir / "history")

    if fallback:
        new_strategy, hypothesis_text, hypothesis = fallback_reflection(
            strategy, metrics, goal
        )
    else:
        prompt = format_hermes_prompt(trades, strategy, goal, metrics)
        result = subprocess.run(
            ["hermes"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Hermes failed: {result.stderr}")

        hypothesis = parse_hermes_output(result.stdout)
        if not hypothesis:
            raise RuntimeError(f"Could not parse Hermes output: {result.stdout}")

        new_strategy, hypothesis_text = apply_hermes_hypothesis(strategy, hypothesis)
        hypothesis["strategy_version"] = new_strategy["version"]
        hypothesis["timestamp"] = datetime.utcnow().isoformat() + "Z"
        hypothesis["metrics"] = metrics

    save_strategy(strategy_file, new_strategy)
    append_hypothesis(hypotheses_file, hypothesis)

    print(f"[reflect] Strategy v{strategy['version']} -> v{new_strategy['version']}")
    print(f"[reflect] {hypothesis_text}")


def main():
    parser = argparse.ArgumentParser(description="Run reflection cycle")
    parser.add_argument("--fallback", action="store_true", help="Use deterministic fallback")
    parser.add_argument("--hermes", action="store_true", help="Use Hermes (production)")
    parser.add_argument(
        "--state-dir", type=Path, default=Path("/app/state"), help="State directory"
    )
    args = parser.parse_args()

    goal_path = args.state_dir / "goal.yaml"
    trades_file = args.state_dir / "trades.jsonl"
    strategy_file = args.state_dir / "strategy.yaml"
    hypotheses_file = args.state_dir / "hypotheses.jsonl"

    if not goal_path.exists():
        print(f"Goal file not found: {goal_path}")
        sys.exit(1)

    goal = load_goal(goal_path)

    import asyncio
    asyncio.run(
        run_reflection(
            args.state_dir,
            goal,
            trades_file,
            strategy_file,
            hypotheses_file,
            fallback=args.fallback or not args.hermes,
        )
    )


if __name__ == "__main__":
    main()