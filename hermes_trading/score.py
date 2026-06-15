#!/usr/bin/env python3
"""
Scoring module: scores trades against goal.yaml.
composite of (realised return vs target), (drawdown vs max), (Sharpe vs min).
Returns float in [-1, +1].
"""
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import yaml


def load_goal(goal_path: Path) -> dict:
    with open(goal_path, "r") as f:
        return yaml.safe_load(f)


def load_trades(trades_path: Path) -> List[dict]:
    trades = []
    if trades_path.exists():
        with open(trades_path, "r") as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
    return trades


def calculate_sharpe(returns: List[float], risk_free: float = 0.02 / 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess_returns = np.array(returns) - risk_free
    if np.std(excess_returns) == 0:
        return 0.0
    return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def score_trades(trades: List[dict], goal: dict) -> float:
    """
    Score trades against goal.
    Returns float in [-1, +1]:
      +1 = perfect (hit target return, zero drawdown, infinite Sharpe)
       0 = neutral
      -1 = catastrophic (blowup)
    """
    closed_trades = [t for t in trades if t.get("status") == "closed"]
    if not closed_trades:
        return 0.0

    returns = [t.get("pnl_pct", 0) / 100 for t in closed_trades]
    total_return = sum(returns)

    equity_curve = [1.0]
    for r in returns:
        equity_curve.append(equity_curve[-1] * (1 + r))

    max_dd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe(returns)

    target_return = goal.get("target_return_30d", 0.05)
    max_drawdown_allowed = goal.get("max_drawdown", 0.08)
    min_sharpe = goal.get("min_sharpe", 1.2)

    return_score = min(total_return / target_return, 2.0) if target_return > 0 else 0
    return_score = max(return_score, -2.0)

    dd_score = 1.0 - (max_dd / max_drawdown_allowed) if max_drawdown_allowed > 0 else 0
    dd_score = max(dd_score, -1.0)

    sharpe_score = min(sharpe / min_sharpe, 2.0) if min_sharpe > 0 else 0
    sharpe_score = max(sharpe_score, -1.0)

    composite = (return_score + dd_score + sharpe_score) / 3.0
    composite = max(min(composite, 1.0), -1.0)

    failure_below = goal.get("failure_below", -0.04)
    if composite < failure_below:
        composite = composite * 2.0

    return float(composite)