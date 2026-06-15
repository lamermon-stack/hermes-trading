#!/usr/bin/env python3
"""
Single iteration of the trading agent monitor.
Run every 30 minutes via cron.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests
import yaml

STATE_DIR = Path.home() / "hermes-trading" / "state"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
GOAL_FILE = STATE_DIR / "goal.yaml"
HISTORY_DIR = STATE_DIR / "history"
HYPOTHESES_FILE = STATE_DIR / "hypotheses.jsonl"
TRADES_FILE = STATE_DIR / "trades.jsonl"
LAST_REFLECTION_FILE = STATE_DIR / "last_reflection.json"

TRADES_URL = "https://hermes-trading-production-0864.up.railway.app/trades"
STRATEGY_URL = "https://hermes-trading-production-0864.up.railway.app/strategy"
REPO_DIR = Path.home() / "hermes-trading"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr, result.returncode


def get_last_reflection_count():
    if LAST_REFLECTION_FILE.exists():
        with open(LAST_REFLECTION_FILE) as f:
            data = json.load(f)
            return data.get("trade_count", 0)
    return 0


def save_last_reflection_count(count):
    with open(LAST_REFLECTION_FILE, "w") as f:
        json.dump({"trade_count": count, "timestamp": datetime.utcnow().isoformat()}, f)


def get_railway_logs(tail=200):
    stdout, stderr, code = run_cmd(f"railway logs --tail {tail}")
    return stdout


def count_closed_trades_in_logs(logs):
    closed_patterns = ["closed", "exit", "profit", "loss", "trade closed", "position closed"]
    count = 0
    for line in logs.split("\n"):
        if any(p in line.lower() for p in closed_patterns):
            count += 1
    return count


def fetch_trades():
    try:
        resp = requests.get(TRADES_URL, timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            return []
        data = resp.json()
        if isinstance(data, dict) and "trades" in data:
            return data["trades"]
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"Error fetching trades: {e}", file=sys.stderr)
        return []


def fetch_strategy():
    try:
        resp = requests.get(STRATEGY_URL, timeout=10)
        resp.raise_for_status()
        return yaml.safe_load(resp.text)
    except Exception as e:
        print(f"Error fetching strategy: {e}", file=sys.stderr)
        return None


def classify_regime(trade):
    entry_time = trade.get("entry_time", "")
    if not entry_time:
        return "unknown"
    try:
        rsi = trade.get("entry_rsi", 50)
        if rsi > 70:
            return "overbought"
        elif rsi < 30:
            return "oversold"
        elif rsi > 50:
            return "bullish"
        else:
            return "bearish"
    except:
        return "unknown"


def score_trades(trades, goal):
    if not trades:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
        }

    returns = [t.get("pnl_pct", 0) for t in trades]
    total_return = sum(returns)
    trade_count = len(trades)
    win_rate = sum(1 for r in returns if r > 0) / trade_count if trade_count > 0 else 0
    avg_return = total_return / trade_count if trade_count > 0 else 0

    peak = 0
    max_dd = 0
    running = 0
    for r in returns:
        running += r
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    import statistics
    sharpe = 0
    if len(returns) > 1:
        std = statistics.stdev(returns)
        if std > 0:
            sharpe = (statistics.mean(returns) / std) * (252**0.5)

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }


def generate_hypotheses(metrics, strategy, goal):
    hypotheses = []

    current_threshold = strategy.get("entry", {}).get("threshold", 30)
    current_stop_loss = strategy.get("stop_loss_pct", 2.0)
    current_position = strategy.get("position_size_r", 0.5)

    total_return = metrics["total_return"]
    max_drawdown = metrics["max_drawdown"]
    sharpe = metrics["sharpe"]
    trade_count = metrics["trade_count"]
    win_rate = metrics["win_rate"]

    target_return = goal["target_return_30d"]
    max_allowed_dd = goal["max_drawdown"]
    min_sharpe = goal["min_sharpe"]

    # Hypothesis 1: Adjust entry threshold
    if total_return < target_return and trade_count < 10:
        new_threshold = max(10, current_threshold - 2)
        hypotheses.append({
            "variable": "entry.threshold",
            "old_value": current_threshold,
            "new_value": new_threshold,
            "reasoning": f"Total return {total_return:.2%} below target {target_return:.2%} with only {trade_count} trades. Lowering RSI threshold from {current_threshold} to {new_threshold} to capture more entry opportunities.",
            "confidence": 0.7,
            "predicted_effect": "increase_trade_count"
        })
    elif max_drawdown > max_allowed_dd:
        new_threshold = min(90, current_threshold + 2)
        hypotheses.append({
            "variable": "entry.threshold",
            "old_value": current_threshold,
            "new_value": new_threshold,
            "reasoning": f"Max drawdown {max_drawdown:.2%} exceeds limit {max_allowed_dd:.2%}. Raising RSI threshold from {current_threshold} to {new_threshold} to be more selective.",
            "confidence": 0.75,
            "predicted_effect": "reduce_drawdown"
        })
    elif sharpe < min_sharpe and win_rate < 0.5:
        new_threshold = min(90, current_threshold + 2)
        hypotheses.append({
            "variable": "entry.threshold",
            "old_value": current_threshold,
            "new_value": new_threshold,
            "reasoning": f"Sharpe {sharpe:.2f} below minimum {min_sharpe} and win rate {win_rate:.2%} low. Raising RSI threshold from {current_threshold} to {new_threshold} for higher quality entries.",
            "confidence": 0.7,
            "predicted_effect": "improve_sharpe"
        })

    # Hypothesis 2: Adjust stop loss
    if max_drawdown > max_allowed_dd:
        new_sl = max(0.5, current_stop_loss - 0.5)
        hypotheses.append({
            "variable": "stop_loss_pct",
            "old_value": current_stop_loss,
            "new_value": new_sl,
            "reasoning": f"Max drawdown {max_drawdown:.2%} exceeds limit {max_allowed_dd:.2%}. Tightening stop loss from {current_stop_loss}% to {new_sl}% to cut losses faster.",
            "confidence": 0.65,
            "predicted_effect": "reduce_drawdown"
        })
    elif total_return < target_return and win_rate > 0.6:
        new_sl = min(5.0, current_stop_loss + 0.5)
        hypotheses.append({
            "variable": "stop_loss_pct",
            "old_value": current_stop_loss,
            "new_value": new_sl,
            "reasoning": f"Win rate {win_rate:.2%} good but total return {total_return:.2%} low. Widening stop loss from {current_stop_loss}% to {new_sl}% to give trades more room.",
            "confidence": 0.6,
            "predicted_effect": "increase_return"
        })

    # Hypothesis 3: Adjust position size
    if sharpe > min_sharpe * 1.5 and max_drawdown < max_allowed_dd * 0.5:
        new_pos = min(1.0, current_position + 0.1)
        hypotheses.append({
            "variable": "position_size_r",
            "old_value": current_position,
            "new_value": new_pos,
            "reasoning": f"Sharpe {sharpe:.2f} well above minimum {min_sharpe} and drawdown {max_drawdown:.2%} well controlled. Increasing position size from {current_position} to {new_pos} to amplify returns.",
            "confidence": 0.55,
            "predicted_effect": "increase_return"
        })
    elif max_drawdown > max_allowed_dd * 0.8:
        new_pos = max(0.1, current_position - 0.1)
        hypotheses.append({
            "variable": "position_size_r",
            "old_value": current_position,
            "new_value": new_pos,
            "reasoning": f"Max drawdown {max_drawdown:.2%} approaching limit {max_allowed_dd:.2%}. Reducing position size from {current_position} to {new_pos} to reduce risk.",
            "confidence": 0.65,
            "predicted_effect": "reduce_drawdown"
        })

    hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
    return hypotheses[:3]


def apply_hypothesis(hypothesis, strategy, goal, metrics):
    var_path = hypothesis["variable"]
    new_value = hypothesis["new_value"]
    old_value = hypothesis["old_value"]

    if var_path == "entry.threshold":
        strategy["entry"]["threshold"] = new_value
    elif var_path == "stop_loss_pct":
        strategy["stop_loss_pct"] = new_value
    elif var_path == "position_size_r":
        strategy["position_size_r"] = new_value
    else:
        print(f"Unknown variable: {var_path}", file=sys.stderr)
        return False

    old_version = int(strategy["version"])
    new_version = old_version + 1
    strategy["version"] = str(new_version)

    history_file = HISTORY_DIR / f"v{old_version:04d}.yaml"
    save_yaml(history_file, strategy)
    save_yaml(STRATEGY_FILE, strategy)

    hypothesis_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "reflection",
        "variable_changed": var_path,
        "old_value": old_value,
        "new_value": new_value,
        "reasoning": hypothesis["reasoning"],
        "metrics": {
            "total_return": metrics.get("total_return", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "sharpe": metrics.get("sharpe", 0),
            "trade_count": metrics.get("trade_count", 0),
        },
        "strategy_version": str(new_version),
    }
    append_jsonl(HYPOTHESES_FILE, hypothesis_record)

    print(f"Applied hypothesis: {var_path} {old_value} -> {new_value} (v{new_version})")
    return True


def push_to_railway():
    stdout, stderr, code = run_cmd("git add -A", cwd=REPO_DIR)
    version = load_yaml(STRATEGY_FILE)["version"]
    commit_msg = f"Hermes: strategy v{version}"
    stdout, stderr, code = run_cmd(f'git commit -m "{commit_msg}"', cwd=REPO_DIR)
    if code != 0 and "nothing to commit" not in stderr:
        print(f"Git commit failed: {stderr}", file=sys.stderr)
        return False
    stdout, stderr, code = run_cmd("git push", cwd=REPO_DIR)
    if code != 0:
        print(f"Git push failed: {stderr}", file=sys.stderr)
        return False
    print(f"Pushed strategy v{version} to Railway")
    return True


def main():
    print(f"=== Monitor check at {datetime.now()} ===")

    # Check railway logs for activity
    logs = get_railway_logs(200)
    if logs:
        closed_count = count_closed_trades_in_logs(logs)
        if closed_count > 0:
            print(f"Detected ~{closed_count} trade closure mentions in logs")

    # Fetch current data
    trades = fetch_trades()
    strategy = fetch_strategy()
    goal = load_yaml(GOAL_FILE)

    if not trades:
        print("No trades yet, skipping reflection")
        return 0

    if not strategy:
        print("Failed to fetch strategy, using local")
        strategy = load_yaml(STRATEGY_FILE)

    # Score trades
    metrics = score_trades(trades, goal)
    print(f"Metrics: {metrics}")

    # Check if we have 6 new trades since last reflection
    last_count = get_last_reflection_count()
    new_trades = metrics["trade_count"] - last_count
    if new_trades < 6:
        print(f"Only {new_trades} new trades since last reflection, need 6")
        return 0

    print(f"Triggering reflection: {new_trades} new trades since last cycle")

    # Tag trades with regime
    for trade in trades:
        trade["regime"] = classify_regime(trade)

    # Generate hypotheses
    hypotheses = generate_hypotheses(metrics, strategy, goal)
    if not hypotheses:
        print("No hypotheses generated")
        return 0

    print(f"Generated {len(hypotheses)} hypotheses:")
    for i, h in enumerate(hypotheses):
        print(f"  {i+1}. {h['variable']}: {h['old_value']} -> {h['new_value']} (confidence: {h['confidence']})")

    # Pick highest confidence
    chosen = hypotheses[0]
    print(f"Chosen: {chosen['variable']} {chosen['old_value']} -> {chosen['new_value']}")

    # Apply
    if not apply_hypothesis(chosen, strategy, goal, metrics):
        return 1

    # Update last reflection count
    save_last_reflection_count(metrics["trade_count"])

    # Push to Railway
    if not push_to_railway():
        return 1

    print(f"Reflection cycle complete. Strategy updated to v{strategy['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())