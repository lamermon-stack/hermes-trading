#!/usr/bin/env python3
"""
Self-Improving Trading Agent - Monitor Loop
Runs every 30 minutes, checks for closed trades, triggers reflection every 6 trades.
"""

import json
import os
import subprocess
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

print("MONITOR SCRIPT STARTING...", flush=True)

import requests
import yaml

print("IMPORTS DONE", flush=True)

STATE_DIR = Path.home() / "hermes-trading" / "state"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
GOAL_FILE = STATE_DIR / "goal.yaml"
HISTORY_DIR = STATE_DIR / "history"
HYPOTHESES_FILE = STATE_DIR / "hypotheses.jsonl"
TRADES_FILE = STATE_DIR / "trades.jsonl"

TRADES_URL = "https://hermes-trading-production-0864.up.railway.app/trades"
STRATEGY_URL = "https://hermes-trading-production-0864.up.railway.app/strategy"
REPO_DIR = Path.home() / "hermes-trading"

last_reflection_trade_count = 0
last_log_check = ""


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


def get_railway_logs(tail=200):
    """Get recent railway logs."""
    stdout, stderr, code = run_cmd(f"railway logs --tail {tail}")
    return stdout


def count_closed_trades_in_logs(logs):
    """Count closed trades mentioned in logs since last check."""
    global last_log_check
    # Look for trade closure patterns in logs
    # This is a simple heuristic - look for "closed", "exit", "profit", "loss"
    closed_patterns = ["closed", "exit", "profit", "loss", "trade closed", "position closed"]
    count = 0
    for line in logs.split("\n"):
        if any(p in line.lower() for p in closed_patterns):
            count += 1
    return count


def fetch_trades():
    """Fetch last 25 trades from worker."""
    try:
        resp = requests.get(TRADES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Handle both list and dict responses
        if isinstance(data, dict) and "trades" in data:
            return data["trades"]
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"Error fetching trades: {e}")
        return []


def fetch_strategy():
    """Fetch current strategy from worker."""
    try:
        resp = requests.get(STRATEGY_URL, timeout=10)
        resp.raise_for_status()
        return yaml.safe_load(resp.text)
    except Exception as e:
        print(f"Error fetching strategy: {e}")
        return None


def classify_regime(trade, price_history=None):
    """Simple 20-day rolling return classifier for market regime."""
    # Without price history, use a simple heuristic based on trade timing
    # In production, this would use actual rolling returns
    entry_time = trade.get("entry_time", "")
    if not entry_time:
        return "unknown"
    try:
        dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        # Simple regime: high RSI = overbought/bearish, low RSI = oversold/bullish
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
    """Score trades against goal metrics."""
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

    # Max drawdown
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

    # Sharpe (simplified)
    import statistics
    sharpe = 0
    if len(returns) > 1:
        std = statistics.stdev(returns)
        if std > 0:
            sharpe = (statistics.mean(returns) / std) * (252**0.5)  # annualized

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }


def generate_hypotheses(metrics, strategy, goal):
    """Generate 1-3 hypotheses, each changing exactly ONE variable."""
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
        # Too few trades, not enough return - loosen entry (lower threshold for long)
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
        # Too much drawdown - tighten entry (raise threshold)
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
        # Low Sharpe and low win rate - tighten entry
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
        # Tighten stop loss
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
        # Good win rate but low return - widen stop to let winners run
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
        # Performing well - increase position size slightly
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
        # Drawdown getting close to limit - reduce position
        new_pos = max(0.1, current_position - 0.1)
        hypotheses.append({
            "variable": "position_size_r",
            "old_value": current_position,
            "new_value": new_pos,
            "reasoning": f"Max drawdown {max_drawdown:.2%} approaching limit {max_allowed_dd:.2%}. Reducing position size from {current_position} to {new_pos} to reduce risk.",
            "confidence": 0.65,
            "predicted_effect": "reduce_drawdown"
        })

    # Sort by confidence, return top 3
    hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
    return hypotheses[:3]


def apply_hypothesis(hypothesis, strategy, goal):
    """Apply the chosen hypothesis to strategy.yaml."""
    global last_reflection_trade_count

    var_path = hypothesis["variable"]
    new_value = hypothesis["new_value"]
    old_value = hypothesis["old_value"]

    # Update strategy
    if var_path == "entry.threshold":
        strategy["entry"]["threshold"] = new_value
    elif var_path == "stop_loss_pct":
        strategy["stop_loss_pct"] = new_value
    elif var_path == "position_size_r":
        strategy["position_size_r"] = new_value
    else:
        print(f"Unknown variable: {var_path}")
        return False

    # Bump version
    old_version = int(strategy["version"])
    new_version = old_version + 1
    strategy["version"] = str(new_version)

    # Save prior version to history
    history_file = HISTORY_DIR / f"v{old_version:04d}.yaml"
    save_yaml(history_file, strategy)

    # Save new strategy
    save_yaml(STRATEGY_FILE, strategy)

    # Append hypothesis to log
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

    print(f"Applied hypothesis: {var_path} {old_value} -> {new_value} (v{new_version})", flush=True)
    return True


def push_to_railway():
    """Commit and push strategy change to Railway."""
    stdout, stderr, code = run_cmd("git add -A", cwd=REPO_DIR)
    version = load_yaml(STRATEGY_FILE)["version"]
    commit_msg = f"Hermes: strategy v{version}"
    stdout, stderr, code = run_cmd(f'git commit -m "{commit_msg}"', cwd=REPO_DIR)
    if code != 0 and "nothing to commit" not in stderr:
        print(f"Git commit failed: {stderr}", flush=True)
        return False
    stdout, stderr, code = run_cmd("git push", cwd=REPO_DIR)
    if code != 0:
        print(f"Git push failed: {stderr}", flush=True)
        return False
    print(f"Pushed strategy v{version} to Railway", flush=True)
    return True


def reflection_cycle():
    """Run one reflection cycle: fetch trades, analyze, hypothesize, apply, push."""
    global last_reflection_trade_count, metrics

    print(f"\n[{datetime.now()}] Starting reflection cycle...", flush=True)

    # Fetch current data
    trades = fetch_trades()
    strategy = fetch_strategy()
    goal = load_yaml(GOAL_FILE)

    if not trades:
        print("No trades yet, skipping reflection")
        return False

    if not strategy:
        print("Failed to fetch strategy, using local")
        strategy = load_yaml(STRATEGY_FILE)

    # Score trades
    metrics = score_trades(trades, goal)
    print(f"Metrics: {metrics}", flush=True)

    # Check if we have 6 new trades since last reflection
    new_trades = metrics["trade_count"] - last_reflection_trade_count
    if new_trades < 6:
        print(f"Only {new_trades} new trades since last reflection, need 6", flush=True)
        return False

    print(f"Triggering reflection: {new_trades} new trades since last cycle", flush=True)

    # Tag trades with regime
    for trade in trades:
        trade["regime"] = classify_regime(trade)

    # Generate hypotheses
    hypotheses = generate_hypotheses(metrics, strategy, goal)
    if not hypotheses:
        print("No hypotheses generated", flush=True)
        return False

    print(f"Generated {len(hypotheses)} hypotheses:", flush=True)
    for i, h in enumerate(hypotheses):
        print(f"  {i+1}. {h['variable']}: {h['old_value']} -> {h['new_value']} (confidence: {h['confidence']})", flush=True)

    # Pick highest confidence
    chosen = hypotheses[0]
    print(f"Chosen: {chosen['variable']} {chosen['old_value']} -> {chosen['new_value']}", flush=True)

    # Apply
    if not apply_hypothesis(chosen, strategy, goal):
        return False

    # Update last reflection count
    last_reflection_trade_count = metrics["trade_count"]

    # Push to Railway
    if not push_to_railway():
        return False

    print(f"Reflection cycle complete. Strategy updated to v{strategy['version']}", flush=True)
    return True


def monitor_loop():
    """Main monitoring loop - runs every 30 minutes."""
    global last_log_check

    print("=== Self-Improving Trading Agent Monitor Started ===", flush=True)
    print(f"Target: BTC/USDT, +5% in 30d, max 8% DD, min Sharpe 1.2", flush=True)
    print(f"Reflection every 6 trades, one variable per cycle", flush=True)
    print(f"Checking railway logs every 30 minutes...", flush=True)
    print("", flush=True)

    while True:
        try:
            # Check railway logs for activity
            logs = get_railway_logs(200)
            if logs:
                closed_count = count_closed_trades_in_logs(logs)
                if closed_count > 0:
                    print(f"[{datetime.now()}] Detected ~{closed_count} trade closure mentions in logs", flush=True)

            # Run reflection cycle check
            reflection_cycle()

        except KeyboardInterrupt:
            print("\nMonitor stopped by user", flush=True)
            break
        except Exception as e:
            print(f"[{datetime.now()}] Error in monitor loop: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Wait 30 minutes
        print(f"[{datetime.now()}] Sleeping 30 minutes...", flush=True)
        time.sleep(30 * 60)


if __name__ == "__main__":
    monitor_loop()