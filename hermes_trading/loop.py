#!/usr/bin/env python3
"""
24/7 reliability loop for the trading worker.
Every minute: pull data via adapters, evaluate strategy, decide (paper trade), log outcome.
Per-adapter retries (3, exponential). Circuit-break after 5 consecutive failures.
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from hermes_trading.adapters.price import PriceAdapter
from hermes_trading.adapters.onchain import OnchainAdapter
from hermes_trading.adapters.news import NewsAdapter
from hermes_trading.adapters.macro import MacroAdapter
from hermes_trading.score import score_trades
from hermes_trading.reflect import run_reflection


@dataclass
class Trade:
    timestamp: str
    asset: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_signal: str = ""
    exit_signal: str = ""
    status: str = "open"  # "open" or "closed"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "entry_signal": self.entry_signal,
            "exit_signal": self.exit_signal,
            "status": self.status,
        }


@dataclass
class Strategy:
    version: str
    entry: dict
    stop_loss_pct: float
    position_size_r: float


class CircuitBreaker:
    def __init__(self, max_failures: int = 5):
        self.max_failures = max_failures
        self.consecutive_failures = 0
        self.is_open = False

    def record_success(self):
        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_failures:
            self.is_open = True

    def can_proceed(self) -> bool:
        return not self.is_open


class TradingLoop:
    def __init__(self, asset: str, goal: dict, goal_path: Path):
        self.asset = asset
        self.goal = goal
        self.goal_path = goal_path
        self.state_dir = goal_path.parent
        self.trades_file = self.state_dir / "trades.jsonl"
        self.strategy_file = self.state_dir / "strategy.yaml"
        self.heartbeat_file = self.state_dir / "heartbeat.json"
        self.hypotheses_file = self.state_dir / "hypotheses.jsonl"

        self.price_adapter = PriceAdapter()
        self.onchain_adapter = OnchainAdapter()
        self.news_adapter = NewsAdapter()
        self.macro_adapter = MacroAdapter()

        self.circuit_breakers = {
            "price": CircuitBreaker(),
            "onchain": CircuitBreaker(),
            "news": CircuitBreaker(),
            "macro": CircuitBreaker(),
        }

        self.position: Optional[Trade] = None
        self.closed_trades_count = 0
        self.last_reflection_count = 0

        self._load_state()

    def _load_state(self):
        if self.trades_file.exists():
            with open(self.trades_file, "r") as f:
                for line in f:
                    if line.strip():
                        trade_data = json.loads(line)
                        if trade_data["status"] == "closed":
                            self.closed_trades_count += 1
        if self.strategy_file.exists():
            with open(self.strategy_file, "r") as f:
                self.strategy = Strategy(**yaml.safe_load(f))
        else:
            self.strategy = self._default_strategy()
            self._save_strategy()

    def _default_strategy(self) -> Strategy:
        return Strategy(
            version="01",
            entry={"indicator": "rsi", "threshold": 30, "direction": "long"},
            stop_loss_pct=2.0,
            position_size_r=0.5,
        )

    def _save_strategy(self):
        data = {
            "version": self.strategy.version,
            "entry": self.strategy.entry,
            "stop_loss_pct": self.strategy.stop_loss_pct,
            "position_size_r": self.strategy.position_size_r,
        }
        with open(self.strategy_file, "w") as f:
            yaml.dump(data, f)

    def _write_heartbeat(self):
        hb = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "asset": self.asset,
            "position": self.position.to_dict() if self.position else None,
            "closed_trades": self.closed_trades_count,
            "strategy_version": self.strategy.version,
        }
        with open(self.heartbeat_file, "w") as f:
            json.dump(hb, f)

    async def _fetch_with_retry(self, adapter_name: str, fetch_fn, max_retries: int = 3):
        cb = self.circuit_breakers[adapter_name]
        if not cb.can_proceed():
            raise RuntimeError(f"Circuit breaker open for {adapter_name}")

        for attempt in range(max_retries):
            try:
                result = await fetch_fn()
                cb.record_success()
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    cb.record_failure()
                    raise
                wait_time = 2**attempt
                await asyncio.sleep(wait_time)
        cb.record_failure()
        raise RuntimeError(f"All retries exhausted for {adapter_name}")

    async def _fetch_all_data(self) -> dict:
        tasks = {
            "price": self._fetch_with_retry("price", self.price_adapter.fetch),
            "onchain": self._fetch_with_retry("onchain", self.onchain_adapter.fetch),
            "news": self._fetch_with_retry("news", self.news_adapter.fetch),
            "macro": self._fetch_with_retry("macro", self.macro_adapter.fetch),
        }
        results = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as e:
                print(f"[{datetime.utcnow().isoformat()}] Adapter {name} failed: {e}")
                results[name] = {"error": str(e), "schema_version": 1}
        return results

    def _evaluate_entry(self, data: dict) -> bool:
        if self.position is not None:
            return False

        price_data = data.get("price", {})
        current_price = price_data.get("close", price_data.get("price", 0))
        rsi = price_data.get("rsi", 50)

        entry_cfg = self.strategy.entry
        if entry_cfg["indicator"] == "rsi":
            threshold = entry_cfg["threshold"]
            direction = entry_cfg["direction"]
            if direction == "long" and rsi <= threshold:
                return True
            if direction == "short" and rsi >= (100 - threshold):
                return True
        return False

    def _evaluate_exit(self, data: dict) -> Optional[str]:
        if self.position is None:
            return None

        price_data = data.get("price", {})
        current_price = price_data.get("close", price_data.get("price", 0))
        entry_price = self.position.entry_price

        if self.position.side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
            if pnl_pct <= -self.strategy.stop_loss_pct:
                return "stop_loss"
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            if pnl_pct <= -self.strategy.stop_loss_pct:
                return "stop_loss"

        rsi = price_data.get("rsi", 50)
        if self.position.side == "long" and rsi >= 70:
            return "rsi_overbought"
        if self.position.side == "short" and rsi <= 30:
            return "rsi_oversold"

        return None

    def _open_position(self, data: dict, signal: str):
        price_data = data.get("price", {})
        current_price = price_data.get("close", price_data.get("price", 0))
        side = self.strategy.entry["direction"]
        size = self.strategy.position_size_r

        self.position = Trade(
            timestamp=datetime.utcnow().isoformat() + "Z",
            asset=self.asset,
            side=side,
            entry_price=current_price,
            size=size,
            entry_signal=signal,
        )
        print(f"[{datetime.utcnow().isoformat()}] OPEN {side} {self.asset} @ {current_price:.2f}")

    def _close_position(self, data: dict, signal: str):
        price_data = data.get("price", {})
        current_price = price_data.get("close", price_data.get("price", 0))

        if self.position.side == "long":
            pnl = (current_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - current_price) * self.position.size

        pnl_pct = (pnl / (self.position.entry_price * self.position.size)) * 100

        self.position.exit_price = current_price
        self.position.pnl = pnl
        self.position.pnl_pct = pnl_pct
        self.position.exit_signal = signal
        self.position.status = "closed"

        self._log_trade(self.position)
        self.closed_trades_count += 1

        print(
            f"[{datetime.utcnow().isoformat()}] CLOSE {self.position.side} {self.asset} @ {current_price:.2f} | PnL: {pnl:.2f} ({pnl_pct:.2f}%)"
        )

        self.position = None

    def _log_trade(self, trade: Trade):
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")

    def _check_reflection_trigger(self) -> bool:
        return (self.closed_trades_count - self.last_reflection_count) >= self.goal.get(
            "reflection_every", 6
        )

    async def _run_reflection(self):
        print(f"[{datetime.utcnow().isoformat()}] Running reflection cycle...")
        await run_reflection(
            self.state_dir,
            self.goal,
            self.trades_file,
            self.strategy_file,
            self.hypotheses_file,
            fallback=True,
        )
        self._load_state()
        self.last_reflection_count = self.closed_trades_count
        print(f"[{datetime.utcnow().isoformat()}] Reflection complete. Strategy v{self.strategy.version}")

    async def run(self):
        print(f"[{datetime.utcnow().isoformat()}] Booting hermes-trading worker")
        iteration = 0

        while True:
            iteration += 1
            loop_start = time.time()

            try:
                data = await self._fetch_all_data()

                if self._evaluate_entry(data):
                    self._open_position(data, "entry_signal")

                exit_signal = self._evaluate_exit(data)
                if exit_signal and self.position:
                    self._close_position(data, exit_signal)

                if self._check_reflection_trigger():
                    await self._run_reflection()

                self._write_heartbeat()

            except Exception as e:
                print(f"[{datetime.utcnow().isoformat()}] Loop error: {e}")

            elapsed = time.time() - loop_start
            sleep_time = max(60 - elapsed, 1)
            await asyncio.sleep(sleep_time)


async def run_loop(asset: str, goal: dict, goal_path: Path):
    loop = TradingLoop(asset, goal, goal_path)
    await loop.run()