#!/usr/bin/env python3
"""
Price data adapter using ccxt (free public endpoints) and yfinance as fallback.
Schema v1: {close, high, low, open, volume, rsi, timestamp}
"""
import asyncio
import os
from typing import Any

import ccxt
import numpy as np
import yfinance as yf

from hermes_trading.adapters import BaseAdapter


class PriceAdapter(BaseAdapter):
    EXPECTED_SCHEMA_VERSION = 1

    def __init__(self):
        self.exchange = None
        self._init_exchange()

    def _init_exchange(self):
        api_key = os.getenv("EXCHANGE_API_KEY")
        api_secret = os.getenv("EXCHANGE_API_SECRET")

        if api_key and api_secret:
            self.exchange = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
        else:
            self.exchange = ccxt.binance(
                {"enableRateLimit": True, "options": {"defaultType": "spot"}}
            )

    def _calculate_rsi(self, closes: list, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _to_yfinance_symbol(self, ccxt_symbol: str) -> str:
        """Convert ccxt symbol (e.g., 'BTC/USDT') to yfinance symbol (e.g., 'BTC-USD')."""
        if "/" in ccxt_symbol:
            base, quote = ccxt_symbol.split("/")
            if quote in ("USDT", "USDC", "USD", "BUSD"):
                return f"{base}-USD"
        return ccxt_symbol.replace("/", "-")

    def _fetch_ccxt_sync(self, ccxt_symbol: str) -> dict:
        """Synchronous ccxt fetch for thread pool."""
        ticker = self.exchange.fetch_ticker(ccxt_symbol)
        ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, "1h", limit=100)
        closes = [c[4] for c in ohlcv]
        rsi = self._calculate_rsi(closes)
        return {
            "close": ticker["last"],
            "high": ticker["high"],
            "low": ticker["low"],
            "open": ticker["open"],
            "volume": ticker["baseVolume"],
            "rsi": rsi,
            "timestamp": ticker["timestamp"],
            "source": "ccxt",
        }

    def _fetch_yfinance_sync(self, yf_symbol: str) -> dict:
        """Synchronous yfinance fetch for thread pool."""
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="5d", interval="1h")
        if hist.empty:
            raise ValueError(f"yfinance returned empty data for {yf_symbol}")
        closes = hist["Close"].tolist()
        rsi = self._calculate_rsi(closes)
        last = hist.iloc[-1]
        return {
            "close": float(last["Close"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "open": float(last["Open"]),
            "volume": float(last["Volume"]),
            "rsi": rsi,
            "timestamp": int(last.name.timestamp() * 1000),
            "source": "yfinance",
        }

    async def fetch(self) -> dict:
        ccxt_symbol = os.getenv("TRADING_ASSET", "BTC/USDT")
        yf_symbol = self._to_yfinance_symbol(ccxt_symbol)
        ccxt_symbol_clean = ccxt_symbol.replace("/", "")

        # Try ccxt in thread pool (blocking call)
        if self.exchange:
            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None, self._fetch_ccxt_sync, ccxt_symbol_clean
                )
                print(f"[price] ccxt success: close={data['close']}, rsi={data['rsi']:.1f}")
                return self._wrap_response(data)
            except Exception as e:
                print(f"[price] ccxt failed: {e}")

        # Try yfinance in thread pool (blocking call)
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._fetch_yfinance_sync, yf_symbol)
            print(f"[price] yfinance success: close={data['close']}, rsi={data['rsi']:.1f}")
            return self._wrap_response(data)
        except Exception as e:
            print(f"[price] yfinance failed: {e}")

        # Fallback
        print("[price] using fallback data")
        return self._wrap_response(
            {
                "close": 0,
                "high": 0,
                "low": 0,
                "open": 0,
                "volume": 0,
                "rsi": 50,
                "timestamp": 0,
                "source": "fallback",
                "error": "All price sources failed",
            }
        )