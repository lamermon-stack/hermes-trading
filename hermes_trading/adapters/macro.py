#!/usr/bin/env python3
"""
Macro data adapter using free public endpoints (FRED, Yahoo Finance, etc.)
Schema v1: {dxy, vix, fed_funds_rate, cpi_yoy, yield_10y, timestamp}
"""
import os
from typing import Any

import httpx
import yfinance as yf

from hermes_trading.adapters import BaseAdapter


class MacroAdapter(BaseAdapter):
    EXPECTED_SCHEMA_VERSION = 1

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.fred_key = os.getenv("FRED_API_KEY")

    async def fetch(self) -> dict:
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            vix = yf.Ticker("^VIX")
            tnx = yf.Ticker("^TNX")

            dxy_hist = dxy.history(period="1d")
            vix_hist = vix.history(period="1d")
            tnx_hist = tnx.history(period="1d")

            dxy_val = float(dxy_hist["Close"].iloc[-1]) if not dxy_hist.empty else 0
            vix_val = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else 0
            yield_10y = float(tnx_hist["Close"].iloc[-1]) if not tnx_hist.empty else 0

            return self._wrap_response(
                {
                    "dxy": dxy_val,
                    "vix": vix_val,
                    "fed_funds_rate": 0,
                    "cpi_yoy": 0,
                    "yield_10y": yield_10y,
                    "timestamp": int(dxy_hist.index[-1].timestamp() * 1000) if not dxy_hist.empty else 0,
                    "source": "yfinance",
                }
            )
        except Exception:
            pass

        try:
            if self.fred_key:
                series = ["DGS10", "FEDFUNDS", "CPIAUCSL"]
                results = {}
                for s in series:
                    url = f"https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": s,
                        "api_key": self.fred_key,
                        "file_type": "json",
                        "limit": 1,
                        "sort_order": "desc",
                    }
                    resp = await self.client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        obs = data.get("observations", [])
                        if obs:
                            results[s] = float(obs[0].get("value", 0))

                return self._wrap_response(
                    {
                        "dxy": 0,
                        "vix": 0,
                        "fed_funds_rate": results.get("FEDFUNDS", 0),
                        "cpi_yoy": 0,
                        "yield_10y": results.get("DGS10", 0),
                        "timestamp": 0,
                        "source": "fred",
                    }
                )
        except Exception:
            pass

        return self._wrap_response(
            {
                "dxy": 0,
                "vix": 0,
                "fed_funds_rate": 0,
                "cpi_yoy": 0,
                "yield_10y": 0,
                "timestamp": 0,
                "source": "fallback",
                "error": "All macro sources failed",
            }
        )