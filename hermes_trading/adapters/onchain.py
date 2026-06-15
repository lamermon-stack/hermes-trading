#!/usr/bin/env python3
"""
On-chain data adapter using free public endpoints (Glassnode, Blockchain.com, etc.)
Schema v1: {mvrv, nvt, hash_rate, difficulty, active_addresses, timestamp}
"""
import os
from typing import Any

import httpx

from hermes_trading.adapters import BaseAdapter


class OnchainAdapter(BaseAdapter):
    EXPECTED_SCHEMA_VERSION = 1

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.api_key = os.getenv("GLASSNODE_API_KEY")

    async def fetch(self) -> dict:
        try:
            if self.api_key:
                url = "https://api.glassnode.com/v1/metrics/indicators/mvrv"
                params = {"a": "BTC", "api_key": self.api_key, "i": "24h"}
                resp = await self.client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    latest = data[-1] if data else {}
                    return self._wrap_response(
                        {
                            "mvrv": latest.get("v", 0),
                            "nvt": 0,
                            "hash_rate": 0,
                            "difficulty": 0,
                            "active_addresses": 0,
                            "timestamp": latest.get("t", 0),
                            "source": "glassnode",
                        }
                    )
        except Exception:
            pass

        try:
            url = "https://api.blockchain.info/stats"
            resp = await self.client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return self._wrap_response(
                    {
                        "mvrv": 0,
                        "nvt": 0,
                        "hash_rate": data.get("hash_rate", 0),
                        "difficulty": data.get("difficulty", 0),
                        "active_addresses": 0,
                        "timestamp": data.get("timestamp", 0),
                        "source": "blockchain.info",
                    }
                )
        except Exception:
            pass

        return self._wrap_response(
            {
                "mvrv": 0,
                "nvt": 0,
                "hash_rate": 0,
                "difficulty": 0,
                "active_addresses": 0,
                "timestamp": 0,
                "source": "fallback",
                "error": "All on-chain sources failed",
            }
        )