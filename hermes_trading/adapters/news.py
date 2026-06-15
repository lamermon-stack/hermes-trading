#!/usr/bin/env python3
"""
News sentiment adapter using free public endpoints (NewsAPI, CryptoPanic, etc.)
Schema v1: {sentiment_score, article_count, top_headlines, timestamp}
"""
import os
from typing import Any

import httpx

from hermes_trading.adapters import BaseAdapter


class NewsAdapter(BaseAdapter):
    EXPECTED_SCHEMA_VERSION = 1

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.api_key = os.getenv("NEWS_API_KEY")

    def _simple_sentiment(self, text: str) -> float:
        positive = ["bull", "surge", "rally", "gain", "up", "high", "growth", "adopt", "partnership"]
        negative = ["bear", "crash", "fall", "drop", "down", "low", "decline", "hack", "ban", "regulation"]
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    async def fetch(self) -> dict:
        try:
            if self.api_key:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": "bitcoin OR crypto OR btc",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10,
                    "apiKey": self.api_key,
                }
                resp = await self.client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    articles = data.get("articles", [])
                    headlines = [a.get("title", "") for a in articles]
                    sentiments = [self._simple_sentiment(h) for h in headlines]
                    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                    return self._wrap_response(
                        {
                            "sentiment_score": avg_sentiment,
                            "article_count": len(articles),
                            "top_headlines": headlines[:5],
                            "timestamp": 0,
                            "source": "newsapi",
                        }
                    )
        except Exception:
            pass

        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {"auth_token": "public", "currencies": "BTC", "public": "true"}
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                posts = data.get("results", [])
                headlines = [p.get("title", "") for p in posts[:10]]
                sentiments = [self._simple_sentiment(h) for h in headlines]
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                return self._wrap_response(
                    {
                        "sentiment_score": avg_sentiment,
                        "article_count": len(posts),
                        "top_headlines": headlines[:5],
                        "timestamp": 0,
                        "source": "cryptopanic",
                    }
                )
        except Exception:
            pass

        return self._wrap_response(
            {
                "sentiment_score": 0.0,
                "article_count": 0,
                "top_headlines": [],
                "timestamp": 0,
                "source": "fallback",
                "error": "All news sources failed",
            }
        )