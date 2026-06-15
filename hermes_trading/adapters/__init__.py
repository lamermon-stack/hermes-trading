#!/usr/bin/env python3
"""
Base adapter class and schema versioning.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class SchemaError(Exception):
    """Raised when adapter response schema doesn't match expected version."""
    pass


@dataclass
class AdapterResponse:
    data: dict
    schema_version: int

    def __post_init__(self):
        if self.schema_version != self.EXPECTED_SCHEMA_VERSION:
            raise SchemaError(
                f"Schema mismatch: expected v{self.EXPECTED_SCHEMA_VERSION}, got v{self.schema_version}"
            )


class BaseAdapter(ABC):
    EXPECTED_SCHEMA_VERSION = 1

    @abstractmethod
    async def fetch(self) -> dict:
        """Fetch data and return dict with schema_version field."""
        pass

    def _wrap_response(self, data: dict) -> dict:
        return {"data": data, "schema_version": self.EXPECTED_SCHEMA_VERSION}