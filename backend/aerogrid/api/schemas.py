"""Request / response schemas for the HTTP API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SourceKind = Literal["live", "cached", "simulated"]


class PredictRequest(BaseModel):
    origin: str = Field(min_length=3, max_length=4, description="IATA or ICAO code of an airport in the seed list")
    destination: str = Field(min_length=3, max_length=4)
    airline: Optional[str] = Field(default=None, min_length=2, max_length=3, description="BTS reporting-airline code, e.g. AA, DL, UA")
    departure_time: Optional[datetime] = Field(default=None, description="Scheduled departure (UTC if no offset); default now + 2 h")
    aircraft: Literal["regional", "narrowbody", "widebody", "ultra_long"] = "narrowbody"

    @field_validator("origin", "destination", "airline")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if v else v


class WhatIfRequest(BaseModel):
    airport: str = Field(min_length=3, max_length=4)
    kind: Literal["storm", "runway_closure"]
    severity: float = Field(ge=0.0, le=1.0)

    @field_validator("airport")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class SourceBadge(BaseModel):
    source: SourceKind
    age_s: Optional[float] = None
    fallback_reason: Optional[str] = None


class Health(BaseModel):
    status: Literal["ok"]
    time: float
    models: dict[str, Any]
    sources: dict[str, Any]
    version: str


class Cause(BaseModel):
    key: str
    label: str
    minutes: float
    share: float


class PredictResponse(BaseModel):
    origin: str
    destination: str
    airline: Optional[str]
    departure_time: str
    horizon_h: int
    delay_min: float
    interval80: list[float]
    baseline_min: float
    causes: list[Cause]
    drivers: dict[str, Any]
    route: dict[str, Any]
    bird: dict[str, Any]
    advice: list[str]
    nas: dict[str, Any]
    sources: dict[str, Any]
    models: dict[str, Any]
    confidence: dict[str, Any]
    trained_on: str
