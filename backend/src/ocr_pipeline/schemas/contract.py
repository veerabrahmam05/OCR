from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ContractParty(BaseModel):
    name: str
    role: str | None = Field(None, description="e.g. buyer, seller, licensor, employer")


class Contract(BaseModel):
    title: str | None = None
    parties: list[ContractParty] = Field(default_factory=list)
    effective_date: date | None = None
    end_date: date | None = None
    governing_law: str | None = Field(None, description="Jurisdiction whose law governs the contract")
    payment_terms: str | None = None
    termination_terms: str | None = Field(None, description="How and when either party can terminate")
    total_value: float | None = None
    currency: str | None = Field(None, description="ISO 4217 code")
