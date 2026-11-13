from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Party(BaseModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = Field(None, description="VAT / GST / tax registration number")


class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = Field(None, description="Line total as printed")


class Invoice(BaseModel):
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    currency: str | None = Field(None, description="ISO 4217 code, e.g. USD, EUR, INR")
    seller: Party | None = None
    buyer: Party | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax_total: float | None = None
    total: float | None = Field(None, description="Total amount due")
    payment_terms: str | None = None
