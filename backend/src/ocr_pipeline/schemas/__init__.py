"""Ready-made extraction schemas for common business documents."""

from ocr_pipeline.schemas.contract import Contract
from ocr_pipeline.schemas.invoice import Invoice

BUILTIN_SCHEMAS = {"invoice": Invoice, "contract": Contract}

__all__ = ["BUILTIN_SCHEMAS", "Contract", "Invoice"]
