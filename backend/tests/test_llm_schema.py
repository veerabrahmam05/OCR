"""Strict-schema conversion and local validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from ocr_pipeline.llm.schema import SchemaError, TargetSchema, envelope, to_strict_schema
from ocr_pipeline.schemas import BUILTIN_SCHEMAS, Invoice


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("name", list(BUILTIN_SCHEMAS))
def test_builtin_schemas_become_strict(name):
    strict = TargetSchema(BUILTIN_SCHEMAS[name]).strict
    for node in _walk(strict):
        assert "$ref" not in node and "$defs" not in node
        assert not {"default", "title", "minimum", "maxLength"} & set(node)
        if "properties" in node:
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])


def test_optional_fields_become_nullable():
    props = TargetSchema(Invoice).strict["properties"]
    assert {"type": "null"} in props["invoice_number"]["anyOf"]
    # nested Party (via $ref) is inlined and nullable
    party = next(s for s in props["seller"]["anyOf"] if s.get("type") == "object")
    assert set(party["properties"]) == {"name", "address", "tax_id"}


def test_unsupported_constraints_still_enforced_locally():
    schema = {
        "type": "object",
        "properties": {"qty": {"type": "integer", "minimum": 0}},
        "required": ["qty"],
    }
    target = TargetSchema(schema)
    assert "minimum" not in target.strict["properties"]["qty"]
    assert target.validate({"qty": 3}) == {"qty": 3}
    with pytest.raises(ValueError, match="qty"):
        target.validate({"qty": -1})


def test_optional_property_null_is_dropped_before_validation():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
    }
    target = TargetSchema(schema)
    assert target.strict["required"] == ["a", "b"]  # forced required ...
    assert {"type": "null"} in target.strict["properties"]["b"]["anyOf"]  # ... but nullable
    assert target.validate({"a": "x", "b": None}) == {"a": "x"}


def test_pydantic_defaults_fill_dropped_nulls():
    data = TargetSchema(Invoice).validate(
        {"invoice_number": "INV-1", "invoice_date": "2026-11-12", "line_items": [], "total": None}
    )
    assert data["invoice_number"] == "INV-1"
    assert data["invoice_date"] == "2026-11-12"
    assert data["total"] is None and data["line_items"] == []


def test_nested_list_items_get_nulls_dropped():
    data = TargetSchema(Invoice).validate(
        {"line_items": [{"description": "Bolts", "quantity": None, "unit_price": 0.5, "amount": None}]}
    )
    assert data["line_items"][0]["unit_price"] == 0.5


def test_recursive_schema_rejected():
    class Node(BaseModel):
        name: str
        children: list[Node] = []

    with pytest.raises(SchemaError, match="recursive"):
        TargetSchema(Node)


def test_top_level_must_be_object():
    with pytest.raises(SchemaError):
        TargetSchema({"type": "array", "items": {"type": "string"}})


def test_invalid_json_schema_reported():
    with pytest.raises(SchemaError, match="invalid JSON schema"):
        TargetSchema({"type": "object", "properties": {"a": {"type": "no-such-type"}}})


def test_envelope_wraps_data_and_sources():
    env = envelope(to_strict_schema({"type": "object", "properties": {}}))
    assert env["required"] == ["data", "sources"]
    item = env["properties"]["sources"]["items"]
    assert item["required"] == ["field", "page", "quote"]
    assert item["additionalProperties"] is False
