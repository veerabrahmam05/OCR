"""Turn a user schema into the strict form every provider accepts, and validate results.

Strict structured outputs (Anthropic, OpenAI) require every object to list all
properties as required with ``additionalProperties: false``, and reject numeric /
length constraints. Ollama accepts the same strict schema. So:

* ``$ref`` s are inlined (recursive schemas are rejected),
* unsupported keywords are dropped (the original schema is still enforced locally),
* properties that were optional become required-but-nullable, and nulls the model
  returns for them are removed again before validation, so defaults apply.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import jsonschema
from pydantic import BaseModel, ValidationError

# Keywords strict mode does not support; enforced client-side by the original schema.
_DROP = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties", "default", "examples", "title", "$schema",
    "patternProperties", "propertyNames", "contains", "if", "then", "else", "not",
}  # fmt: skip


class SchemaError(ValueError):
    pass


def _resolve_refs(node: Any, defs: dict, stack: tuple[str, ...] = ()) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if ref is not None:
            name = ref.rsplit("/", 1)[-1]
            if name in stack:
                raise SchemaError(f"recursive schemas are not supported ({' -> '.join(stack + (name,))})")
            if name not in defs:
                raise SchemaError(f"unresolved $ref: {ref}")
            merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
            return _resolve_refs(merged, defs, stack + (name,))
        return {k: _resolve_refs(v, defs, stack) for k, v in node.items() if k not in ("$defs", "definitions")}
    if isinstance(node, list):
        return [_resolve_refs(v, defs, stack) for v in node]
    return node


def _nullable(node: dict) -> dict:
    if node.get("type") == "null" or any(s.get("type") == "null" for s in node.get("anyOf", [])):
        return node
    description = node.get("description")
    wrapped = {"anyOf": [node, {"type": "null"}]}
    if description:
        wrapped["description"] = description
    return wrapped


def _strictify(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: _strictify(v) for k, v in node.items() if k not in _DROP}
    if out.get("type") == "object" or "properties" in out:
        props = out.get("properties", {})
        originally_required = set(node.get("required", []))
        out["properties"] = {
            name: (sub if name in originally_required else _nullable(sub))
            for name, sub in props.items()
        }
        out["required"] = list(props)
        out["additionalProperties"] = False
        out.setdefault("type", "object")
    return out


def to_strict_schema(schema: dict) -> dict:
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}
    return _strictify(_resolve_refs(copy.deepcopy(schema), defs))


def drop_optional_nulls(data: Any, schema: dict) -> Any:
    """Remove null values the strict schema forced onto originally-optional properties."""
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}
    return _drop(data, _resolve_refs(copy.deepcopy(schema), defs))


def _drop(data: Any, node: dict) -> Any:
    if isinstance(data, dict) and "properties" in node:
        required = set(node.get("required", []))
        out = {}
        for k, v in data.items():
            if v is None and k not in required:
                continue
            sub = node["properties"].get(k)
            out[k] = _drop(v, _non_null_branch(sub)) if sub else v
        return out
    if isinstance(data, list) and isinstance(node.get("items"), dict):
        return [_drop(v, _non_null_branch(node["items"])) for v in data]
    return data


def _non_null_branch(node: dict) -> dict:
    branches = [s for s in node.get("anyOf", []) if s.get("type") != "null"]
    return branches[0] if len(branches) == 1 else node


class TargetSchema:
    """A user-supplied schema: a Pydantic model class or a JSON-schema dict."""

    def __init__(self, schema: type[BaseModel] | dict):
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            self.model: type[BaseModel] | None = schema
            self.json_schema = schema.model_json_schema()
        elif isinstance(schema, dict):
            self.model = None
            try:
                jsonschema.Draft202012Validator.check_schema(schema)
            except jsonschema.SchemaError as e:
                raise SchemaError(f"invalid JSON schema: {e.message}") from e
            self.json_schema = schema
        else:
            raise TypeError("schema must be a Pydantic model class or a JSON-schema dict")
        if self.json_schema.get("type", "object") != "object":
            raise SchemaError("the top-level schema must be an object")
        self.strict = to_strict_schema(self.json_schema)

    def validate(self, data: Any) -> dict:
        """Validate against the *original* schema; returns cleaned data or raises ValueError."""
        data = drop_optional_nulls(data, self.json_schema)
        if self.model is not None:
            try:
                return self.model.model_validate(data).model_dump(mode="json")
            except ValidationError as e:
                raise ValueError(str(e)) from e
        errors = sorted(
            jsonschema.Draft202012Validator(self.json_schema).iter_errors(data), key=lambda e: e.path
        )
        if errors:
            raise ValueError(
                "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:10])
            )
        return data


# Free-text mode: no user schema, the model lists whatever the instructions ask for.
FREE_FORM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "description": "One entry per requested value, in document order.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short snake_case name of the value."},
                    "value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["name", "value"],
            },
        }
    },
    "required": ["fields"],
}

_SOURCES_SCHEMA = {
    "type": "array",
    "description": "One entry per non-null extracted value.",
    "items": {
        "type": "object",
        "properties": {
            "field": {"type": "string", "description": "JSON path of the value, e.g. line_items[0].amount"},
            "page": {"type": "integer", "description": "Page number the value was taken from."},
            "quote": {"type": "string", "description": "Short verbatim excerpt of that page containing the value."},
        },
        "required": ["field", "page", "quote"],
        "additionalProperties": False,
    },
}


def envelope(data_schema: dict) -> dict:
    """The schema actually sent to the provider: the user's data plus source references."""
    return {
        "type": "object",
        "properties": {"data": data_schema, "sources": _SOURCES_SCHEMA},
        "required": ["data", "sources"],
        "additionalProperties": False,
    }


Validator = Callable[[Any], dict]
