"""Whatever a language model returns is untrusted input."""

import pytest

from app.services.entity_schema import (
    EntityRejected,
    coerce_value,
    missing_required,
    required_names,
    validate,
)

SCHEMA = {
    "counterparty": {"type": "string", "required": True},
    "status": {"type": "enum", "values": ["ACTIVE", "EXPIRED"]},
    "expiration_year": {"type": "integer"},
    "signed_after": {"type": "date"},
    "amount": {"type": "number"},
    "is_renewal": {"type": "boolean"},
    "tags": {"type": "array", "items": {"type": "string"}},
}


def test_undeclared_keys_are_dropped():
    accepted, rejected = validate({"counterparty": "Microsoft", "secret": "x"}, SCHEMA)
    assert accepted == {"counterparty": "Microsoft"}
    assert any("secret" in note for note in rejected)


def test_types_are_coerced_from_strings():
    accepted, _ = validate(
        {"expiration_year": "2026", "amount": "1500.50", "is_renewal": "yes"}, SCHEMA
    )
    assert accepted == {"expiration_year": 2026, "amount": 1500.5, "is_renewal": True}


def test_enum_is_normalized_to_the_declared_spelling():
    accepted, _ = validate({"status": "active"}, SCHEMA)
    assert accepted == {"status": "ACTIVE"}


def test_value_outside_an_enum_is_rejected():
    accepted, rejected = validate({"status": "PENDING"}, SCHEMA)
    assert accepted == {}
    assert any("PENDING" in note for note in rejected)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("2026-03-01", "2026-03-01"),
        ("March 1, 2026", "2026-03-01"),
        ("01/03/2026", "2026-03-01"),
        ("2026", "2026"),
    ],
)
def test_dates_are_normalized(given, expected):
    accepted, _ = validate({"signed_after": given}, SCHEMA)
    assert accepted["signed_after"] == expected


def test_unparseable_date_is_rejected():
    accepted, rejected = validate({"signed_after": "sometime soon"}, SCHEMA)
    assert accepted == {}
    assert rejected


def test_a_scalar_is_wrapped_into_a_declared_array():
    accepted, _ = validate({"tags": "urgent"}, SCHEMA)
    assert accepted["tags"] == ["urgent"]


def test_nulls_are_rejected_rather_than_passed_through():
    accepted, rejected = validate({"counterparty": None}, SCHEMA)
    assert accepted == {}
    assert rejected


def test_a_non_object_response_is_rejected_whole():
    accepted, rejected = validate(["Microsoft"], SCHEMA)
    assert accepted == {}
    assert rejected == ["extractor did not return an object"]


def test_required_fields_are_reported():
    assert required_names(SCHEMA) == ["counterparty"]
    assert missing_required({}, SCHEMA) == ["counterparty"]
    assert missing_required({"counterparty": "Microsoft"}, SCHEMA) == []


def test_an_object_where_a_scalar_is_declared_is_rejected():
    with pytest.raises(EntityRejected):
        coerce_value({"nested": "object"}, {"type": "string"})
