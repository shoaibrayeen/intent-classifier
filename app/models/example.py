"""Training example: a natural-language utterance belonging to an intent."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, field_validator

from app.models.common import new_id, now_ts


def text_hash(normalized: str) -> str:
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class ExampleCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class Example(BaseModel):
    id: str = Field(default_factory=new_id)
    domain_id: str
    intent_id: str
    intent_name: str
    text: str
    text_hash: str
    created_at: float = Field(default_factory=now_ts)


class ExampleRead(Example):
    pass
