"""Domain: an isolated intent namespace."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.common import Status, name_key, new_id, now_ts, to_iso

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 _.\-]*$"


class DomainBase(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=NAME_PATTERN)
    description: str = Field(default="", max_length=2000)
    status: Status = Status.ACTIVE
    #: Appended to the extraction system prompt. Where a domain states its
    #: vocabulary, conventions and what must never be inferred.
    system_instructions: str = Field(default="", max_length=4000)
    #: Prepended to the user turn sent to the extractor. Where a domain gives
    #: examples of how its users phrase things.
    user_instructions: str = Field(default="", max_length=4000)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class DomainCreate(DomainBase):
    #: Draft system/user instructions from the name and description with the
    #: configured LLM provider at creation time. Explicitly supplied
    #: instructions always win over generated ones.
    generate_instructions: bool = False


class InstructionGenerateRequest(BaseModel):
    """Regenerate a domain's extraction instructions.

    ``brief`` overrides the stored description as the input; useful when the
    description is empty or the admin wants to steer the draft.
    """

    brief: str | None = Field(default=None, max_length=2000)


class InstructionGenerateResponse(BaseModel):
    domain_id: str
    status: str
    detail: str | None = None
    provider: str = ""
    system_instructions: str = ""
    user_instructions: str = ""
    saved: bool = False


class DomainUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120, pattern=NAME_PATTERN)
    description: str | None = Field(default=None, max_length=2000)
    status: Status | None = None
    system_instructions: str | None = Field(default=None, max_length=4000)
    user_instructions: str | None = Field(default=None, max_length=4000)


class DomainConfig(DomainBase):
    """The persisted record (stored as the Chroma document, JSON encoded)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id)
    created_at: float = Field(default_factory=now_ts)
    updated_at: float = Field(default_factory=now_ts)

    @property
    def name_key(self) -> str:
        return name_key(self.name)


class DomainRead(DomainConfig):
    intent_count: int = 0
    example_count: int = 0

    @property
    def created_at_iso(self) -> str:
        return to_iso(self.created_at)
