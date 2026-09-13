"""Conversation memory for multi-turn classification.

A session is just a caller-chosen id. Turns in the same session and domain
inform each other in two deliberate, inspectable ways:

* **Contextual retrieval.** "when do they expire" cannot be classified on its
  own. If a turn comes back UNKNOWN and there is a previous turn, it is retried
  as the previous question plus the new one. The contextual result is used only
  if it actually resolves, so context can rescue a follow-up but never overrule
  a confident answer.

  Two guards keep that from absorbing a change of subject. The retry only runs
  for text that reads as referential -- a pronoun, or a fragment too short to
  stand alone -- and its result must clear the threshold by a margin. Without
  them, "what is the weather today" asked after "summarize contract C-1042"
  came back as CONTRACT_SUMMARY, because the previous question alone was enough
  to match the concatenation.
* **Entity carry-over.** Values named earlier ("Microsoft") flow into a later
  intent that accepts the same entity, unless the new turn names its own. The
  response lists which entities were carried, so the caller can see it.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.models.classification import SessionTurn
from app.models.common import now_ts
from app.repositories.chroma_sessions import ChromaSessionRepository
from app.services.normalizer import tokenize

logger = logging.getLogger(__name__)

#: Words that point at something already said. Their presence is what makes a
#: turn a follow-up rather than a new question.
REFERENTIAL_WORDS = frozenset(
    {
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "those",
        "these",
        "that",
        "this",
        "there",
        "same",
        "one",
        "ones",
        "another",
        "other",
        "others",
        "him",
        "her",
        "his",
        "hers",
        "he",
        "she",
        "above",
        "previous",
        "earlier",
        "instead",
        "too",
        "also",
        "as",
        "well",
    }
)


class SessionService:
    def __init__(self, settings: Settings, repo: ChromaSessionRepository) -> None:
        self._settings = settings
        self._repo = repo
        self._writes_since_expiry = 0

    @property
    def enabled(self) -> bool:
        return self._settings.sessions_enabled

    def history(self, session_id: str, domain_id: str) -> list[SessionTurn]:
        if not self.enabled:
            return []
        return self._repo.history(session_id, domain_id, self._settings.session_history_turns)

    def all_turns(self, session_id: str) -> list[SessionTurn]:
        return self._repo.all_turns(session_id)

    def next_turn_number(self, history: list[SessionTurn]) -> int:
        return (history[-1].turn + 1) if history else 1

    def reads_as_followup(self, text: str) -> bool:
        """Whether this turn points back at the conversation.

        A complete question that simply does not match the domain is a change of
        subject, not a follow-up, and must not be rescued by being glued to the
        previous one.
        """
        words = tokenize(text)
        if not words:
            return False
        if len(words) <= self._settings.context_followup_max_words:
            return True
        return bool(set(words) & REFERENTIAL_WORDS)

    def contextual_text(self, history: list[SessionTurn], text: str) -> str | None:
        """The previous question prepended to a follow-up, or None if nothing to add."""
        if not history or not self._settings.context_retrieval_enabled:
            return None
        if not self.reads_as_followup(text):
            return None
        previous = history[-1].text.strip().rstrip("?.!")
        if not previous or previous.casefold() == text.strip().casefold():
            return None
        return f"{previous} {text.strip()}"

    @property
    def rescue_floor(self) -> float:
        """What a contextual retry must score before it is believed."""
        return self._settings.confidence_threshold + self._settings.context_rescue_margin

    def carry_over(
        self,
        history: list[SessionTurn],
        entities: dict[str, Any],
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Fill schema entities the new turn did not name from earlier turns.

        Most recent turn wins. Only entities the *new* intent declares are
        eligible, so a value never leaks into a tool that has no use for it.
        """
        if not history or not schema or not self._settings.entity_carry_over_enabled:
            return dict(entities), []
        merged = dict(entities)
        carried: list[str] = []
        for turn in reversed(history):
            for name, value in turn.entities.items():
                if name in schema and name not in merged and value not in (None, "", [], {}):
                    merged[name] = value
                    carried.append(name)
        return merged, carried

    def record(
        self,
        session_id: str,
        domain_id: str,
        turn: SessionTurn,
        principal: str = "",
    ) -> None:
        if not self.enabled:
            return
        turn.created_at = turn.created_at or now_ts()
        self._repo.append(session_id, domain_id, turn, principal)
        self._repo.prune(session_id, self._settings.session_max_turns)
        # Expiry is a store-wide sweep, so amortise it rather than doing it on
        # every write.
        self._writes_since_expiry += 1
        if self._writes_since_expiry >= 50:
            self._writes_since_expiry = 0
            removed = self._repo.expire(self._settings.session_ttl_seconds)
            if removed:
                logger.info("expired %d stale session turn(s)", removed)

    def clear(self, session_id: str) -> int:
        return self._repo.delete(session_id)

    def expire_now(self) -> int:
        return self._repo.expire(self._settings.session_ttl_seconds)
