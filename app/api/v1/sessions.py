"""Conversation sessions.

A session is created implicitly by the first classify call that names it.
These endpoints let a caller inspect or forget one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from app.dependencies import get_session_service, require_classify, require_read
from app.errors import NotFoundError
from app.models.classification import SessionTurn
from app.security.auth import Principal
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionView(BaseModel):
    session_id: str
    turns: list[SessionTurn]
    last_intent: str | None = None
    entities_in_play: dict = {}


@router.get("/{session_id}", response_model=SessionView)
def get_session(
    session_id: str,
    request: Request,
    principal: Principal = Depends(require_read),
    sessions: SessionService = Depends(get_session_service),
) -> SessionView:
    turns = sessions.all_turns(session_id)
    if not turns:
        raise NotFoundError(f"session '{session_id}' has no turns")
    in_play: dict = {}
    for turn in turns:
        in_play.update(turn.entities)
    return SessionView(
        session_id=session_id,
        turns=turns,
        last_intent=turns[-1].intent,
        entities_in_play=in_play,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def clear_session(
    session_id: str,
    request: Request,
    principal: Principal = Depends(require_classify),
    sessions: SessionService = Depends(get_session_service),
) -> None:
    removed = sessions.clear(session_id)
    request.app.state.container.audit.record(
        action="session.clear", resource=f"session:{session_id}", turns=removed
    )
