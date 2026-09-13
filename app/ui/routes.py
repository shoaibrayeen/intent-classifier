"""Server-rendered management UI.

Every mutation calls the same service objects as the JSON API and returns the
affected list partial plus an out-of-band flash message, so htmx can swap both
in one response.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.dependencies import (
    get_container,
    get_domain_service,
    get_example_service,
    get_index_manager,
    get_intent_service,
)
from app.errors import InvalidInputError, NotFoundError
from app.models.classification import ClassifyRequest, SessionTurn
from app.models.domain import DomainCreate, DomainUpdate
from app.models.example import ExampleCreate
from app.models.intent import IntentCreate, IntentUpdate, ToolRef
from app.models.mcp import McpImportRequest, McpToolCreate, McpToolRead
from app.observability import tracing
from app.services.container import Container
from app.services.domain_service import DomainService
from app.services.evaluation import run_evaluation
from app.services.example_service import ExampleService
from app.services.index_manager import IndexManager
from app.services.intent_service import IntentService
from app.services.mcp_service import schema_summary
from app.services.strategies import BUILTIN_STRATEGIES
from app.services.turn_details import snapshot_of, snapshot_of_turn
from app.ui.templating import templates

router = APIRouter(tags=["ui"], include_in_schema=False)

#: The reference pages, served as a directory at /ui/docs so that the links
#: between them resolve the same way in the browser as they do on disk.
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
DOCS_MOUNT = "/ui/docs"
#: Short, stable URLs for the pages people go looking for.
DOC_ALIASES = {
    "/ui/api": "api-documentation.html",
    "/ui/changelog": "changelog.html",
    "/changelog": "changelog.html",
    "/ui/properties": "properties.html",
    "/ui/architecture": "architecture.html",
    "/ui/demo": "demo.html",
}


def _flash(message: str, level: str = "success") -> str:
    return f'<div id="flash" class="flash {level}" hx-swap-oob="true">{message}</div>'


def _partial(request: Request, name: str, context: dict, flash: str | None = None) -> HTMLResponse:
    response = templates.TemplateResponse(request=request, name=name, context=context)
    if flash:
        body = response.body.decode() + flash
        return HTMLResponse(body)
    return response


# --------------------------------------------------------------------- pages
@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    domains: DomainService = Depends(get_domain_service),
    container: Container = Depends(get_container),
):
    return templates.TemplateResponse(
        request=request,
        name="pages/dashboard.html",
        context={
            "domains": domains.list(),
            "generator_available": container.instruction_generator.available,
            "provider": container.instruction_generator.provider_name,
        },
    )


@router.get("/ui/domains/{domain_id}", response_class=HTMLResponse)
def domain_page(
    request: Request,
    domain_id: str,
    domains: DomainService = Depends(get_domain_service),
    intents: IntentService = Depends(get_intent_service),
    indexes: IndexManager = Depends(get_index_manager),
):
    domain = domains.read(domain_id)
    container = request.app.state.container
    return templates.TemplateResponse(
        request=request,
        name="pages/domain.html",
        context={
            "domain": domain,
            "intents": intents.list(domain_id),
            "status": indexes.status(domain_id),
            "generator_available": container.instruction_generator.available,
            "provider": container.instruction_generator.provider_name,
        },
    )


@router.get("/ui/domains/{domain_id}/intents/{intent_id}", response_class=HTMLResponse)
def intent_page(
    request: Request,
    domain_id: str,
    intent_id: str,
    domains: DomainService = Depends(get_domain_service),
    intents: IntentService = Depends(get_intent_service),
    examples: ExampleService = Depends(get_example_service),
    indexes: IndexManager = Depends(get_index_manager),
):
    container = request.app.state.container
    return templates.TemplateResponse(
        request=request,
        name="pages/intent.html",
        context={
            "domain": domains.get(domain_id),
            "intent": intents.read(domain_id, intent_id),
            "examples": examples.list(domain_id, intent_id),
            "status": indexes.status(domain_id),
            "generator_available": container.intent_authoring.available,
            "provider": container.intent_authoring.provider_name,
            "mcp_tools": _mcp_rows(container),
            "bound": container.mcp_tools.resolve(
                intents.get(domain_id, intent_id).tool.mcp_tool_id or ""
            ),
        },
    )


@router.get("/ui/playground", response_class=HTMLResponse)
def playground(
    request: Request,
    domain: str | None = None,
    session: str | None = None,
    domains: DomainService = Depends(get_domain_service),
    container: Container = Depends(get_container),
):
    """The chat workbench.

    With ``?session=`` it reopens an existing conversation, its turns and their
    recorded details included, so a session listed elsewhere is one click from
    being inspected rather than being a dead row.
    """
    session_id = (session or "").strip() or f"chat-{uuid.uuid4().hex[:8]}"
    turns = container.sessions.all_turns(session_id) if session else []
    selected = domain
    if turns and not selected:
        # Reopen the conversation against the domain it actually happened in.
        selected = _domain_of_session(container, session_id)
    return templates.TemplateResponse(
        request=request,
        name="pages/playground.html",
        context={
            "domains": domains.list(),
            "selected": selected,
            "session_id": session_id,
            "turns": turns,
            "details": snapshot_of_turn(turns[-1]) if turns else None,
            "strategies": list(BUILTIN_STRATEGIES.values()),
            "extraction_available": container.entity_extractor.available,
            "extraction_enabled": container.entity_extractor.enabled,
            "provider": container.entity_extractor.provider_name,
            "settings": container.settings,
        },
    )


@router.get("/ui/operations", response_class=HTMLResponse)
def operations(
    request: Request,
    container: Container = Depends(get_container),
    domains: DomainService = Depends(get_domain_service),
):
    """Index health, configuration and recent audit activity in one place."""
    statuses = _index_statuses(container, domains)
    return templates.TemplateResponse(
        request=request,
        name="pages/operations.html",
        context={
            "statuses": statuses,
            "settings": container.settings,
            "auth_enabled": container.authenticator.enabled,
            "key_count": container.authenticator.key_count,
            "extraction_available": container.entity_extractor.available,
            "extraction_enabled": container.entity_extractor.enabled,
            "provider": container.entity_extractor.provider_name,
            "tracing_enabled": tracing.enabled(),
            "ab_enabled": container.strategies.enabled,
            "variants": container.strategies.variants,
            "audit_entries": container.audit.tail(25),
            "audit_enabled": container.audit.enabled,
        },
    )


def _doc_redirect(filename: str):
    """A named route that redirects to the file under the docs mount."""

    def handler() -> RedirectResponse:
        return RedirectResponse(f"{DOCS_MOUNT}/{filename}", status_code=307)

    return handler


for _path, _filename in DOC_ALIASES.items():
    router.add_api_route(
        _path,
        _doc_redirect(_filename),
        methods=["GET"],
        include_in_schema=_path.startswith("/ui/"),
        name=f"doc-{_filename.removesuffix('.html')}",
    )


@router.get("/ui/domains", include_in_schema=False)
def domains_index() -> RedirectResponse:
    """The dashboard lists domains, and /ui/domains is where people try first.

    Without this it answered 405, because only the create handler was bound to
    that path.
    """
    return RedirectResponse("/", status_code=307)


@router.get("/ui/{name}.html", include_in_schema=False)
def doc_by_filename(name: str) -> RedirectResponse:
    """Forgive /ui/architecture.html, which is where the links used to point."""
    if not (DOCS_DIR / f"{name}.html").is_file():
        raise NotFoundError(f"no document named '{name}.html'")
    return RedirectResponse(f"{DOCS_MOUNT}/{name}.html", status_code=307)


@router.get("/ui/operations/index-health", response_class=HTMLResponse)
def operations_index_health(
    request: Request,
    container: Container = Depends(get_container),
    domains: DomainService = Depends(get_domain_service),
):
    """Refresh target: index health polls rather than needing a page reload."""
    return templates.TemplateResponse(
        request=request,
        name="partials/index_health.html",
        context={"statuses": _index_statuses(container, domains)},
    )


@router.get("/ui/operations/audit", response_class=HTMLResponse)
def operations_audit(request: Request, container: Container = Depends(get_container)):
    return templates.TemplateResponse(
        request=request,
        name="partials/audit_feed.html",
        context={"audit_entries": container.audit.tail(25)},
    )


@router.get("/ui/evaluation", response_class=HTMLResponse)
def evaluation_page(request: Request, container: Container = Depends(get_container)):
    from app.services.evaluation import dataset_path

    return templates.TemplateResponse(
        request=request,
        name="pages/evaluation.html",
        context={"dataset": dataset_path(container.settings)},
    )


@router.post("/ui/evaluation/run", response_class=HTMLResponse)
async def ui_run_evaluation(
    request: Request,
    variant: Annotated[str, Form()] = "",
    container: Container = Depends(get_container),
):
    """Run the held-out evaluation set against the live catalogue."""
    report = await run_evaluation(container, variant or None)
    return templates.TemplateResponse(
        request=request,
        name="partials/evaluation_result.html",
        context={"report": report, "settings": container.settings},
    )


# ----------------------------------------------------------------- mutations
@router.post("/ui/domains", response_class=HTMLResponse)
async def ui_create_domain(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    system_instructions: Annotated[str, Form()] = "",
    user_instructions: Annotated[str, Form()] = "",
    generate_instructions: Annotated[str, Form()] = "",
    domains: DomainService = Depends(get_domain_service),
    container: Container = Depends(get_container),
):
    domain = domains.create(
        DomainCreate(
            name=name,
            description=description,
            system_instructions=system_instructions,
            user_instructions=user_instructions,
        )
    )
    message = f"Domain '{name}' created."
    if generate_instructions and not (system_instructions.strip() or user_instructions.strip()):
        result = await container.instruction_generator.generate(domain.name, domain.description)
        if result.status == "ok":
            domains.update(
                domain.id,
                DomainUpdate(
                    system_instructions=result.system_instructions,
                    user_instructions=result.user_instructions,
                ),
            )
            message = f"Domain '{name}' created; extraction instructions drafted and saved."
        else:
            message = (
                f"Domain '{name}' created, but instruction generation "
                f"{result.status}: {result.detail}"
            )
    return _partial(
        request,
        "partials/domain_list.html",
        {"domains": domains.list()},
        _flash(message),
    )


@router.post("/ui/domains/{domain_id}/instructions", response_class=HTMLResponse)
def ui_update_instructions(
    request: Request,
    domain_id: str,
    system_instructions: Annotated[str, Form()] = "",
    user_instructions: Annotated[str, Form()] = "",
    domains: DomainService = Depends(get_domain_service),
):
    domains.update(
        domain_id,
        DomainUpdate(system_instructions=system_instructions, user_instructions=user_instructions),
    )
    return HTMLResponse(_flash("Extraction instructions saved."))


@router.post("/ui/domains/{domain_id}/instructions/generate", response_class=HTMLResponse)
async def ui_generate_instructions(
    request: Request,
    domain_id: str,
    brief: Annotated[str, Form()] = "",
    domains: DomainService = Depends(get_domain_service),
    container: Container = Depends(get_container),
):
    """Draft, save, and re-render the editor with the draft loaded for refining."""
    domain = domains.get(domain_id)
    generator = container.instruction_generator
    if not generator.available:
        return HTMLResponse(
            _flash("No LLM provider configured. Set OPENAI_API_KEY or LLM_PROVIDER=mock.", "error")
        )
    effective_brief = brief.strip() or domain.description.strip()
    if not effective_brief:
        return HTMLResponse(
            _flash("Give the generator a brief (or set a domain description) first.", "error")
        )
    result = await generator.generate(domain.name, effective_brief)
    if result.status != "ok":
        return HTMLResponse(_flash(f"Generation {result.status}: {result.detail}", "error"))
    updated = domains.update(
        domain_id,
        DomainUpdate(
            system_instructions=result.system_instructions,
            user_instructions=result.user_instructions,
        ),
    )
    return _partial(
        request,
        "partials/instructions_form.html",
        {
            "domain": updated,
            "generator_available": generator.available,
            "provider": generator.provider_name,
        },
        _flash("Instructions drafted and saved. Refine them below if needed."),
    )


@router.delete("/ui/domains/{domain_id}", response_class=HTMLResponse)
def ui_delete_domain(
    request: Request, domain_id: str, domains: DomainService = Depends(get_domain_service)
):
    domain = domains.get(domain_id)
    domains.delete(domain_id)
    return _partial(
        request,
        "partials/domain_list.html",
        {"domains": domains.list()},
        _flash(f"Domain '{domain.name}' deleted."),
    )


@router.post("/ui/domains/{domain_id}/intents/generate", response_class=HTMLResponse)
async def ui_generate_intents(
    request: Request,
    domain_id: str,
    count: Annotated[int, Form()] = 5,
    examples_per_intent: Annotated[int, Form()] = 6,
    brief: Annotated[str, Form()] = "",
    domains: DomainService = Depends(get_domain_service),
    intents: IntentService = Depends(get_intent_service),
    container: Container = Depends(get_container),
):
    """Auto mode: draft, save and index this domain's intents."""
    domain = domains.get(domain_id)
    authoring = container.intent_authoring
    if not authoring.available:
        return HTMLResponse(
            _flash("No LLM provider configured. Set OPENAI_API_KEY or LLM_PROVIDER=mock.", "error")
        )
    result = await authoring.generate_intents(
        domain,
        brief=brief.strip(),
        count=max(1, min(count, 12)),
        examples_per_intent=max(0, min(examples_per_intent, 25)),
    )
    if result.status != "ok":
        return HTMLResponse(_flash(f"Generation {result.status}: {result.detail}", "error"))
    message = (
        f"Generated {result.created_count} intent(s) with "
        f"{result.example_count} example(s). Edit any of them below."
    )
    if result.skipped:
        message += f" Skipped {len(result.skipped)}: {result.skipped[0]}"
    return _partial(
        request,
        "partials/intent_list.html",
        {"domain": domain, "intents": intents.list(domain_id)},
        _flash(message),
    )


@router.post(
    "/ui/domains/{domain_id}/intents/{intent_id}/examples/generate",
    response_class=HTMLResponse,
)
async def ui_generate_examples(
    request: Request,
    domain_id: str,
    intent_id: str,
    count: Annotated[int, Form()] = 6,
    domains: DomainService = Depends(get_domain_service),
    intents: IntentService = Depends(get_intent_service),
    examples: ExampleService = Depends(get_example_service),
    container: Container = Depends(get_container),
):
    domain = domains.get(domain_id)
    intent = intents.get(domain_id, intent_id)
    authoring = container.intent_authoring
    if not authoring.available:
        return HTMLResponse(
            _flash("No LLM provider configured. Set OPENAI_API_KEY or LLM_PROVIDER=mock.", "error")
        )
    result = await authoring.generate_examples(domain, intent, count=max(1, min(count, 25)))
    if result.status != "ok":
        return HTMLResponse(_flash(f"Generation {result.status}: {result.detail}", "error"))
    message = f"Added {result.created_count} example(s)."
    if result.skipped:
        message += f" Skipped {len(result.skipped)} duplicate(s)."
    return _partial(
        request,
        "partials/example_list.html",
        {
            "domain": domain,
            "intent": intent,
            "examples": examples.list(domain_id, intent_id),
        },
        _flash(message),
    )


@router.post("/ui/domains/{domain_id}/intents", response_class=HTMLResponse)
def ui_create_intent(
    request: Request,
    domain_id: str,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    tool_name: Annotated[str, Form()] = "",
    tool_version: Annotated[str, Form()] = "v1",
    entity_schema: Annotated[str, Form()] = "",
    extraction_hints: Annotated[str, Form()] = "",
    domains: DomainService = Depends(get_domain_service),
    intents: IntentService = Depends(get_intent_service),
):
    domains.get(domain_id)
    intents.create(
        domain_id,
        IntentCreate(
            name=name,
            description=description,
            tool=ToolRef(name=tool_name, version=tool_version or "v1"),
            entity_schema=_parse_schema(entity_schema),
            extraction_hints=extraction_hints,
        ),
    )
    return _partial(
        request,
        "partials/intent_list.html",
        {"domain": domains.read(domain_id), "intents": intents.list(domain_id)},
        _flash(f"Intent '{name}' created."),
    )


@router.delete("/ui/domains/{domain_id}/intents/{intent_id}", response_class=HTMLResponse)
def ui_delete_intent(
    request: Request,
    domain_id: str,
    intent_id: str,
    domains: DomainService = Depends(get_domain_service),
    intents: IntentService = Depends(get_intent_service),
):
    intent = intents.get(domain_id, intent_id)
    intents.delete(domain_id, intent_id)
    return _partial(
        request,
        "partials/intent_list.html",
        {"domain": domains.read(domain_id), "intents": intents.list(domain_id)},
        _flash(f"Intent '{intent.name}' deleted."),
    )


@router.post("/ui/domains/{domain_id}/intents/{intent_id}/examples", response_class=HTMLResponse)
def ui_add_example(
    request: Request,
    domain_id: str,
    intent_id: str,
    text: Annotated[str, Form()],
    domains: DomainService = Depends(get_domain_service),
    examples: ExampleService = Depends(get_example_service),
):
    domains.get(domain_id)
    examples.add(domain_id, intent_id, ExampleCreate(text=text))
    return _partial(
        request,
        "partials/example_list.html",
        {
            "domain_id": domain_id,
            "intent_id": intent_id,
            "examples": examples.list(domain_id, intent_id),
        },
        _flash("Example added and index rebuilt."),
    )


@router.delete(
    "/ui/domains/{domain_id}/intents/{intent_id}/examples/{example_id}",
    response_class=HTMLResponse,
)
def ui_delete_example(
    request: Request,
    domain_id: str,
    intent_id: str,
    example_id: str,
    examples: ExampleService = Depends(get_example_service),
):
    examples.delete(domain_id, intent_id, example_id)
    return _partial(
        request,
        "partials/example_list.html",
        {
            "domain_id": domain_id,
            "intent_id": intent_id,
            "examples": examples.list(domain_id, intent_id),
        },
        _flash("Example deleted and index rebuilt."),
    )


@router.post("/ui/domains/{domain_id}/reindex", response_class=HTMLResponse)
def ui_reindex(
    request: Request,
    domain_id: str,
    domains: DomainService = Depends(get_domain_service),
    indexes: IndexManager = Depends(get_index_manager),
):
    domains.get(domain_id)
    indexes.rebuild(domain_id)
    return _partial(
        request,
        "partials/index_status.html",
        {"status": indexes.status(domain_id)},
        _flash("Index rebuilt."),
    )


@router.post("/ui/playground/classify", response_class=HTMLResponse)
async def ui_classify(
    request: Request,
    domain: Annotated[str, Form()],
    text: Annotated[str, Form()],
    extract_entities: Annotated[str, Form()] = "",
    variant: Annotated[str, Form()] = "",
    session_id: Annotated[str, Form()] = "",
    container: Container = Depends(get_container),
):
    """Classify one chat message.

    Returns the new exchange to append to the thread, and swaps the detail pane
    out of band, so the answer and the reasoning arrive together.
    """
    payload = ClassifyRequest(
        domain=domain,
        text=text,
        extract_entities=True if extract_entities else None,
        variant=variant or None,
        session_id=session_id.strip() or None,
    )
    result = await container.classification.classify(payload, debug=True)

    details = snapshot_of(result, result.debug)
    details["text"] = payload.text
    details["turn"] = result.context.turn if result.context else 0
    turn = SessionTurn(
        turn=details["turn"],
        text=payload.text,
        intent=result.intent,
        intent_id=result.intent_id,
        confidence=result.confidence,
        entities=result.entities,
        details=details,
    )
    bubble = templates.TemplateResponse(
        request=request,
        name="partials/chat_turn.html",
        context={"turn": turn, "session_id": payload.session_id or ""},
    )
    panel = templates.TemplateResponse(
        request=request,
        name="partials/turn_details.html",
        context={"details": details, "settings": container.settings},
    )
    return HTMLResponse(
        bubble.body.decode()
        + '<div id="detail-panel" class="card detail-pane" hx-swap-oob="true">'
        + panel.body.decode()
        + "</div>"
    )


@router.get("/ui/playground/details/{session_id}/{turn}", response_class=HTMLResponse)
def playground_turn_details(
    request: Request,
    session_id: str,
    turn: int,
    container: Container = Depends(get_container),
):
    """The detail pane for one recorded turn."""
    match = next((t for t in container.sessions.all_turns(session_id) if t.turn == turn), None)
    if match is None:
        raise NotFoundError(f"turn {turn} not found in session '{session_id}'")
    return templates.TemplateResponse(
        request=request,
        name="partials/turn_details.html",
        context={"details": snapshot_of_turn(match), "settings": container.settings},
    )


@router.delete("/ui/playground/session/{session_id}", response_class=HTMLResponse)
def ui_clear_session(
    request: Request, session_id: str, container: Container = Depends(get_container)
):
    removed = container.sessions.clear(session_id)
    return _partial(
        request,
        "partials/chat_thread.html",
        {"turns": [], "session_id": session_id},
        _flash(f"Session cleared ({removed} turn(s) forgotten)."),
    )


def _parse_schema(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"entity schema is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise InvalidInputError("entity schema must be a JSON object")
    return parsed


# --------------------------------------------------------------- MCP registry
@router.get("/ui/mcp", response_class=HTMLResponse)
def mcp_page(request: Request, container: Container = Depends(get_container)):
    return templates.TemplateResponse(
        request=request,
        name="pages/mcp.html",
        context={"tools": _mcp_rows(container)},
    )


@router.get("/ui/mcp/{tool_id}", response_class=HTMLResponse)
def mcp_tool_page(request: Request, tool_id: str, container: Container = Depends(get_container)):
    tool = container.mcp_tools.get(tool_id)
    bound = []
    for domain in container.domains.list():
        for intent in container.intents.list(domain.id):
            resolved = container.mcp_tools.resolve(intent.tool.mcp_tool_id or "")
            if resolved is not None and resolved.id == tool.id:
                bound.append({"domain": domain, "intent": intent})
    return templates.TemplateResponse(
        request=request,
        name="pages/mcp_tool.html",
        context={
            "tool": McpToolRead.of(tool, len(bound)),
            "schema_rows": schema_summary(tool),
            "bound": bound,
        },
    )


@router.post("/ui/mcp", response_class=HTMLResponse)
def ui_register_mcp_tool(
    request: Request,
    server: Annotated[str, Form()],
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    transport: Annotated[str, Form()] = "unknown",
    endpoint: Annotated[str, Form()] = "",
    input_schema: Annotated[str, Form()] = "",
    container: Container = Depends(get_container),
):
    tool = container.mcp_tools.create(
        McpToolCreate(
            server=server,
            name=name,
            description=description,
            transport=transport,
            endpoint=endpoint,
            input_schema=_parse_schema(input_schema),
        )
    )
    return _partial(
        request,
        "partials/mcp_list.html",
        {"tools": _mcp_rows(container)},
        _flash(f"Registered {tool.qualified_name}."),
    )


@router.post("/ui/mcp/import", response_class=HTMLResponse)
def ui_import_mcp_tools(
    request: Request,
    server: Annotated[str, Form()],
    tools: Annotated[str, Form()],
    transport: Annotated[str, Form()] = "unknown",
    endpoint: Annotated[str, Form()] = "",
    replace: Annotated[str, Form()] = "",
    container: Container = Depends(get_container),
):
    try:
        payload = json.loads(tools)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"tools/list must be valid JSON: {exc}") from exc
    result = container.mcp_tools.import_tools(
        McpImportRequest(
            server=server,
            transport=transport,
            endpoint=endpoint,
            tools=payload,
            replace=bool(replace),
        )
    )
    message = (
        f"Imported {len(result.created)} new and updated {len(result.updated)} "
        f"tool(s) for '{server}'."
    )
    if result.removed:
        message += f" Removed {len(result.removed)}."
    if result.skipped:
        message += f" Skipped {len(result.skipped)}: {result.skipped[0]}"
    return _partial(
        request, "partials/mcp_list.html", {"tools": _mcp_rows(container)}, _flash(message)
    )


@router.delete("/ui/mcp/{tool_id}", response_class=HTMLResponse)
def ui_delete_mcp_tool(
    request: Request, tool_id: str, container: Container = Depends(get_container)
):
    tool = container.mcp_tools.get(tool_id)
    container.mcp_tools.delete(tool_id)
    return _partial(
        request,
        "partials/mcp_list.html",
        {"tools": _mcp_rows(container)},
        _flash(f"Deleted {tool.qualified_name}."),
    )


@router.post("/ui/domains/{domain_id}/intents/{intent_id}/mcp", response_class=HTMLResponse)
def ui_bind_mcp_tool(
    request: Request,
    domain_id: str,
    intent_id: str,
    mcp_tool_id: Annotated[str, Form()] = "",
    intents: IntentService = Depends(get_intent_service),
    container: Container = Depends(get_container),
):
    """Bind this intent to a registered MCP tool, or clear the binding."""
    reference = mcp_tool_id.strip()
    intent = intents.get(domain_id, intent_id)
    updated = intents.update(
        domain_id,
        intent_id,
        IntentUpdate(
            tool=ToolRef(
                name=intent.tool.name,
                version=intent.tool.version,
                mcp_tool_id=reference or None,
            )
        ),
    )
    if reference:
        tool = container.mcp_tools.resolve(reference)
        message = (
            f"Bound {updated.name} to {tool.qualified_name}."
            if tool
            else f"Saved, but no registered MCP tool matches '{reference}'."
        )
    else:
        message = f"Cleared the MCP binding on {updated.name}."
    return _partial(
        request,
        "partials/mcp_binding.html",
        {
            "domain": container.domains.get(domain_id),
            "intent": updated,
            "mcp_tools": container.mcp_tools.list(),
            "bound": container.mcp_tools.resolve(updated.tool.mcp_tool_id or ""),
        },
        _flash(message),
    )


# ------------------------------------------------------------- global listings
@router.get("/ui/intents", response_class=HTMLResponse)
def intents_page(request: Request, container: Container = Depends(get_container)):
    rows = []
    for domain in container.domains.list():
        for intent in container.intents.list(domain.id):
            rows.append(
                {
                    "domain": domain,
                    "intent": intent,
                    "mcp": container.mcp_tools.resolve(intent.tool.mcp_tool_id or ""),
                }
            )
    rows.sort(key=lambda r: (r["domain"].name.casefold(), r["intent"].name.casefold()))
    return templates.TemplateResponse(
        request=request, name="pages/intents.html", context={"rows": rows}
    )


@router.get("/ui/sessions", response_class=HTMLResponse)
def sessions_page(request: Request, container: Container = Depends(get_container)):
    return templates.TemplateResponse(
        request=request,
        name="pages/sessions.html",
        context={"sessions": _session_rows(container)},
    )


@router.get("/ui/sessions/list", response_class=HTMLResponse)
def sessions_list(request: Request, container: Container = Depends(get_container)):
    """Refresh target for the sessions page."""
    return templates.TemplateResponse(
        request=request,
        name="partials/session_list.html",
        context={"sessions": _session_rows(container)},
    )


@router.delete("/ui/sessions/{session_id}", response_class=HTMLResponse)
def ui_clear_session_row(
    request: Request, session_id: str, container: Container = Depends(get_container)
):
    removed = container.sessions.clear(session_id)
    return _partial(
        request,
        "partials/session_list.html",
        {"sessions": _session_rows(container)},
        _flash(f"Cleared '{session_id}' ({removed} turn(s))."),
    )


def _mcp_rows(container: Container) -> list[McpToolRead]:
    counts: dict[str, int] = {}
    for domain in container.domains.list():
        for intent in container.intents.list(domain.id):
            tool = container.mcp_tools.resolve(intent.tool.mcp_tool_id or "")
            if tool is not None:
                counts[tool.id] = counts.get(tool.id, 0) + 1
    return [McpToolRead.of(tool, counts.get(tool.id, 0)) for tool in container.mcp_tools.list()]


def _session_rows(container: Container) -> list[dict]:
    """Recent conversations, newest activity first."""
    store = container.store
    with store.lock:
        result = store.sessions.get(include=["metadatas"])
    latest: dict[str, dict] = {}
    for meta in result.get("metadatas") or []:
        session_id = str(meta.get("session_id", ""))
        if not session_id:
            continue
        row = latest.setdefault(
            session_id, {"session_id": session_id, "turns": 0, "created_at": 0.0}
        )
        row["turns"] += 1
        if float(meta.get("created_at", 0)) >= row["created_at"]:
            row["created_at"] = float(meta.get("created_at", 0))
            row["last_intent"] = str(meta.get("intent", ""))
    rows = []
    for session_id, row in latest.items():
        turns = container.sessions.all_turns(session_id)
        entities: dict = {}
        for turn in turns:
            entities.update(turn.entities)
        rows.append(
            {
                **row,
                "last_text": turns[-1].text if turns else "",
                "last_intent": turns[-1].intent if turns else row.get("last_intent", ""),
                "entities": entities,
            }
        )
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows[:50]


def _index_statuses(container: Container, domains: DomainService) -> list[dict]:
    rows = []
    for domain in domains.list():
        status = container.index_manager.status(domain.id)
        rows.append(
            {
                "domain": domain,
                "status": status,
                "in_sync": status.bm25_doc_count == status.dense_count,
            }
        )
    return rows


def _domain_of_session(container: Container, session_id: str) -> str | None:
    """Which domain a recorded conversation belongs to."""
    store = container.store
    with store.lock:
        result = store.sessions.get(where={"session_id": session_id}, include=["metadatas"])
    for meta in result.get("metadatas") or []:
        domain_id = str(meta.get("domain_id", ""))
        if domain_id:
            return domain_id
    return None
