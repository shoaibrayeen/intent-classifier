"""Render the hiring demo as an animation of the application's pages.

Drawn from the captured run rather than screen-recorded, so it regenerates
anywhere and always matches docs/demo-hiring.html. Each frame is one page of
the service, in the order someone would actually visit them: domains, the
domain, MCP tools, intents, one intent, the chat as it builds, sessions, and
operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

W, H = 1000, 640
BG, PANEL, INK, MUTED, LINE = "#f7f6f3", "#ffffff", "#1a1917", "#6b6a66", "#e3e1dc"
ACCENT, ACCENT_SOFT, WARN, DANGER = "#2f6f4f", "#e8f1eb", "#8a6d1f", "#9c2f2f"

NAV = ["Domains", "Intents", "MCP tools", "Playground", "Sessions", "Evaluation", "Operations"]


def _font(size: int, bold: bool = False, mono: bool = False):
    from PIL import ImageFont

    paths = (
        ["/System/Library/Fonts/Menlo.ttc"]
        if mono
        else [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ]
    )
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def colour_for(value: float) -> str:
    return ACCENT if value >= 0.8 else (WARN if value >= 0.5 else DANGER)


class Page:
    """One frame: the application shell, with a content area to draw into."""

    def __init__(self, active: str, title: str, subtitle: str = ""):
        from PIL import Image, ImageDraw

        self.image = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.image)
        self.f_nav = _font(13)
        self.f_title = _font(23, True)
        self.f_sub = _font(13)
        self.f_h = _font(14, True)
        self.f_body = _font(13)
        self.f_small = _font(11)
        self.f_mono = _font(12, mono=True)
        self.f_mono_b = _font(13, mono=True)

        self.d.rectangle([0, 0, W, 46], fill=PANEL)
        self.d.line([0, 46, W, 46], fill=LINE)
        self.d.text((24, 15), "Intent Classifier", font=_font(14, True), fill=INK)
        x = 190
        for item in NAV:
            width = self.d.textlength(item, font=self.f_nav)
            if item == active:
                self.d.rounded_rectangle([x - 9, 12, x + width + 9, 34], 11, fill=ACCENT_SOFT)
                self.d.text((x, 16), item, font=self.f_nav, fill=ACCENT)
            else:
                self.d.text((x, 16), item, font=self.f_nav, fill=MUTED)
            x += width + 26

        self.d.text((24, 62), title, font=self.f_title, fill=INK)
        if subtitle:
            self.d.text((24, 94), subtitle, font=self.f_sub, fill=MUTED)
        self.y = 122 if subtitle else 104

    def card(self, height: int, x0: int = 24, x1: int = W - 24) -> tuple[int, int]:
        top = self.y
        self.d.rounded_rectangle([x0, top, x1, top + height], 9, fill=PANEL, outline=LINE)
        self.y = top + height + 14
        return x0 + 18, top + 14

    def wrap(self, text: str, font, width: int) -> list[str]:
        words, lines, line = text.split(), [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if self.d.textlength(trial, font=font) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    def header_row(self, x: int, y: int, columns: list[tuple[str, int]]) -> None:
        for label, offset in columns:
            self.d.text((x + offset, y), label.upper(), font=self.f_small, fill=MUTED)
        self.d.line([x - 8, y + 16, W - 42, y + 16], fill=LINE)

    def badge(self, x: int, y: int, text: str, colour: str = ACCENT) -> int:
        width = self.d.textlength(text, font=self.f_small)
        self.d.rounded_rectangle([x, y, x + width + 16, y + 18], 9, fill=ACCENT_SOFT)
        self.d.text((x + 8, y + 3), text, font=self.f_small, fill=colour)
        return x + int(width) + 24


def render_gif(run: dict[str, Any], path: Path) -> None:
    from PIL import Image  # noqa: F401  (imported for the type, used via Page)

    steps = {s["stage"]: s for s in run["steps"]}
    domain = run["domain"]
    intents = steps["intents"]["payload"]["intents"]
    tools = steps["mcp"]["payload"]["tools"]
    session = steps["session"]["payload"]
    index = steps["index"]["payload"]
    turns = run["turns"]

    frames: list = []

    # 1. Domains ------------------------------------------------------------
    p = Page(
        "Domains",
        "Domains",
        "An isolated intent namespace. The same words can mean different things in each.",
    )
    x, y = p.card(150)
    p.header_row(x, y, [("Domain", 0), ("Description", 190), ("Intents", 690), ("Examples", 790)])
    y += 26
    rows = [
        (
            domain["name"],
            domain["description"][:52] + "…",
            run["intent_count"],
            run["example_count"],
            True,
        ),
        ("contract", "contract and agreement operations", 4, 28, False),
        ("employee", "people and workforce operations", 3, 18, False),
    ]
    for name, desc, ni, ne, current in rows:
        if current:
            p.d.rounded_rectangle([x - 10, y - 5, W - 42, y + 21], 6, fill=ACCENT_SOFT)
        p.d.text((x, y), name, font=p.f_mono_b, fill=ACCENT if current else INK)
        p.d.text((x + 190, y + 1), desc, font=p.f_body, fill=MUTED)
        p.d.text((x + 700, y + 1), str(ni), font=p.f_body, fill=INK)
        p.d.text((x + 800, y + 1), str(ne), font=p.f_body, fill=INK)
        y += 30
    frames.append(p.image)

    # 2. The domain, with drafted instructions -------------------------------
    p = Page(
        "Domains",
        domain["name"],
        "Created from a name and one sentence. The instructions below were drafted from them.",
    )
    x, y = p.card(112)
    p.d.text((x, y), "SYSTEM INSTRUCTIONS", font=p.f_small, fill=MUTED)
    y += 20
    for line in p.wrap(domain["system_instructions"], p.f_body, W - 90)[:4]:
        p.d.text((x, y), line, font=p.f_body, fill=INK)
        y += 19
    x, y = p.card(96)
    p.d.text((x, y), "USER INSTRUCTIONS", font=p.f_small, fill=MUTED)
    y += 20
    for line in p.wrap(domain["user_instructions"], p.f_body, W - 90)[:3]:
        p.d.text((x, y), line, font=p.f_body, fill=INK)
        y += 19
    x, y = p.card(64)
    nx = p.badge(x, y, f"{run['intent_count']} intents")
    nx = p.badge(nx, y, f"{run['example_count']} examples")
    nx = p.badge(nx, y, "active")
    p.d.text(
        (x, y + 30),
        "Ordinary configuration: editable here, layered onto every extraction prompt in this domain.",
        font=p.f_small,
        fill=MUTED,
    )
    frames.append(p.image)

    # 3. MCP tools -----------------------------------------------------------
    p = Page(
        "MCP tools",
        "MCP tools",
        f"Imported from the ATS server's tools/list at {steps['mcp']['payload']['endpoint']}",
    )
    x, y = p.card(430)
    p.header_row(x, y, [("Qualified name", 0), ("What it does", 330), ("Arguments", 690)])
    y += 26
    for tool in tools[:12]:
        p.d.text((x, y), tool["qualified_name"][:40], font=p.f_mono, fill=ACCENT)
        p.d.text((x + 330, y), tool["description"][:42], font=p.f_body, fill=MUTED)
        p.d.text((x + 690, y), ", ".join(tool["arguments"])[:26] or "—", font=p.f_mono, fill=INK)
        y += 26
    p.d.text((x, y + 6), f"… and {max(0, len(tools) - 12)} more", font=p.f_small, fill=MUTED)
    frames.append(p.image)

    # 4. Intents -------------------------------------------------------------
    p = Page(
        "Intents",
        "Intents",
        f"{run['intent_count']} operations a hiring team asks for, each wired to the tool that serves it.",
    )
    x, y = p.card(430)
    p.header_row(x, y, [("Intent", 0), ("MCP tool", 250), ("Accepts", 590), ("Ex.", 870)])
    y += 26
    for intent in intents[:12]:
        p.d.text((x, y), intent["name"], font=p.f_mono_b, fill=INK)
        p.d.text((x + 250, y), (intent["mcp"] or "—")[:36], font=p.f_mono, fill=ACCENT)
        p.d.text(
            (x + 590, y), ", ".join(intent["entity_schema"])[:30] or "—", font=p.f_mono, fill=MUTED
        )
        p.d.text((x + 875, y), str(len(intent["examples"])), font=p.f_body, fill=INK)
        y += 26
    p.d.text((x, y + 6), f"… and {max(0, len(intents) - 12)} more", font=p.f_small, fill=MUTED)
    frames.append(p.image)

    # 5. One intent ----------------------------------------------------------
    first = intents[0]
    p = Page("Intents", first["name"], first["description"])
    x, y = p.card(120)
    p.d.text((x, y), "ACCEPTS", font=p.f_small, fill=MUTED)
    y += 20
    for key, spec in list(first["entity_schema"].items())[:4]:
        kind = spec.get("type", "string") if isinstance(spec, dict) else str(spec)
        required = " · required" if isinstance(spec, dict) and spec.get("required") else ""
        p.d.text((x, y), f"{key}", font=p.f_mono, fill=INK)
        p.d.text((x + 220, y), f"{kind}{required}", font=p.f_mono, fill=MUTED)
        y += 20
    x, y = p.card(240)
    p.d.text((x, y), f"TRAINING EXAMPLES · {len(first['examples'])}", font=p.f_small, fill=MUTED)
    y += 20
    for example in first["examples"][:8]:
        p.d.text((x, y), f"· {example}", font=p.f_body, fill=INK)
        y += 22
    x, y = p.card(52)
    p.badge(x, y, first["mcp"] or "no tool")
    p.d.text(
        (x, y + 26),
        "Bound to the ATS tool. Calls are resolved and reported, never made.",
        font=p.f_small,
        fill=MUTED,
    )
    frames.append(p.image)

    # 6..n. The chat, building up -------------------------------------------
    def chat_frame(count: int):
        turn = turns[count - 1]
        page = Page(
            "Playground",
            "Playground",
            "Ask on the right; everything behind the decision is on the left.",
        )
        # left: the decision
        page.d.rounded_rectangle([24, page.y, 400, H - 24], 9, fill=PANEL, outline=LINE)
        ly = page.y + 16
        label = "No matching intent" if turn["intent"] == "UNKNOWN" else turn["intent"]
        for line in page.wrap(label, _font(16, True), 340):
            page.d.text((42, ly), line, font=_font(16, True), fill=INK)
            ly += 21
        colour = colour_for(turn["confidence"])
        page.d.text(
            (42, ly + 4), f"{round(turn['confidence'] * 100)}%", font=_font(22, True), fill=colour
        )
        page.d.rounded_rectangle([42, ly + 36, 382, ly + 42], 3, fill=LINE)
        page.d.rounded_rectangle(
            [42, ly + 36, 42 + int(340 * turn["confidence"]), ly + 42], 3, fill=colour
        )
        ly += 58
        page.d.text(
            (42, ly),
            f"{turn['strategy']} · {turn['latency_ms']:.0f} ms",
            font=page.f_small,
            fill=MUTED,
        )
        ly += 24
        if turn["entities"]:
            page.d.text((42, ly), "ENTITIES", font=page.f_small, fill=MUTED)
            ly += 18
            for key, value in list(turn["entities"].items())[:4]:
                page.d.text((42, ly), f"{key}: {value}", font=page.f_mono, fill=INK)
                ly += 17
            ly += 6
        mcp = turn.get("mcp") or {}
        if mcp.get("qualified_name") and not mcp.get("unresolved"):
            page.d.text((42, ly), "MCP CALL", font=page.f_small, fill=MUTED)
            ly += 18
            for line in page.wrap(mcp["qualified_name"], page.f_mono, 330):
                page.d.text((42, ly), line, font=page.f_mono, fill=ACCENT)
                ly += 16
            for key, value in list((mcp.get("arguments") or {}).items())[:3]:
                page.d.text((42, ly), f"  {key} = {value}", font=page.f_mono, fill=INK)
                ly += 16
        if turn.get("breakdown"):
            ly += 6
            page.d.text((42, ly), "CONFIDENCE", font=page.f_small, fill=MUTED)
            ly += 18
            bd = turn["breakdown"]
            for name in ("s_dense", "s_rrf", "s_margin", "s_support"):
                value = bd.get(name, 0.0)
                if ly > H - 46:
                    break
                page.d.ellipse([42, ly + 4, 50, ly + 12], fill=colour_for(value))
                page.d.text((58, ly), f"{name}", font=page.f_mono, fill=INK)
                page.d.text((210, ly), f"{value:.3f}", font=page.f_mono, fill=MUTED)
                ly += 17

        # right: the thread, scrolled so the newest is visible
        page.d.rounded_rectangle([416, page.y, W - 24, H - 24], 9, fill=PANEL, outline=LINE)

        def height_of(t) -> int:
            return 24 * len(page.wrap(t["query"], page.f_body, 300)) + 4 + 46

        available = (H - 24) - (page.y + 16) - 10
        visible = list(range(count))
        while len(visible) > 1 and sum(height_of(turns[i]) for i in visible) > available:
            visible.pop(0)
        ry = page.y + 16
        if visible and visible[0] > 0:
            page.d.text((434, ry), f"… {visible[0]} earlier turn(s)", font=page.f_small, fill=MUTED)
            ry += 20
        for index_ in visible:
            current = turns[index_]
            for line in page.wrap(current["query"], page.f_body, 300):
                width = page.d.textlength(line, font=page.f_body)
                page.d.rounded_rectangle(
                    [W - 40 - width - 16, ry - 4, W - 40, ry + 18], 9, fill=ACCENT
                )
                page.d.text((W - 40 - width - 8, ry), line, font=page.f_body, fill="#ffffff")
                ry += 24
            ry += 4
            name = "No matching intent" if current["intent"] == "UNKNOWN" else current["intent"]
            width = max(page.d.textlength(name, font=page.f_h), 130)
            page.d.rounded_rectangle(
                [434, ry - 4, 434 + width + 70, ry + 34], 9, fill=BG, outline=LINE
            )
            page.d.text((442, ry), name, font=page.f_h, fill=INK)
            page.d.ellipse([442, ry + 22, 450, ry + 30], fill=colour_for(current["confidence"]))
            page.d.text(
                (456, ry + 19),
                f"{round(current['confidence'] * 100)}%",
                font=page.f_small,
                fill=MUTED,
            )
            ry += 46
        return page.image

    for count in range(1, len(turns) + 1):
        frames.append(chat_frame(count))

    # n+1. Sessions -----------------------------------------------------------
    p = Page(
        "Sessions",
        "Conversations",
        "Read back from Chroma. Opening one shows what happened, not a re-classification.",
    )
    x, y = p.card(360)
    p.header_row(x, y, [("#", 0), ("Request", 40), ("Intent", 560), ("Conf", 810)])
    y += 26
    for turn in session["turns"]:
        p.d.text((x, y), str(turn["turn"]), font=p.f_body, fill=MUTED)
        p.d.text((x + 40, y), turn["text"][:62], font=p.f_body, fill=INK)
        p.d.text((x + 560, y), turn["intent"][:26], font=p.f_mono, fill=INK)
        p.d.ellipse([x + 810, y + 4, x + 818, y + 12], fill=colour_for(turn["confidence"]))
        p.d.text((x + 826, y), f"{round(turn['confidence'] * 100)}%", font=p.f_body, fill=MUTED)
        y += 26
    x, y = p.card(70)
    p.d.text((x, y), "ENTITIES STILL IN PLAY", font=p.f_small, fill=MUTED)
    entities = session.get("entities_in_play", {})
    p.d.text(
        (x, y + 20),
        ", ".join(f"{k}={v}" for k, v in list(entities.items())[:5]) or "none",
        font=p.f_mono,
        fill=INK,
    )
    frames.append(p.image)

    # n+2. Operations ---------------------------------------------------------
    p = Page("Operations", "Operations", "Index health and the configuration actually in force.")
    x, y = p.card(120)
    p.header_row(
        x,
        y,
        [
            ("Domain", 0),
            ("State", 250),
            ("Version", 400),
            ("BM25", 530),
            ("Vectors", 650),
            ("In sync", 780),
        ],
    )
    y += 26
    p.d.text((x, y), domain["name"], font=p.f_mono_b, fill=INK)
    p.badge(x + 250, y - 2, index.get("state", "?"))
    p.d.text((x + 405, y), f"v{index.get('version', '?')}", font=p.f_body, fill=INK)
    p.d.text((x + 535, y), str(index.get("bm25_doc_count", "?")), font=p.f_body, fill=INK)
    p.d.text((x + 655, y), str(index.get("dense_count", "?")), font=p.f_body, fill=INK)
    in_sync = index.get("bm25_doc_count") == index.get("dense_count")
    p.badge(x + 780, y - 2, "yes" if in_sync else "drifted", ACCENT if in_sync else WARN)
    y += 34
    p.d.text(
        (x, y),
        "Both indexes are written together. A drift means a mutation failed partway.",
        font=p.f_small,
        fill=MUTED,
    )
    health = run["health"]
    x, y = p.card(150)
    p.d.text((x, y), "IN FORCE", font=p.f_small, fill=MUTED)
    y += 22
    for label, value in [
        ("embedding model", health.get("model", "")),
        ("extraction provider", health.get("entity_extraction", {}).get("provider", "")),
        ("retrieval strategy", health.get("strategy", "")),
        ("sessions", "on" if health.get("sessions_enabled") else "off"),
        ("authentication", "on" if health.get("auth_enabled") else "off"),
    ]:
        p.d.text((x, y), label, font=p.f_body, fill=MUTED)
        p.d.text((x + 260, y), str(value), font=p.f_mono, fill=INK)
        y += 22
    frames.append(p.image)

    durations = [2200] * 5 + [1500] * len(turns) + [2600, 3400]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations[: len(frames)],
        loop=0,
        optimize=True,
    )
