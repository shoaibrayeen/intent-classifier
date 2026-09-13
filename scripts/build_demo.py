"""Record a demo against a running service and render it to docs/.

Produces two artefacts from one real run, so neither can drift from what the
service actually does:

* ``docs/demo.html`` -- the transcript, interactive: click any turn to see how
  it was decided.
* ``docs/demo.gif``  -- the same conversation as a short animation.

    uv run python -m scripts.build_demo --base-url http://localhost:8000

The service must be reachable. Use LLM_PROVIDER=mock so the run is
deterministic and makes no external API calls.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DOCS = Path(__file__).resolve().parents[1] / "docs"

MCP_CATALOGUE = {
    "server": "contracts",
    "transport": "stdio",
    "endpoint": "npx -y @acme/contracts-mcp",
    "tools": {
        "tools": [
            {
                "name": "search_contracts",
                "description": "Search contracts by counterparty and status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "counterparty": {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "required": ["counterparty"],
                },
            },
            {
                "name": "get_expiring_contracts",
                "description": "Contracts expiring in a window",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "counterparty": {"type": "string"},
                        "period": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "summarize_contract",
                "description": "Summarize one contract",
                "inputSchema": {
                    "type": "object",
                    "properties": {"contract_id": {"type": "string"}},
                    "required": ["contract_id"],
                },
            },
        ]
    },
}

BINDING = {
    "CONTRACT_SEARCH": "search_contracts",
    "CONTRACT_EXPIRY": "get_expiring_contracts",
    "CONTRACT_SUMMARY": "summarize_contract",
}

#: Chosen to show the things that are easy to claim and hard to do: a plain
#: match, a follow-up that needs the conversation, a change of subject that must
#: be refused, and a second domain.
CONVERSATION = [
    (
        "contract",
        "demo-contracts",
        "Find all active contracts with Microsoft",
        "A plain question. Entities are extracted and mapped onto the MCP tool's arguments.",
    ),
    (
        "contract",
        "demo-contracts",
        "which of them expire next month",
        "A follow-up. 'them' refers back, and this one still resolves on its own.",
    ),
    (
        "contract",
        "demo-contracts",
        "and for Oracle",
        "Three words with no anchor. Alone it is UNKNOWN; read against the previous "
        "question it resolves, and the new counterparty wins over the carried one.",
    ),
    (
        "contract",
        "demo-contracts",
        "summarize contract C-1042",
        "A different intent in the same conversation, with the contract id as its argument.",
    ),
    (
        "contract",
        "demo-contracts",
        "what is the weather today",
        "A change of subject. It must stay UNKNOWN rather than inheriting the last topic.",
    ),
    (
        "employee",
        "demo-people",
        "how many leave days do I have left",
        "A different domain entirely, with its own intents and its own vocabulary.",
    ),
]


def call(base: str, method: str, path: str, body: Any = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, method=method, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def record(base: str) -> dict[str, Any]:
    """Drive the service and capture everything the demo needs.

    Sessions are cleared first. Re-recording against a service that already
    holds these conversations would carry entities in from the previous run and
    quietly produce a transcript that never happened.
    """
    for session in {session for _, session, _, _ in CONVERSATION}:
        try:
            call(base, "DELETE", f"/api/v1/sessions/{session}")
        except urllib.error.HTTPError:
            pass  # nothing recorded under that id yet

    call(base, "POST", "/api/v1/mcp/tools/import", MCP_CATALOGUE)
    contract = next(d for d in call(base, "GET", "/api/v1/domains") if d["name"] == "contract")
    tools = {t["name"]: t["id"] for t in call(base, "GET", "/api/v1/mcp/tools")}
    for intent in call(base, "GET", f"/api/v1/domains/{contract['id']}/intents"):
        tool = BINDING.get(intent["name"])
        if tool:
            call(
                base,
                "PUT",
                f"/api/v1/domains/{contract['id']}/intents/{intent['id']}",
                {"tool": {"name": tool, "mcp_tool_id": tools[tool]}},
            )

    steps = []
    for domain, session, text, note in CONVERSATION:
        result = call(
            base,
            "POST",
            "/api/v1/classify/debug",
            {"domain": domain, "text": text, "session_id": session},
        )
        debug = result.get("debug") or {}
        steps.append(
            {
                "domain": domain,
                "session": session,
                "query": text,
                "note": note,
                "intent": result["intent"],
                "confidence": result["confidence"],
                "reason": result.get("reason"),
                "strategy": result.get("strategy"),
                "latency_ms": result.get("latency_ms"),
                "entities": result.get("entities", {}),
                "tool": result.get("tool"),
                "context": result.get("context"),
                "breakdown": debug.get("confidence_breakdown"),
                "top_intents": result.get("top_intents", [])[:4],
                "dense_hits": [
                    {"text": h["text"], "intent": h["intent_name"], "score": h["similarity"]}
                    for h in debug.get("dense_hits", [])[:4]
                ],
                "bm25_hits": [
                    {"text": h["text"], "intent": h["intent_name"], "score": h["score"]}
                    for h in debug.get("bm25_hits", [])[:4]
                ],
            }
        )
    return {
        "steps": steps,
        "health": call(base, "GET", "/api/v1/health"),
        "index": call(base, "GET", "/api/v1/health/index"),
        "mcp_tools": call(base, "GET", "/api/v1/mcp/tools"),
    }


# ------------------------------------------------------------------------ html
def band(value: float) -> str:
    return "high" if value >= 0.8 else ("medium" if value >= 0.5 else "low")


def render_html(run: dict[str, Any]) -> str:
    payload = json.dumps(run["steps"], ensure_ascii=False)
    health = run["health"]
    tool_count = len(run["mcp_tools"])
    examples = sum(d.get("bm25_doc_count", 0) for d in run["index"].get("domains", []))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo · Intent Classifier</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#faf9f7; --panel:#fff; --ink:#1a1917; --muted:#6b6a66; --line:#e3e1dc;
  --accent:#2f6f4f; --accent-soft:#e8f1eb; --warn:#8a6d1f; --danger:#9c2f2f;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16171a; --panel:#1e2023; --ink:#eceae6; --muted:#9b9a95;
    --line:#32343a; --accent:#6fbf94; --accent-soft:#223029; --warn:#d6b45c; --danger:#e0806f; }}
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:3rem 1.5rem 4rem}}
header{{border-bottom:1px solid var(--line);padding-bottom:1.5rem;margin-bottom:1.5rem}}
h1{{font-size:2rem;margin:0 0 .4rem;letter-spacing:-.025em}}
.lede{{font-size:1.05rem;color:var(--muted);margin:0;max-width:62ch}}
.chips{{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:1rem}}
.badge{{display:inline-block;padding:.15rem .6rem;border-radius:999px;
  background:var(--accent-soft);font-size:12.5px}}
.badge.mono{{font-family:var(--mono)}}
.badge.warn{{background:none;border:1px solid var(--warn);color:var(--warn)}}
.badge.danger{{background:none;border:1px solid var(--danger);color:var(--danger)}}
.badge.ok{{background:var(--accent-soft)}}
.grid{{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:1rem;align-items:start}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1.1rem}}
.thread{{display:flex;flex-direction:column;gap:.9rem}}
.turn{{display:flex;flex-direction:column;gap:.35rem}}
.bubble{{max-width:88%;padding:.55rem .8rem;border-radius:14px;font-size:14px;line-height:1.45}}
.bubble.user{{align-self:flex-end;background:var(--accent);color:#fff;border-bottom-right-radius:4px}}
.bubble.assistant{{align-self:flex-start;background:var(--bg);border:1px solid var(--line);
  border-bottom-left-radius:4px;cursor:pointer;display:flex;flex-direction:column;gap:.2rem;
  text-align:left;font:inherit;width:auto}}
.bubble.assistant:hover,.bubble.assistant[aria-current="true"]{{border-color:var(--accent)}}
.bubble.assistant.unknown{{border-style:dashed}}
.intent{{font-weight:650}}
.meta{{display:flex;align-items:center;gap:.35rem;font-size:12px;color:var(--muted)}}
.dot{{width:8px;height:8px;border-radius:50%;display:inline-block}}
.dot.high{{background:var(--accent)}} .dot.medium{{background:var(--warn)}} .dot.low{{background:var(--danger)}}
.bar{{height:6px;background:var(--line);border-radius:999px;overflow:hidden;margin:.35rem 0 .1rem}}
.bar span{{display:block;height:100%}}
.bar.high span{{background:var(--accent)}} .bar.medium span{{background:var(--warn)}} .bar.low span{{background:var(--danger)}}
.conf{{font-size:1.3rem;font-variant-numeric:tabular-nums}}
.conf.high{{color:var(--accent)}} .conf.medium{{color:var(--warn)}} .conf.low{{color:var(--danger)}}
h2{{font-size:1.1rem;margin:0 0 .5rem}}
h3{{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  margin:1.1rem 0 .4rem;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:.4rem .5rem;border-bottom:1px solid var(--line)}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
pre{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:.7rem;
  overflow-x:auto;font-family:var(--mono);font-size:12px;margin:.3rem 0}}
.mono{{font-family:var(--mono);font-size:13px}}
.muted{{color:var(--muted)}} .small{{font-size:12.5px}}
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:.6rem .9rem;border-radius:0 8px 8px 0;font-size:13.5px;margin:.6rem 0 0}}
footer{{margin-top:2.5rem;color:var(--muted);font-size:.85rem;
  border-top:1px solid var(--line);padding-top:1.25rem}}
@media (max-width:880px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Intent Classifier, demonstrated</h1>
  <p class="lede">
    A real conversation against a running service. Every number below was produced by the run,
    not written by hand. Click any reply to see how that turn was decided.
  </p>
  <div class="chips">
    <span class="badge">{len(run["steps"])} turns</span>
    <span class="badge">{examples} indexed examples</span>
    <span class="badge">{tool_count} MCP tools</span>
    <span class="badge mono">{html.escape(str(health.get("model", "")))}</span>
    <span class="badge">provider: {html.escape(str(health.get("entity_extraction", {}).get("provider", "")))}</span>
  </div>
  <p class="muted small" style="margin-top:.75rem">
    Recorded with the offline provider, so the run is deterministic and makes no external calls.
    <a href="architecture.html">Architecture</a> &middot;
    <a href="api-documentation.html">API reference</a> &middot;
    <a href="changelog.html">Changelog</a>
  </p>
</header>

<div class="grid">
  <section class="card" id="detail"></section>
  <section class="card">
    <h2>Conversation</h2>
    <div class="thread" id="thread"></div>
  </section>
</div>

<footer>
  Tool calls are resolved and reported, never executed. The MCP block shows exactly what the
  caller would invoke.
</footer>
</div>

<script>
const STEPS = {payload};
const pct = v => Math.round(v * 100) + "%";
const band = v => v >= 0.8 ? "high" : (v >= 0.5 ? "medium" : "low");
const esc = s => String(s).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}})[c]);

function rows(list, key) {{
  if (!list || !list.length) return '<p class="muted small">None.</p>';
  return '<table><tbody>' + list.map(h =>
    `<tr><td>${{esc(h.text)}}</td><td class="mono small">${{esc(h.intent)}}</td>` +
    `<td class="num">${{h.score.toFixed(3)}}</td></tr>`).join('') + '</tbody></table>';
}}

function detail(i) {{
  const s = STEPS[i], b = band(s.confidence);
  const mcp = (s.tool || {{}}).mcp;
  let html = `<h2>${{esc(s.intent)}}</h2>
    <div class="conf ${{b}}">${{pct(s.confidence)}}</div>
    <div class="bar ${{b}}"><span style="width:${{pct(s.confidence)}}"></span></div>
    <p class="muted small">${{esc(s.query)}}</p>
    <div class="chips">
      <span class="badge">${{esc(s.domain)}}</span>
      <span class="badge">${{esc(s.strategy || '')}}</span>
      <span class="badge">${{(s.latency_ms||0).toFixed(1)}} ms</span>
      ${{s.reason ? `<span class="badge warn">${{esc(s.reason)}}</span>` : ''}}
    </div>
    <p class="note">${{esc(s.note)}}</p>`;

  html += '<h3>Entities</h3>';
  const keys = Object.keys(s.entities || {{}});
  html += keys.length
    ? '<table><tbody>' + keys.map(k =>
        `<tr><td class="mono small">${{esc(k)}}</td><td>${{esc(s.entities[k])}}</td></tr>`).join('') + '</tbody></table>'
    : '<p class="muted small">None extracted.</p>';

  html += '<h3>MCP call</h3>';
  if (mcp && !mcp.unresolved) {{
    html += `<div class="chips"><span class="badge mono">${{esc(mcp.qualified_name)}}</span>` +
      (mcp.ready ? '<span class="badge ok">ready</span>' : '<span class="badge warn">not ready</span>') +
      `<span class="badge">${{esc(mcp.transport)}}</span></div>` +
      `<pre>${{esc(JSON.stringify({{server: mcp.server, tool: mcp.tool, arguments: mcp.arguments}}, null, 2))}}</pre>` +
      '<p class="muted small">Reported, never called.</p>';
  }} else {{
    html += '<p class="muted small">This intent is not bound to an MCP tool.</p>';
  }}

  if (s.context && (s.context.used_for_retrieval || (s.context.carried_entities||[]).length)) {{
    html += '<h3>Conversation</h3>';
    if (s.context.used_for_retrieval)
      html += `<p class="small">Read against the previous turn: <span class="mono">${{esc(s.context.retrieval_text)}}</span></p>`;
    if ((s.context.carried_entities||[]).length)
      html += `<p class="small">Carried forward: <span class="mono">${{esc(s.context.carried_entities.join(', '))}}</span></p>`;
  }}

  if (s.breakdown) {{
    const bd = s.breakdown;
    const signals = [["s_dense",bd.s_dense],["s_rrf",bd.s_rrf],["s_margin",bd.s_margin],
                     ["s_support",bd.s_support],["best similarity",bd.best_similarity]];
    html += '<h3>Confidence breakdown</h3><table><tbody>' + signals.map(([n,v]) =>
      `<tr><td><span class="dot ${{band(v)}}"></span> ${{n}}</td><td class="num">${{v.toFixed(3)}}</td></tr>`
    ).join('') + '</tbody></table>';
  }}

  if ((s.top_intents||[]).length) {{
    html += '<h3>Ranked intents after fusion</h3><table><tbody>' + s.top_intents.map((t,n) =>
      `<tr><td>${{n+1}}</td><td><span class="dot ${{band(t.best_similarity)}}"></span> ${{esc(t.intent)}}</td>` +
      `<td class="num">${{t.score.toFixed(5)}}</td><td class="num">${{t.best_similarity.toFixed(3)}}</td></tr>`
    ).join('') + '</tbody></table>';
  }}

  html += '<h3>Dense retrieval</h3>' + rows(s.dense_hits);
  html += '<h3>BM25 retrieval</h3>' + rows(s.bm25_hits);
  document.getElementById('detail').innerHTML = html;
  document.querySelectorAll('.bubble.assistant').forEach((el, n) =>
    el.setAttribute('aria-current', n === i ? 'true' : 'false'));
}}

document.getElementById('thread').innerHTML = STEPS.map((s, i) => {{
  const b = band(s.confidence), unknown = s.intent === 'UNKNOWN';
  const entities = Object.keys(s.entities||{{}}).length
    ? `<span class="muted small mono">${{esc(JSON.stringify(s.entities))}}</span>` : '';
  return `<div class="turn">
    <div class="bubble user">${{esc(s.query)}}</div>
    <button class="bubble assistant ${{unknown ? 'unknown' : ''}}" onclick="detail(${{i}})">
      <span class="intent">${{unknown ? 'No matching intent' : esc(s.intent)}}</span>
      ${{entities}}
      <span class="meta"><span class="dot ${{b}}"></span>${{pct(s.confidence)}}
      ${{s.tool && s.tool.name ? '· <span class="mono">'+esc(s.tool.name)+'</span>' : ''}}</span>
    </button>
  </div>`;
}}).join('');
detail(0);
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------- gif
def render_gif(run: dict[str, Any], path: Path) -> None:
    """Draw the conversation building up, frame by frame.

    Rendered from the captured run rather than screen-recorded, so it can be
    regenerated anywhere and always matches the data in demo.html.
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 900, 560
    BG, PANEL, INK, MUTED, LINE = "#faf9f7", "#ffffff", "#1a1917", "#6b6a66", "#e3e1dc"
    ACCENT, WARN, DANGER = "#2f6f4f", "#8a6d1f", "#9c2f2f"

    def font(size: int, bold: bool = False, mono: bool = False):
        candidates = (
            ["/System/Library/Fonts/Menlo.ttc"]
            if mono
            else [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                if bold
                else "/System/Library/Fonts/Supplemental/Arial.ttf"
            ]
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f_title, f_body, f_small = font(20, True), font(14), font(12)
    f_bold, f_mono = font(15, True), font(12, mono=True)

    def colour_for(value: float) -> str:
        return ACCENT if value >= 0.8 else (WARN if value >= 0.5 else DANGER)

    def wrap(draw, text, fnt, width):
        words, lines, line = text.split(), [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=fnt) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    frames = []
    for shown in range(1, len(run["steps"]) + 1):
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        draw.text((28, 22), "Intent Classifier", font=f_title, fill=INK)
        draw.text(
            (28, 50),
            "hybrid retrieval · entities · MCP call resolved, never executed",
            font=f_small,
            fill=MUTED,
        )

        step = run["steps"][shown - 1]
        # left: the decision
        draw.rounded_rectangle([28, 80, 400, H - 28], 10, fill=PANEL, outline=LINE)
        y = 100
        intent = "No matching intent" if step["intent"] == "UNKNOWN" else step["intent"]
        for line in wrap(draw, intent, f_bold, 330):
            draw.text((46, y), line, font=f_bold, fill=INK)
            y += 20
        confidence = step["confidence"]
        colour = colour_for(confidence)
        draw.text((46, y + 4), f"{round(confidence * 100)}%", font=f_title, fill=colour)
        draw.rounded_rectangle([46, y + 34, 382, y + 40], 3, fill=LINE)
        draw.rounded_rectangle([46, y + 34, 46 + int(336 * confidence), y + 40], 3, fill=colour)
        y += 56
        draw.text((46, y), f"domain: {step['domain']}", font=f_small, fill=MUTED)
        y += 18
        draw.text(
            (46, y), f"{step['strategy']} · {step['latency_ms']:.0f} ms", font=f_small, fill=MUTED
        )
        y += 24

        if step["entities"]:
            draw.text((46, y), "ENTITIES", font=f_small, fill=MUTED)
            y += 18
            for key, value in step["entities"].items():
                draw.text((46, y), f"{key}: {value}", font=f_mono, fill=INK)
                y += 17
            y += 6

        mcp = (step.get("tool") or {}).get("mcp")
        if mcp and not mcp.get("unresolved"):
            draw.text((46, y), "MCP CALL", font=f_small, fill=MUTED)
            y += 18
            for line in wrap(draw, mcp["qualified_name"], f_mono, 330):
                draw.text((46, y), line, font=f_mono, fill=ACCENT)
                y += 16
            for key, value in (mcp.get("arguments") or {}).items():
                draw.text((46, y), f"  {key} = {value}", font=f_mono, fill=INK)
                y += 16
        elif step["intent"] == "UNKNOWN":
            for line in wrap(draw, "Nothing in this domain was close enough.", f_small, 330):
                draw.text((46, y), line, font=f_small, fill=MUTED)
                y += 16

        # right: the conversation so far
        draw.rounded_rectangle([416, 80, W - 28, H - 28], 10, fill=PANEL, outline=LINE)

        def turn_height(entry, draw=draw) -> int:
            return 26 * len(wrap(draw, entry["query"], f_body, 300)) + 4 + 52

        # Scroll like a chat: drop the oldest turns until the rest fit, rather
        # than drawing past the bottom of the panel.
        available = (H - 28) - 98 - 10
        visible = list(range(shown))
        while len(visible) > 1 and sum(turn_height(run["steps"][i]) for i in visible) > available:
            visible.pop(0)

        y = 98
        if visible and visible[0] > 0:
            draw.text((434, y), f"… {visible[0]} earlier turn(s)", font=f_small, fill=MUTED)
            y += 22
        for index in visible:
            current = run["steps"][index]
            for line in wrap(draw, current["query"], f_body, 300):
                width = draw.textlength(line, font=f_body)
                draw.rounded_rectangle([W - 44 - width - 16, y - 4, W - 44, y + 20], 9, fill=ACCENT)
                draw.text((W - 44 - width - 8, y), line, font=f_body, fill="#ffffff")
                y += 26
            y += 4
            label = "No matching intent" if current["intent"] == "UNKNOWN" else current["intent"]
            width = max(draw.textlength(label, font=f_bold), 120)
            draw.rounded_rectangle([434, y - 4, 434 + width + 74, y + 38], 9, fill=BG, outline=LINE)
            draw.text((442, y), label, font=f_bold, fill=INK)
            value = current["confidence"]
            draw.ellipse([442, y + 24, 450, y + 32], fill=colour_for(value))
            draw.text((456, y + 21), f"{round(value * 100)}%", font=f_small, fill=MUTED)
            y += 52

        frames.append(image)

    # Hold each frame long enough to read, and the last one longer.
    durations = [1600] * (len(frames) - 1) + [3200]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record and render the demo")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args(argv)

    try:
        run = record(args.base_url.rstrip("/"))
    except urllib.error.URLError as exc:
        print(f"cannot reach {args.base_url}: {exc}", file=sys.stderr)
        return 1

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "demo.html").write_text(render_html(run), "utf-8")
    print(f"wrote {DOCS / 'demo.html'}")
    (DOCS / "demo-run.json").write_text(json.dumps(run, indent=1), "utf-8")
    print(f"wrote {DOCS / 'demo-run.json'}")

    if not args.skip_gif:
        try:
            render_gif(run, DOCS / "demo.gif")
            print(f"wrote {DOCS / 'demo.gif'}")
        except ImportError:
            print("Pillow not installed; skipping the GIF", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
