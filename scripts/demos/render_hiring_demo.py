"""Render the hiring demo to an interactive page and an animation.

Reads docs/demo-hiring-run.json, written by build_hiring_demo.py, so both
artefacts describe one real run against a live service.

    uv run --with pillow python -m scripts.demos.render_hiring_demo
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

DOCS = Path(__file__).resolve().parents[2] / "docs"
RUN = DOCS / "demo-hiring-run.json"


def band(value: float) -> str:
    return "high" if value >= 0.8 else ("medium" if value >= 0.5 else "low")


# ------------------------------------------------------------------------ html
def render_html(run: dict[str, Any]) -> str:
    steps = {s["stage"]: s for s in run["steps"]}
    domain = run["domain"]
    intents = steps["intents"]["payload"]["intents"]
    tools = steps["mcp"]["payload"]["tools"]
    session = steps["session"]["payload"]
    index = steps["index"]["payload"]
    turns = run["turns"]

    def esc(value: Any) -> str:
        return html.escape(str(value))

    intent_rows = "".join(
        f"""<tr>
          <td><span class="mono">{esc(i["name"])}</span><div class="muted small">{esc(i["description"])}</div></td>
          <td class="mono small">{esc(i["mcp"] or "—")}</td>
          <td class="mono small">{esc(", ".join(i["entity_schema"]) or "—")}</td>
          <td class="num">{len(i["examples"])}</td>
        </tr>"""
        for i in intents
    )
    tool_rows = "".join(
        f"""<tr>
          <td class="mono small">{esc(t["qualified_name"])}</td>
          <td class="muted small">{esc(t["description"])}</td>
          <td class="mono small">{esc(", ".join(t["arguments"]) or "—")}</td>
          <td class="mono small">{esc(", ".join(t["required"]) or "—")}</td>
        </tr>"""
        for t in tools
    )
    example_blocks = "".join(
        f"""<details><summary><span class="mono">{esc(i["name"])}</span>
        <span class="muted small">{len(i["examples"])} examples</span></summary>
        <ul class="examples">{"".join(f"<li>{esc(e)}</li>" for e in i["examples"])}</ul></details>"""
        for i in intents
    )
    session_rows = "".join(
        f"""<tr><td class="num">{t["turn"]}</td><td>{esc(t["text"])}</td>
        <td class="mono small">{esc(t["intent"])}</td><td class="num">{t["confidence"]:.2f}</td>
        <td class="mono small">{esc(json.dumps(t["entities"]) if t["entities"] else "—")}</td></tr>"""
        for t in session["turns"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hiring demo · Intent Classifier</title>
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
.wrap{{max-width:1120px;margin:0 auto;padding:3rem 1.5rem 5rem}}
header{{border-bottom:1px solid var(--line);padding-bottom:1.5rem;margin-bottom:1.5rem}}
h1{{font-size:2.1rem;margin:0 0 .4rem;letter-spacing:-.025em}}
.lede{{font-size:1.05rem;color:var(--muted);margin:0;max-width:66ch}}
h2{{font-size:1.2rem;margin:2.5rem 0 .35rem;letter-spacing:-.015em}}
.stage{{display:flex;align-items:baseline;gap:.6rem;margin-top:2.5rem}}
.stage .n{{width:1.7rem;height:1.7rem;border-radius:50%;background:var(--accent);color:#fff;
  display:inline-flex;align-items:center;justify-content:center;font-size:13px;font-weight:650;flex:none}}
.stage h2{{margin:0}}
.chips{{display:flex;flex-wrap:wrap;gap:.4rem;margin:.6rem 0}}
.badge{{display:inline-block;padding:.15rem .6rem;border-radius:999px;
  background:var(--accent-soft);font-size:12.5px}}
.badge.mono{{font-family:var(--mono)}}
.badge.warn{{background:none;border:1px solid var(--warn);color:var(--warn)}}
.badge.ok{{background:var(--accent-soft)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1.1rem;margin:.9rem 0}}
.grid{{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:1rem;align-items:start}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
.scroll{{overflow-x:auto;max-height:26rem;overflow-y:auto}}
.thread{{display:flex;flex-direction:column;gap:.85rem}}
.turn{{display:flex;flex-direction:column;gap:.3rem}}
.bubble{{max-width:88%;padding:.55rem .8rem;border-radius:14px;font-size:14px;line-height:1.45}}
.bubble.user{{align-self:flex-end;background:var(--accent);color:#fff;border-bottom-right-radius:4px}}
.bubble.assistant{{align-self:flex-start;background:var(--bg);border:1px solid var(--line);
  border-bottom-left-radius:4px;cursor:pointer;display:flex;flex-direction:column;gap:.2rem;
  text-align:left;font:inherit;width:auto}}
.bubble.assistant:hover,.bubble.assistant[aria-current="true"]{{border-color:var(--accent)}}
.bubble.assistant.unknown{{border-style:dashed}}
.intent{{font-weight:650}}
.meta{{display:flex;align-items:center;gap:.35rem;font-size:12px;color:var(--muted)}}
.dot{{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}}
.dot.high{{background:var(--accent)}} .dot.medium{{background:var(--warn)}} .dot.low{{background:var(--danger)}}
.bar{{height:6px;background:var(--line);border-radius:999px;overflow:hidden;margin:.35rem 0 .1rem}}
.bar span{{display:block;height:100%}}
.bar.high span{{background:var(--accent)}} .bar.medium span{{background:var(--warn)}} .bar.low span{{background:var(--danger)}}
.conf{{font-size:1.35rem;font-variant-numeric:tabular-nums}}
.conf.high{{color:var(--accent)}} .conf.medium{{color:var(--warn)}} .conf.low{{color:var(--danger)}}
h3{{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  margin:1.1rem 0 .4rem;font-weight:600}}
pre{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:.7rem;
  overflow-x:auto;font-family:var(--mono);font-size:12px;margin:.3rem 0}}
.mono{{font-family:var(--mono);font-size:13px}}
.muted{{color:var(--muted)}} .small{{font-size:12.5px}}
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:.6rem .9rem;border-radius:0 8px 8px 0;font-size:13.5px;margin:.6rem 0 0}}
details{{border-bottom:1px solid var(--line);padding:.4rem 0}}
details:last-child{{border-bottom:none}}
summary{{cursor:pointer;display:flex;gap:.6rem;align-items:baseline}}
ul.examples{{margin:.5rem 0 .2rem;padding-left:1.2rem;font-size:13.5px;color:var(--muted)}}
ul.examples li{{margin:.15rem 0}}
footer{{margin-top:3rem;color:var(--muted);font-size:.85rem;
  border-top:1px solid var(--line);padding-top:1.25rem}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>A hiring platform, configured and classified</h1>
  <p class="lede">
    One real run against a live service: a domain created from a one-line brief, an applicant
    tracking system's tools registered over MCP, {run["intent_count"]} intents with
    {run["example_count"]} examples indexed, then a conversation classified against them.
    Every number here came out of the run.
  </p>
  <div class="chips">
    <span class="badge">{run["intent_count"]} intents</span>
    <span class="badge">{run["example_count"]} examples</span>
    <span class="badge">{len(tools)} MCP tools</span>
    <span class="badge">{len(turns)} turns</span>
    <span class="badge mono">{esc(run["health"].get("model", ""))}</span>
    <span class="badge">provider: {esc(run["health"].get("entity_extraction", {}).get("provider", ""))}</span>
  </div>
  <p class="muted small">
    Recorded with the offline provider, so it is deterministic and makes no external calls.
    The data is in Chroma and can be browsed in the running service.
    <a href="architecture.html">Architecture</a> &middot;
    <a href="properties.html">Configuration</a> &middot;
    <a href="changelog.html">Changelog</a>
  </p>
</header>

<div class="stage"><span class="n">1</span><h2>The domain</h2></div>
<p class="muted">An administrator supplies a name and a sentence. The role and vocabulary the
extractor works from are drafted from those and saved.</p>
<div class="card">
  <p class="mono">{esc(domain["name"])}</p>
  <p class="muted small">{esc(domain["description"])}</p>
  <h3>System instructions, drafted</h3>
  <p class="small">{esc(domain["system_instructions"])}</p>
  <h3>User instructions, drafted</h3>
  <p class="small">{esc(domain["user_instructions"])}</p>
  <p class="note">Both are ordinary configuration: editable on the domain page, and layered onto
  every extraction prompt in this domain.</p>
</div>

<div class="stage"><span class="n">2</span><h2>The tools it can reach</h2></div>
<p class="muted">Imported from the ATS server's <span class="mono">tools/list</span> response at
<span class="mono">{esc(steps["mcp"]["payload"]["endpoint"])}</span>. Each tool's own schema is
what extracted entities are later matched against.</p>
<div class="card scroll">
  <table>
    <thead><tr><th>Tool</th><th>What it does</th><th>Arguments</th><th>Required</th></tr></thead>
    <tbody>{tool_rows}</tbody>
  </table>
</div>

<div class="stage"><span class="n">3</span><h2>The intents</h2></div>
<p class="muted">{run["intent_count"]} operations a hiring team actually asks for, each wired to
the tool that serves it. Several are deliberately close in wording, because a real applicant
tracking system looks like that.</p>
<div class="card scroll">
  <table>
    <thead><tr><th>Intent</th><th>MCP tool</th><th>Accepts</th><th class="num">Examples</th></tr></thead>
    <tbody>{intent_rows}</tbody>
  </table>
</div>

<div class="stage"><span class="n">4</span><h2>The training signal</h2></div>
<p class="muted">{run["example_count"]} example utterances. This is the whole of it: no model was
fine-tuned, and adding an intent means adding examples.</p>
<div class="card scroll">{example_blocks}</div>
<div class="card">
  <div class="chips">
    <span class="badge ok">index {esc(index.get("state", ""))}</span>
    <span class="badge">v{index.get("version", "?")}</span>
    <span class="badge">{index.get("bm25_doc_count", "?")} BM25 documents</span>
    <span class="badge">{index.get("dense_count", "?")} vectors</span>
    <span class="badge {"ok" if index.get("bm25_doc_count") == index.get("dense_count") else "warn"}">
      {"in step" if index.get("bm25_doc_count") == index.get("dense_count") else "drifted"}</span>
  </div>
  <p class="muted small">Both indexes are built as the examples land and must agree.</p>
</div>

<div class="stage"><span class="n">5</span><h2>The conversation</h2></div>
<p class="muted">Nine turns in one session. Click any reply to see how that turn was decided.</p>
<div class="grid">
  <section class="card" id="detail"></section>
  <section class="card"><div class="thread" id="thread"></div></section>
</div>

<div class="stage"><span class="n">6</span><h2>What Chroma kept</h2></div>
<p class="muted">Read back from the store, not recomputed. Opening this session in the playground
shows exactly this.</p>
<div class="card">
  <div class="chips">
    <span class="badge mono">{esc(session["turns"][0]["text"][:0] or "hiring-demo")}</span>
    <span class="badge">last intent: {esc(session.get("last_intent", ""))}</span>
  </div>
  <table>
    <thead><tr><th>#</th><th>Request</th><th>Intent</th><th class="num">Conf</th><th>Entities</th></tr></thead>
    <tbody>{session_rows}</tbody>
  </table>
  <h3>Entities still in play</h3>
  <pre>{esc(json.dumps(session.get("entities_in_play", {}), indent=2))}</pre>
</div>

<footer>
  Tool calls are resolved and reported, never executed: the MCP block shows exactly what the
  caller would invoke. The catalogue and the conversation are in Chroma under the
  <span class="mono">hiring</span> domain.
</footer>
</div>

<script>
const TURNS = {json.dumps(turns)};
const pct = v => Math.round(v * 100) + "%";
const band = v => v >= 0.8 ? "high" : (v >= 0.5 ? "medium" : "low");
const esc = s => String(s).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}})[c]);

function hits(list) {{
  if (!list || !list.length) return '<p class="muted small">None.</p>';
  return '<table><tbody>' + list.map(h =>
    `<tr><td>${{esc(h.text)}}</td><td class="mono small">${{esc(h.intent)}}</td>` +
    `<td class="num">${{h.score.toFixed(3)}}</td></tr>`).join('') + '</tbody></table>';
}}

function detail(i) {{
  const t = TURNS[i], b = band(t.confidence), mcp = t.mcp || {{}};
  let out = `<h2>${{esc(t.intent)}}</h2>
    <div class="conf ${{b}}">${{pct(t.confidence)}}</div>
    <div class="bar ${{b}}"><span style="width:${{pct(t.confidence)}}"></span></div>
    <p class="muted small">${{esc(t.query)}}</p>
    <div class="chips"><span class="badge">${{esc(t.strategy||'')}}</span>
      <span class="badge">${{(t.latency_ms||0).toFixed(1)}} ms</span>
      ${{t.reason ? `<span class="badge warn">${{esc(t.reason)}}</span>` : ''}}</div>
    <p class="note">${{esc(t.note)}}</p>`;

  out += '<h3>Entities</h3>';
  const keys = Object.keys(t.entities || {{}});
  out += keys.length
    ? '<table><tbody>' + keys.map(k =>
        `<tr><td class="mono small">${{esc(k)}}</td><td>${{esc(t.entities[k])}}</td></tr>`).join('') + '</tbody></table>'
    : '<p class="muted small">None extracted.</p>';

  out += '<h3>MCP call</h3>';
  out += (mcp.qualified_name && !mcp.unresolved)
    ? `<div class="chips"><span class="badge mono">${{esc(mcp.qualified_name)}}</span>` +
      (mcp.ready ? '<span class="badge ok">ready</span>' : '<span class="badge warn">not ready</span>') +
      `</div><pre>${{esc(JSON.stringify({{server: mcp.server, tool: mcp.tool, arguments: mcp.arguments}}, null, 2))}}</pre>` +
      (mcp.missing_required && mcp.missing_required.length
        ? `<p class="small">missing: <span class="mono">${{esc(mcp.missing_required.join(', '))}}</span></p>` : '') +
      '<p class="muted small">Reported, never called.</p>'
    : '<p class="muted small">No MCP tool for this intent.</p>';

  if (t.context && (t.context.used_for_retrieval || (t.context.carried_entities||[]).length)) {{
    out += '<h3>Conversation</h3>';
    if (t.context.used_for_retrieval)
      out += `<p class="small">Read against the previous turn: <span class="mono">${{esc(t.context.retrieval_text)}}</span></p>`;
    if ((t.context.carried_entities||[]).length)
      out += `<p class="small">Carried forward: <span class="mono">${{esc(t.context.carried_entities.join(', '))}}</span></p>`;
  }}

  if (t.breakdown) {{
    const bd = t.breakdown;
    out += '<h3>Confidence breakdown</h3><table><tbody>' +
      [["s_dense",bd.s_dense],["s_rrf",bd.s_rrf],["s_margin",bd.s_margin],
       ["s_support",bd.s_support],["best similarity",bd.best_similarity]].map(([n,v]) =>
      `<tr><td><span class="dot ${{band(v)}}"></span> ${{n}}</td><td class="num">${{v.toFixed(3)}}</td></tr>`
    ).join('') + '</tbody></table>';
  }}
  if ((t.top_intents||[]).length) {{
    out += '<h3>Ranked intents after fusion</h3><table><tbody>' + t.top_intents.map((x,n) =>
      `<tr><td>${{n+1}}</td><td><span class="dot ${{band(x.best_similarity)}}"></span> ${{esc(x.intent)}}</td>` +
      `<td class="num">${{x.score.toFixed(5)}}</td><td class="num">${{x.best_similarity.toFixed(3)}}</td></tr>`
    ).join('') + '</tbody></table>';
  }}
  out += '<h3>Dense retrieval</h3>' + hits(t.dense_hits);
  out += '<h3>BM25 retrieval</h3>' + hits(t.bm25_hits);
  document.getElementById('detail').innerHTML = out;
  document.querySelectorAll('.bubble.assistant').forEach((el, n) =>
    el.setAttribute('aria-current', n === i ? 'true' : 'false'));
}}

document.getElementById('thread').innerHTML = TURNS.map((t, i) => {{
  const b = band(t.confidence), unknown = t.intent === 'UNKNOWN';
  const ents = Object.keys(t.entities||{{}}).length
    ? `<span class="muted small mono">${{esc(JSON.stringify(t.entities))}}</span>` : '';
  return `<div class="turn">
    <div class="bubble user">${{esc(t.query)}}</div>
    <button class="bubble assistant ${{unknown ? 'unknown' : ''}}" onclick="detail(${{i}})">
      <span class="intent">${{unknown ? 'No matching intent' : esc(t.intent)}}</span>
      ${{ents}}
      <span class="meta"><span class="dot ${{b}}"></span>${{pct(t.confidence)}}
      ${{t.tool && t.tool.name ? '· <span class="mono">'+esc(t.tool.name)+'</span>' : ''}}</span>
    </button>
  </div>`;
}}).join('');
detail(0);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the hiring demo")
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args(argv)

    if not RUN.exists():
        print(f"{RUN} is missing: run scripts.demos.build_hiring_demo first", file=sys.stderr)
        return 1
    run = json.loads(RUN.read_text("utf-8"))

    (DOCS / "demo-hiring.html").write_text(render_html(run), "utf-8")
    print(f"wrote {DOCS / 'demo-hiring.html'}")

    if not args.skip_gif:
        from scripts.demos.render_hiring_gif import render_gif

        render_gif(run, DOCS / "demo-hiring.gif")
        print(f"wrote {DOCS / 'demo-hiring.gif'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
