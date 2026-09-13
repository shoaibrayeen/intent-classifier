"""Generate docs/properties.html from the Settings model.

Every setting, its default, its constraints and what it does, read out of
``app/config.py`` rather than transcribed. A hand-written configuration table
drifts the first time someone adds a field; a generated one cannot.

    uv run python -m scripts.build_properties
    uv run python -m scripts.build_properties --check
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo

from app.config import Settings

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "properties.html"

#: The order groups are presented in: what you reach for first, first.
GROUP_ORDER = [
    "Core",
    "Entity extraction and generation",
    "Security",
    "Storage",
    "Embedding model",
    "Retrieval",
    "Confidence and UNKNOWN",
    "Retrieval strategy",
    "Multi-turn sessions",
    "Observability",
]

GROUP_NOTES = {
    "Core": "Nothing here needs setting to run the service.",
    "Entity extraction and generation": (
        "Classification never calls a language model. These control the parts that do: entity "
        "extraction, and drafting domain instructions and intents."
    ),
    "Security": "Off by default so the service runs out of the box. Turn it on before exposing it.",
    "Confidence and UNKNOWN": (
        "The thresholds that decide whether to answer at all. Calibrate them against your own "
        "evaluation set rather than by feel."
    ),
    "Retrieval strategy": (
        "Changing retrieval is the riskiest change here, because a regression is invisible until "
        "the wrong tool gets called."
    ),
    "Multi-turn sessions": "Only active when a request carries a session id.",
    "Observability": "Metrics and the audit trail are on; tracing needs a collector, so it is not.",
}


def type_name(annotation: Any) -> str:
    if get_origin(annotation) is not None:
        args = get_args(annotation)
        if all(isinstance(a, str) for a in args):  # Literal["a", "b"]
            return " | ".join(f'"{a}"' for a in args)
    return getattr(annotation, "__name__", str(annotation))


def constraint_text(field: FieldInfo) -> str:
    parts = []
    for item in field.metadata:
        for attribute, symbol in (("ge", "≥"), ("gt", ">"), ("le", "≤"), ("lt", "<")):
            value = getattr(item, attribute, None)
            if value is not None:
                parts.append(f"{symbol} {value}")
    return ", ".join(parts)


def default_text(field: FieldInfo) -> str:
    value = field.default
    if value == "":
        return "(empty)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def collect() -> dict[str, list[tuple[str, FieldInfo]]]:
    groups: dict[str, list[tuple[str, FieldInfo]]] = {}
    for name, field in Settings.model_fields.items():
        extra = field.json_schema_extra or {}
        group = str(extra.get("group", "Core")) if isinstance(extra, dict) else "Core"
        groups.setdefault(group, []).append((name, field))
    return groups


def render() -> str:
    groups = collect()
    ordered = [g for g in GROUP_ORDER if g in groups] + [
        g for g in sorted(groups) if g not in GROUP_ORDER
    ]

    required = [
        (name, field)
        for _, fields in groups.items()
        for name, field in fields
        if isinstance(field.json_schema_extra, dict)
        and field.json_schema_extra.get("required_when")
    ]

    sections = []
    for group in ordered:
        note = GROUP_NOTES.get(group, "")
        rows = []
        for name, field in sorted(groups[group]):
            extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
            when = extra.get("required_when")
            constraints = constraint_text(field)
            rows.append(
                f"""      <tr>
        <td class="mono">{html.escape(name.upper())}"""
                + (f'<div class="tag">required when {html.escape(str(when))}</div>' if when else "")
                + f"""</td>
        <td class="mono default">{html.escape(default_text(field))}</td>
        <td class="type">{html.escape(type_name(field.annotation))}"""
                + (f'<div class="muted">{html.escape(constraints)}</div>' if constraints else "")
                + f"""</td>
        <td>{html.escape(field.description or "")}</td>
      </tr>"""
            )
        sections.append(
            f"""<section>
  <h2 id="{group.lower().replace(" ", "-")}">{html.escape(group)}</h2>
  {f'<p class="muted">{html.escape(note)}</p>' if note else ""}
  <div class="scroll"><table>
    <thead><tr><th>Variable</th><th>Default</th><th>Type</th><th>What it does</th></tr></thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table></div>
</section>"""
        )

    required_rows = "".join(
        f'<li><span class="mono">{html.escape(name.upper())}</span> — '
        f"{html.escape(str(field.json_schema_extra['required_when']))}</li>"
        for name, field in sorted(required)
    )
    nav = " · ".join(
        f'<a href="#{g.lower().replace(" ", "-")}">{html.escape(g)}</a>' for g in ordered
    )
    total = sum(len(v) for v in groups.values())

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Configuration · Intent Classifier</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#faf9f7; --panel:#fff; --ink:#1a1917; --muted:#6b6a66; --line:#e3e1dc;
  --accent:#2f6f4f; --accent-soft:#e8f1eb; --warn:#8a6d1f;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16171a; --panel:#1e2023; --ink:#eceae6; --muted:#9b9a95;
    --line:#32343a; --accent:#6fbf94; --accent-soft:#223029; --warn:#d6b45c; }}
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:3rem 1.5rem 5rem}}
header{{border-bottom:1px solid var(--line);padding-bottom:1.5rem;margin-bottom:1rem}}
h1{{font-size:2rem;margin:0 0 .4rem;letter-spacing:-.025em}}
.lede{{font-size:1.05rem;color:var(--muted);margin:0;max-width:64ch}}
h2{{font-size:1.15rem;margin:2.5rem 0 .5rem;padding-top:1.25rem;
  border-top:1px solid var(--line);letter-spacing:-.015em}}
section:first-of-type h2{{border-top:none;padding-top:0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:.55rem .65rem;border-bottom:1px solid var(--line);
  vertical-align:top}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  font-weight:600}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:var(--accent-soft)}}
.mono{{font-family:var(--mono);font-size:13px}}
.default{{white-space:nowrap;color:var(--accent)}}
.type{{font-size:12.5px;color:var(--muted);white-space:nowrap}}
.muted{{color:var(--muted)}}
.tag{{display:inline-block;margin-top:.25rem;padding:.1rem .45rem;border-radius:999px;
  border:1px solid var(--warn);color:var(--warn);font-size:11px;font-family:inherit;
  white-space:normal}}
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:.85rem 1.1rem;border-radius:0 8px 8px 0;margin:1.5rem 0}}
.note strong{{display:block;margin-bottom:.25rem}}
.note ul{{margin:.5rem 0 0;padding-left:1.1rem}}
.note li{{margin:.25rem 0}}
.toc{{font-size:13.5px;color:var(--muted);margin:.75rem 0 0;line-height:2}}
.toc a{{color:var(--accent);text-decoration:none}}
.scroll{{overflow-x:auto}}
pre{{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:.9rem 1rem;overflow-x:auto;font-family:var(--mono);font-size:13px}}
footer{{margin-top:3rem;border-top:1px solid var(--line);padding-top:1.25rem;
  color:var(--muted);font-size:.85rem}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Configuration</h1>
  <p class="lede">
    Every setting the service reads, with the value it uses when you say nothing.
    All {total} have working defaults: the service starts, classifies and serves its UI with no
    <span class="mono">.env</span> at all.
  </p>
  <p class="toc">{nav}</p>
</header>

<div class="note">
  <strong>Nothing is required to start.</strong>
  Two values become mandatory once you turn on the feature that needs them:
  <ul>{required_rows}</ul>
  Set a variable in <span class="mono">.env</span> only to move it off the default below.
</div>

<p>Settings come from the environment, or from <span class="mono">.env</span> beside the
application. Environment variables win. Names are case-insensitive:</p>

<pre>CONFIDENCE_THRESHOLD=0.65
LLM_PROVIDER=mock</pre>

{chr(10).join(sections)}

<footer>
  Generated from <span class="mono">app/config.py</span> by
  <span class="mono">scripts/build_properties.py</span>, so this page cannot drift from the code.
  <a href="architecture.html">Architecture</a> ·
  <a href="api-documentation.html">API reference</a> ·
  <a href="changelog.html">Changelog</a> ·
  <a href="demo.html">Demo</a>
</footer>
</div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the configuration reference")
    parser.add_argument("--check", action="store_true", help="fail if the page is stale")
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text("utf-8") != rendered:
            print(
                "docs/properties.html is out of date: run scripts/build_properties.py",
                file=sys.stderr,
            )
            return 1
        print("properties.html is up to date")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, "utf-8")
    print(f"wrote {OUTPUT} ({len(rendered):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
