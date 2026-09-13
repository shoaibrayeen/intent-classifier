"""The API reference is generated, so it must stay in step with the code."""

from scripts.build_api_docs import build_schema, render


def test_the_checked_in_reference_matches_the_current_schema():
    """If this fails, run: uv run python -m scripts.build_api_docs"""
    from scripts.build_api_docs import OUTPUT

    assert OUTPUT.exists(), "docs/api-documentation.html has not been generated"
    assert OUTPUT.read_text("utf-8") == render(build_schema()), (
        "docs/api-documentation.html is out of date; regenerate it with "
        "'uv run python -m scripts.build_api_docs'"
    )


def test_every_endpoint_in_the_schema_appears_in_the_reference():
    schema = build_schema()
    page = render(schema)
    for path in schema["paths"]:
        assert path in page, path


def test_request_and_response_models_are_documented():
    page = render(build_schema())
    for model in ["ClassifyRequest", "ClassifyResponse", "ToolCall", "SessionTurn", "ContextInfo"]:
        assert f'id="model-{model}"' in page, model


def test_the_documentation_lives_under_docs():
    """Everything a reader opens is in one place, not scattered at the root."""
    from pathlib import Path

    docs = Path(__file__).resolve().parents[2] / "docs"
    names = {p.name for p in docs.glob("*.html")}
    assert names == {
        "api-documentation.html",
        "architecture.html",
        "changelog.html",
        "demo.html",
        "demo-hiring.html",
        "properties.html",
    }
    # The demo ships its recording and its animation alongside the page.
    assert (docs / "demo.gif").exists()
    assert (docs / "demo-run.json").exists()
    # README.md stays at the root: pyproject declares it as the project readme,
    # so the package build fails without it there.
    assert (docs.parent / "README.md").exists()


def test_the_properties_page_matches_the_settings_model():
    """A configuration table drifts the first time someone adds a field."""
    from scripts.build_properties import OUTPUT, render

    assert OUTPUT.exists(), "docs/properties.html has not been generated"
    assert OUTPUT.read_text("utf-8") == render(), (
        "docs/properties.html is out of date; regenerate it with "
        "'uv run python -m scripts.build_properties'"
    )


def test_every_setting_is_documented_with_a_group():
    from app.config import Settings

    for name, field in Settings.model_fields.items():
        assert field.description, f"{name} has no description"
        extra = field.json_schema_extra
        assert isinstance(extra, dict) and extra.get("group"), f"{name} has no group"


def test_the_properties_page_lists_every_setting():
    from app.config import Settings
    from scripts.build_properties import render

    page = render()
    for name in Settings.model_fields:
        assert name.upper() in page, name


def test_the_env_files_carry_only_what_differs_from_the_defaults():
    """A .env that repeats the defaults is noise, and goes stale silently."""
    from pathlib import Path

    from app.config import Settings

    defaults = {n: f.default for n, f in Settings.model_fields.items()}

    def is_default(key: str, value: str) -> bool:
        default = defaults.get(key.lower())
        if default is None and key.lower() not in defaults:
            return False
        value = value.strip()
        if isinstance(default, bool):
            return value.lower() == str(default).lower()
        if isinstance(default, int | float):
            try:
                return float(value) == float(default)
            except ValueError:
                return False
        return value == str(default)

    root = Path(__file__).resolve().parents[2]
    for name in [".env", ".env.example"]:
        path = root / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # An empty OPENAI_API_KEY is the one placeholder worth keeping:
            # it is the value a deployment is most likely to supply.
            if key.strip() == "OPENAI_API_KEY":
                continue
            assert not is_default(key.strip(), value), (
                f"{name} sets {key.strip()} to its default; remove it"
            )


def test_the_memory_file_documents_the_current_invariants():
    """MEMORY.md records what is expensive to rediscover. If a guard named
    there is gone, the file is lying and the bug is probably back."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    memory = (root / "MEMORY.md").read_text()

    from app.config import Settings

    for name in ["context_rescue_margin", "auto_rescue_margin", "seed_on_startup"]:
        assert name in Settings.model_fields
        assert name.upper() in memory, f"MEMORY.md does not mention {name}"

    # The paths it points at must exist.
    for path in [
        "app/evaluation/dataset.json",
        "app/services/turn_details.py",
        "app/services/mcp_executor.py",
        "scripts/build_properties.py",
        "app/repositories/base.py",
    ]:
        assert path in memory, f"MEMORY.md does not mention {path}"
        assert (root / path).exists(), f"MEMORY.md points at a missing {path}"


def test_generated_artefacts_are_not_committed():
    """graphify-out is ~4MB and rebuilt in seconds."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "check-ignore", "graphify-out/graph.json"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "graphify-out/ is not gitignored"


def test_the_hiring_demo_artefacts_are_present_and_consistent():
    """Both artefacts must describe the same recorded run."""
    import json
    from pathlib import Path

    docs = Path(__file__).resolve().parents[2] / "docs"
    run_file = docs / "demo-hiring-run.json"
    assert run_file.exists(), "the hiring demo has not been recorded"
    assert (docs / "demo-hiring.html").exists()
    assert (docs / "demo-hiring.gif").exists()

    run = json.loads(run_file.read_text())
    page = (docs / "demo-hiring.html").read_text()

    assert run["domain"]["name"] == "hiring"
    assert run["intent_count"] >= 15
    assert run["example_count"] >= 100
    assert len(run["turns"]) >= 8
    # The page reports the run's own numbers, not hand-written ones.
    assert f"{run['intent_count']} intents" in page
    assert f"{run['example_count']} examples" in page


def test_the_hiring_catalogue_is_internally_consistent():
    """Every intent names a tool the MCP server actually offers."""
    from scripts.demos.hiring_catalogue import INTENTS, MCP_SERVER

    offered = {t["name"] for t in MCP_SERVER["tools"]["tools"]}
    for intent in INTENTS:
        assert intent["tool"] in offered, f"{intent['name']} maps to an unknown tool"
        assert len(intent["examples"]) >= 8, f"{intent['name']} has too few examples"
        assert intent["description"], intent["name"]
    assert len({i["name"] for i in INTENTS}) == len(INTENTS), "duplicate intent name"
