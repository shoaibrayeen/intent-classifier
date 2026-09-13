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
    assert names == {"api-documentation.html", "architecture.html", "changelog.html"}
    # README.md stays at the root: pyproject declares it as the project readme,
    # so the package build fails without it there.
    assert (docs.parent / "README.md").exists()
