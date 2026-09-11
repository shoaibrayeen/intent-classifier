"""The API reference is generated, so it must stay in step with the code."""

from scripts.build_api_docs import build_schema, render


def test_the_checked_in_reference_matches_the_current_schema():
    """If this fails, run: uv run python -m scripts.build_api_docs"""
    from scripts.build_api_docs import OUTPUT

    assert OUTPUT.exists(), "api-documentation.html has not been generated"
    assert OUTPUT.read_text("utf-8") == render(build_schema()), (
        "api-documentation.html is out of date; regenerate it with "
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
