"""The evaluation page, including the states where it cannot run.

The dataset used to live under tests/, which is not copied into the Docker
image, so in a container the page ran zero cases and rendered a table of zeroes
that looked like a successful run.
"""

import json

from tests.conftest import seed_contract_domain


def test_the_dataset_ships_with_the_application(client):
    """Not under tests/: the running service needs it."""
    from app.services.evaluation import DATASET_PATH, load_cases

    assert DATASET_PATH.exists()
    assert "app" in DATASET_PATH.parts and "tests" not in DATASET_PATH.parts
    assert len(load_cases()) > 0


def test_the_dockerfile_copies_what_the_dataset_lives_in(client):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert "COPY app ./app" in (root / "Dockerfile").read_text()


def test_running_the_evaluation_reports_real_numbers(client, contract_domain):
    response = client.post("/ui/evaluation/run", data={}, headers={"HX-Request": "true"})

    assert response.status_code == 200
    body = response.text
    assert "Top-1 accuracy" in body
    assert "cases ran" in body
    assert "Nothing was evaluated" not in body


def test_it_says_so_when_there_is_no_dataset(app_factory, tmp_path):
    """A run that could not happen must not look like a run that matched nothing."""
    missing = tmp_path / "absent.json"
    _, client = app_factory({"evaluation_dataset_path": str(missing)})
    seed_contract_domain(client)

    body = client.post("/ui/evaluation/run", data={}, headers={"HX-Request": "true"}).text

    assert "Nothing was evaluated" in body
    assert "No evaluation cases found" in body
    assert str(missing) in body


def test_it_says_so_when_no_domains_are_configured(client):
    body = client.post("/ui/evaluation/run", data={}, headers={"HX-Request": "true"}).text

    assert "Nothing was evaluated" in body
    assert "No domains are configured" in body


def test_it_says_so_when_every_case_is_for_another_domain(app_factory, tmp_path):
    dataset = tmp_path / "other.json"
    dataset.write_text(
        json.dumps(
            {"cases": [{"domain": "aviation", "query": "book a flight", "expected": "FLIGHT_BOOK"}]}
        )
    )
    _, client = app_factory({"evaluation_dataset_path": str(dataset)})
    seed_contract_domain(client)

    body = client.post("/ui/evaluation/run", data={}, headers={"HX-Request": "true"}).text

    assert "Every case was skipped" in body
    assert "aviation" in body


def test_a_custom_dataset_is_used(app_factory, tmp_path):
    dataset = tmp_path / "mine.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "domain": "contract",
                        "query": "Find all contracts with Microsoft",
                        "expected": "CONTRACT_SEARCH",
                    },
                ]
            }
        )
    )
    _, client = app_factory({"evaluation_dataset_path": str(dataset)})
    seed_contract_domain(client)

    body = client.post("/ui/evaluation/run", data={}, headers={"HX-Request": "true"}).text

    assert "1 of 1 cases ran" in body
    assert "100%" in body


def test_the_page_says_where_its_cases_come_from(client):
    page = client.get("/ui/evaluation").text
    assert "dataset.json" in page
    assert "EVALUATION_DATASET_PATH" in page
