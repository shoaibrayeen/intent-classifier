"""Every link the UI offers has to go somewhere.

The reference pages link to each other by filename, which is how they read from
disk. Served under their own routes, those links resolved beside the UI and
404ed, so the whole directory is mounted at one prefix instead.
"""

import re

DOCS = [
    "architecture.html",
    "api-documentation.html",
    "changelog.html",
    "properties.html",
    "demo.html",
]


def test_every_document_is_served(client):
    for name in DOCS:
        assert client.get(f"/ui/docs/{name}").status_code == 200, name


def test_the_short_urls_redirect_to_the_documents(client):
    for path, target in [
        ("/ui/api", "api-documentation.html"),
        ("/ui/changelog", "changelog.html"),
        ("/changelog", "changelog.html"),
        ("/ui/properties", "properties.html"),
        ("/ui/architecture", "architecture.html"),
    ]:
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 307, path
        assert response.headers["location"] == f"/ui/docs/{target}"
        assert client.get(path).status_code == 200, path


def test_a_filename_under_ui_is_forgiven(client):
    """Where the links used to point, and where people will try."""
    for name in DOCS:
        assert client.get(f"/ui/{name}").status_code == 200, name


def test_an_unknown_document_is_a_404_not_a_redirect_loop(client):
    assert client.get("/ui/nonsense.html").status_code == 404


def test_the_links_between_documents_resolve(client):
    """The actual bug: architecture.html from a served page must not 404."""
    for name in DOCS:
        body = client.get(f"/ui/docs/{name}").text
        for target in set(re.findall(r'href="([a-z0-9-]+\.html)"', body)):
            assert client.get(f"/ui/docs/{target}").status_code == 200, f"{name} -> {target}"


def test_every_link_on_every_ui_page_resolves(client, contract_domain):
    """A page that offers a dead link is worse than one that offers none."""
    pages = [
        "/",
        "/ui/intents",
        "/ui/mcp",
        "/ui/playground",
        "/ui/sessions",
        "/ui/evaluation",
        "/ui/operations",
    ]
    checked, broken = set(), []
    for page in pages:
        body = client.get(page).text
        for href in re.findall(r'href="(/[^"#?]*)"', body):
            if href in checked:
                continue
            checked.add(href)
            if client.get(href).status_code >= 400:
                broken.append(f"{page} -> {href}")
    assert not broken, broken
    assert len(checked) > 10  # the crawl actually found links


def test_the_navigation_offers_every_reference_page(client):
    body = client.get("/ui/playground").text
    for href in ["/ui/architecture", "/ui/api", "/ui/changelog", "/docs"]:
        assert f'href="{href}"' in body, href


def test_the_demo_assets_are_served(client):
    assert client.get("/ui/docs/demo.gif").status_code == 200
    assert client.get("/ui/docs/demo-run.json").status_code == 200
