"""One definition of the confidence bands, used everywhere they are shown."""

import pytest

from app.ui.templating import HIGH_BAND, MEDIUM_BAND, asset, band


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "high"),
        (0.95, "high"),
        (0.80, "high"),
        (0.799, "medium"),
        (0.65, "medium"),
        (0.50, "medium"),
        (0.499, "low"),
        (0.0, "low"),
        (None, "low"),
    ],
)
def test_the_bands_are_green_above_80_amber_to_50_red_below(value, expected):
    assert band(value) == expected


def test_the_boundaries_are_inclusive_at_the_top_of_each_band():
    assert band(HIGH_BAND) == "high"
    assert band(MEDIUM_BAND) == "medium"


def test_a_static_asset_url_carries_a_version():
    """Without it a cached stylesheet survives a deploy and the change is invisible."""
    url = asset("app.css")
    assert url.startswith("/static/app.css?v=")
    assert url.split("v=")[1].isdigit()


def test_a_missing_asset_still_produces_a_usable_url():
    assert asset("does-not-exist.css") == "/static/does-not-exist.css"
