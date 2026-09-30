"""The record of the published models against the frozen tree's loaders.

Every artifact a loader reaches is on the record, at the URL the loader
downloads it from, so a re-pointed alias or a new model fails here rather
than going unrecorded.
"""

from __future__ import annotations

import inspect

import pytest
from mace_core.foundation_roster import ROSTER, roster_entry


def _tables():
    from mace.calculators import foundations_models as loaders

    yield from (("mace_mp", name, url) for name, url in loaders.mace_mp_urls.items())
    yield from (("mace_off", name, url) for name, url in loaders.mace_off_urls.items())
    yield from (
        ("mace_polar", name, url) for name, url in loaders.polar_model_urls.items()
    )
    yield ("mace_mdp", "default", loaders.mace_mdp_default_url)


@pytest.mark.parametrize(("loader", "name", "url"), list(_tables()))
def test_every_tabled_model_is_on_the_record_at_its_url(loader, name, url):
    assert roster_entry(loader, name).url == url


@pytest.mark.parametrize("loader", ["mace_omol", "mace_anicc"])
def test_the_loaders_with_their_url_inline_download_the_recorded_one(loader):
    from mace.calculators import foundations_models as loaders

    source = inspect.getsource(getattr(loaders, loader))
    (entry,) = [entry for entry in ROSTER if entry.loader == loader]
    assert entry.url in source


def test_the_record_holds_nothing_the_loaders_do_not_reach():
    tabled = {(loader, name) for loader, name, _ in _tables()}
    inline = {("mace_omol", "extra_large"), ("mace_anicc", "default")}
    assert {(entry.loader, entry.name) for entry in ROSTER} == tabled | inline
