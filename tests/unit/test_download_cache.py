"""A failed model download must not leave anything at the cache path.

_urlretrieve_with_timeout renames the body into place as soon as the HTTP fetch
succeeds, which is before the caller inspects the content type. An error page
served with 200 therefore lands exactly where the checkpoint belongs. Since
every loader guards its download with `if not os.path.isfile(path)`, leaving it
there poisons the cache for good: later calls skip the download and hand the
HTML to torch.load, which fails with a zip-archive error bearing no relation to
the original problem.
"""

import os

import pytest

from mace.calculators import foundations_models as fm


@pytest.fixture(name="html_download")
def fixture_html_download(tmp_path, monkeypatch):
    """Point the cache at tmp_path and serve an HTML error page for every fetch."""
    monkeypatch.setattr(fm, "get_cache_dir", lambda: str(tmp_path))

    def fake_download(url, filename, timeout=None):
        # the real helper has already renamed the body into place by this point
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write("<html>404 not found</html>")
        return filename, "Content-Type: text/html\nContent-Length: 26"

    monkeypatch.setattr(fm, "_urlretrieve_with_timeout", fake_download)
    return tmp_path


def test_html_response_leaves_no_cached_file(html_download):
    with pytest.raises(RuntimeError, match="Model download failed"):
        fm.download_mace_mp_checkpoint("medium")

    leftovers = os.listdir(html_download)
    assert not leftovers, f"cache poisoned with {leftovers}"


def test_second_attempt_retries_instead_of_reusing_the_error_page(html_download):
    """The failure has to be reproducible, not masked by the first attempt."""
    calls = []
    original = fm._urlretrieve_with_timeout

    def counting(url, filename, timeout=None):
        calls.append(url)
        return original(url, filename, timeout)

    fm._urlretrieve_with_timeout = counting
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Model download failed"):
                fm.download_mace_mp_checkpoint("medium")
    finally:
        fm._urlretrieve_with_timeout = original

    assert len(calls) == 2, (
        "the second call reused the cached error page instead of retrying"
    )


def test_discard_is_quiet_when_the_file_is_already_gone(tmp_path):
    """Cleanup must not raise on top of the error it is reporting."""
    fm._discard_cached_download(str(tmp_path / "not-there"))
