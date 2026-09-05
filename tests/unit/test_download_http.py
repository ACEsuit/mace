"""Exercise the real checkpoint downloader with offline HTTP responses."""

import io
from email.message import Message
from pathlib import Path
from unittest.mock import Mock

import pytest

from mace.calculators import foundations_models as fm


class Response(io.BytesIO):
    """Minimal context-managed HTTP response; no network or model needed."""

    def __init__(self, body, content_type="application/octet-stream"):
        super().__init__(body)
        self.headers = Message()
        if content_type is not None:
            self.headers["content-type"] = content_type
        self.headers["Content-Length"] = str(len(body))

    def info(self):
        return self.headers


@pytest.mark.parametrize(
    "content_type", ["text/html", "text/html; charset=utf-8", "Text/HTML"]
)
@pytest.mark.parametrize("existing", [False, True])
def test_html_is_rejected_before_reading_or_replacing_checkpoint(
    tmp_path, monkeypatch, content_type, existing
):
    """A successful HTTP status must not publish an HTML body as a model."""
    destination = tmp_path / "model.model"
    if existing:
        destination.write_bytes(b"existing checkpoint")
    response = Response(b"<html>login required</html>", content_type)
    response.read = Mock(wraps=response.read)
    urlopen = Mock(return_value=response)
    monkeypatch.setattr(fm.urllib.request, "urlopen", urlopen)
    url = "https://example.test/model.model"

    with pytest.raises(RuntimeError, match="Model download failed") as error:
        fm._urlretrieve_with_timeout(url, str(destination))

    assert url in str(error.value)
    urlopen.assert_called_once_with(url, timeout=fm._DOWNLOAD_TIMEOUT)
    response.read.assert_not_called()
    assert response.closed
    assert not destination.with_suffix(".model.part").exists()
    if existing:
        assert destination.read_bytes() == b"existing checkpoint"
    else:
        assert not destination.exists()


@pytest.mark.parametrize(
    "loader",
    [fm.mace_mp, fm.mace_polar, fm.mace_off, fm.mace_omol, fm.mace_mdp, fm.mace_anicc],
)
def test_loader_retries_html_then_reuses_successful_cache(
    tmp_path, monkeypatch, loader
):
    """Every loader uses the common guard, retries, and then stays offline."""
    monkeypatch.setattr(fm, "get_cache_dir", lambda: str(tmp_path))
    calculator = Mock()
    monkeypatch.setattr(fm, "MACECalculator", calculator)
    html = Response(b"<html>login required</html>", "text/html; charset=utf-8")
    checkpoint = Response(b"checkpoint bytes")
    urlopen = Mock(side_effect=[html, checkpoint])
    monkeypatch.setattr(fm.urllib.request, "urlopen", urlopen)
    if loader is fm.mace_anicc:
        kwargs = {"model_path": str(tmp_path / "model.model")}
    else:
        kwargs = {"model": "https://example.test/model.model"}

    with pytest.raises(RuntimeError, match="download"):
        loader(device="cpu", **kwargs)

    calculator.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    assert urlopen.call_count == 1

    loader(device="cpu", **kwargs)
    saved = Path(calculator.call_args.kwargs["model_paths"])
    assert saved.parent == tmp_path
    assert saved.read_bytes() == b"checkpoint bytes"
    assert not Path(str(saved) + ".part").exists()
    assert urlopen.call_count == 2

    loader(device="cpu", **kwargs)
    assert urlopen.call_count == 2
    assert calculator.call_count == 2
    assert Path(calculator.call_args.kwargs["model_paths"]) == saved


@pytest.mark.parametrize("content_type", ["application/octet-stream", None])
def test_binary_download_preserves_bytes_headers_timeout_and_progress(
    tmp_path, monkeypatch, capsys, content_type
):
    """Valid downloads still publish exact bytes and report progress."""
    body = b"checkpoint bytes"
    response = Response(body, content_type)
    urlopen = Mock(return_value=response)
    monkeypatch.setattr(fm.urllib.request, "urlopen", urlopen)
    destination = tmp_path / "model.model"
    url = "https://github.com/example/models/blob/main/model.model?raw=true"

    path, info = fm._urlretrieve_with_timeout(url, str(destination), timeout=17)

    assert path == str(destination)
    assert info is response.headers
    assert destination.read_bytes() == body
    assert not Path(str(destination) + ".part").exists()
    assert response.closed
    urlopen.assert_called_once_with(
        "https://raw.githubusercontent.com/example/models/main/model.model", timeout=17
    )
    assert "Downloading: 100.0%" in capsys.readouterr().out


def test_interrupted_binary_download_is_cleaned_up_and_can_retry(tmp_path, monkeypatch):
    """Keep atomic-download cleanup while adding the content-type guard."""
    destination = tmp_path / "model.model"
    interrupted = Response(b"unused")
    interrupted.read = Mock(
        side_effect=[b"partial bytes", TimeoutError("read timed out")]
    )
    success = Response(b"complete checkpoint")
    urlopen = Mock(side_effect=[interrupted, success])
    monkeypatch.setattr(fm.urllib.request, "urlopen", urlopen)
    url = "https://example.test/model.model"

    with pytest.raises(TimeoutError, match="read timed out"):
        fm._urlretrieve_with_timeout(url, str(destination))

    assert list(tmp_path.iterdir()) == []
    assert interrupted.closed
    fm._urlretrieve_with_timeout(url, str(destination))
    assert destination.read_bytes() == b"complete checkpoint"
    assert not Path(str(destination) + ".part").exists()
    assert urlopen.call_count == 2
