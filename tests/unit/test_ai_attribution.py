"""The authorship gate's rules, pinned against what the tools really emit.

The policy is narrow on purpose: an assistant may be used, it may not be
credited as an author. So the rejected cases are all authorship claims, and
the accepted ones include every way of saying a tool helped.

Samples marked "seen on this repo" are copied from real commits, so a regex
edit that stops catching them fails here rather than on the next pull request.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "check_ai_attribution.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("check_ai_attribution", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through sys.modules, so the module has
    # to be registered before it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load()

HUMAN = "Ada Lovelace <ada@example.org>"


def _commit(message="Do a thing", author=HUMAN, committer=None):
    return checker.Commit(
        sha="0" * 40,
        author=author,
        committer=committer if committer is not None else author,
        message=message,
    )


REJECTED_COAUTHORS = [
    # Claude Code's default trailer. Seen on this repo in #1244 and #1711, and
    # on #1711 it is the *only* thing crediting an assistant: the author field
    # there is already the contributor's own.
    "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>",
    "Co-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
    "Co-authored-by: Claude Code <noreply@anthropic.com>",
    # Other assistants.
    "Co-authored-by: Copilot <198982749+Copilot@users.noreply.github.com>",
    "Co-authored-by: Cursor Agent <cursoragent@cursor.com>",
    "Co-authored-by: devin-ai-integration[bot] <devin@example.com>",
    "Co-authored-by: claude[bot] <claude@example.com>",
    "Co-authored-by: ChatGPT <noreply@openai.com>",
]


@pytest.mark.parametrize("line", REJECTED_COAUTHORS)
def test_an_assistant_credited_as_coauthor_is_rejected(line):
    report = checker.inspect([_commit(message=f"Add a block\n\n{line}\n")])
    assert not report.ok, f"missed: {line}"


@pytest.mark.parametrize(
    "identity",
    [
        "Claude <noreply@anthropic.com>",
        "Claude Code <noreply@anthropic.com>",
        "claude[bot] <bot@users.noreply.github.com>",
        "Copilot <copilot@example.com>",
    ],
)
def test_an_assistant_as_author_or_committer_is_rejected(identity):
    assert not checker.inspect([_commit(author=identity)]).ok
    assert not checker.inspect([_commit(author=HUMAN, committer=identity)]).ok


ALLOWED_TOOL_DISCLOSURE = [
    # Cursor's trailer, seen on this repo in #1391. It discloses a tool rather
    # than claiming authorship, so it passes.
    "Made-with: Cursor",
    "Generated-by: Devin",
    "Assisted-by: Gemini",
    # Claude Code's footer.
    "🤖 Generated with [Claude Code](https://claude.ai/code)",
    "Generated with Claude Code",
    # Saying so in prose is fine too.
    "Wrote the first draft of this block with Claude, then rewrote the loop",
    "Reviewed-by: Copilot",
]


@pytest.mark.parametrize("line", ALLOWED_TOOL_DISCLOSURE)
def test_disclosing_a_tool_is_allowed(line):
    report = checker.inspect([_commit(message=f"Add a block\n\n{line}\n")])
    assert report.ok, f"tool disclosure must pass: {line} -> {report.findings}"


ACCEPTED_PEOPLE = [
    # A contributor whose given name is Claude. The gate keys on the vendor
    # address or a product name, never on a bare first name.
    "Co-authored-by: Claude Dupont <claude.dupont@univ-lyon1.fr>",
    "Co-authored-by: Claude Bernard <cbernard@example.org>",
    "Co-authored-by: Claude <claude@univ-lyon1.fr>",
    "Co-authored-by: Ilyes Batatia <ilyes@example.org>",
    # Prose that happens to contain the words.
    "Move the cursor to the next atom in the readout loop",
    "Fix: cursor position drifts when the basis is truncated",
]


@pytest.mark.parametrize("line", ACCEPTED_PEOPLE)
def test_people_and_prose_are_accepted(line):
    report = checker.inspect([_commit(message=f"Add a block\n\n{line}\n")])
    assert report.ok, f"false positive on: {line} -> {report.findings}"


#: People the gate must leave alone. Every one of these was a real false
#: positive on the first cut of the patterns, found in review of #1716.
PEOPLE = [
    "Claude Dupont <claude.dupont@univ-lyon1.fr>",
    "Claude <claude@univ-lyon1.fr>",
    # A digit right after the name is an email local part, not a model
    # version. These read as "Claude 1" and "Gemini 1" without the separator.
    "Claude <claude1@example.org>",
    "Claude Bernard <claude2024@gmail.com>",
    "Gemini Rossi <gemini1@example.org>",
    # A token has to be a whole word, or every longer word starting with it
    # matches. Devine is a real surname.
    "Mary Devine <mdevine@example.org>",
    "Raj Devinder <raj@example.org>",
    "Ann Codexis <ann@example.org>",
    "Joe Windsurfer <joe@example.org>",
    "Tom Copilots <tom@example.org>",
    # Devin is a product and a common given name, so the bare token cannot
    # stand on its own any more than Claude can.
    "Devin Smith <devin@example.org>",
    "Devin Booker <dbooker@example.org>",
    "Devin <devin.smith@univ-lyon1.fr>",
]


@pytest.mark.parametrize("identity", PEOPLE)
def test_a_person_can_author_a_commit(identity):
    assert checker.inspect([_commit(author=identity)]).ok, identity


@pytest.mark.parametrize("identity", PEOPLE)
def test_a_person_can_be_credited_as_coauthor(identity):
    report = checker.inspect([_commit(message=f"x\n\nCo-authored-by: {identity}\n")])
    assert report.ok, f"{identity} -> {report.findings}"


def test_a_product_name_is_read_from_the_name_not_the_address():
    """The split is the fix for the email-local-part false positives.

    An assistant is still caught when the address is innocent, which is what
    stops the narrowing from becoming a hole.
    """
    assert checker.display_name("Claude Opus 5 <x@gmail.com>") == "Claude Opus 5"
    assert checker.display_name("plain@example.org") == "plain@example.org"
    assert not checker.inspect([_commit(author="Claude Opus 4.7 <someone@gmail.com>")]).ok
    assert not checker.inspect([_commit(author="Claude 3.5 Sonnet <x@gmail.com>")]).ok


def test_a_clean_commit_passes():
    assert checker.inspect([_commit()]).ok


def test_it_reports_every_offending_line_not_just_the_first():
    message = (
        "Add a block\n\n"
        "Co-authored-by: Claude Opus 5 <noreply@anthropic.com>\n"
        "Co-authored-by: Copilot <1+Copilot@users.noreply.github.com>\n"
    )
    assert len(checker.inspect([_commit(message=message)]).findings) == 2


def test_findings_name_the_commit_and_the_reason():
    report = checker.inspect(
        [_commit(message="x\n\nCo-authored-by: Claude Code <noreply@anthropic.com>\n")]
    )
    finding = report.findings[0]
    assert finding.sha == "0" * 40
    assert "trailer" in finding.where
    assert finding.why


def test_trailers_survive_crlf_and_indentation():
    crlf = "x\r\n\r\nCo-authored-by: Claude Opus 5 <noreply@anthropic.com>\r\n"
    indented = "x\n\n  Co-authored-by: Claude Opus 5 <noreply@anthropic.com>\n"
    assert not checker.inspect([_commit(message=crlf)]).ok
    assert not checker.inspect([_commit(message=indented)]).ok


def test_a_commit_with_no_body_passes():
    assert checker.inspect([_commit(message="Fix it")]).ok
    assert checker.inspect([_commit(message="")]).ok


REJECTED_ASSISTANTS = [
    # Devin still has to be caught, by a product form, a bot account or the
    # vendor's own domain.
    "Devin AI <x@gmail.com>",
    "devin-ai <x@gmail.com>",
    "devin-ai-integration[bot] <devin@example.com>",
    "Devin <noreply@cognition.ai>",
    "Devin <noreply@devin.ai>",
]


@pytest.mark.parametrize("identity", REJECTED_ASSISTANTS)
def test_devin_is_still_caught_by_product_bot_or_domain(identity):
    assert not checker.inspect([_commit(author=identity)]).ok, identity


def test_a_bare_token_is_only_used_where_nobody_is_called_it():
    """The rule the patterns follow, so an addition cannot quietly break it.

    A name that people actually have needs a product word, a version number
    or a vendor address. A name nobody has may stand alone.
    """
    for human_name in ("Claude", "Gemini", "Devin"):
        identity = f"{human_name} Smith <smith@example.org>"
        assert checker.inspect([_commit(author=identity)]).ok, human_name
    for product_only in ("Copilot", "Codex", "CodeWhisperer", "ChatGPT"):
        identity = f"{product_only} <bot@example.org>"
        assert not checker.inspect([_commit(author=identity)]).ok, product_only
