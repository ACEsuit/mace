#!/usr/bin/env python3
"""Require that a commit's authors are people.

Using an assistant is fine and needs no disclosure rules here. Claiming one as
an author is the problem: GitHub builds the contributor list from the author
and committer identities and from `Co-authored-by:` trailers, so any of the
three can put an assistant on MACE's contributor list. A squash merge then
copies the branch's trailers into the permanent history, where removing them
means rewriting it. Two commits on develop carry nineteen such lines between
them.

Scope is authorship and nothing else. Disclosing a tool is left alone: a
`Made-with: Cursor` trailer, a generated-with footer and any mention in the
message body all pass, because none of them claims authorship.

A contributor whose own name is Claude is unaffected. Only an AI vendor
address, a bot account, or an assistant's product name counts, never a bare
first name.

Usage:
    check_ai_attribution.py <base-ref> <head-ref>   # inspect a commit range
    check_ai_attribution.py --stdin                 # read `git log` output
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field

#: Vendor addresses. An email is the one unambiguous signal: nobody receives
#: mail at anthropic.com by being called Claude.
VENDOR_EMAIL = re.compile(
    r"@(?:anthropic\.com|openai\.com|cursor\.(?:com|sh)|cognition(?:-?labs)?\.ai"
    r"|devin\.ai"
    r"|codeium\.com|sourcegraph\.com|tabnine\.com|aider\.chat)\b",
    re.IGNORECASE,
)

#: Bot accounts, which GitHub renders with a `[bot]` suffix.
BOT_ACCOUNT = re.compile(
    r"\b(?:claude|copilot|chatgpt|codex|chatgpt-codex-connector|cursor|devin"
    r"|devin-ai-integration|gemini|jules|google-labs-jules|openhands|sweep"
    r"|codegen|cody|continue|augment|qodo|codiumai|coderabbitai|greptile"
    r"|ellipsis|korbit|sourcery|bito|tembo|factory-droid|charlie"
    r"|blackboxai)(?:-ai)?\[bot\]",
    re.IGNORECASE,
)

#: An assistant's product name. `Claude Dupont` is a person; `Claude Opus 4.7`,
#: `Claude Code` and `GPT-4` are not. Requiring this form is what lets an
#: identity naming a real human through without a vendor address.
#:
#: Both boundaries are anchored, or a token matches any longer word starting
#: with it and the surname Devine reads as Devin. A version digit needs a
#: separator in front of it for the same reason: `Claude 3` is a model,
#: `claude3` is the start of somebody's email address.
#:
#: A token stands alone only when nobody is called it. `Copilot`, `Codex` and
#: `CodeWhisperer` are safe on their own; `Claude`, `Gemini` and `Devin` are
#: people's names, so each needs a product word, a version number or a vendor
#: address before it counts. Apply that test to anything added here.
ASSISTANT_PRODUCT = re.compile(
    r"\b(?:"
    r"claude[\s_-]*(?:code|opus|sonnet|haiku|instant)"
    r"|claude[\s_-]+\d"
    r"|gemini[\s_-]*(?:pro|flash)"
    r"|gemini[\s_-]+\d"
    r"|gpt-?\d"
    r"|o[1-4]-(?:mini|preview)"
    r"|devin[\s_-]*ai"
    r"|chatgpt|copilot|codex|codewhisperer|windsurf|openhands"
    r"|cursor[\s_-]*agent"
    r"|swe-?agent"
    r"|amazon\s+q"
    r"|replit\s+agent"
    r")\b",
    re.IGNORECASE,
)

#: `Name <email>`, so a product name can be matched against the name alone.
_IDENTITY = re.compile(r"^(?P<name>.*?)\s*<[^>]*>$")

#: The trailer GitHub reads to add a co-author. Attribution trailers that do
#: not claim authorship are deliberately not listed: `Made-with:` discloses a
#: tool, and disclosure is allowed.
COAUTHOR_TRAILERS = ("co-authored-by", "coauthored-by")

_TRAILER = re.compile(r"^\s*([A-Za-z][A-Za-z0-9-]*)\s*:\s*(.+?)\s*$")


@dataclass
class Commit:
    """A commit reduced to the fields that can carry authorship."""

    sha: str
    author: str = ""
    committer: str = ""
    message: str = ""


@dataclass
class Finding:
    sha: str
    where: str
    text: str
    why: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings


def display_name(identity: str) -> str:
    """The name half of `Name <email>`, or all of it when there is no address."""
    identity = identity.strip()
    match = _IDENTITY.match(identity)
    return match.group("name") if match else identity


def assistant_reason(identity: str) -> str:
    """Why `identity` names an assistant rather than a person, or ""."""
    if VENDOR_EMAIL.search(identity):
        return "an AI vendor email address"
    if BOT_ACCOUNT.search(identity):
        return "an AI bot account"
    # Against the display name only. An email's local part is not a name:
    # reading `claude1@` as `Claude 1` would block the very people the vendor
    # address and product name rules are shaped to leave alone.
    if ASSISTANT_PRODUCT.search(display_name(identity)):
        return "an assistant's product name"
    return ""


def inspect(commits: list[Commit]) -> Report:
    """Findings for every commit that names an assistant as an author.

    Pure: takes commit records, returns findings. The git plumbing lives in
    `commits_in_range`, so the rules are testable without a repository.
    """
    report = Report()
    for commit in commits:
        for label, identity in (("author", commit.author),
                                ("committer", commit.committer)):
            why = assistant_reason(identity)
            if why:
                report.findings.append(
                    Finding(commit.sha, label, identity, why))

        for line in commit.message.splitlines():
            match = _TRAILER.match(line)
            if match is None or match.group(1).lower() not in COAUTHOR_TRAILERS:
                continue
            why = assistant_reason(match.group(2))
            if why:
                report.findings.append(
                    Finding(commit.sha, f"{match.group(1)} trailer",
                            match.group(2), why))
    return report


def commits_in_range(base: str, head: str) -> list[Commit]:
    """The commits `head` adds to `base`, newest last."""
    sep, rec = "\x1f", "\x1e"
    out = subprocess.run(
        ["git", "log", f"--format=%H{sep}%an <%ae>{sep}%cn <%ce>{sep}%B{rec}",
         f"{base}..{head}"],
        capture_output=True, text=True, check=True).stdout
    return _parse(out)


def _parse(raw: str) -> list[Commit]:
    sep, rec = "\x1f", "\x1e"
    commits = []
    for chunk in raw.split(rec):
        if not chunk.strip():
            continue
        sha, author, committer, message = chunk.lstrip("\n").split(sep, 3)
        commits.append(Commit(sha, author, committer, message))
    return commits


def main(argv: list[str]) -> int:
    if argv[:1] == ["--stdin"]:
        commits = _parse(sys.stdin.read())
    elif len(argv) == 2:
        commits = commits_in_range(argv[0], argv[1])
    else:
        print(__doc__, file=sys.stderr)
        return 2

    report = inspect(commits)
    if report.ok:
        print(f"[ok] every author is a person in {len(commits)} commit(s)")
        return 0

    print(f"FAIL an assistant is named as an author in "
          f"{len(report.findings)} place(s) across {len(commits)} commit(s):\n")
    for f in report.findings:
        print(f"  {f.sha[:12]}  {f.where}: {f.text}")
        print(f"                {f.why}")
    print(
        "\nUse whatever tools you like. The authors have to be people.\n"
        "Drop these lines and amend:\n"
        "  git commit --amend              # for the tip commit\n"
        "  git rebase -i <base>            # reword, further back\n"
        "Claude Code stops adding the trailer with "
        "`includeCoAuthoredBy: false`.\n"
        "Saying a tool helped is fine: `Made-with:` and a generated-with "
        "footer both pass."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
