#!/usr/bin/env python3
"""Validate `debt_book.md`: schema, burn steps, and both directions of the markers.

The debt book is the ledger of every compromise the coexistence window
creates, and the failure mode it exists to prevent is a row nobody is holding
to anything. So this checks the things a reader cannot check by looking:

* the five-column schema, exactly, in order;
* every row names a burn-step ticket as an id and an issue number, so it can
  be looked up by someone who was not in the room;
* every row names a fitness test that is actually **defined** in
  `tests/architecture` -- a row pointing at a test nobody wrote reads as
  coverage, which is worse than a row admitting there is none;
* every row is claimed by an `open_debt` marker, and every marker matches a
  row. Deleting one without the other is how a burn-step pull request goes
  half-finished, and each direction fails differently.

Run from anywhere:  python3 tests/architecture/check_debt_book.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.architecture import debt  # noqa: E402


def main() -> int:
    problems = debt.problems()
    rows = debt.rows()
    if problems:
        print(f"{debt.BOOK.name}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{debt.BOOK.name}: {len(rows)} open debt row(s), all valid")
    for debt_id, row in rows.items():
        print(f"  {debt_id:38s} burned by {row.burn_step:16s} {row.fitness_test}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
