# Debt book

Every compromise the coexistence window creates, and the ticket whose merge removes it.

The rewrite runs as a coexistence: the new `packages/` stack is built beside a byte-frozen legacy
`mace/`, and legacy is retired capability by capability through deletion-only pull requests. That
window buys parity, and it costs compromises: a tolerance instead of bit-exactness, a capability
whose `--engine` default is still `legacy`, TorchScript decorators live on the path a v1 user hits,
a deployment gap. The known failure mode is not that any of those is wrong. It is that with no
business pressure the retirement never finishes and the dual state becomes permanent.

So each compromise is a row here, and each row has a test that fails until it is burned. There are
**no dates**. A date would be a promise to nobody: `main` stays on v0.3 until v1 ships, and no end
of life for v0.3 is announced. The trigger is the ticket, and the mechanics are in the next
section.

## How a row goes red

Each row names a fitness test in `tests/architecture/test_debt_book.py`. That test asserts the state
of the tree **after** the debt is burned, and it carries `xfail(strict=True)` for as long as its row
is in this file:

| the tree | the row | pytest says | reading |
|---|---|---|---|
| debt still there | open | `xfail` | expected, and green |
| debt burned | open | `XPASS(strict)` → **red** | the burn-step ticket closed and nobody removed the row |
| debt burned | removed | `passed` | done, and the assertion stays as a permanent guard |
| debt still there | removed | `failed` | the row was removed without burning the debt |

The third column is the whole mechanism, and it needs no calendar and no GitHub query: `strict=True`
turns an unexpected pass into a failure, so "the burn-step ticket is closed while the row is still
open" is exactly an `XPASS`. The burn-step pull request therefore deletes the row *and* the
`@open_debt(...)` decorator in the same change; deleting either alone is one of the two red rows
above.

`burn-check` is the prose statement of what the fitness test asserts. It is not a second copy of the
assertion: it says what would have to be true of the tree for the debt to be gone, in the terms a
reviewer of the burn-step pull request can check.

Run the meta-checks:

```bash
python3 tests/architecture/check_debt_book.py
python -m pytest tests/architecture/test_debt_book.py
```

## The ledger

The legacy implementation and launcher have been retired on the unpublished V1
reference branch. The permanent fitness tests remain without xfail markers.
External release prerequisites are recorded in `docs/reforge/reference_branch.md`.

| debt_id | description | burn-step (ticket) | burn-check | fitness-test |
|---|---|---|---|---|
