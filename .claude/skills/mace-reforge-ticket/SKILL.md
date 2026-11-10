---
name: mace-reforge-ticket
description: Take a MACE Reforge board ticket from start to a reviewable PR - session ritual, branch naming, scope discipline, the pre-PR verification commands for both toolchains, the feature-inventory gate, deletion-only RET tickets, and the review gate. Use when starting, implementing, or wrapping up work on a reforge ticket.
---

# mace-reforge-ticket

Read the ticket and [reference scope](../../../docs/reforge/reference_branch.md).
Keep changes reviewable and test every changed contract. Run the native package
suite, relevant frozen gates, ruff and ty before preparing a PR. V1 changes
are reference work; final PRs target mace-reforge. Publishing and remote ticket
transitions require the user’s requested workflow.
