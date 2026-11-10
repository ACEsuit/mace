# Frozen numerical evidence

The original values, tolerance table, structures and pickled oracle snapshots
remain under tests/golden. Native checkpoints and parameter-coordinate maps
under tests/retirement reproduce energies, derivatives and the original
all-parameter training-gradient digests without a legacy import. Frozen
electrostatic cases live in tests/parity. Historical feature-inventory pins
record the removed tests at the pre-retirement commit; they are not live gates.

Do not regenerate references to make a test pass. If a recapture is explicitly
requested, use a separate checkout of
`fdef2fb577925cd8e7b66876403870128e69ec97` and its pinned legacy environment.
The electrostatic capture recipes additionally require their pinned external
source checkouts. Set `MACE_GOLDEN_ORACLE_CHECKOUT` to that historical checkout
when invoking those recipes. Capture there and review any proposed change
separately. Normal testing needs neither a legacy install nor external sources.
