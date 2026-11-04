"""Legacy model spellings are rejected before they can coerce architecture."""

import pytest
from mace_torch.cli.run_train import parse


@pytest.mark.parametrize(
    "first",
    [
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticAttResidualInteractionBlock",
        "misspelled",
    ],
)
def test_legacy_plain_model_never_coerces_the_requested_first_interaction(first):
    with pytest.raises(ValueError):
        parse(["--model", "MACE", "--interaction_first", first])
    # The refusal is specific to the ambiguous model/unsupported interaction;
    # the retained residual choice still reaches the native configuration.
    if first == "RealAgnosticResidualInteractionBlock":
        assert (
            parse(["--interaction_first", first]).model.interaction_first == "residual"
        )
