# MACE governance

The **steering committee** sets project direction and represents the interests
of core developers and scientific stakeholders. The **technical committee**
consists of core developers responsible for technical decisions, code review
and final technical approval. The contribution procedure is in
[CONTRIBUTING.md](CONTRIBUTING.md).

Committee membership, named approval owners, appointment procedures and the
question of external funding beyond MPI require a human decision. This document
does not appoint people or settle funding arrangements.

Releases are feature-driven, not time-boxed. Out-of-cycle bug-fix releases are
supported. Tagged releases should be citable through Zenodo under the existing
[concept DOI](https://doi.org/10.5281/zenodo.14103332); the release process must
verify the version DOI before announcing it.

V1 is an unpublished reference branch. Its `release.yaml` is a manually
triggered build and wheel-smoke check with read-only repository permissions;
it creates no tag, GitHub Release, PyPI upload or DOI. The final publishing
pipeline and model publication remain pending on `mace-reforge`.

Automated review and issue-management tools, if adopted, must remain advisory.
A missing token or an unavailable service must not block contributions, close
issues or supply final approval.
