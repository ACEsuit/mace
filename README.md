# MACE V1 reference implementation

> **AI-generated prototype — reference only.** `v1-reference` is preserved to
> help the team understand the proposed implementation and review PRs.
> The reviewed implementation is being developed
> on `mace-reforge` through one PR per ticket; releases will come from that work.

MACE builds equivariant machine-learning interatomic potentials. Model hosting
and version publication remain pending.

Use Python 3.12+ for the complete workspace (core/torch also support 3.11):

```sh
python -m pip install --group dev -e packages/mace-core -e packages/mace-torch -e packages/mace-jax -e packages/mace-backend-example
mace --help
```

The `mace-torch` distribution owns the `mace` CLI and imports as `mace_torch`.
The retired v0.3 import `mace`, its pickle runtime and the transitional launcher
are absent. Convert trusted published checkpoints in a separate pinned legacy
environment; native inference reads JSON plus safetensors.

- [Documentation](docs/index.md): configuration, training, evaluation and deployment.
- [Migration guide](docs/reforge/migration.md): old command and Python API mappings.
- [Examples](examples/README.md) and [JAX teaching forward](tutorials/README.md).
- [Contributing](CONTRIBUTING.md), [governance](GOVERNANCE.md), [agent guidance](CLAUDE.md).

Run `python -m pytest tests/retirement tests/architecture tests/golden` for the
frozen numerical anchors and architecture gates. Package tests live beside
their implementation. GPU and optional integration tests use capability markers;
missing capabilities skip locally and fail when promised by CI.

Install agent skills with `npx skills add ACEsuit/mace --list`; for this
unpublished checkout, use `npx skills add .`. The tracked index is
[skills.json](skills.json). See [MIT license](LICENSE.md) for licensing.


## References

If you use this code, please cite our papers:

```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr.

For bugs or feature requests, please use [GitHub Issues](https://github.com/ACEsuit/mace/issues).

## License

The MACE code is published and distributed under the [MIT License](LICENSE.md). (Note that some of the models linked above come with different licenses).
