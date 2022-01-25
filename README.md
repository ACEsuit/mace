# ACE_torch

ACE implementation in PyTorch.

## Requirements

* Python >= 3.8
* PyTorch >= 1.8
* PyTorch geometric >= 1.7.1


## Development

We use `yapf` and `flake8` for code formatting.
Run the following to check formatting:

```bash
yapf --style=.style.yapf --in-place --recursive .
flake8 --config=.flake8 .
```

We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.
