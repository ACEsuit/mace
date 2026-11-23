"""L2-SP follows the selected optimizer's native regularization rule."""

import math

import torch


class L2SPCheckpointError(RuntimeError):
    """A checkpoint cannot restore the original fine-tuning objective."""


class L2SPRegularization:
    """Apply the optimizer to differences from immutable pretrained weights.

    Model weights and optimizer parameter-valued state remain in absolute
    coordinates between steps. L-BFGS has no native weight decay, so its
    closure instead includes ``strength / 2 * ||theta - theta_0||^2``.
    """

    def __init__(self, optimizer, strength):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("L2-SP strength must be finite and non-negative")
        if any(group.get("weight_decay", 0.0) != 0 for group in optimizer.param_groups):
            raise ValueError(
                "L2-SP requires optimizer groups with weight_decay=0; "
                "the coefficient is managed by L2-SP"
            )
        self.strength = float(strength)
        self.uses_loss = isinstance(optimizer, torch.optim.LBFGS)
        self._schedulefree = type(optimizer).__module__.startswith("schedulefree.")
        if self._schedulefree and type(optimizer).__name__ != "AdamWScheduleFree":
            raise ValueError("L2-SP supports MACE's AdamWScheduleFree optimizer")
        for group in optimizer.param_groups:
            if not self.uses_loss and "weight_decay" not in group:
                raise ValueError("L2-SP requires optimizer weight decay or L-BFGS")
            if any(
                group.get(option, False)
                for option in ("differentiable", "capturable", "fused")
            ):
                raise ValueError(
                    "L2-SP requires eager, non-differentiable optimizer steps"
                )
        self.optimizer_kind = (
            f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        )
        self.parameters = [list(group["params"]) for group in optimizer.param_groups]
        self.names = [
            list(group.get("param_names", [])) for group in optimizer.param_groups
        ]
        names = [name for group in self.names for name in group]
        parameters = [param for group in self.parameters for param in group]
        if (
            not parameters
            or any(
                len(params) != len(names)
                for params, names in zip(self.parameters, self.names)
            )
            or len(set(names)) != len(names)
            or len({id(p) for p in parameters}) != len(parameters)
            or any(not p.requires_grad for p in parameters)
        ):
            raise ValueError(
                "L2-SP requires unique names for selected trainable parameters"
            )
        self.references = [
            [p.detach().clone() for p in group] for group in self.parameters
        ]

    def squared_distance(self):
        return torch.stack(
            [
                (param - reference).square().sum()
                for params, references in zip(self.parameters, self.references)
                for param, reference in zip(params, references)
            ]
        ).sum()

    def penalty(self):
        return (self.strength / 2) * self.squared_distance()

    @torch.no_grad()
    def step(self, optimizer):
        """Use native decay on relative weights, then restore absolute weights.

        ScheduleFree's auxiliary iterate z is a parameter value rather than a
        gradient moment, so it must be translated along with the model weights.
        Its train/eval interpolation then continues to work without wrappers.
        Gradients have already been computed and clipped in model coordinates.
        """
        if any(
            len(group["params"]) != len(params)
            or any(p is not q for p, q in zip(group["params"], params))
            for group, params in zip(optimizer.param_groups, self.parameters)
        ) or len(optimizer.param_groups) != len(self.parameters):
            raise ValueError("L2-SP parameter selection must remain fixed")
        if self.uses_loss or self.strength == 0:
            return optimizer.step()
        translated = []
        decay = [group["weight_decay"] for group in optimizer.param_groups]
        try:
            for group, refs in zip(optimizer.param_groups, self.references):
                group["weight_decay"] = self.strength
                for param, reference in zip(group["params"], refs):
                    if param.grad is None:
                        continue
                    original = param.detach().clone()
                    z = (
                        optimizer.state.get(param, {}).get("z")
                        if self._schedulefree
                        else None
                    )
                    original_z = z.detach().clone() if z is not None else original
                    translated.append((param, reference, original, original_z))
                    param.sub_(reference)
                    if z is not None:
                        z.sub_(reference)
            return optimizer.step()
        finally:
            for param, reference, original, original_z in translated:
                # Apply the change to the original value to preserve exact
                # no-op updates (e.g. lr=0), avoiding a subtract/add round trip.
                param.copy_(original + (param - (original - reference)))
                if self._schedulefree:
                    z = optimizer.state.get(param, {}).get("z")
                    if z is not None:
                        z.copy_(original_z + (z - (original_z - reference)))
            for group, weight_decay in zip(optimizer.param_groups, decay):
                group["weight_decay"] = weight_decay

    @torch.no_grad()
    def add_gradients(self):
        """Add the penalty gradient once per full-dataset L-BFGS closure.

        Each DDP rank has identical parameters/references, so this contribution
        needs no gradient reduction. Adding it directly avoids firing DDP's
        backward hooks a second time outside the model's forward pass.
        """
        if self.strength == 0:
            return
        for params, references in zip(self.parameters, self.references):
            for param, reference in zip(params, references):
                gradient = self.strength * (param - reference)
                if param.grad is None:
                    param.grad = gradient
                else:
                    param.grad.add_(gradient)

    def state_dict(self):
        return {
            "version": 3,
            "rule": "loss" if self.uses_loss else "native_decay",
            "strength": self.strength,
            "optimizer_kind": self.optimizer_kind,
            "references": self.references,
        }

    def validate_state_dict(self, checkpoint):
        metadata = checkpoint.get("l2sp", {})
        version = metadata.get("version")
        if version == 1 and self.optimizer_kind == "torch.optim.adamw.AdamW":
            compatible = metadata.get("mode", "decoupled") == "decoupled"
        elif version == 2 and self.uses_loss:
            compatible = (
                metadata.get("mode") == "loss"
                and metadata.get("optimizer_kind") == self.optimizer_kind
            )
        else:
            compatible = (
                version == 3
                and metadata.get("rule")
                == ("loss" if self.uses_loss else "native_decay")
                and metadata.get("optimizer_kind") == self.optimizer_kind
            )
        if not compatible:
            raise L2SPCheckpointError(
                "Checkpoint is missing L2-SP references or uses a different optimizer/regularization rule"
            )
        if metadata.get("strength") != self.strength:
            raise L2SPCheckpointError("L2-SP strength differs from the checkpoint")
        groups = checkpoint.get("param_groups", [])
        references = metadata.get("references", [])
        if not len(groups) == len(references) == len(self.parameters):
            raise L2SPCheckpointError(
                "L2-SP parameter groups differ from the checkpoint"
            )
        for params, names, saved, refs in zip(
            self.parameters, self.names, groups, references
        ):
            if saved.get("param_names") != names or len(refs) != len(params):
                raise L2SPCheckpointError(
                    "L2-SP selected parameters differ from the checkpoint"
                )
            if saved.get("weight_decay", 0.0) != 0:
                raise L2SPCheckpointError("Invalid L2-SP checkpoint weight decay")
            for param, reference in zip(params, refs):
                if (
                    not isinstance(reference, torch.Tensor)
                    or reference.shape != param.shape
                ):
                    raise L2SPCheckpointError(
                        "L2-SP reference shape differs from the model"
                    )

    def load_state_dict(self, checkpoint):
        self.validate_state_dict(checkpoint)
        self.references = [
            [ref.detach().to(param).clone() for param, ref in zip(params, refs)]
            for params, refs in zip(self.parameters, checkpoint["l2sp"]["references"])
        ]


def get_regularization(optimizer):
    return getattr(optimizer, "l2sp", None)


def optimizer_step(optimizer):
    regularization = get_regularization(optimizer)
    if regularization is None:
        return optimizer.step()
    return regularization.step(optimizer)


def validate_checkpoint(optimizer, checkpoint):
    regularization = get_regularization(optimizer)
    if regularization is not None:
        regularization.validate_state_dict(checkpoint)
    elif "l2sp" in checkpoint:
        raise L2SPCheckpointError("An L2-SP checkpoint requires --l2sp=True to resume")


def optimizer_state_dict(optimizer):
    state = optimizer.state_dict()
    penalty = get_regularization(optimizer)
    if penalty is not None:
        state["l2sp"] = penalty.state_dict()
    return state


def load_optimizer_state_dict(optimizer, checkpoint):
    validate_checkpoint(optimizer, checkpoint)
    optimizer.load_state_dict(checkpoint)
    penalty = get_regularization(optimizer)
    if penalty is not None:
        penalty.load_state_dict(checkpoint)
