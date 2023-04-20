###########################################################################################
# Checkpointing
# Authors: Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch

from .torch_tools import TensorDict

Checkpoint = Dict[str, TensorDict]


@dataclasses.dataclass
class CheckpointState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR


class CheckpointBuilder:
    @staticmethod
    def create_checkpoint(state: CheckpointState) -> Checkpoint:
        return {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "lr_scheduler": state.lr_scheduler.state_dict(),
        }

    @staticmethod
    def load_checkpoint(
        state: CheckpointState, checkpoint: Checkpoint, strict: bool
    ) -> None:
        state.model.load_state_dict(checkpoint["model"], strict=strict)  # type: ignore
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        state.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


@dataclasses.dataclass
class CheckpointPathInfo:
    path: str
    tag: str
    epochs: int
    swa: bool


class CheckpointIO:
    def __init__(
        self, directory: str, tag: str, keep: bool = False, swa_start: int = None
    ) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None
        self.swa_start = swa_start

        self._epochs_string = "_epoch-"
        self._filename_extension = "pt"

    def _get_checkpoint_filename(self, epochs: int, swa_start=None) -> str:
        if swa_start is not None and epochs > swa_start:
            return (
                self.tag
                + self._epochs_string
                + str(epochs)
                + "_swa"
                + "."
                + self._filename_extension
            )
        return (
            self.tag
            + self._epochs_string
            + str(epochs)
            + "."
            + self._filename_extension
        )

    def _list_file_paths(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        all_paths = [
            os.path.join(self.directory, f) for f in os.listdir(self.directory)
        ]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_checkpoint_path(self, path: str) -> Optional[CheckpointPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)\.{self._filename_extension}$"
        )
        regex2 = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)_swa\.{self._filename_extension}$"
        )
        match = regex.match(filename)
        match2 = regex2.match(filename)
        swa = False
        if not match:
            if not match2:
                return None
            match = match2
            swa = True

        return CheckpointPathInfo(
            path=path,
            tag=match.group("tag"),
            epochs=int(match.group("epochs")),
            swa=swa,
        )

    def _get_latest_checkpoint_path(self, swa) -> Optional[str]:
        all_file_paths = self._list_file_paths()
        checkpoint_info_list = [
            self._parse_checkpoint_path(path) for path in all_file_paths
        ]
        selected_checkpoint_info_list = [
            info for info in checkpoint_info_list if info and info.tag == self.tag
        ]

        if len(selected_checkpoint_info_list) == 0:
            logging.warning(
                f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'"
            )
            return None

        selected_checkpoint_info_list_swa = []
        selected_checkpoint_info_list_no_swa = []

        for ckp in selected_checkpoint_info_list:
            if ckp.swa:
                selected_checkpoint_info_list_swa.append(ckp)
            else:
                selected_checkpoint_info_list_no_swa.append(ckp)
        if swa:
            latest_checkpoint_info = max(
                selected_checkpoint_info_list_swa, key=lambda info: info.epochs
            )
        else:
            latest_checkpoint_info = max(
                selected_checkpoint_info_list_no_swa, key=lambda info: info.epochs
            )
        return latest_checkpoint_info.path

    def save(
        self, checkpoint: Checkpoint, epochs: int, keep_last: bool = False
    ) -> None:
        if not self.keep and self.old_path and not keep_last:
            logging.debug(f"Deleting old checkpoint file: {self.old_path}")
            os.remove(self.old_path)

        filename = self._get_checkpoint_filename(epochs, self.swa_start)
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving checkpoint: {path}")
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)
        self.old_path = path

    def load_latest(
        self, swa: Optional[bool] = False, device: Optional[torch.device] = None
    ) -> Optional[Tuple[Checkpoint, int]]:
        path = self._get_latest_checkpoint_path(swa=swa)
        if path is None:
            return None

        return self.load(path, device=device)

    def load(
        self, path: str, device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, int]:
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        logging.info(f"Loading checkpoint: {checkpoint_info.path}")
        return (
            torch.load(f=checkpoint_info.path, map_location=device),
            checkpoint_info.epochs,
        )


class CheckpointHandler:
    def __init__(self, *args, **kwargs) -> None:
        self.io = CheckpointIO(*args, **kwargs)
        self.builder = CheckpointBuilder()

    def save(
        self, state: CheckpointState, epochs: int, keep_last: bool = False
    ) -> None:
        checkpoint = self.builder.create_checkpoint(state)
        self.io.save(checkpoint, epochs, keep_last)

    def load_latest(
        self,
        state: CheckpointState,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
        strict=False,
    ) -> Optional[int]:
        result = self.io.load_latest(swa=swa, device=device)
        if result is None:
            return None

        checkpoint, epochs = result
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint, strict=strict)
        return epochs

    def load(
        self,
        state: CheckpointState,
        path: str,
        strict=False,
        device: Optional[torch.device] = None,
    ) -> int:
        checkpoint, epochs = self.io.load(path, device=device)
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint, strict=strict)
        return epochs
