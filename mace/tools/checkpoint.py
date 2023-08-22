###########################################################################################
# Checkpointing
# Authors: Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import os
import re
from typing import Dict, List, Optional

from accelerate import Accelerator

from .torch_tools import TensorDict

Checkpoint = Dict[str, TensorDict]

@dataclasses.dataclass
class CheckpointPathInfo:
    path: str
    tag: str
    epochs: int
    swa: bool


class CheckpointHandler:
    def __init__(
        self, accelerator: Accelerator, directory: str, tag: str, keep: bool = False, swa_start: int = None,
    ) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None
        self.swa_start = swa_start
        self.accelerator = accelerator

        self._epochs_string = "_epoch-"

    def _get_checkpoint_dirname(self, epochs: int, swa_start=None) -> str:
        if swa_start is not None and epochs > swa_start:
            return (
                self.tag
                + self._epochs_string
                + str(epochs)
                + "_swa"
            )
        return (
            self.tag
            + self._epochs_string
            + str(epochs)
        )

    def _list_file_paths(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        all_paths = [
            os.path.join(self.directory, f) for f in os.listdir(self.directory)
        ]
        return [path for path in all_paths if os.path.isdir(path)]

    def _parse_checkpoint_path(self, path: str) -> Optional[CheckpointPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)$"
        )
        regex2 = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)_swa$"
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

    def _get_latest_checkpoint_path_info(self, swa) -> Optional[str]:
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
        return latest_checkpoint_info

    def delete_checkpoint(self, path):
        for filename in os.listdir(path):
            full_filename = os.path.join(path, filename)
            os.remove(full_filename)
        os.rmdir(path)

    def save(
        self, epochs: int, keep_last: bool = False
    ) -> None:

        if self.accelerator.process_index == 0 and not self.keep and self.old_path and not keep_last:
            logging.debug(f"Deleting old checkpoint: {self.old_path}")
            self.delete_checkpoint(self.old_path)

        filename = self._get_checkpoint_dirname(epochs, self.swa_start)
        path = os.path.join(self.directory, filename)
        self.old_path = path

        logging.debug(f"Saving checkpoint: {path}")
        self.accelerator.save_state(path)

    def load_latest(self, swa: Optional[bool] = False):

        checkpoint_info = self._get_latest_checkpoint_path_info(swa=swa)

        if checkpoint_info is None:
            return None

        self.accelerator.load_state(checkpoint_info.path)

        return checkpoint_info.epochs

    def load(self, path: str):

        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        self.accelerator.load_state(path)

        return checkpoint_info.epochs
