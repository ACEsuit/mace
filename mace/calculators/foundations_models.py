"""
"""

import logging
from .mace import MACECalculator
import os

path = os.path.dirname(__file__)


def mace_mp(
    device="cuda",
    model_path=None,
) -> MACECalculator:
    """
    Constructs a MACECalculator with a pretrained model based on the Materials Project (89 elements).
    The model is released under the MIT license.
    Note:
        If you are using this function, please cite the relevant paper for the Materials Project,
        any paper associated with the MACE model, and also the following:
        - "MACE-Universal by Yuan Chiang, 2023, Hugging Face, Revision e5ebd9b, DOI: 10.57967/hf/1202, URL: https://huggingface.co/cyrusyc/mace-universal"
        - "Matbench Discovery by Janosh Riebesell, Rhys EA Goodall, Anubhav Jain, Philipp Benner, Kristin A Persson, Alpha A Lee, 2023, arXiv:2308.14920"
    """
    if model_path is None:
        model_path = os.path.join(
            path, "foundations_models/2023-08-14-mace-universal.model"
        )
        print(
            "Using Materials Project model for MACECalculator, see https://huggingface.co/cyrusyc/mace-universal"
        )
    return MACECalculator(model_path, device=device, default_dtype="float32")


def mace_anicc(
    device="cuda",
    model_path=None,
) -> MACECalculator:
    """
    Constructs a MACECalculator with a pretrained model based on the ANI (H, C, N, O).
    The model is released under the MIT license.
    Note:
        If you are using this function, please cite the relevant paper associated with the MACE model, ANI dataset, and also the following:
        - "Evaluation of the MACE Force Field Architecture by Dávid Péter Kovács, Ilyes Batatia, Eszter Sára Arany, and Gábor Csányi, The Journal of Chemical Physics, 2023, URL: https://doi.org/10.1063/5.0155322
    """
    if model_path is None:
        model_path = os.path.join(path, "foundations_models/ani500k_large_CC.model")
        print(
            "Using ANI couple cluster model for MACECalculator, see https://doi.org/10.1063/5.0155322"
        )
    return MACECalculator(model_path, device=device, default_dtype="float64")
