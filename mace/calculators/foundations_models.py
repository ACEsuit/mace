import os
import urllib.request
from pathlib import Path
from typing import Union

import torch

from .mace import MACECalculator

module_dir = os.path.dirname(__file__)


def mace_mp(
    model_path: Union[str, Path] = None,
    device: str = "",
    default_dtype: str = "float32",
    **kwargs,
) -> MACECalculator:
    """
    Constructs a MACECalculator with a pretrained model based on the Materials Project (89 elements).
    The model is released under the MIT license.
    Note:
        If you are using this function, please cite the relevant paper for the Materials Project,
        any paper associated with the MACE model, and also the following:
        - "MACE-Universal by Yuan Chiang, 2023, Hugging Face, Revision e5ebd9b, DOI: 10.57967/hf/1202, URL: https://huggingface.co/cyrusyc/mace-universal"
        - "Matbench Discovery by Janosh Riebesell, Rhys EA Goodall, Anubhav Jain, Philipp Benner, Kristin A Persson, Alpha A Lee, 2023, arXiv:2308.14920"

    Args:
        device (str, optional): Device to use for the model. Defaults to "cuda".
        model_path (str, optional): Path to the model. Defaults to "https://figshare.com/ndownloader/files/42374049".
        default_dtype (str, optional): Default dtype for the model. Defaults to "float32".
        **kwargs: Passed to MACECalculator.

    Returns:
        MACECalculator: trained on the MPtrj dataset (unless model_path otherwise specified).
    """
    if model_path is None or str(model_path).startswith("https:"):
        try:
            # default URL points to 2023-08-14-mace-yuan-trained-mptrj-04.model (16 MB, 2M params)
            model_path = model_path or "https://figshare.com/ndownloader/files/42374049"
            cache_dir = os.path.expanduser("~/.cache/mace")
            cached_model_path = f"{cache_dir}/{os.path.basename(model_path)}"
            if not os.path.exists(cached_model_path):
                os.makedirs(cache_dir, exist_ok=True)
                # download and save to disk
                print(f"Downloading MACE model from {model_path!r}")
                urllib.request.urlretrieve(model_path, cached_model_path)
                print(f"Cached MACE model to {cached_model_path}")
            model_path = cached_model_path
            print(
                "Using Materials Project model for MACECalculator, see https://figshare.com/articles/dataset/22715158"
            )
        except Exception:
            # if download fails, use the default model in the repo
            model_path = os.path.join(
                module_dir, "foundations_models/2023-08-14-mace-universal.model"
            )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    return MACECalculator(
        model_path, device=device, default_dtype=default_dtype, **kwargs
    )


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
        model_path = os.path.join(
            module_dir, "foundations_models/ani500k_large_CC.model"
        )
        print(
            "Using ANI couple cluster model for MACECalculator, see https://doi.org/10.1063/5.0155322"
        )
    return MACECalculator(model_path, device=device, default_dtype="float64")
