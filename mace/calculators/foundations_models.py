import os
import urllib.request
from pathlib import Path
from typing import Union

import torch

from .mace import MACECalculator

module_dir = os.path.dirname(__file__)


def mace_mp(
    model: Union[str, Path] = None,
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
        model (str, optional): Path to the model. Defaults to None which first checks for
            a local model and then downloads the default model from figshare. Specify "medium"
            or "large" to download a smaller or larger model from figshare.
        default_dtype (str, optional): Default dtype for the model. Defaults to "float32".
        **kwargs: Passed to MACECalculator.

    Returns:
        MACECalculator: trained on the MPtrj dataset (unless model otherwise specified).
    """
    local_model_path = os.path.join(
        module_dir, "foundations_models/2023-08-14-mace-universal.model"
    )
    if model in (None, "medium") and os.path.exists(local_model_path):
        model = local_model_path
    elif model in (None, "medium", "large") or str(model).startswith("https:"):
        try:
            urls = dict(
                medium="https://figshare.com/ndownloader/files/42374049",
                large="https://figshare.com/ndownloader/files/43117273",
            )
            # default URL points to 2023-08-14-mace-yuan-trained-mptrj-04.model (16 MB, 2M params)
            checkpoint_url = (
                urls.get(model, urls["medium"])
                if model in (None, "medium", "large")
                else model
            )
            cache_dir = os.path.expanduser("~/.cache/mace")
            cached_model_path = f"{cache_dir}/{os.path.basename(checkpoint_url)}"
            if not os.path.exists(cached_model_path):
                os.makedirs(cache_dir, exist_ok=True)
                # download and save to disk
                print(f"Downloading MACE model from {model!r}")
                urllib.request.urlretrieve(model, cached_model_path)
                print(f"Cached MACE model to {cached_model_path}")
            model = cached_model_path
            print(
                "Using Materials Project model for MACECalculator, see https://figshare.com/articles/dataset/22715158"
            )
        except Exception as exc:
            raise RuntimeError(
                "Model download failed and no local model found"
            ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    return MACECalculator(model, device=device, default_dtype=default_dtype, **kwargs)


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
