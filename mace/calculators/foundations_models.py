import os
import urllib.request
import warnings
from pathlib import Path
from typing import Union

import torch
from ase import units
from ase.calculators.mixing import SumCalculator

from .mace import MACECalculator

module_dir = os.path.dirname(__file__)
local_model_path = os.path.join(
    module_dir, "foundations_models/2023-12-03-mace-mp.model"
)


def mace_mp(
    model: Union[str, Path] = None,
    device: str = "",
    dtype: str = "float32",
    dispersion: bool = False,
    dispersion_xc="pbe",
    dispersion_cutoff=40.0 * units.Bohr,
    **kwargs,
) -> MACECalculator:
    """
    Constructs a MACECalculator with a pretrained model based on the Materials Project (89 elements).
    The model is released under the MIT license.
    Note:
        If you are using this function, please cite the relevant paper for the Materials Project,
        any paper associated with the MACE model, and also the following:
        - MACE-Universal by Yuan Chiang, 2023, Hugging Face, Revision e5ebd9b,
            DOI: 10.57967/hf/1202, URL: https://huggingface.co/cyrusyc/mace-universal
        - Matbench Discovery by Janosh Riebesell, Rhys EA Goodall, Philipp Benner, Yuan Chiang,
            Alpha A Lee, Anubhav Jain, Kristin A Persson, 2023, arXiv:2308.14920

    Args:
        model (str, optional): Path to the model. Defaults to None which first checks for
            a local model and then downloads the default model from figshare. Specify "small",
            "medium" or "large" to download a smaller or larger model from figshare.
        device (str, optional): Device to use for the model. Defaults to "cuda".
        dtype (str, optional): Default dtype for the model. Defaults to "float32".
        dispersion (bool, optional): Whether to use D3 dispersion corrections. Defaults to False.
        dispersion_xc (str, optional): Exchange-correlation functional for D3 dispersion corrections.
        dispersion_cutoff (float, optional): Cutoff radius in Bhor for D3 dispersion corrections.
        **kwargs: Passed to MACECalculator and TorchDFTD3Calculator.

    Returns:
        MACECalculator: trained on the MPtrj dataset (unless model otherwise specified).
    """
    if "default_dtype" in kwargs:
        dtype = kwargs.pop("default_dtype")
        warnings.warn(
            "default_dtype is deprecated, use dtype instead!", DeprecationWarning
        )

    if model in (None, "medium") and os.path.isfile(local_model_path):
        model = local_model_path
        print(
            f"Using local medium Materials Project MACE model for MACECalculator {model}"
        )
    elif model in (None, "small", "medium", "large") or str(model).startswith("https:"):
        try:
            urls = dict(
                small="https://tinyurl.com/2jmmb8b7",  # 2023-12-10-mace-128-L0_energy_epoch-249.model
                medium="https://tinyurl.com/y7uhwpje",  # 2023-12-03-mace-128-L1_epoch-199.model
                large="https://figshare.com/ndownloader/files/43117273",
            )
            checkpoint_url = (
                urls.get(model, urls["medium"])
                if model in (None, "small", "medium", "large")
                else model
            )
            cache_dir = os.path.expanduser("~/.cache/mace")
            checkpoint_url_name = "".join(
                c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
            )
            cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
            if not os.path.isfile(cached_model_path):
                os.makedirs(cache_dir, exist_ok=True)
                # download and save to disk
                print(f"Downloading MACE model from {checkpoint_url!r}")
                urllib.request.urlretrieve(checkpoint_url, cached_model_path)
                print(f"Cached MACE model to {cached_model_path}")
            model = cached_model_path
            msg = f"Using Materials Project MACE for MACECalculator with {model}"
            print(msg)
        except Exception as exc:
            raise RuntimeError(
                "Model download failed and no local model found"
            ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "float64":
        print(
            "Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization."
        )
    if dtype == "float32":
        print(
            "Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization."
        )
    mace_calc = MACECalculator(model_paths=model, device=device, dtype=dtype, **kwargs)
    if dispersion:
        gh_url = "https://github.com/pfnet-research/torch-dftd"
        try:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        except ImportError:
            raise RuntimeError(
                f"Please install torch-dftd to use dispersion corrections (see {gh_url})"
            )
        print(
            f"Using TorchDFTD3Calculator for D3 dispersion corrections (see {gh_url})"
        )
        d3_calc = TorchDFTD3Calculator(
            device=device,
            damping="bj",
            dtype=dtype,
            xc=dispersion_xc,
            cutoff=dispersion_cutoff,
            **kwargs,
        )
    calc = mace_calc if not dispersion else SumCalculator([mace_calc, d3_calc])
    return calc


def mace_off(
    model: Union[str, Path] = None,
    device: str = "",
    dtype: str = "float64",
    return_raw_model: bool = False,
    **kwargs,
) -> MACECalculator:
    """
    Constructs a MACECalculator with a pretrained model based on the MACE-OFF23 models.
    The model is released under the ASL license.
    Note:
        If you are using this function, please cite the relevant paper by Kovacs et.al., arXiv:2312.15211

    Args:
        model (str, optional): Path to the model. Defaults to None which first checks for
            a local model and then downloads the default medium model from https://github.com/ACEsuit/mace-off.
            Specify "small", "medium" or "large" to download a smaller or larger model.
        device (str, optional): Device to use for the model. Defaults to "cuda".
        dtype (str, optional): Default dtype for the model. Defaults to "float64".
        return_raw_model (bool, optional): Whether to return the raw model or an ASE calculator. Defaults to False.
        **kwargs: Passed to MACECalculator.

    Returns:
        MACECalculator: trained on the MACE-OFF23 dataset
    """
    if "default_dtype" in kwargs:
        dtype = kwargs.pop("default_dtype")
        warnings.warn(
            "default_dtype is deprecated, use dtype instead!", DeprecationWarning
        )
    try:
        urls = dict(
            small="https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true",
            medium="https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true",
            large="https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model?raw=true",
        )
        checkpoint_url = (
            urls.get(model, urls["medium"])
            if model in (None, "small", "medium", "large")
            else model
        )
        cache_dir = os.path.expanduser("~/.cache/mace")
        checkpoint_url_name = os.path.basename(checkpoint_url).split("?")[0]
        cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
        if not os.path.isfile(cached_model_path):
            os.makedirs(cache_dir, exist_ok=True)
            # download and save to disk
            print(f"Downloading MACE model from {checkpoint_url!r}")
            print(
                "By downloading the model you accept the ASL license, see https://github.com/gabor1/ASL"
            )
            urllib.request.urlretrieve(checkpoint_url, cached_model_path)
            print(f"Cached MACE model to {cached_model_path}")
        model = cached_model_path
        msg = f"Using MACE-OFF23 MODEL for MACECalculator with {model}"
        print(msg)
    except Exception as exc:
        raise RuntimeError("Model download failed") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if return_raw_model:
        return torch.load(model, map_location=device)

    if dtype == "float64":
        print(
            "Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization."
        )
    if dtype == "float32":
        print(
            "Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization."
        )
    mace_calc = MACECalculator(model_paths=model, device=device, dtype=dtype, **kwargs)
    return mace_calc


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
    return MACECalculator(model_path, device=device, dtype="float64")
