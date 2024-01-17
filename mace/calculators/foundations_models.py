import os
import urllib.request
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
    default_dtype: str = "float32",
    dispersion: bool = False,
    dispersion_xc="pbe",
    dispersion_cutoff=40.0 * units.Bohr,
    **kwargs,
) -> MACECalculator:
    """
    Constructs a MACECalculator with a pretrained model based on the Materials Project (89 elements).
    The model is released under the MIT license. See https://github.com/ACEsuit/mace-mp for all models.
    Note:
        If you are using this function, please cite the relevant paper for the Materials Project,
        any paper associated with the MACE model, and also the following:
        - MACE-MP by Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena,
            Dávid P. Kovács, Janosh Riebesell, et al., 2023, arXiv:2401.00096
        - MACE-Universal by Yuan Chiang, 2023, Hugging Face, Revision e5ebd9b,
            DOI: 10.57967/hf/1202, URL: https://huggingface.co/cyrusyc/mace-universal
        - Matbench Discovery by Janosh Riebesell, Rhys EA Goodall, Philipp Benner, Yuan Chiang,
            Alpha A Lee, Anubhav Jain, Kristin A Persson, 2023, arXiv:2308.14920

    Args:
        model (str, optional): Path to the model. Defaults to None which first checks for
            a local model and then downloads the default model from figshare. Specify "small",
            "medium" or "large" to download a smaller or larger model from figshare.
        device (str, optional): Device to use for the model. Defaults to "cuda".
        default_dtype (str, optional): Default dtype for the model. Defaults to "float32".
        dispersion (bool, optional): Whether to use D3 dispersion corrections. Defaults to False.
        dispersion_xc (str, optional): Exchange-correlation functional for D3 dispersion corrections.
        dispersion_cutoff (float, optional): Cutoff radius in Bhor for D3 dispersion corrections.
        **kwargs: Passed to MACECalculator and TorchDFTD3Calculator.

    Returns:
        MACECalculator: trained on the MPtrj dataset (unless model otherwise specified).
    """
    if model in (None, "medium") and os.path.isfile(local_model_path):
        model = local_model_path
        print(
            f"Using local medium Materials Project MACE model for MACECalculator {model}"
        )
    elif model in (None, "small", "medium", "large") or str(model).startswith("https:"):
        try:
            urls = dict(
                small="http://tinyurl.com/46jrkm3v",  # 2023-12-10-mace-128-L0_energy_epoch-249.model
                medium="http://tinyurl.com/5yyxdm76",  # 2023-12-03-mace-128-L1_epoch-199.model
                large="http://tinyurl.com/5f5yavf3",  # MACE_MPtrj_2022.9.model
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
                _, http_msg = urllib.request.urlretrieve(
                    checkpoint_url, cached_model_path
                )
                if "Content-Type: text/html" in http_msg:
                    raise RuntimeError(
                        f"Model download failed, please check the URL {checkpoint_url}"
                    )
                print(f"Cached MACE model to {cached_model_path}")
            model = cached_model_path
            msg = f"Using Materials Project MACE for MACECalculator with {model}"
            print(msg)
        except Exception as exc:
            raise RuntimeError(
                "Model download failed and no local model found"
            ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if default_dtype == "float64":
        print(
            "Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization."
        )
    if default_dtype == "float32":
        print(
            "Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization."
        )
    mace_calc = MACECalculator(
        model_paths=model, device=device, default_dtype=default_dtype, **kwargs
    )
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
        dtype = torch.float32 if default_dtype == "float32" else torch.float64
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
    default_dtype: str = "float64",
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
        default_dtype (str, optional): Default dtype for the model. Defaults to "float64".
        return_raw_model (bool, optional): Whether to return the raw model or an ASE calculator. Defaults to False.
        **kwargs: Passed to MACECalculator.

    Returns:
        MACECalculator: trained on the MACE-OFF23 dataset
    """
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
                "The model is distributed under the Academic Software License (ASL) license, see https://github.com/gabor1/ASL \n To use the model you accept the terms of the license."
            )
            print(
                "ASL is based on the Gnu Public License, but does not permit commercial use"
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

    if default_dtype == "float64":
        print(
            "Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization."
        )
    if default_dtype == "float32":
        print(
            "Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization."
        )
    mace_calc = MACECalculator(
        model_paths=model, device=device, default_dtype=default_dtype, **kwargs
    )
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
    return MACECalculator(model_path, device=device, default_dtype="float64")
