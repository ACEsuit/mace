import os
import urllib.request
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import torch
from ase import units
from ase.calculators.mixing import SumCalculator

from mace.tools.utils import get_cache_dir

from .mace import MACECalculator

module_dir = os.path.dirname(__file__)
local_model_path = os.path.join(
    module_dir, "foundations_models/mace-mpa-0-medium.model"
)

mace_mp_urls = {
    "small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model",
    "small-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model",
    "medium-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model",
    "small-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model",
    "medium-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
    "large-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
    "medium-0b3": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
    "medium-mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "small-omat-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model",
    "medium-omat-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "mace-matpes-pbe-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "mace-matpes-r2scan-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
}
mace_mp_names = [None] + list(mace_mp_urls.keys())


def download_mace_mp_checkpoint(model: Optional[Union[str, Path]] = None) -> str:
    """
    Downloads or locates the MACE-MP checkpoint file.

    Args:
        model (str, optional): Path to the model or size specification.
            Defaults to None which uses the medium model.

    Returns:
        str: Path to the downloaded (or cached, if previously loaded) checkpoint file.
    """
    if model in (None, "medium-mpa-0") and os.path.isfile(local_model_path):
        return local_model_path

    checkpoint_url = (
        mace_mp_urls.get(model, mace_mp_urls["medium-mpa-0"])
        if model in mace_mp_names
        else model
    )

    if checkpoint_url == mace_mp_urls["medium-mpa-0"]:
        print(
            "Using medium MPA-0 model as default MACE-MP model, to use previous (before 3.10) default model please specify 'medium' as model argument"
        )
    ASL_checkpoint_urls = {
        mace_mp_urls["small-omat-0"],
        mace_mp_urls["medium-omat-0"],
        mace_mp_urls["mace-matpes-pbe-0"],
        mace_mp_urls["mace-matpes-r2scan-0"],
    }
    if checkpoint_url in ASL_checkpoint_urls:
        print(
            "Using model under Academic Software License (ASL) license, see https://github.com/gabor1/ASL \n To use this model you accept the terms of the license."
        )

    cache_dir = get_cache_dir()
    checkpoint_url_name = "".join(
        c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
    )
    cached_model_path = f"{cache_dir}/{checkpoint_url_name}"

    if not os.path.isfile(cached_model_path):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Downloading MACE model from {checkpoint_url!r}")
        _, http_msg = urllib.request.urlretrieve(checkpoint_url, cached_model_path)
        if "Content-Type: text/html" in http_msg:
            raise RuntimeError(
                f"Model download failed, please check the URL {checkpoint_url}"
            )
        print(f"Cached MACE model to {cached_model_path}")

    return cached_model_path


@overload
def mace_mp(*, return_raw_model: Literal[True], **kwargs: Any) -> torch.nn.Module: ...


@overload
def mace_mp(
    *, return_raw_model: Literal[False] = False, **kwargs: Any
) -> MACECalculator: ...


def mace_mp(
    model: Optional[Union[str, Path]] = None,
    device: str = "",
    default_dtype: str = "float32",
    dispersion: bool = False,
    damping: str = "bj",  # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc: str = "pbe",
    dispersion_cutoff: float = 40.0 * units.Bohr,
    return_raw_model: bool = False,
    **kwargs,
) -> Union[MACECalculator, torch.nn.Module, SumCalculator]:
    """
    Constructs a MACECalculator with a pretrained model based on the Materials Project (89 elements).
    The model is released under the MIT license. See https://github.com/ACEsuit/mace-foundations for all models.
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
        device (str, optional): Device to use for the model. Defaults to "cuda" if available.
        default_dtype (str, optional): Default dtype for the model. Defaults to "float32".
        dispersion (bool, optional): Whether to use D3 dispersion corrections. Defaults to False.
        damping (str): The damping function associated with the D3 correction. Defaults to "bj" for D3(BJ).
        dispersion_xc (str, optional): Exchange-correlation functional for D3 dispersion corrections.
        dispersion_cutoff (float, optional): Cutoff radius in Bohr for D3 dispersion corrections.
        return_raw_model (bool, optional): Whether to return the raw model or an ASE calculator. Defaults to False.
        **kwargs: Passed to MACECalculator and TorchDFTD3Calculator.

    Returns:
        MACECalculator: trained on the MPtrj dataset (unless model otherwise specified).
    """
    try:
        if model in mace_mp_names or str(model).startswith("https:"):
            model_path = download_mace_mp_checkpoint(model)
            print(f"Using Materials Project MACE for MACECalculator with {model_path}")
        else:
            if not Path(model).exists():
                raise FileNotFoundError(f"{model} not found locally")
            model_path = model
    except Exception as exc:
        raise RuntimeError("Model download failed and no local model found") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if default_dtype == "float64":
        print(
            "Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization."
        )
    if default_dtype == "float32":
        print(
            "Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization."
        )

    if return_raw_model:
        return torch.load(model_path, map_location=device)

    mace_calc = MACECalculator(
        model_paths=model_path, device=device, default_dtype=default_dtype, **kwargs
    )

    if not dispersion:
        return mace_calc

    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as exc:
        raise RuntimeError(
            "Please install torch-dftd to use dispersion corrections (see https://github.com/pfnet-research/torch-dftd)"
        ) from exc

    print("Using TorchDFTD3Calculator for D3 dispersion corrections")
    dtype = torch.float32 if default_dtype == "float32" else torch.float64
    d3_calc = TorchDFTD3Calculator(
        device=device,
        damping=damping,
        dtype=dtype,
        xc=dispersion_xc,
        cutoff=dispersion_cutoff,
        **kwargs,
    )

    return SumCalculator([mace_calc, d3_calc])


@overload
def mace_off(*, return_raw_model: Literal[True], **kwargs: Any) -> torch.nn.Module: ...


@overload
def mace_off(
    *, return_raw_model: Literal[False] = False, **kwargs: Any
) -> MACECalculator: ...


def mace_off(
    model: Optional[Union[str, Path]] = None,
    device: str = "",
    default_dtype: str = "float64",
    return_raw_model: bool = False,
    **kwargs,
) -> Union[MACECalculator, torch.nn.Module]:
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
        if model in (None, "small", "medium", "large") or str(model).startswith(
            "https:"
        ):
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
            cache_dir = get_cache_dir()
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
        else:
            if not Path(model).exists():
                raise FileNotFoundError(f"{model} not found locally")
    except Exception as exc:
        raise RuntimeError("Model download failed and no local model found") from exc

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


@overload
def mace_anicc(
    *, return_raw_model: Literal[True], **kwargs: Any
) -> torch.nn.Module: ...


@overload
def mace_anicc(
    *, return_raw_model: Literal[False] = False, **kwargs: Any
) -> MACECalculator: ...


def mace_anicc(
    device: str = "cuda",
    model_path: Optional[str] = None,
    return_raw_model: bool = False,
) -> Union[MACECalculator, torch.nn.Module]:
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

    if not os.path.exists(model_path):
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        # Download the model
        print(f"Model not found at {model_path}. Downloading...")
        model_url = "https://github.com/ACEsuit/mace/raw/main/mace/calculators/foundations_models/ani500k_large_CC.model"

        try:

            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if total_size > 0:
                    print(
                        f"\rDownloading model: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)",
                        end="",
                    )

            urllib.request.urlretrieve(
                model_url, model_path, reporthook=report_progress
            )
            print("\nDownload complete!")

        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}") from e

    if return_raw_model:
        return torch.load(model_path, map_location=device)
    return MACECalculator(
        model_paths=model_path, device=device, default_dtype="float64"
    )


@overload
def mace_omol(*, return_raw_model: Literal[True], **kwargs: Any) -> torch.nn.Module: ...


@overload
def mace_omol(
    *, return_raw_model: Literal[False] = False, **kwargs: Any
) -> MACECalculator: ...


def mace_omol(
    model: Optional[Union[str, Path]] = None,
    device: str = "",
    default_dtype: str = "float64",
    return_raw_model: bool = False,
    **kwargs,
) -> Union[MACECalculator, torch.nn.Module]:
    """
    Constructs a MACECalculator with a pretrained model based on the MACE-OMOL models.
    The model is released under the ASL license.
    Note:
        If you are using this function, please cite the relevant OMOL paper.

    Args:
        model (str or Path, optional): Either a path to a local model file or a string specifier.
            Use "extra_large" or None to download the default OMOL model.
        device (str, optional): Device to use for the model. Defaults to "cuda" if available.
        default_dtype (str, optional): Default dtype for the model. Defaults to "float64".
        return_raw_model (bool, optional): Whether to return the raw model or an ASE calculator. Defaults to False.
        **kwargs: Passed to MACECalculator.

    Returns:
        MACECalculator: trained on the OMOL dataset.
    """
    urls = {
        "extra_large": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model"
    }

    try:
        if model is None or model == "extra_large":
            checkpoint_url = urls["extra_large"]
        elif isinstance(model, str) and model.startswith("https:"):
            checkpoint_url = model
        elif isinstance(model, (str, Path)) and Path(model).exists():
            checkpoint_url = str(model)
        else:
            raise ValueError(
                f"Invalid model specification: {model}. "
                f"Supported options: {list(urls.keys())}, a local file path, or a direct URL."
            )

        if checkpoint_url.startswith("http"):
            cache_dir = get_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            checkpoint_url_name = os.path.basename(checkpoint_url).split("?")[0]
            cached_model_path = os.path.join(cache_dir, checkpoint_url_name)

            if not os.path.isfile(cached_model_path):
                print(f"Downloading MACE model from {checkpoint_url!r}")
                print(
                    "The model is distributed under the Academic Software License (ASL), see https://github.com/gabor1/ASL\n"
                    "To use the model, you accept the terms of the license.\n"
                    "ASL is based on the GNU Public License, but does not permit commercial use."
                )
                urllib.request.urlretrieve(checkpoint_url, cached_model_path)
                print(f"Cached MACE model to {cached_model_path}")
            model = cached_model_path
        else:
            model = checkpoint_url

    except Exception as exc:
        raise RuntimeError("Model download failed and no local model found") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if return_raw_model:
        return torch.load(model, map_location=device)

    if default_dtype == "float64":
        print(
            "Using float64 for MACECalculator, recommended for geometry optimization."
        )
    elif default_dtype == "float32":
        print("Using float32 for MACECalculator, recommended for MD.")

    return MACECalculator(
        model_paths=model,
        device=device,
        default_dtype=default_dtype,
        **kwargs,
        head="omol",
    )
