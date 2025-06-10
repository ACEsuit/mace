"""
This module contains the AseDBDataset class and its dependencies.
It is extracted from the fairchem codebase and adapted to remove dependencies on fairchem.

Original code copyright:
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import bisect
import logging
import numbers
import os
import zlib
from abc import ABC, abstractmethod

try:
    from functools import cache, cached_property
except ImportError:
    from functools import cached_property, lru_cache

    cache = lru_cache(maxsize=None)
from glob import glob
from pathlib import Path
from typing import Any, Callable, TypeVar

import ase
import ase.db.core
import ase.db.row
import ase.io
import lmdb
import numpy as np
import orjson
import torch

# Type variable for generic dataset return type
T_co = TypeVar("T_co", covariant=True)


def _decode_ndarrays(obj):
    """Recursively turn {"__ndarray__": [...] } blobs back into NumPy arrays."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            shape, dtype, flat = obj["__ndarray__"]
            return np.asarray(flat, dtype=dtype).reshape(shape)
        # recurse into dict values
        return {k: _decode_ndarrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_ndarrays(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_decode_ndarrays(v) for v in obj)
    return obj  # everything else is left untouched


def rename_data_object_keys(data_object, key_mapping: dict[str, str | list[str]]):
    """Rename data object keys

    Args:
        data_object: data object
        key_mapping: dictionary specifying keys to rename and new names {prev_key: new_key}

        new_key can be a list of new keys, for example,
        prev_key: energy
        new_key: [common_energy, oc20_energy]

        This is currently required when we use a single target/label for multiple tasks
    """
    for _property in key_mapping:
        # catch for test data not containing labels
        if _property in data_object:
            list_of_new_keys = key_mapping[_property]
            if isinstance(list_of_new_keys, str):
                list_of_new_keys = [list_of_new_keys]
            for new_property in list_of_new_keys:
                if new_property == _property:
                    continue
                assert new_property not in data_object
                data_object[new_property] = data_object[_property]
            if _property not in list_of_new_keys:
                del data_object[_property]
    return data_object


def apply_one_tags(
    atoms: ase.Atoms, skip_if_nonzero: bool = True, skip_always: bool = False
):
    """
    This function will apply tags of 1 to an ASE atoms object.
    It is used as an atoms_transform in the datasets contained in this file.

    Certain models will treat atoms differently depending on their tags.
    For example, GemNet-OC by default will only compute triplet and quadruplet interactions
    for atoms with non-zero tags. This model throws an error if there are no tagged atoms.
    For this reason, the default behavior is to tag atoms in structures with no tags.

    args:
        skip_if_nonzero (bool): If at least one atom has a nonzero tag, do not tag any atoms

        skip_always (bool): Do not apply any tags. This arg exists so that this function can be disabled
                without needing to pass a callable (which is currently difficult to do with main.py)
    """
    if skip_always:
        return atoms

    if np.all(atoms.get_tags() == 0) or not skip_if_nonzero:
        atoms.set_tags(np.ones(len(atoms)))

    return atoms


class UnsupportedDatasetError(ValueError):
    pass


class BaseDataset(ABC):
    """Base Dataset class for all ASE datasets."""

    def __init__(self, config: dict):
        """Initialize

        Args:
            config (dict): dataset configuration
        """
        self.config = config
        self.paths = []

        if "src" in self.config:
            if isinstance(config["src"], str):
                self.paths = [Path(self.config["src"])]
            else:
                self.paths = tuple(Path(path) for path in sorted(config["src"]))

        self.lin_ref = None
        if self.config.get("lin_ref", False):
            lin_ref = torch.tensor(
                np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            )
            self.lin_ref = torch.nn.Parameter(lin_ref, requires_grad=False)

    def __len__(self) -> int:
        return self.num_samples

    def metadata_hasattr(self, attr) -> bool:
        return attr in self._metadata

    @cached_property
    def indices(self):
        return np.arange(self.num_samples, dtype=int)

    @cached_property
    def _metadata(self) -> dict[str, np.ndarray]:
        # logic to read metadata file here
        metadata_npzs = []
        if self.config.get("metadata_path", None) is not None:
            metadata_npzs.append(
                np.load(self.config["metadata_path"], allow_pickle=True)
            )

        else:
            for path in self.paths:
                if path.is_file():
                    metadata_file = path.parent / "metadata.npz"
                else:
                    metadata_file = path / "metadata.npz"
                if metadata_file.is_file():
                    metadata_npzs.append(np.load(metadata_file, allow_pickle=True))

        if len(metadata_npzs) == 0:
            logging.warning(
                f"Could not find dataset metadata.npz files in '{self.paths}'"
            )
            return {}

        metadata = {
            field: np.concatenate([metadata[field] for metadata in metadata_npzs])
            for field in metadata_npzs[0]
        }

        assert np.issubdtype(
            metadata["natoms"].dtype, np.integer
        ), f"Metadata natoms must be an integer type! not {metadata['natoms'].dtype}"
        assert metadata["natoms"].shape[0] == len(
            self
        ), "Loaded metadata and dataset size mismatch."

        return metadata

    def get_metadata(self, attr, idx):
        if attr in self._metadata:
            metadata_attr = self._metadata[attr]
            if isinstance(idx, list):
                return [metadata_attr[_idx] for _idx in idx]
            return metadata_attr[idx]
        return None


class Subset(BaseDataset):
    """A subset that also takes metadata if given."""

    def __init__(
        self,
        dataset: BaseDataset,
        indices: list[int],
        metadata: dict[str, np.ndarray],
    ) -> None:
        super().__init__(dataset.config)
        self.dataset = dataset
        self.metadata = metadata
        self.indices = indices
        self.num_samples = len(indices)
        self.config = dataset.config

    @cached_property
    def _metadata(self) -> dict[str, np.ndarray]:
        return self.dataset._metadata  # pylint: disable=protected-access

    def get_metadata(self, attr, idx):
        if isinstance(idx, list):
            return self.dataset.get_metadata(attr, [[self.indices[i] for i in idx]])
        return self.dataset.get_metadata(attr, self.indices[idx])


class LMDBDatabase(ase.db.core.Database):
    """
    This module is modified from the ASE db json backend
    and is thus licensed under the corresponding LGPL2.1 license.

    The ASE notice for the LGPL2.1 license is available here:
    https://gitlab.com/ase/ase/-/blob/master/LICENSE
    """

    def __init__(  # pylint: disable=keyword-arg-before-vararg
        self,
        filename: str | Path | None = None,
        create_indices: bool = True,
        use_lock_file: bool = False,
        serial: bool = False,
        readonly: bool = False,  # Moved after *args to make it keyword-only
        *args,
        **kwargs,
    ) -> None:
        """
        For the most part, this is identical to the standard ase db initiation
        arguments, except that we add a readonly flag.
        """
        super().__init__(
            Path(filename),
            create_indices,
            use_lock_file,
            serial,
            *args,
            **kwargs,
        )

        # Add a readonly mode for when we're only training
        # to make sure there's no parallel locks
        self.readonly = readonly

        if self.readonly:
            # Open a new env
            self.env = lmdb.open(
                str(self.filename),
                subdir=False,
                meminit=False,
                map_async=True,
                readonly=True,
                lock=False,
            )

            # Open a transaction and keep it open for fast read/writes!
            self.txn = self.env.begin(write=False)

        else:
            # Open a new env with write access
            self.env = lmdb.open(
                str(self.filename),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )

            self.txn = self.env.begin(write=True)

        # Load all ids based on keys in the DB.
        self.ids = []
        self.deleted_ids = []
        self._load_ids()

    def __enter__(self) -> "LMDBDatabase":
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.close()

    def close(self) -> None:
        # Close the lmdb environment and transaction
        self.txn.commit()
        self.env.close()

    def _write(
        self,
        atoms: ase.Atoms | ase.db.row.AtomsRow,
        key_value_pairs: dict,
        data: dict | None,
        id: int | None = None,  # pylint: disable=redefined-builtin
    ) -> None:

        # 1) dump the entire atoms.info dict into key_value_pairs
        key_value_pairs = dict(key_value_pairs or {}, **atoms.info)
        scalar_types = (numbers.Real, str, bool, np.bool_)
        key_value_pairs = {
            k: v
            for k, v in (key_value_pairs or {}).items()
            if isinstance(v, scalar_types)
        }

        if data is None:
            data = {}
        for k, v in atoms.info.items():
            if isinstance(v, scalar_types):
                key_value_pairs[k] = v
            else:
                # If the value is not serializable, we store it in data
                data.setdefault("__info__", {})[k] = v
        arrays_to_dump = {}
        for name, arr in atoms.arrays.items():
            if name not in (
                "numbers",
                "positions",
                "tags",
                "momenta",
                "masses",
                "charges",
                "magmoms",
                "velocities",
            ):
                arrays_to_dump[name] = arr
        if arrays_to_dump:
            data.setdefault("__arrays__", {}).update(arrays_to_dump)

        # 3) also save all extra calculator results (if any)
        if hasattr(atoms, "calc") and getattr(atoms.calc, "results", None):
            for k, v in atoms.calc.results.items():
                if k in ("energy", "forces", "stress", "free_energy"):
                    continue  # ASE already stores these
            data.setdefault(k, v)
        # Call parent method with the original parameter name
        super()._write(atoms, key_value_pairs, data)

        mtime = ase.db.core.now()

        if isinstance(atoms, ase.db.row.AtomsRow):
            row = atoms
        else:
            row = ase.db.row.AtomsRow(atoms)
            row.ctime = mtime
            row.user = os.getenv("USER")

        dct = {}
        for key in row.__dict__:
            # Use getattr to avoid accessing protected member directly
            if key[0] == "_" or key == "id" or key in getattr(row, "_keys", []):
                continue
            dct[key] = row[key]

        dct["mtime"] = mtime

        if key_value_pairs:
            dct["key_value_pairs"] = key_value_pairs

        if data:
            dct["data"] = data

        constraints = row.get("constraints")
        if constraints:
            dct["constraints"] = [constraint.todict() for constraint in constraints]

        # json doesn't like Cell objects, so make it an array
        dct["cell"] = np.asarray(dct["cell"])

        if id is None:
            id = self._nextid
            nextid = id + 1
        else:
            data = self.txn.get(f"{id}".encode("ascii"))
            assert data is not None

        # Add the new entry
        self.txn.put(
            f"{id}".encode("ascii"),
            zlib.compress(orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY)),
        )
        # only append if idx is not in ids
        if id not in self.ids:
            self.ids.append(id)
            self.txn.put(
                "nextid".encode("ascii"),
                zlib.compress(orjson.dumps(nextid, option=orjson.OPT_SERIALIZE_NUMPY)),
            )
        # check if id is in removed ids and remove accordingly
        if id in self.deleted_ids:
            self.deleted_ids.remove(id)
            self._write_deleted_ids()

        return id

    def _update(
        self,
        idx: int,
        key_value_pairs: dict | None = None,
        data: dict | None = None,
    ):
        # hack this to play nicely with ASE code
        row = self._get_row(idx, include_data=True)
        if data is not None or key_value_pairs is not None:
            self._write(
                atoms=row, key_value_pairs=key_value_pairs, data=data, id=idx
            )  # Fixed E1123 by using id=idx

    def _write_deleted_ids(self):
        self.txn.put(
            "deleted_ids".encode("ascii"),
            zlib.compress(
                orjson.dumps(self.deleted_ids, option=orjson.OPT_SERIALIZE_NUMPY)
            ),
        )

    def delete(self, ids: list[int]) -> None:
        for idx in ids:
            self.txn.delete(f"{idx}".encode("ascii"))
            self.ids.remove(idx)

        self.deleted_ids += ids
        self._write_deleted_ids()

    def _get_row(self, idx: int, include_data: bool = True):
        if idx is None:
            assert len(self.ids) == 1
            idx = self.ids[0]
        data = self.txn.get(f"{idx}".encode("ascii"))

        if data is not None:
            dct = orjson.loads(zlib.decompress(data))
        else:
            raise KeyError(f"Id {idx} missing from the database!")

        if not include_data:
            dct.pop("data", None)

        dct["id"] = idx
        return ase.db.row.AtomsRow(dct)

    def _get_row_by_index(self, index: int, include_data: bool = True):
        """Auxiliary function to get the ith entry, rather than a specific id"""
        data = self.txn.get(f"{self.ids[index]}".encode("ascii"))

        if data is not None:
            dct = orjson.loads(zlib.decompress(data))
        else:
            raise KeyError(f"Id {id} missing from the database!")

        if not include_data:
            dct.pop("data", None)

        dct["id"] = id
        return ase.db.row.AtomsRow(dct)

    def _select(
        self,
        keys,
        cmps: list[tuple[str, str, str]],
        explain: bool = False,
        _verbosity: int = 0,  # Unused parameter marked with underscore
        limit: int | None = None,
        offset: int = 0,
        sort: str | None = None,
        include_data: bool = True,
        _columns: str = "all",  # Unused parameter marked with underscore
    ):
        if explain:
            yield {"explain": (0, 0, 0, "scan table")}
            return

        if sort is not None:
            if sort[0] == "-":
                reverse = True
                sort = sort[1:]
            else:
                reverse = False

            rows = []
            missing = []
            for row in self._select(keys, cmps):
                key = row.get(sort)
                if key is None:
                    missing.append((0, row))
                else:
                    rows.append((key, row))

            rows.sort(reverse=reverse, key=lambda x: x[0])
            rows += missing

            if limit:
                rows = rows[offset : offset + limit]
            for _, row in rows:
                yield row
            return

        if not limit:
            limit = -offset - 1

        cmps = [(key, ase.db.core.ops[op], val) for key, op, val in cmps]
        n = 0
        for idx in self.ids:
            if n - offset == limit:
                return
            row = self._get_row(idx, include_data=include_data)

            for key in keys:
                if key not in row:
                    break
            else:
                for key, op, val in cmps:
                    if isinstance(key, int):
                        value = np.equal(row.numbers, key).sum()
                    else:
                        value = row.get(key)
                        if key == "pbc":
                            assert op in [ase.db.core.ops["="], ase.db.core.ops["!="]]
                            value = "".join("FT"[x] for x in value)
                    if value is None or not op(value, val):
                        break
                else:
                    if n >= offset:
                        yield row
                    n += 1

    @property
    def metadata(self):
        """Override abstract metadata method from Database class."""
        return self.db_metadata

    @property
    def db_metadata(self):
        """Load the metadata from the DB if present"""
        if self._metadata is None:
            metadata = self.txn.get("metadata".encode("ascii"))
            if metadata is None:
                self._metadata = {}
            else:
                self._metadata = orjson.loads(zlib.decompress(metadata))

        return self._metadata.copy()

    @db_metadata.setter
    def db_metadata(self, dct):
        self._metadata = dct

        # Put the updated metadata dictionary
        self.txn.put(
            "metadata".encode("ascii"),
            zlib.compress(orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY)),
        )

    @property
    def _nextid(self):
        """Get the id of the next row to be written"""
        # Get the nextid
        nextid_data = self.txn.get("nextid".encode("ascii"))
        if nextid_data:
            return orjson.loads(zlib.decompress(nextid_data))
        return 1  # Removed unnecessary else (R1705)

    def count(self, selection=None, **kwargs) -> int:
        """Count rows.

        See the select() method for the selection syntax.  Use db.count() or
        len(db) to count all rows.
        """
        if selection is not None:
            n = 0
            for _row in self.select(selection, **kwargs):
                n += 1
            return n
        return len(self.ids)

    def _load_ids(self) -> None:
        """Load ids from the DB

        Since ASE db ids are mostly 1-N integers, but can be missing entries
        if ids have been deleted. To save space and operating under the assumption
        that there will probably not be many deletions in most OCP datasets,
        we just store the deleted ids.
        """
        # Load the deleted ids
        deleted_ids_data = self.txn.get("deleted_ids".encode("ascii"))
        if deleted_ids_data is not None:
            self.deleted_ids = orjson.loads(zlib.decompress(deleted_ids_data))

        # Reconstruct the full id list
        self.ids = [i for i in range(1, self._nextid) if i not in set(self.deleted_ids)]


# Placeholder for AtomsToGraphs class
# This is a minimal implementation without the full functionality
class AtomsToGraphs:
    """Enhanced AtomsToGraphs implementation with proper property handling."""

    def __init__(
        self,
        r_edges=False,
        r_pbc=True,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_data_keys=None,
        **kwargs,
    ):
        self.r_edges = r_edges
        self.r_pbc = r_pbc
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_stress = r_stress
        self.r_data_keys = r_data_keys or {}
        self.kwargs = kwargs

    def convert(self, atoms, sid=None):
        """
        Convert ASE atoms to graph data format with proper property handling.
        """
        from mace.tools.torch_geometric.data import Data

        # Create a minimal data object with required properties
        data = Data()

        # Set positions
        data.pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

        # Set atomic numbers
        data.atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

        # Set cell if available
        if atoms.cell is not None:
            data.cell = torch.tensor(atoms.get_cell(), dtype=torch.float)

        # Set PBC if requested
        if self.r_pbc:
            data.pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool)

        # Set energy if requested
        if self.r_energy:
            energy = self._get_property(atoms, "energy")
            if energy is not None:
                data.energy = torch.tensor(energy, dtype=torch.float)

        # Set forces if requested
        if self.r_forces:
            forces = self._get_property(atoms, "forces")
            if forces is not None:
                data.forces = torch.tensor(forces, dtype=torch.float)

        # Set stress if requested
        if self.r_stress:
            stress = self._get_property(atoms, "stress")
            if stress is not None:
                data.stress = torch.tensor(stress, dtype=torch.float)

        # Set sid if provided
        if sid is not None:
            data.sid = sid

        return data

    def _get_property(self, atoms, prop_name):
        """Get property from atoms, checking custom names first then standard methods."""
        # Check if we have a custom name for this property
        custom_name = self.r_data_keys.get(prop_name)

        # Try custom name in info dict
        if custom_name and custom_name in atoms.info:
            return atoms.info[custom_name]

        # Try custom name in arrays dict
        if custom_name and custom_name in atoms.arrays:
            return atoms.arrays[custom_name]

        # Try standard name in info dict
        if prop_name in atoms.info:
            return atoms.info[prop_name]

        # Try standard name in arrays dict
        if prop_name in atoms.arrays:
            return atoms.arrays[prop_name]

        # Try standard ASE methods
        method_map = {
            "energy": "get_potential_energy",
            "forces": "get_forces",
            "stress": "get_stress",
        }

        if prop_name in method_map and hasattr(atoms, method_map[prop_name]):
            try:
                method = getattr(atoms, method_map[prop_name])
                return method()
            except (
                AttributeError,
                RuntimeError,
            ) as exc:  # Fixed W0718 by specifying exceptions
                logging.debug(f"Error getting property {prop_name}: {exc}")
                # Removed unnecessary pass (W0107)

        return None


# Placeholder for DataTransforms class
class DataTransforms:
    """Minimal implementation of DataTransforms to satisfy dependencies."""

    def __init__(self, transforms_config=None):
        self.transforms_config = transforms_config or {}

    def __call__(self, data):
        """Apply transforms to data"""
        # No transforms applied in this minimal implementation
        return data


class AseAtomsDataset(BaseDataset, ABC):
    """
    This is an abstract Dataset that includes helpful utilities for turning
    ASE atoms objects into OCP-usable data objects. This should not be instantiated directly
    as get_atoms_object and load_dataset_get_ids are not implemented in this base class.

    Derived classes must add at least two things:
        self.get_atoms_object(id): a function that takes an identifier and returns a corresponding atoms object

        self.load_dataset_get_ids(config: dict): This function is responsible for any initialization/loads
            of the dataset and importantly must return a list of all possible identifiers that can be passed into
            self.get_atoms_object(id)

    Identifiers need not be any particular type.
    """

    def __init__(
        self,
        config: dict,
        atoms_transform: Callable[[ase.Atoms, Any], ase.Atoms] = apply_one_tags,
    ) -> None:
        super().__init__(config)

        a2g_args = config.get("a2g_args", {}) or {}

        # set default to False if not set by user, assuming otf_graph will be used
        if "r_edges" not in a2g_args:
            a2g_args["r_edges"] = False

        # Make sure we always include PBC info in the resulting atoms objects
        a2g_args["r_pbc"] = True
        self.a2g = AtomsToGraphs(**a2g_args)

        self.key_mapping = self.config.get("key_mapping", None)
        self.transforms = DataTransforms(self.config.get("transforms", {}))

        self.atoms_transform = atoms_transform

        if self.config.get("keep_in_memory", False):
            self.__getitem__ = cache(self.__getitem__)

        self.ids = self._load_dataset_get_ids(config)
        self.num_samples = len(self.ids)

        if len(self.ids) == 0:
            raise ValueError(
                rf"No valid ase data found! \n"
                f"Double check that the src path and/or glob search pattern gives ASE compatible data: {config['src']}"
            )

    def __getitem__(self, idx):  # pylint: disable=method-hidden
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        # Get atoms object via derived class method
        atoms = self.get_atoms(self.ids[idx])

        # Transform atoms object
        if self.atoms_transform is not None:
            atoms = self.atoms_transform(
                atoms, **self.config.get("atoms_transform_args", {})
            )

        sid = atoms.info.get("sid", self.ids[idx])
        fid = atoms.info.get("fid", torch.tensor([0]))

        # Convert to data object
        data_object = self.a2g.convert(atoms, sid)
        data_object.fid = fid
        data_object.natoms = len(atoms)

        # apply linear reference
        if self.a2g.r_energy is True and self.lin_ref is not None:
            data_object.energy -= sum(self.lin_ref[data_object.atomic_numbers.long()])

        # Transform data object
        data_object = self.transforms(data_object)

        if self.key_mapping is not None:
            data_object = rename_data_object_keys(data_object, self.key_mapping)

        if self.config.get("include_relaxed_energy", False):
            data_object.energy_relaxed = self.get_relaxed_energy(self.ids[idx])

        return data_object

    @abstractmethod
    def get_atoms(self, idx: str | int) -> ase.Atoms:
        # This function should return an ASE atoms object.
        raise NotImplementedError(
            "Returns an ASE atoms object. Derived classes should implement this function."
        )

    @abstractmethod
    def _load_dataset_get_ids(self, config):
        # This function should return a list of ids that can be used to index into the database
        raise NotImplementedError(
            "Every ASE dataset needs to declare a function to load the dataset and return a list of ids."
        )

    def get_relaxed_energy(self, identifier):
        raise NotImplementedError(
            "Reading relaxed energy from trajectory or file is not implemented with this dataset. "
            "If relaxed energies are saved with the atoms info dictionary, they can be used by passing the keys in "
            "the r_data_keys argument under a2g_args."
        )

    def get_metadata(self, attr, idx):
        # try the parent method
        metadata = super().get_metadata(attr, idx)
        if metadata is not None:
            return metadata
        # try to resolve it here
        if attr != "natoms":
            return None
        if isinstance(idx, (list, np.ndarray)):
            return np.array([self.get_metadata(attr, i) for i in idx])
        return len(self.get_atoms(idx))


class AseDBDataset(AseAtomsDataset):
    """
    This Dataset connects to an ASE Database, allowing the storage of atoms objects
    with a variety of backends including JSON, SQLite, and database server options.
    """

    def _load_dataset_get_ids(self, config: dict) -> list[int]:
        if isinstance(config["src"], list):
            filepaths = []
            for path in sorted(config["src"]):
                if os.path.isdir(path):
                    filepaths.extend(sorted(glob(f"{path}/*")))
                elif os.path.isfile(path):
                    filepaths.append(path)
                else:
                    raise RuntimeError(f"Error reading dataset in {path}!")
        elif os.path.isfile(config["src"]):
            filepaths = [config["src"]]
        elif os.path.isdir(config["src"]):
            filepaths = sorted(glob(f'{config["src"]}/*'))
        else:
            filepaths = sorted(glob(config["src"]))

        self.dbs = []

        for path in filepaths:
            try:
                self.dbs.append(self.connect_db(path, config.get("connect_args", {})))
            except ValueError:
                logging.debug(
                    f"Tried to connect to {path} but it's not an ASE database!"
                )

        self.select_args = config.get("select_args", {})
        if self.select_args is None:
            self.select_args = {}

        # Get all unique IDs from the databases
        self.db_ids = []
        for db in self.dbs:
            if hasattr(db, "ids") and self.select_args == {}:
                self.db_ids.append(db.ids)
            else:
                # this is the slow alternative
                self.db_ids.append([row.id for row in db.select(**self.select_args)])

        idlens = [len(ids) for ids in self.db_ids]
        self._idlen_cumulative = np.cumsum(idlens).tolist()

        return list(range(sum(idlens)))

    def get_atoms(self, idx: int) -> ase.Atoms:
        """
        Return an `ase.Atoms` object for the dataset entry `idx`, decoding any
        JSON‐encoded ndarrays encountered anywhere in the row.
        """
        # ------------------------------------------------------------------ #
        # 1.  Locate the correct database and row                            #
        # ------------------------------------------------------------------ #
        db_idx = bisect.bisect(self._idlen_cumulative, idx)
        local_idx = idx - self._idlen_cumulative[db_idx - 1] if db_idx else idx
        row = self.get_row_from_db(db_idx, local_idx)

        # ------------------------------------------------------------------ #
        # 2.  Fast path if ASE can already parse the row natively            #
        # ------------------------------------------------------------------ #
        if not (isinstance(row.numbers, dict) and "__ndarray__" in row.numbers):
            atoms = row.toatoms()
        else:
            # -------------------------------------------------------------- #
            # 3.  Decode *everything* that might hide __ndarray__ blobs      #
            # -------------------------------------------------------------- #
            atom_numbers = _decode_ndarrays(row.numbers)
            positions = _decode_ndarrays(row.positions)
            cell = _decode_ndarrays(getattr(row, "cell", None))
            pbc = _decode_ndarrays(getattr(row, "pbc", None))

            atoms = ase.Atoms(
                numbers=atom_numbers,
                positions=positions,
                cell=cell if cell is not None else None,
                pbc=pbc if pbc is not None else None,
            )

        # ------------------------------------------------------------------ #
        # 4.  Row-level dictionaries (data / key_value_pairs) – deep decode  #
        # ------------------------------------------------------------------ #
        data_dict = _decode_ndarrays(row.data) if isinstance(row.data, dict) else {}
        kvp_dict = (
            _decode_ndarrays(row.key_value_pairs)
            if getattr(row, "key_value_pairs", None)
            else {}
        )

        atoms.info.update(data_dict)
        atoms.info.update(kvp_dict)

        # ------------------------------------------------------------------ #
        # 5.  Energy, forces, stress → atoms.calc (decode if needed)         #
        # ------------------------------------------------------------------ #
        calc_kwargs = {}
        for prop in ("energy", "forces", "stress", "free_energy"):
            val = getattr(row, prop, None)
            if val is not None:
                calc_kwargs[prop] = _decode_ndarrays(val)
                atoms.info[prop] = calc_kwargs[prop]

        if calc_kwargs:
            from ase.calculators.singlepoint import SinglePointCalculator

            atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)

        # ------------------------------------------------------------------ #
        # 6.  Extra arrays & info stored under __arrays__ / __info__         #
        # ------------------------------------------------------------------ #
        extra_arrays = data_dict.pop("__arrays__", {})
        for name, arr in extra_arrays.items():
            atoms.new_array(name, np.asarray(arr))  # already decoded above

        extra_info = data_dict.pop("__info__", {})
        atoms.info.update(extra_info)

        # ------------------------------------------------------------------ #
        # 7.  Respect any user-defined r_data_keys renamings                 #
        # ------------------------------------------------------------------ #
        a2g_args = self.config.get("a2g_args", {}) or {}
        r_data_keys = a2g_args.get("r_data_keys", {})
        for custom, standard in r_data_keys.items():
            if standard in atoms.info:
                atoms.info[custom] = atoms.info[standard]
            elif standard in atoms.arrays:
                atoms.arrays[custom] = atoms.arrays[standard]

        return atoms

    def get_row_from_db(self, db_idx, el_idx):
        """Get a row from the database at the given indices."""
        db = self.dbs[db_idx]
        row_id = self.db_ids[db_idx][el_idx]
        if isinstance(db, LMDBDatabase):
            return db._get_row(row_id)  # pylint: disable=protected-access
        return db.get(row_id)

    @staticmethod
    def connect_db(
        address: str | Path, connect_args: dict | None = None
    ) -> ase.db.core.Database:
        if connect_args is None:
            connect_args = {}
        db_type = connect_args.get("type", "extract_from_name")
        if db_type in ("lmdb", "aselmdb") or (
            db_type == "extract_from_name"
            and str(address).rsplit(".", maxsplit=1)[-1] in ("lmdb", "aselmdb")
        ):
            return LMDBDatabase(address, readonly=True, **connect_args)

        return ase.db.connect(address, **connect_args)

    def __del__(self):
        for db in self.dbs:
            if hasattr(db, "close"):
                db.close()

    def sample_property_metadata(
        self,
    ) -> dict:  # Removed unused argument num_samples (W0613)
        """
        Sample property metadata from the database.

        This method was previously using the copy module which is now removed.
        """
        logging.warning(
            "You specified a folder of ASE dbs, so it's impossible to know which metadata to use. Using the first!"
        )
        if self.dbs[0].metadata == {}:
            return {}

        # Fixed unnecessary comprehension (R1721)
        return dict(self.dbs[0].metadata.items())
