Heterogeneous Data Training
=============================================================

MACE supports loading multiple files for training, validation, and testing, with flexible support for different file formats within a single training run. This capability is particularly useful when working with data from multiple sources or when your dataset is split across several files.

Supported File Formats
----------------------

MACE can handle a mix of the following file formats:

- **ASE files**: Standard ASE-readable atomic structure files
- **HDF5 files**: Pre-processed data in ``.h5`` or ``.hdf5`` format
- **LMDB databases**: Efficient storage and access for large datasets

Configuration
-------------

Basic Usage
~~~~~~~~~~~

You can provide multiple files as a list in your configuration:

.. code-block:: yaml

    heads:
      DFT:
        train_file: ["path/to/training1.xyz", "path/to/training2.xyz"]
        valid_file: ["path/to/validation1.xyz", "path/to/validation2/"]
        test_file: "path/to/test.xyz"

Mixing Different File Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MACE automatically detects file types and uses the appropriate loader:

.. code-block:: yaml

    heads:
      QM7:
        train_file: ["data/qm7/train.xyz", "data/qm7/train.h5", "data/qm7/database/"]
        valid_file: ["data/qm7/valid.xyz", "data/qm7_h5/valid/"]
        test_file:  ["data/qm7/test.xyz", "data/qm7/test.h5"]