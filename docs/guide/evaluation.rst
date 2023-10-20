.. _evaluation:

============
Evaluation
============

To evaluate your MACE model on an XYZ file, run the `eval_configs.py`:

.. code-block:: bash
    
    python3 <mace_repo_dir>/mace/cli/eval_configs.py \
        --configs="your_configs.xyz" \
        --model="your_model.model" \
        --output="./your_output.xyz" 
