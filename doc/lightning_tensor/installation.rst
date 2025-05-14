Lightning-Tensor installation
*****************************

Standard installation
=====================
For the majority of cases, Lightning-Tensor can be installed by following our installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__.

Install Lightning-Tensor from source
====================================

Lightning-Tensor requires CUDA 12 and the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ (only the `cutensornet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`_ library is required).
The SDK may be installed within the Python environment ``site-packages`` directory using ``pip`` or ``conda`` or the SDK library path appended to the ``LD_LIBRARY_PATH`` environment variable.
Please see the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ install guide for more information.


.. note::

    The section below contains instructions for installing Lightning-Tensor **from source**. For most cases, *this is not required* and one can simply use the installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__. If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

Lightning-Qubit needs to be 'installed' by ``pip`` before Lightning-Tensor (compilation is not necessary):

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    pip install cutensornet-cu12
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    SKIP_COMPILATION=True pip install -e . --config-settings editable_mode=compat

Note that `cutensornet-cu12` is a requirement for Lightning-Tensor, and is installed by ``pip`` separately. After `cutensornet-cu12` is installed, the ``CUQUANTUM_SDK`` environment variable should be set to enable discovery during installation:

.. code-block:: bash

    export CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum')")

The Lightning-Tensor can then be installed with ``pip``:

.. code-block:: bash

    PL_BACKEND="lightning_tensor" python scripts/configure_pyproject_toml.py
    pip install -e . --config-settings editable_mode=compat -vv

Lightning-Tensor also requires additional NVIDIA libraries including ``nvJitLink``, ``cuSOLVER``, ``cuSPARSE``, ``cuBLAS``, and ``CUDA runtime``. These can be installed through the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit/>`_ or from ``pip``.

Please refer to the `plugin documentation <https://docs.pennylane.ai/projects/lightning/>`_ as
well as to the `PennyLane documentation <https://docs.pennylane.ai/>`_ for further reference.
