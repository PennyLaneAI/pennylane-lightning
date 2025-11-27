Lightning GPU device
====================

The ``lightning.amdgpu`` device is an extension of PennyLane's built-in ``lightning.qubit`` device.
It extends the CPU-focused Lightning simulator to enable GPU-accelerated simulation of quantum state-vector evolution specifically on AMD GPUs.

A ``lightning.amdgpu`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.amdgpu", wires=2)

The pre-built ``lightning.amdgpu`` device wheels are available for ROCm 7.0+ and MI300 series GPUs. For other versions of ROCm or AMD GPUs series, please refer to the :doc:`/lightning_amdgpu/installation` guide for instructions on building from source.

The ``lightning.amdgpu`` device is an instantiation of the ``lightning.kokkos`` device specifically for AMD GPUs. It inherits all features from the ``lightning.kokkos`` device, including support for a wide range of quantum operations and observables, as well as compatibility with PennyLane's quantum functions and QNodes. For a comprehensive list of supported operations and observables, please refer to the :doc:`/lightning_kokkos/device` documentation.

.. note ::

    It is not recommended to install both ``lightning.amdgpu`` and ``lightning.kokkos`` devices in the same environment, as this may lead to conflicts. Choose the device that best fits your hardware and simulation needs.
