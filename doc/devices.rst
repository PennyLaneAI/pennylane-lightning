Honeywell Quantum Solutions Devices
===================================

The PennyLane-Honeywell plugin provides the ability for PennyLane to access
devices available via Honeywell Quantum Solutions' cloud hardware service.

.. raw::html
    <section id="hqs">

Cloud ion-trap machines
-----------------------

This PennyLane device connects you to ion-trap machines available from Honeywell Quantum Solutions.
Once the plugin has been installed, you can use this device
directly in PennyLane by specifying ``"honeywell.hqs"`` and providing the name of the online hardware machine
you wish to access:

.. code-block:: python

    import pennylane as qml

    dev = qml.device("honeywell.hqs", "machine_name", wires=2)

    @qml.qnode(dev)
    def circuit(w, x, y, z):
        qml.RX(w, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(0.5, wires=0)
        return qml.expval(qml.PauliZ(0))

where ``machine_name`` is a string specifying the specific online hardware you wish to use.
Contact Honeywell Quantum Solutions to receive platform access and machine names.

Remote backend access
---------------------

The user will need to obtain access credentials for the Honeywell Quantum
Solutions platform in order to use these remote devices.
These credentials should be provided to PennyLane via a
`configuration file or environment variable <https://pennylane.readthedocs.io/en/stable/introduction/configuration.html>`_.
Specifically, the variable ``HQS_TOKEN`` must contain a valid access key for Honeywell's online platform.
