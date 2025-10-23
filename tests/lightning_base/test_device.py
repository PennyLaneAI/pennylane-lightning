# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains unit tests for new device API Lightning classes.
"""
# pylint: disable=too-many-arguments, unused-argument

import itertools
from dataclasses import replace

import numpy as np
import pennylane as qml
import pytest
from conftest import (
    PHI,
    THETA,
    VARPHI,
    LightningAdjointJacobian,
    LightningDevice,
    LightningMeasurements,
    LightningStateVector,
    device_name,
)
from pennylane.devices import DefaultQubit, ExecutionConfig, MCMConfig
from pennylane.devices.default_qubit import adjoint_ops
from pennylane.devices.preprocess import device_resolve_dynamic_wires
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.measurements import ProbabilityMP
from pennylane.tape import QuantumScript
from pennylane.transforms import defer_measurements, dynamic_one_shot

if device_name == "lightning.qubit":
    from pennylane_lightning.lightning_qubit.lightning_qubit import (
        _add_adjoint_transforms,
        _adjoint_ops,
        _supports_adjoint,
        accepted_observables,
        adjoint_measurements,
        adjoint_observables,
        allow_mcms_stopping_condition,
        decompose,
        no_mcms_stopping_condition,
        no_sampling,
        stopping_condition,
        validate_adjoint_trainable_params,
        validate_device_wires,
        validate_measurements,
        validate_observables,
    )
elif device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos.lightning_kokkos import (
        _add_adjoint_transforms,
        _adjoint_ops,
        _supports_adjoint,
        accepted_observables,
        adjoint_measurements,
        adjoint_observables,
        allow_mcms_stopping_condition,
        decompose,
        no_mcms_stopping_condition,
        no_sampling,
        stopping_condition,
        validate_adjoint_trainable_params,
        validate_device_wires,
        validate_measurements,
        validate_observables,
    )
elif device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu.lightning_gpu import (
        _add_adjoint_transforms,
        _adjoint_ops,
        _supports_adjoint,
        accepted_observables,
        adjoint_measurements,
        adjoint_observables,
        allow_mcms_stopping_condition,
        decompose,
        no_mcms_stopping_condition,
        no_sampling,
        stopping_condition,
        validate_adjoint_trainable_params,
        validate_device_wires,
        validate_measurements,
        validate_observables,
    )
elif device_name == "lightning.tensor":
    from pennylane_lightning.lightning_tensor.lightning_tensor import (
        accepted_observables,
        stopping_condition,
    )
else:
    raise TypeError(f"The device name: {device_name} is not a valid name")

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

fixture_params = list(
    itertools.product([3, None], [np.complex64, np.complex128])
)  # wires x c_dtype


@pytest.fixture(params=fixture_params)
def dev(request):
    return LightningDevice(wires=request.param[0], c_dtype=request.param[1])


@pytest.fixture()
def enable_disable_plxpr():
    """Fixture to enable and disable the plxpr capture"""
    pytest.importorskip("jax")
    qml.capture.enable()
    yield
    qml.capture.disable()


@pytest.fixture(params=[False, True], ids=["graph_disabled", "graph_enabled"])
def enable_and_disable_graph_decomp(request):
    """
    A fixture that parametrizes a test to run twice: once with graph
    decomposition disabled and once with it enabled.

    It automatically handles the setup (enabling/disabling) before the
    test runs and the teardown (always disabling) after the test completes.
    """
    try:
        use_graph_decomp = request.param

        # --- Setup Phase ---
        # This code runs before the test function is executed.
        if use_graph_decomp:
            qml.decomposition.enable_graph()
        else:
            # Explicitly disable to ensure a clean state
            qml.decomposition.disable_graph()

        # Yield control to the test function
        yield use_graph_decomp

    finally:
        # --- Teardown Phase ---
        # This code runs after the test function has finished,
        # regardless of whether it passed or failed.
        qml.decomposition.disable_graph()


class TestHelpers:
    """Unit tests for helper functions"""

    class DummyOperator(qml.operation.Operation):
        """Dummy operator"""

        num_wires = 1

    @pytest.mark.parametrize(
        "valid_op",
        [
            qml.RX(1.23, 0),
        ],
    )
    def test_stopping_condition_valid(self, valid_op):
        """Test that stopping_condition returns True for operations unsupported by the device."""

        assert stopping_condition(valid_op) is True

    def test_stopping_condition_invalid(self):
        """Test that stopping_condition returns False for operations unsupported by the device."""

        invalid_op = self.DummyOperator(0)

        assert stopping_condition(invalid_op) is False

    def test_accepted_observables(self):
        """Test that accepted_observables returns whether or not an observable
        is supported by the device."""
        valid_obs = qml.Projector([0], 0)
        invalid_obs = self.DummyOperator(0)
        result = True if device_name != "lightning.tensor" else False
        assert accepted_observables(valid_obs) is result
        assert accepted_observables(invalid_obs) is False

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support adjoint_observables",
    )
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.prod(qml.Projector([0], 0), qml.PauliZ(1)), False),
            (qml.prod(qml.Projector([0], 0), qml.PauliZ(1)), False),
            (qml.s_prod(1.5, qml.Projector([0], 0)), False),
            (qml.sum(qml.Projector([0], 0), qml.Hadamard(1)), False),
            (qml.sum(qml.prod(qml.Projector([0], 0), qml.Y(1)), qml.PauliX(1)), False),
            (qml.prod(qml.Y(0), qml.Z(1)), True),
            (qml.prod(qml.Y(0), qml.PauliZ(1)), True),
            (qml.s_prod(1.5, qml.Y(1)), True),
            (qml.sum(qml.Y(1), qml.Hadamard(1)), True),
            (qml.X(0), True),
            (qml.Hermitian(np.eye(4), [0, 1]), True),
        ],
    )
    def test_adjoint_observables(self, obs, expected):
        """Test that adjoint_observables returns the expected boolean result for
        a given observable"""
        assert adjoint_observables(obs) == expected

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support adjoint",
    )
    def test_add_adjoint_transforms(self):
        """Test that the correct transforms are added to the program by _add_adjoint_transforms"""
        expected_program = qml.transforms.core.TransformProgram()

        name = f"adjoint + {device_name}"
        expected_program.add_transform(no_sampling, name=name)
        expected_program.add_transform(qml.transforms.broadcast_expand)
        expected_program.add_transform(
            decompose,
            stopping_condition=_adjoint_ops,
            name=name,
            skip_initial_state_prep=False,
            device_wires=None,
            target_gates=LightningDevice.capabilities.gate_set(differentiable=True),
        )
        expected_program.add_transform(validate_observables, accepted_observables, name=name)
        expected_program.add_transform(
            validate_measurements,
            analytic_measurements=adjoint_measurements,
            name=name,
        )
        expected_program.add_transform(validate_adjoint_trainable_params)

        actual_program = qml.transforms.core.TransformProgram()
        _add_adjoint_transforms(actual_program)
        assert actual_program == expected_program

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support adjoint",
    )
    @pytest.mark.parametrize(
        "circuit, expected",
        [
            (None, True),
            (QuantumScript([], [qml.state()]), False),
            (QuantumScript([qml.RX(1.23, 0)], [qml.expval(qml.Z(0))]), True),
            (
                QuantumScript([qml.CRot(1.23, 4.56, 7.89, [0, 1])], [qml.expval(qml.Z(0))]),
                True,
            ),
            (QuantumScript([qml.Rot(1.23, 4.56, 7.89, 1)], [qml.var(qml.X(0))]), False),
        ],
    )
    def test_supports_adjoint(self, circuit, expected):
        """Test that _supports_adjoint returns the correct boolean value."""
        assert _supports_adjoint(circuit) == expected

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not contain a state vector",
    )
    @pytest.mark.parametrize("device_wires", [None, 2])
    def test_state_vector_init(self, device_wires):
        """Test that the state-vector is not created during initialization"""
        dev = LightningDevice(wires=device_wires)
        assert dev._statevector == None

    @pytest.mark.parametrize(
        "circuit_in, n_wires, expected_circuit_out",
        [
            (
                QuantumScript(
                    [
                        qml.RX(0.1, 0),
                        qml.CNOT([1, 0]),
                        qml.RZ(0.1, 1),
                        qml.CNOT([2, 1]),
                    ],
                    [qml.expval(qml.Z(0))],
                ),
                3,
                QuantumScript(
                    [
                        qml.RX(0.1, 0),
                        qml.CNOT([1, 0]),
                        qml.RZ(0.1, 1),
                        qml.CNOT([2, 1]),
                    ],
                    [qml.expval(qml.Z(0))],
                ),
            ),
            (
                QuantumScript(
                    [
                        qml.RX(0.1, 0),
                        qml.CNOT([1, 4]),
                        qml.RZ(0.1, 4),
                        qml.CNOT([2, 1]),
                    ],
                    [qml.expval(qml.Z(6))],
                ),
                5,
                QuantumScript(
                    [
                        qml.RX(0.1, 0),
                        qml.CNOT([1, 2]),
                        qml.RZ(0.1, 2),
                        qml.CNOT([3, 1]),
                    ],
                    [qml.expval(qml.Z(4))],
                ),
            ),
        ],
    )
    def test_dynamic_wires_from_circuit(self, circuit_in, n_wires, expected_circuit_out):
        """Test that dynamic_wires_from_circuit returns correct circuit and creates state-vectors properly"""
        device = LightningDevice(wires=None)

        circuit_out = device.dynamic_wires_from_circuit(circuit_in)

        assert circuit_out.num_wires == n_wires
        assert circuit_out.wires == qml.wires.Wires(range(n_wires))
        assert circuit_out.operations == expected_circuit_out.operations
        assert circuit_out.measurements == expected_circuit_out.measurements

        if device_name != "lightning.tensor":
            assert device._statevector._num_wires == n_wires
            assert device._statevector._wires == qml.wires.Wires(range(n_wires))

    @pytest.mark.parametrize(
        "circuit_in, n_wires, wires_list",
        [
            (
                QuantumScript(
                    [
                        qml.RX(0.1, 0),
                        qml.CNOT([1, 0]),
                        qml.RZ(0.1, 1),
                        qml.CNOT([2, 1]),
                    ],
                    [qml.expval(qml.Z(0))],
                ),
                3,
                [0, 1, 2],
            ),
            (
                QuantumScript(
                    [
                        qml.RX(0.1, 0),
                        qml.CNOT([1, 4]),
                        qml.RZ(0.1, 4),
                        qml.CNOT([2, 1]),
                    ],
                    [qml.expval(qml.Z(6))],
                ),
                7,
                [0, 1, 4, 2, 6],
            ),
        ],
    )
    def test_dynamic_wires_from_circuit_fixed_wires(self, circuit_in, n_wires, wires_list):
        """Test that dynamic_wires_from_circuit does not alter the circuit if wires are fixed and state-vector is created properly"""
        device = LightningDevice(wires=n_wires)

        circuit_out = device.dynamic_wires_from_circuit(circuit_in)

        assert circuit_out.num_wires == circuit_in.num_wires
        assert circuit_out.wires == qml.wires.Wires(wires_list)
        assert circuit_out.operations == circuit_in.operations
        assert circuit_out.measurements == circuit_in.measurements

        if device_name != "lightning.tensor":
            assert device._statevector._num_wires == n_wires
            assert device._statevector._wires == qml.wires.Wires(range(n_wires))

    @pytest.mark.parametrize(
        "circuit_0, n_wires_0",
        [
            (QuantumScript([qml.RX(0.1, 0)], [qml.expval(qml.Z(1))]), 2),
            (
                QuantumScript([qml.RX(0.1, 0), qml.RX(0.1, 1)], [qml.expval(qml.Z(2))]),
                3,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "circuit_1, n_wires_1",
        [
            (QuantumScript([qml.RX(0.1, 0)], [qml.expval(qml.Z(1))]), 2),
            (
                QuantumScript([qml.RX(0.1, 0), qml.RX(0.1, 2)], [qml.expval(qml.Z(1))]),
                3,
            ),
            (
                QuantumScript(
                    [qml.RX(0.1, 0), qml.RX(0.1, 1), qml.RX(0.1, 4), qml.RX(0.1, 6)],
                    [qml.expval(qml.Z(2))],
                ),
                5,
            ),
        ],
    )
    @pytest.mark.parametrize("shots", [None, 10])
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not have state vector",
    )
    def test_dynamic_wires_from_circuit_reset_state(
        self, circuit_0, n_wires_0, circuit_1, n_wires_1, shots, dtype
    ):
        """Test that dynamic_wires_from_circuit resets state when reusing or initializing new state vector"""
        device = LightningDevice(wires=None, c_dtype=dtype)

        # Initialize statevector and apply a state
        device.dynamic_wires_from_circuit(circuit_0.copy(shots=shots))
        state = np.zeros(2**n_wires_0)
        state[-1] = 1.0
        device._statevector._apply_state_vector(state, range(n_wires_0))

        # Dynamic wires again will reset the state
        device.dynamic_wires_from_circuit(circuit_1.copy(shots=shots))
        expected_state = np.zeros(2**n_wires_1)
        expected_state[0] = 1.0
        assert np.allclose(device._statevector.state, expected_state)

    @pytest.mark.parametrize("shots", [None, 10])
    @pytest.mark.skipif(
        device_name not in ("lightning.kokkos", "lightning.gpu"),
        reason="This device state has no additional kwargs",
    )
    def test_dynamic_wires_from_circuit_state_kwargs(self, shots):
        """Test that dynamic_wires_from_circuit sets the state with the correct device init kwargs"""

        if device_name == "lightning.kokkos":
            from pennylane_lightning.lightning_kokkos_ops import InitializationSettings

            sv_init_kwargs = {"kokkos_args": InitializationSettings().set_num_threads(2)}
        if device_name == "lightning.gpu":
            sv_init_kwargs = {"use_async": True}

        device = LightningDevice(wires=None, **sv_init_kwargs)

        circuit = QuantumScript(
            [qml.RX(0.1, 0), qml.RX(0.1, 2)], [qml.expval(qml.Z(1))], shots=shots
        )
        circuit_num_wires = 3

        device.dynamic_wires_from_circuit(circuit)

        if device_name == "lightning.gpu":
            assert device._statevector._use_async == sv_init_kwargs["use_async"]
        if device_name == "lightning.kokkos":
            sv = LightningStateVector(circuit_num_wires, **sv_init_kwargs)
            type(sv) == type(device._statevector)

    @pytest.mark.parametrize("shots", [None, 10])
    @pytest.mark.parametrize("n_wires", [None, 3])
    def test_dynamic_wires_from_circuit_bad_kwargs(self, n_wires, shots):
        """Test that dynamic_wires_from_circuit produce right error when setting the state with the incorrect device init kwargs"""

        if device_name == "lightning.kokkos":
            bad_init_kwargs = {"kokkos_args": np.array([33])}
        else:
            bad_init_kwargs = {"XXX": True}

        circuit = QuantumScript(
            [qml.RX(0.1, 0), qml.RX(0.1, 2)], [qml.expval(qml.Z(1))], shots=shots
        )

        if device_name == "lightning.kokkos":
            with pytest.raises(TypeError, match="Argument kokkos_args must be of type "):
                device = LightningDevice(wires=n_wires, **bad_init_kwargs)
                device.dynamic_wires_from_circuit(circuit)
        else:
            with pytest.raises(
                TypeError,
                match=r"got an unexpected keyword argument|Unexpected argument",
            ):
                device = LightningDevice(wires=n_wires, **bad_init_kwargs)
                device.dynamic_wires_from_circuit(circuit)


class TestInitialization:
    """Unit tests for device initialization"""

    @pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
    def test_property_complex(self, c_dtype):
        """Test that the property complex is set correctly"""
        dev = LightningDevice(wires=2, c_dtype=c_dtype)
        assert dev.c_dtype == c_dtype

    def test_wires_mapping(self):
        """Test that the wires mapping is set correctly"""
        dev = LightningDevice(wires=2)
        assert dev._wire_map == None

        dev = LightningDevice(wires=["a", "b"])
        assert dev._wire_map == {"a": 0, "b": 1}

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor is not a state-vector simulator",
    )
    def test_dummies_definition(self):
        """Test that the dummies are defined correctly"""
        dev = LightningDevice(wires=2)
        assert dev.LightningStateVector == LightningStateVector
        assert dev.LightningMeasurements == LightningMeasurements
        assert dev.LightningAdjointJacobian == LightningAdjointJacobian

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support seeding",
    )
    @pytest.mark.parametrize("n_wires", [None, 3])
    @pytest.mark.parametrize("seed", ["global", None, 42, [42, 43, 44]])
    def test_device_seed(self, n_wires, seed):
        """Test that seeding the lightning device works correctly"""
        dev = LightningDevice(wires=n_wires, seed=seed)
        assert dev._rng is not None


class TestExecution:
    """Unit tests for executing quantum tapes on a device"""

    @staticmethod
    def calculate_reference(tape):
        device = DefaultQubit()
        program, _ = device.preprocess()
        tapes, transf_fn = program([tape])
        results = device.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(device, tape):
        program, _ = device.preprocess()
        tapes, transf_fn = program([tape])
        results = device.execute(tapes)
        return transf_fn(results)

    _default_device_options = {
        "c_dtype": np.complex128,
        "batch_obs": False,
        "mcmc": False,
        "kernel_name": None,
        "num_burnin": 0,
    }

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support rng key",
    )
    @pytest.mark.parametrize(
        "config, expected_config",
        [
            (
                ExecutionConfig(),
                ExecutionConfig(
                    grad_on_execution=None,
                    use_device_gradient=False,
                    use_device_jacobian_product=False,
                    device_options=_default_device_options,
                    mcm_config=MCMConfig(mcm_method="deferred"),
                ),
            ),
            (
                None,
                ExecutionConfig(
                    grad_on_execution=None,
                    use_device_gradient=False,
                    use_device_jacobian_product=False,
                    device_options=_default_device_options,
                    mcm_config=MCMConfig(mcm_method="deferred"),
                ),
            ),
            (
                ExecutionConfig(gradient_method="best"),
                ExecutionConfig(
                    gradient_method="adjoint",
                    grad_on_execution=True,
                    use_device_gradient=True,
                    use_device_jacobian_product=True,
                    device_options=_default_device_options,
                    mcm_config=MCMConfig(mcm_method="deferred"),
                ),
            ),
            pytest.param(
                ExecutionConfig(
                    device_options={
                        "c_dtype": np.complex64,
                        "mcmc": False,
                    }
                ),
                ExecutionConfig(
                    grad_on_execution=None,
                    use_device_gradient=False,
                    use_device_jacobian_product=False,
                    device_options={
                        "c_dtype": np.complex64,
                        "batch_obs": False,
                        "mcmc": False,
                        "kernel_name": None,
                        "num_burnin": 0,
                    },
                    mcm_config=MCMConfig(mcm_method="deferred"),
                ),
                marks=pytest.mark.skipif(
                    device_name != "lightning.qubit",
                    reason=f"The device {device_name} does not support mcmc",
                ),
            ),
            pytest.param(
                ExecutionConfig(
                    device_options={
                        "c_dtype": np.complex64,
                        "mcmc": True,
                        "kernel_name": "Local",
                        "num_burnin": 100,
                    },
                ),
                ExecutionConfig(
                    grad_on_execution=None,
                    use_device_gradient=False,
                    use_device_jacobian_product=False,
                    device_options={
                        "c_dtype": np.complex64,
                        "batch_obs": False,
                        "mcmc": True,
                        "kernel_name": "Local",
                        "num_burnin": 100,
                    },
                    mcm_config=MCMConfig(mcm_method="deferred"),
                ),
                marks=pytest.mark.skipif(
                    device_name != "lightning.qubit",
                    reason=f"The device {device_name} does not support mcmc",
                ),
            ),
            pytest.param(
                ExecutionConfig(
                    device_options={
                        "c_dtype": np.complex64,
                        "mcmc": True,
                        "kernel_name": "NonZeroRandom",
                        "num_burnin": 100,
                    }
                ),
                ExecutionConfig(
                    grad_on_execution=None,
                    use_device_gradient=False,
                    use_device_jacobian_product=False,
                    device_options={
                        "c_dtype": np.complex64,
                        "batch_obs": False,
                        "mcmc": True,
                        "kernel_name": "NonZeroRandom",
                        "num_burnin": 100,
                    },
                    mcm_config=MCMConfig(mcm_method="deferred"),
                ),
                marks=pytest.mark.skipif(
                    device_name != "lightning.qubit",
                    reason=f"The device {device_name} does not support mcmc",
                ),
            ),
            (
                ExecutionConfig(
                    gradient_method="backprop",
                    use_device_gradient=False,
                    grad_on_execution=False,
                ),
                ExecutionConfig(
                    gradient_method="backprop",
                    use_device_gradient=False,
                    grad_on_execution=False,
                    use_device_jacobian_product=False,
                    device_options=_default_device_options,
                    mcm_config=MCMConfig(mcm_method="deferred"),
                ),
            ),
        ],
    )
    def test_preprocess_correct_config_setup(self, config, expected_config):
        """Test that the execution config is set up correctly in preprocess"""
        device = LightningDevice(wires=2)
        new_config = device.setup_execution_config(config)

        # Update the device options to be able to compare
        device_options = new_config.device_options.copy()
        device_options.pop("rng", None)
        new_config = replace(new_config, device_options=device_options)

        assert new_config == expected_config

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device supports new device options",
    )
    def test_preprocess_incorrect_device_config(self):
        """Test that an error is raised if the device options are not valid"""
        config = ExecutionConfig(
            device_options={
                "is_wrong_option": True,
            }
        )
        device = LightningDevice(wires=2)
        with pytest.raises(DeviceError, match="device option is_wrong_option"):
            _ = device.setup_execution_config(config)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device doesn't have support for program capture.",
    )
    @pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
    def test_sbs_and_postselect_warning(self, enable_disable_plxpr, postselect_mode):
        """Test that a warning is raised if post-selection is used with single branch statistics."""
        device = LightningDevice(wires=1)
        config = ExecutionConfig(
            mcm_config=MCMConfig(
                mcm_method="single-branch-statistics", postselect_mode=postselect_mode
            )
        )

        with pytest.warns(
            UserWarning,
            match="Setting 'postselect_mode' is not supported with mcm_method='single-branch-",
        ):
            _ = device.setup_execution_config(config)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device doesn't have support for program capture.",
    )
    def test_preprocess_invalid_mcm_method_error(self, enable_disable_plxpr):
        """Test that an error is raised if mcm_method is invalid."""
        device = LightningDevice(wires=1)
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="foo"))

        with pytest.raises(DeviceError, match="mcm_method='foo' is not supported"):
            _ = device.setup_execution_config(config)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support mcms",
    )
    def test_decompose_conditionals(self):
        """Test that conditional templates are properly decomposed."""

        class NoMatOp(qml.operation.Operation):
            """Dummy operation for expanding circuit."""

            # pylint: disable=arguments-renamed, invalid-overridden-method
            @property
            def has_matrix(self):
                return False

            def decomposition(self):
                return [qml.PauliX(self.wires), qml.PauliY(self.wires)]

        m0 = qml.measure(0)
        tape = qml.tape.QuantumScript(
            [m0.measurements[0], qml.ops.Conditional(m0, NoMatOp(wires=0))], [qml.probs(wires=0)]
        )
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="deferred"))

        prog = LightningDevice(wires=2).preprocess_transforms(config)
        [new_tape], _ = prog((tape,))

        expected = qml.tape.QuantumScript(
            [qml.CNOT((0, 1)), qml.CNOT((1, 0)), qml.CY((1, 0))], [qml.probs(wires=0)]
        )
        qml.assert_equal(new_tape, expected)

    def test_no_mcms_conditionals_defer_measurements(self):
        """Test that an error is raised if an mcm occurs in a decomposition after defer measurements has been applied."""

        m0 = qml.measure(0)

        class MyOp(qml.operation.Operator):
            def decomposition(self):
                return m0.measurements

        tape = qml.tape.QuantumScript([MyOp(0)])
        config = qml.devices.ExecutionConfig(
            mcm_config=qml.devices.MCMConfig(mcm_method="deferred")
        )

        prog = LightningDevice(wires=2).preprocess_transforms(config)

        with pytest.raises(DeviceError, match="not supported with"):
            prog((tape,))

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support mcms",
    )
    @pytest.mark.parametrize("shots, expected", [(None, "deferred"), (10, "one-shot")])
    def test_default_mcm_method_circuit(self, shots, expected):
        """Test that the default mcm method depends on the shots in the circuit."""
        device = LightningDevice(wires=2)
        config = ExecutionConfig()
        processed = device.setup_execution_config(
            config, circuit=qml.tape.QuantumScript(shots=shots)
        )
        assert processed.mcm_config.mcm_method == expected

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support mcms",
    )
    def test_default_mcm_method_no_circuit(self):
        """Test that the default mcm method is deferred if no shots are provided."""
        device = LightningDevice(wires=2)
        config = ExecutionConfig()
        processed = device.setup_execution_config(config)
        assert processed.mcm_config.mcm_method == "deferred"

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device doesn't have support for program capture.",
    )
    def test_transform_program(self, enable_disable_plxpr):
        """Test that the transform program returned by preprocess has the correct transforms."""
        dev = LightningDevice(wires=1)

        # Default config
        config = ExecutionConfig()
        program = dev.preprocess_transforms(execution_config=config)
        assert len(program) == 1
        # pylint: disable=protected-access
        assert program[0].transform == qml.transforms.decompose._transform

        # mcm_method="deferred"
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="deferred"))
        program = dev.preprocess_transforms(execution_config=config)
        assert len(program) == 2
        # pylint: disable=protected-access
        assert program[0].transform == qml.defer_measurements._transform
        assert program[1].transform == qml.transforms.decompose._transform

        # mcm_method="single-branch-statistics"
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="single-branch-statistics"))
        program = dev.preprocess_transforms(execution_config=config)
        assert len(program) == 1
        # pylint: disable=protected-access
        assert program[0].transform == qml.transforms.decompose._transform

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support adjoint",
    )
    @pytest.mark.parametrize("adjoint", [True, False])
    @pytest.mark.parametrize("mcm_method", ("deferred", "one-shot", "tree-traversal"))
    def test_preprocess(self, adjoint, mcm_method):
        """Test that the transform program returned by preprocess is correct"""
        device = LightningDevice(wires=2)

        expected_program = qml.transforms.core.TransformProgram()
        expected_program.add_transform(validate_measurements, name=device.name)
        expected_program.add_transform(validate_observables, accepted_observables, name=device.name)
        if mcm_method == "deferred":
            expected_program.add_transform(defer_measurements, allow_postselect=False)
        expected_program.add_transform(
            decompose,
            stopping_condition=(
                no_mcms_stopping_condition
                if mcm_method == "deferred"
                else allow_mcms_stopping_condition
            ),
            skip_initial_state_prep=True,
            name=device.name,
            device_wires=device.wires,
            target_gates=device.capabilities.gate_set(),
        )
        expected_program.add_transform(
            device_resolve_dynamic_wires, wires=device.wires, allow_resets=mcm_method != "deferred"
        )
        expected_program.add_transform(validate_device_wires, device.wires, name=device.name)
        if mcm_method == "one-shot":
            expected_program.add_transform(dynamic_one_shot, postselect_mode=None)
        expected_program.add_transform(qml.transforms.broadcast_expand)

        if adjoint:
            name = f"adjoint + {device_name}"
            expected_program.add_transform(no_sampling, name=name)
            expected_program.add_transform(qml.transforms.broadcast_expand)
            expected_program.add_transform(
                decompose,
                stopping_condition=_adjoint_ops,
                name=name,
                skip_initial_state_prep=False,
                device_wires=device.wires,
                target_gates=device.capabilities.gate_set(differentiable=True),
            )
            expected_program.add_transform(validate_observables, accepted_observables, name=name)
            expected_program.add_transform(
                validate_measurements,
                analytic_measurements=adjoint_measurements,
                name=name,
            )
            expected_program.add_transform(validate_adjoint_trainable_params)

        gradient_method = "adjoint" if adjoint else None
        config = ExecutionConfig(
            gradient_method=gradient_method, mcm_config=MCMConfig(mcm_method=mcm_method)
        )
        actual_program = device.preprocess_transforms(config)
        assert actual_program == expected_program

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        "op, is_trainable",
        (
            [
                (qml.StatePrep([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0), False),
                (
                    qml.StatePrep(qml.numpy.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), wires=0),
                    True,
                ),
                (qml.StatePrep(np.array([1, 0]), wires=0), False),
                (qml.BasisState([1, 1], wires=[0, 1]), False),
                (qml.BasisState(qml.numpy.array([1, 1]), wires=[0, 1]), True),
            ]
        ),
    )
    def test_preprocess_state_prep_first_op_decomposition(self, op, is_trainable):
        """Test that state prep ops in the beginning of a tape are decomposed with adjoint
        but not otherwise."""
        if device_name == "lightning.tensor" and is_trainable:
            pytest.skip("StatePrep trainable not supported in lightning.tensor")

        tape = qml.tape.QuantumScript([op, qml.RX(1.23, wires=0)], [qml.expval(qml.PauliZ(0))])
        device = LightningDevice(wires=3)

        if is_trainable:
            decomp = op.decomposition()
            # decompose one more time if it's decomposed into a template:
            decomp = decomp[0].decomposition() if len(decomp) == 1 else decomp
        else:
            decomp = [op]

        config = ExecutionConfig(gradient_method="best" if is_trainable else None)
        program, _ = device.preprocess(config)
        [new_tape], _ = program([tape])
        expected_tape = qml.tape.QuantumScript([*decomp, qml.RX(1.23, wires=0)], tape.measurements)
        assert qml.equal(new_tape, expected_tape)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        "op, decomp_depth",
        [
            (qml.StatePrep([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0), 1),
            (qml.StatePrep(np.array([1, 0]), wires=0), 1),
            (qml.BasisState([1, 1], wires=[0, 1]), 1),
            (qml.BasisState(qml.numpy.array([1, 1]), wires=[0, 1]), 1),
            (qml.AmplitudeEmbedding([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0), 1),
            (
                qml.MottonenStatePreparation([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0),
                0,
            ),
        ],
    )
    def test_preprocess_state_prep_middle_op_decomposition(self, op, decomp_depth):
        """Test that state prep ops in the middle of a tape are always decomposed."""
        tape = qml.tape.QuantumScript(
            [qml.RX(1.23, wires=0), op, qml.CNOT([0, 1])], [qml.expval(qml.PauliZ(0))]
        )
        device = LightningDevice(wires=3)

        op = op.decomposition()[0] if decomp_depth and len(op.decomposition()) == 1 else op
        decomp = op.decomposition()

        program, _ = device.preprocess()
        [new_tape], _ = program([tape])
        expected_tape = qml.tape.QuantumScript(
            [qml.RX(1.23, wires=0), *decomp, qml.CNOT([0, 1])], tape.measurements
        )
        assert qml.equal(new_tape, expected_tape)

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "mp",
        (
            [
                qml.probs(wires=[1, 2]),
                qml.probs(op=qml.Z(2)),
                qml.expval(qml.Z(2)),
                qml.var(qml.X(2)),
                qml.expval(qml.X(0) + qml.Z(0)),
                qml.expval(qml.Hamiltonian([-0.5, 1.5], [qml.Y(1), qml.X(1)])),
                qml.expval(2.5 * qml.Z(0)),
                qml.expval(qml.Z(0) @ qml.X(1)),
                qml.expval(qml.prod(qml.Z(0), qml.X(1))),
                qml.expval(
                    qml.SparseHamiltonian(
                        qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]).sparse_matrix(
                            wire_order=[0, 1, 2]
                        ),
                        wires=[0, 1, 2],
                    )
                ),
                qml.expval(qml.Projector([1], wires=2)),
            ]
        ),
    )
    def test_execute_single_measurement(self, theta, phi, mp, dev):
        """Test that execute returns the correct results with a single measurement."""
        if device_name == "lightning.tensor":
            if isinstance(mp.obs, qml.SparseHamiltonian) or isinstance(mp.obs, qml.Projector):
                pytest.skip("SparseHamiltonian/Projector obs not supported in lightning.tensor")

        if isinstance(mp.obs, qml.SparseHamiltonian) and dev.c_dtype == np.complex64:
            pytest.skip(
                reason="The conversion from qml.Hamiltonian to SparseHamiltonian is only possible with np.complex128"
            )

        qs = QuantumScript(
            [
                qml.RX(phi, 0),
                qml.CNOT([0, 2]),
                qml.RZ(theta, 1),
                qml.CNOT([1, 2]),
            ],
            [mp],
        )
        res = self.process_and_execute(dev, qs)[0]
        expected = self.calculate_reference(qs)[0]
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "mp1",
        (
            [
                qml.probs(wires=[1, 2]),
                qml.expval(qml.Z(2)),
                qml.var(qml.X(2)),
                qml.var(qml.Hermitian(qml.Hadamard.compute_matrix(), 0)),
            ]
        ),
    )
    @pytest.mark.parametrize(
        "mp2",
        (
            [
                qml.probs(op=qml.X(2)),
                qml.expval(qml.Y(2)),
                qml.var(qml.Y(2)),
                qml.expval(qml.Hamiltonian([-0.5, 1.5, -1.1], [qml.Y(1), qml.X(1), qml.Z(0)])),
            ]
        ),
    )
    def test_execute_multi_measurement(self, theta, phi, dev, mp1, mp2):
        """Test that execute returns the correct results with multiple measurements."""

        qs = QuantumScript(
            [
                qml.RX(phi, 0),
                qml.CNOT([0, 2]),
                qml.RZ(theta, 1),
                qml.CNOT([1, 2]),
            ],
            [mp1, mp2],
        )
        res = self.process_and_execute(dev, qs)[0]
        expected = self.calculate_reference(qs)[0]
        assert len(res) == 2
        for r, e in zip(res, expected):
            assert np.allclose(r, e)

    def test_execute_tape_batch_with_dynamic_wires(self):
        """Test that execute handles multiple tapes with dynamic number of wires."""

        qs0 = QuantumScript(
            [
                qml.RX(0.1, 0),
                qml.CNOT([1, 0]),
                qml.RZ(0.1, 1),
                qml.CNOT([2, 1]),
            ],
            [qml.state()],
        )
        qs1 = QuantumScript(
            [
                qml.RX(0.1, 0),
                qml.CNOT([1, 0]),
                qml.RZ(0.1, 1),
                qml.CNOT([0, 1]),
            ],
            [qml.state()],
        )
        qs2 = QuantumScript(
            [
                qml.RX(0.1, 4),
                qml.CNOT([2, 4]),
                qml.RZ(0.1, 2),
                qml.CNOT([1, 2]),
                qml.CNOT([0, 2]),
            ],
            [qml.state()],
        )
        dev = LightningDevice(wires=None)
        result = dev.execute([qs0, qs1, qs2])

        dev_ref = DefaultQubit()
        result_ref = dev_ref.execute([qs0, qs1, qs2])

        for r, e in zip(result, result_ref):
            assert np.allclose(r, e)

    @pytest.mark.parametrize("phi, theta", list(zip(PHI, THETA)))
    @pytest.mark.parametrize("wires", (["a", "b", -3], [0, "target", "other_target"]))
    def test_custom_wires_execute(self, phi, theta, wires):
        """Test execution with custom wires"""
        device = LightningDevice(wires=wires)
        qs = QuantumScript(
            [
                qml.RX(phi, wires[0]),
                qml.RY(theta, wires[2]),
                qml.CNOT([wires[0], wires[1]]),
                qml.CNOT([wires[1], wires[2]]),
            ],
            [qml.expval(qml.Z(wires[0])), qml.expval(qml.Z(wires[2]))],
        )

        result = device.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], np.cos(phi))
        assert np.allclose(result[1], np.cos(phi) * np.cos(theta))

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support out of order probs",
    )
    @pytest.mark.parametrize(
        "wires, wire_order", [(3, (0, 1, 2)), (("a", "b", "c"), ("a", "b", "c"))]
    )
    def test_probs_different_wire_orders(self, wires, wire_order):
        """Test that measuring probabilities works with custom wires."""

        dev = LightningDevice(wires=wires)

        op = qml.Hadamard(wire_order[1])

        tape = QuantumScript([op], [qml.probs(wires=(wire_order[0], wire_order[1]))])

        res = dev.execute(tape)
        assert qml.math.allclose(res, np.array([0.5, 0.5, 0.0, 0.0]))

        tape2 = QuantumScript([op], [qml.probs(wires=(wire_order[1], wire_order[2]))])
        res2 = dev.execute(tape2)
        assert qml.math.allclose(res2, np.array([0.5, 0.0, 0.5, 0.0]))

        tape3 = QuantumScript([op], [qml.probs(wires=(wire_order[1], wire_order[0]))])
        res3 = dev.execute(tape3)
        assert qml.math.allclose(res3, np.array([0.5, 0.0, 0.5, 0.0]))

    @pytest.mark.parametrize("device_wires", (None, (0, 1, 2)))
    def test_reuse_without_mcms(self, device_wires):
        """Test that a dynamic allocations that do not require mcms can be executed."""

        dev = LightningDevice(wires=device_wires)

        with qml.queuing.AnnotatedQueue() as q:
            with qml.allocate(1, restored=True) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 0))
                qml.H(wires)

            with qml.allocate(1) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 1))
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        config = dev.setup_execution_config(circuit=tape)
        [new_tape], _ = dev.preprocess_transforms(config)((tape,))
        assert not any(isinstance(w, qml.allocation.DynamicWire) for w in new_tape.wires)
        assert not any(isinstance(op, qml.measurements.MidMeasureMP) for op in new_tape)
        assert len(new_tape.wires) == 3

        res1, res2 = dev.execute(new_tape, config)
        assert qml.math.allclose(res1, 0)
        assert qml.math.allclose(res2, 0)

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("device_wires", (None, (0, 1, 2, 3)))
    @pytest.mark.parametrize(
        "mcm_method", ("tree-traversal", "deferred", "one-shot", "device", None)
    )
    def test_reuse_with_mcms(self, device_wires, mcm_method, seed):
        """Test that a simple dynamic allocation with mcms can be executed."""

        if device_name == "lightning.tensor":
            pytest.skip("lightning.tensor does not support native mcm.")

        dev = LightningDevice(wires=device_wires, seed=seed)

        with qml.queuing.AnnotatedQueue() as q:
            with qml.allocate(1, restored=False) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 0))
                qml.H(wires)

            with qml.allocate(1) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 1))
            qml.expval(qml.Z(0))
            qml.expval(qml.Z(1))

        tape = qml.tape.QuantumScript.from_queue(
            q, shots=5000 if mcm_method == "one-shot" else None
        )
        config = qml.devices.ExecutionConfig(
            mcm_config=qml.devices.MCMConfig(mcm_method=mcm_method)
        )
        config = dev.setup_execution_config(config)
        batch, fn = dev.preprocess_transforms(config)((tape,))

        res1, res2 = fn(dev.execute(batch, config))[0]
        atol = 0.05 if mcm_method == "one-shot" else 1e-6
        assert qml.math.allclose(res1, 0, atol=atol)
        assert qml.math.allclose(res2, 0, atol=atol)


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support derivatives",
)
@pytest.mark.parametrize("batch_obs", [True, False])
class TestDerivatives:
    """Unit tests for calculating derivatives with a device"""

    @staticmethod
    def calculate_reference(tape, execute_and_derivatives=False):
        device = DefaultQubit()
        program, config = device.preprocess(ExecutionConfig(gradient_method="adjoint"))
        tapes, transf_fn = program([tape])

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_derivatives(tapes, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_derivatives(tapes, config)
        return transf_fn(results), jac

    @staticmethod
    def process_and_execute(
        device, tape, execute_and_derivatives=False, obs_batch=False, use_default_config=False
    ):
        program, config = device.preprocess(
            ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": obs_batch})
        )
        tapes, transf_fn = program([tape])

        if use_default_config:
            config = None

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_derivatives(tapes, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_derivatives(tapes, config)
        return transf_fn(results), jac

    @pytest.mark.parametrize(
        "config, tape, expected",
        [
            (None, None, True),
            (ExecutionConfig(), None, False),
            (ExecutionConfig(gradient_method="backprop"), None, False),
            (
                ExecutionConfig(gradient_method="backprop"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))]),
                False,
            ),
            (ExecutionConfig(gradient_method="best"), None, True),
            (ExecutionConfig(gradient_method="adjoint"), None, True),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))]),
                True,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.var(qml.Z(0))]),
                False,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.state()]),
                False,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))], shots=10),
                False,
            ),
        ],
    )
    def test_supports_derivatives(self, dev, config, tape, expected, batch_obs):
        """Test that supports_derivative returns the correct boolean value."""
        assert dev.supports_derivatives(config, tape) == expected

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "obs",
        [
            qml.Z(1),
            2.5 * qml.Z(0),
            qml.Z(0) @ qml.X(1),
            qml.Z(1) + qml.X(1),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Hermitian(qml.Hadamard.compute_matrix(), 0),
            qml.SparseHamiltonian(
                qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]).sparse_matrix(
                    wire_order=[0, 1, 2]
                ),
                wires=[0, 1, 2],
            ),
            qml.Projector([1], 1),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize("use_default_config", [True, False])
    def test_derivatives_single_expval(
        self, theta, phi, dev, obs, execute_and_derivatives, batch_obs, use_default_config
    ):
        """Test that the jacobian is correct when a tape has a single expectation value"""
        if isinstance(obs, qml.SparseHamiltonian) and dev.c_dtype == np.complex64:
            pytest.skip(
                reason="The conversion from qml.Hamiltonian to SparseHamiltonian is only possible with np.complex128"
            )

        qs = QuantumScript(
            [qml.RX(theta, 0), qml.CNOT([0, 1]), qml.RY(phi, 1)],
            [qml.expval(obs)],
            trainable_params=[0, 1],
        )

        res, jac = self.process_and_execute(
            dev,
            qs,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
            use_default_config=use_default_config,
        )
        if isinstance(obs, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [qml.expval(qml.Hermitian(qml.matrix(obs), wires=obs.wires))],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == len(jac) == 1
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta, phi, omega", list(zip(THETA, PHI, VARPHI)))
    @pytest.mark.parametrize(
        "obs1",
        [
            qml.Z(1),
            2.5 * qml.Y(2),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Z(0) @ qml.X(1),
        ],
    )
    @pytest.mark.parametrize(
        "obs2",
        [
            qml.Y(0) @ qml.X(2),
            qml.Z(1) + qml.X(1),
            qml.SparseHamiltonian(
                qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]).sparse_matrix(
                    wire_order=[0, 1, 2]
                ),
                wires=[0, 1, 2],
            ),
            qml.Projector([1], wires=2),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_multi_expval(
        self, theta, phi, omega, dev, obs1, obs2, execute_and_derivatives, batch_obs
    ):
        """Test that the jacobian is correct when a tape has multiple expectation values"""
        if isinstance(obs2, qml.SparseHamiltonian) and dev.c_dtype == np.complex64:
            pytest.skip(
                reason="The conversion from qml.Hamiltonian to SparseHamiltonian is only possible with np.complex128"
            )

        qs = QuantumScript(
            [
                qml.RX(theta, 0),
                qml.CNOT([0, 1]),
                qml.RY(phi, 1),
                qml.CNOT([1, 2]),
                qml.RZ(omega, 2),
            ],
            [qml.expval(obs1), qml.expval(obs2)],
            trainable_params=[0, 1, 2],
        )

        res, jac = self.process_and_execute(
            dev,
            qs,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
        )
        if isinstance(obs1, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [
                    qml.expval(qml.Hermitian(qml.matrix(obs1), wires=obs1.wires)),
                    qml.expval(obs2),
                ],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, execute_and_derivatives=execute_and_derivatives
        )
        res = res[0]
        jac = jac[0]

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == len(jac) == 2
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_no_trainable_params(self, dev, execute_and_derivatives, batch_obs):
        """Test that the derivatives are empty with there are no trainable parameters."""
        qs = QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.S(1), qml.T(1)],
            [qml.expval(qml.Z(1))],
        )
        res, jac = self.process_and_execute(
            dev,
            qs,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
        )
        expected, _ = self.calculate_reference(qs, execute_and_derivatives=execute_and_derivatives)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert len(jac) == 1
        assert qml.math.shape(jac[0]) == (0,)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize(
        "state_prep, params, wires",
        [
            (qml.BasisState, [1, 1], [0, 1]),
            (qml.StatePrep, [0.0, 0.0, 0.0, 1.0], [0, 1]),
            (qml.StatePrep, qml.numpy.array([0.0, 1.0]), [1]),
        ],
    )
    @pytest.mark.parametrize(
        "trainable_params",
        [(0, 1, 2), (1, 2)],
    )
    def test_state_prep_ops(
        self,
        dev,
        state_prep,
        params,
        wires,
        execute_and_derivatives,
        batch_obs,
        trainable_params,
    ):
        """Test that a circuit containing state prep operations is differentiated correctly."""
        qs = QuantumScript(
            [
                state_prep(params, wires),
                qml.RX(1.23, 0),
                qml.CNOT([0, 1]),
                qml.RX(4.56, 1),
            ],
            [qml.expval(qml.PauliZ(1))],
        )

        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        program, new_config = dev.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            res, jac = dev.execute_and_compute_derivatives(tapes, new_config)
            res = fn(res)
        else:
            res, jac = (
                fn(dev.execute(tapes, new_config)),
                dev.compute_derivatives(tapes, new_config),
            )

        dev_ref = DefaultQubit()
        config = ExecutionConfig(gradient_method="adjoint")
        program, new_config = dev_ref.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            expected, expected_jac = dev_ref.execute_and_compute_derivatives(tapes, new_config)
            expected = fn(expected)
        else:
            expected, expected_jac = (
                fn(dev_ref.execute(tapes, new_config)),
                dev_ref.compute_derivatives(tapes, new_config),
            )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    def test_state_jacobian_not_supported(self, dev, batch_obs):
        """Test that an error is raised if derivatives are requested for state measurement"""
        qs = QuantumScript([qml.RX(1.23, 0)], [qml.state()], trainable_params=[0])
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            _ = dev.compute_derivatives(qs, config)

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            _ = dev.execute_and_compute_derivatives(qs, config)

    def test_shots_error_with_derivatives(self, dev, batch_obs):
        """Test that an error is raised if the gradient method is adjoint when the tape has shots"""
        qs = QuantumScript(
            [qml.RX(1.23, 0)], [qml.expval(qml.Z(0))], shots=10, trainable_params=[0]
        )
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        program, _ = dev.preprocess(config)

        with pytest.raises(DeviceError, match="Finite shots are not supported"):
            _, _ = program([qs])

    @pytest.mark.parametrize("device_wires", [None, 4])
    @pytest.mark.parametrize("phi", PHI)
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_tape_batch(self, device_wires, phi, execute_and_derivatives, batch_obs):
        """Test that results are correct when we execute and compute derivatives for a batch of
        tapes with and without dynamic wires."""

        device = LightningDevice(wires=device_wires, batch_obs=batch_obs)

        ops = [qml.X(0), qml.X(1)]
        if device_name == "lightning.qubit":
            ops.append(qml.ctrl(qml.RX(phi, 2), (0, 1, 3), control_values=[1, 1, 0]))
        else:
            ops.append(qml.RX(phi, 2))

        qs1 = QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.Y(2), qml.Z(1))),
                qml.expval(qml.s_prod(3, qml.Z(2))),
            ],
            trainable_params=[0],
        )

        ops = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        qs2 = QuantumScript(ops, [qml.expval(qml.prod(qml.Z(0), qml.Z(1)))], trainable_params=[0])

        if execute_and_derivatives:
            results, jacs = device.execute_and_compute_derivatives((qs1, qs2))

        else:
            results = device.execute((qs1, qs2))
            jacs = device.compute_derivatives((qs1, qs2))

        # Assert results
        expected1 = (-np.sin(phi) - 1, 3 * np.cos(phi))
        x1 = np.cos(phi / 2) ** 2 / 2
        x2 = np.sin(phi / 2) ** 2 / 2
        expected2 = sum([x1, -x2, -x1, x2])  # zero
        expected = (expected1, expected2)

        assert len(results) == len(expected)
        assert len(results[0]) == len(expected[0])
        assert np.allclose(results[0][0], expected[0][0])
        assert np.allclose(results[0][1], expected[0][1])
        assert np.allclose(results[1], expected[1])

        # Assert derivatives
        expected_jac1 = (-np.cos(phi), -3 * np.sin(phi))
        x1_jac = -np.cos(phi / 2) * np.sin(phi / 2) / 2
        x2_jac = np.sin(phi / 2) * np.cos(phi / 2) / 2
        expected_jac2 = sum([x1_jac, -x2_jac, -x1_jac, x2_jac])  # zero
        expected_jac = (expected_jac1, expected_jac2)

        assert len(jacs) == len(expected_jac)
        assert len(jacs[0]) == len(expected_jac[0])
        assert np.allclose(jacs[0], expected_jac[0])
        assert np.allclose(jacs[1], expected_jac[1])

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize("wires", (["a", "b", -3], [0, "target", "other_target"]))
    def test_derivatives_custom_wires(
        self, theta, phi, dev, execute_and_derivatives, batch_obs, wires
    ):
        """Test that the jacobian is correct when set custom wires"""
        device = LightningDevice(wires=wires)

        qs = QuantumScript(
            [
                qml.RX(theta, wires[0]),
                qml.CNOT([wires[0], wires[1]]),
                qml.RY(phi, wires[1]),
            ],
            [qml.expval(qml.Z(wires[1]))],
            trainable_params=[0, 1],
        )

        res, jac = self.process_and_execute(
            device,
            qs,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
        )
        expected, expected_jac = self.calculate_reference(
            qs, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == len(jac) == 1
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support vjp",
)
@pytest.mark.parametrize("batch_obs", [True, False])
class TestVJP:
    """Unit tests for VJP computation with the new device API."""

    @staticmethod
    def calculate_reference(tape, dy, execute_and_derivatives=False):
        device = DefaultQubit()
        program, config = device.preprocess(ExecutionConfig(gradient_method="adjoint"))
        tapes, transf_fn = program([tape])
        dy = [dy]

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_vjp(tapes, dy, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_vjp(tapes, dy, config)
        return transf_fn(results), jac

    @staticmethod
    def process_and_execute(
        device, tape, dy, execute_and_derivatives=False, obs_batch=False, use_default_config=False
    ):
        program, config = device.preprocess(
            ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": obs_batch})
        )
        tapes, transf_fn = program([tape])
        dy = [dy]

        if use_default_config:
            config = None

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_vjp(tapes, dy, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_vjp(tapes, dy, config)
        return transf_fn(results), jac

    @pytest.mark.parametrize(
        "config, tape, expected",
        [
            (None, None, True),
            (ExecutionConfig(), None, False),
            (
                None,
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))]),
                False,
            ),
            (
                None,
                QuantumScript([qml.RX(0.123, 0)], [qml.var(qml.Z(0))]),
                False,
            ),
            (ExecutionConfig(gradient_method="backprop"), None, False),
            (
                ExecutionConfig(gradient_method="backprop"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))]),
                False,
            ),
            (ExecutionConfig(gradient_method="best"), None, True),
            (ExecutionConfig(gradient_method="adjoint"), None, True),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))]),
                True,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.var(qml.Z(0))]),
                False,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.state()]),
                False,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))], shots=10),
                False,
            ),
        ],
    )
    def test_supports_vjp(self, dev, config, tape, expected, batch_obs):
        """Test that supports_vjp returns the correct boolean value."""
        assert dev.supports_vjp(config, tape) == expected

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "obs",
        [
            qml.Z(1),
            2.5 * qml.Z(0),
            qml.Z(0) @ qml.X(1),
            qml.Z(1) + qml.X(1),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Hermitian(qml.Hadamard.compute_matrix(), 0),
            qml.Projector([1], 1),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize("use_default_config", [True, False])
    def test_vjp_single_expval(
        self, theta, phi, dev, obs, execute_and_derivatives, batch_obs, use_default_config
    ):
        """Test that the VJP is correct when a tape has a single expectation value"""

        qs = QuantumScript(
            [qml.RX(theta, 0), qml.CNOT([0, 1]), qml.RY(phi, 1)],
            [qml.expval(obs)],
            trainable_params=[0, 1],
        )

        dy = 1.0
        res, jac = self.process_and_execute(
            dev,
            qs,
            dy,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
            use_default_config=use_default_config,
        )
        if isinstance(obs, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [qml.expval(qml.Hermitian(qml.matrix(obs), wires=obs.wires))],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, dy, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == len(jac) == 1
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta, phi, omega", list(zip(THETA, PHI, VARPHI)))
    @pytest.mark.parametrize(
        "obs1",
        [
            qml.Z(1),
            2.5 * qml.Y(2),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Z(0) @ qml.X(1),
        ],
    )
    @pytest.mark.parametrize(
        "obs2",
        [
            qml.Y(0) @ qml.X(2),
            qml.Z(1) + qml.X(1),
            qml.SparseHamiltonian(
                qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]).sparse_matrix(
                    wire_order=[0, 1, 2]
                ),
                wires=[0, 1, 2],
            ),
            qml.Projector([1], wires=2),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_vjp_multi_expval(
        self, theta, phi, omega, dev, obs1, obs2, execute_and_derivatives, batch_obs
    ):
        """Test that the VJP is correct when a tape has multiple expectation values"""
        if isinstance(obs2, qml.SparseHamiltonian) and dev.c_dtype == np.complex64:
            pytest.skip(
                reason="The conversion from qml.Hamiltonian to SparseHamiltonian is only possible with np.complex128"
            )

        qs = QuantumScript(
            [
                qml.RX(theta, 0),
                qml.CNOT([0, 1]),
                qml.RY(phi, 1),
                qml.CNOT([1, 2]),
                qml.RZ(omega, 2),
            ],
            [qml.expval(obs1), qml.expval(obs2)],
            trainable_params=[0, 1, 2],
        )
        dy = (1.0, 2.0)

        res, jac = self.process_and_execute(
            dev,
            qs,
            dy,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
        )
        if isinstance(obs1, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [
                    qml.expval(qml.Hermitian(qml.matrix(obs1), wires=obs1.wires)),
                    qml.expval(obs2),
                ],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, dy, execute_and_derivatives=execute_and_derivatives
        )
        res = res[0]
        jac = jac[0]

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == 2
        assert len(jac) == 3
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_vjp_no_trainable_params(self, dev, execute_and_derivatives, batch_obs):
        """Test that the VJP is empty with there are no trainable parameters."""
        qs = QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.S(1), qml.T(1)],
            [qml.expval(qml.Z(1))],
        )
        dy = 1.0

        res, jac = self.process_and_execute(
            dev,
            qs,
            dy,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
        )
        expected, _ = self.calculate_reference(
            qs, dy, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert len(jac) == 1
        assert qml.math.shape(jac[0]) == (0,)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize(
        "state_prep, params, wires",
        [
            (qml.BasisState, [1, 1], [0, 1]),
            (qml.StatePrep, [0.0, 0.0, 0.0, 1.0], [0, 1]),
            (qml.StatePrep, qml.numpy.array([0.0, 1.0]), [1]),
        ],
    )
    @pytest.mark.parametrize(
        "trainable_params",
        [(0, 1, 2), (1, 2)],
    )
    def test_state_prep_ops(
        self,
        dev,
        state_prep,
        params,
        wires,
        execute_and_derivatives,
        batch_obs,
        trainable_params,
    ):
        """Test that a circuit containing state prep operations is differentiated correctly."""
        qs = QuantumScript(
            [
                state_prep(params, wires),
                qml.RX(1.23, 0),
                qml.CNOT([0, 1]),
                qml.RX(4.56, 1),
            ],
            [qml.expval(qml.PauliZ(1))],
        )
        dy = [1.0]

        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        program, new_config = dev.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            res, jac = dev.execute_and_compute_vjp(tapes, dy, new_config)
            res = fn(res)
        else:
            res, jac = (
                fn(dev.execute(tapes, new_config)),
                dev.compute_vjp(tapes, dy, new_config),
            )

        dev_ref = DefaultQubit()
        config = ExecutionConfig(gradient_method="adjoint")
        program, new_config = dev_ref.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            expected, expected_jac = dev_ref.execute_and_compute_vjp(tapes, dy, new_config)
            expected = fn(expected)
        else:
            expected, expected_jac = (
                fn(dev_ref.execute(tapes, new_config)),
                dev_ref.compute_vjp(tapes, dy, new_config),
            )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    def test_state_vjp_not_supported(self, dev, batch_obs):
        """Test that an error is raised if VJP are requested for state measurement"""
        qs = QuantumScript([qml.RX(1.23, 0)], [qml.state()], trainable_params=[0])
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        dy = 1.0

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation does not support State measurements",
        ):
            _ = dev.compute_vjp(qs, dy, config)

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation does not support State measurements",
        ):
            _ = dev.execute_and_compute_vjp(qs, dy, config)

    @pytest.mark.parametrize("device_wires", [None, 4])
    @pytest.mark.parametrize("phi", PHI)
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_vjp_tape_batch(self, device_wires, phi, execute_and_derivatives, batch_obs):
        """Test that results are correct when we execute and compute vjp for a batch of
        tapes with and without dynamic wires."""

        device = LightningDevice(wires=device_wires, batch_obs=batch_obs)

        ops = [
            qml.X(0),
            qml.X(1),
        ]
        if device_name == "lightning.qubit":
            ops.append(qml.ctrl(qml.RX(phi, 2), (0, 1, 3), control_values=[1, 1, 0]))
        else:
            ops.append(qml.RX(phi, 2))

        qs1 = QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.Y(2), qml.Z(1))),
                qml.expval(qml.s_prod(3, qml.Z(2))),
            ],
            trainable_params=[0],
        )

        ops = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        qs2 = QuantumScript(ops, [qml.expval(qml.prod(qml.Z(0), qml.Z(1)))], trainable_params=[0])
        dy = [(1.5, 2.5), 1.0]

        if execute_and_derivatives:
            results, jacs = device.execute_and_compute_vjp((qs1, qs2), dy)

        else:
            results = device.execute((qs1, qs2))
            jacs = device.compute_vjp((qs1, qs2), dy)

        # Assert results
        expected1 = (-np.sin(phi) - 1, 3 * np.cos(phi))
        x1 = np.cos(phi / 2) ** 2 / 2
        x2 = np.sin(phi / 2) ** 2 / 2
        expected2 = sum([x1, -x2, -x1, x2])  # zero
        expected = (expected1, expected2)

        assert len(results) == len(expected)
        assert len(results[0]) == len(expected[0])
        assert np.allclose(results[0][0], expected[0][0])
        assert np.allclose(results[0][1], expected[0][1])
        assert np.allclose(results[1], expected[1])

        # Assert derivatives
        expected_jac1 = -1.5 * np.cos(phi) - 2.5 * 3 * np.sin(phi)
        x1_jac = -np.cos(phi / 2) * np.sin(phi / 2) / 2
        x2_jac = np.sin(phi / 2) * np.cos(phi / 2) / 2
        expected_jac2 = sum([x1_jac, -x2_jac, -x1_jac, x2_jac])  # zero
        expected_jac = (expected_jac1, expected_jac2)

        assert len(jacs) == len(expected_jac) == 2
        assert np.allclose(jacs[0], expected_jac[0])
        assert np.allclose(jacs[1], expected_jac[1])

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize("wires", (["a", "b", -3], [0, "target", "other_target"]))
    def test_vjp_custom_wires(self, theta, phi, dev, wires, execute_and_derivatives, batch_obs):
        """Test that the VJP is correct when set a custom wires"""

        device = LightningDevice(wires=wires)

        qs = QuantumScript(
            [
                qml.RX(theta, wires[0]),
                qml.CNOT([wires[0], wires[1]]),
                qml.RY(phi, wires[1]),
            ],
            [qml.expval(qml.Z(wires[1]))],
            trainable_params=[0, 1],
        )

        dy = 1.0
        res, jac = self.process_and_execute(
            device,
            qs,
            dy,
            execute_and_derivatives=execute_and_derivatives,
            obs_batch=batch_obs,
        )

        expected, expected_jac = self.calculate_reference(
            qs, dy, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == len(jac) == 1
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)


class TestLightningDeviceGraphModeExclusive:
    """Tests for LightningDevice features that require graph mode enabled.
    The legacy decomposition mode should not be able to run these tests.

    NOTE: All tests in this suite will auto-enable graph mode via fixture.
    """

    @pytest.fixture(autouse=True)
    def enable_graph_mode_only(self):
        """Auto-enable graph mode for all tests in this class."""
        qml.decomposition.enable_graph()
        yield
        qml.decomposition.disable_graph()

    def test_insufficient_work_wires_causes_fallback(self):
        """Test that if a decomposition requires more work wires than available on lightning device,
        that decomposition is discarded and fallback is used."""

        class MyLightningDeviceOp(qml.operation.Operator):
            num_wires = 1

        @qml.register_resources({qml.H: 2})
        def decomp_fallback(wires):
            qml.H(wires)
            qml.H(wires)

        @qml.register_resources({qml.X: 1}, work_wires={"burnable": 5})
        def decomp_with_work_wire(wires):
            qml.X(wires)

        qml.add_decomps(MyLightningDeviceOp, decomp_fallback, decomp_with_work_wire)

        tape = qml.tape.QuantumScript([MyLightningDeviceOp(0)])
        dev = LightningDevice(wires=1)  # Only 1 wire, but decomp needs 5 burnable
        program = dev.preprocess_transforms()
        (out_tape,), _ = program([tape])

        assert len(out_tape.operations) == 2
        assert out_tape.operations[0].name == "Hadamard"
        assert out_tape.operations[1].name == "Hadamard"
