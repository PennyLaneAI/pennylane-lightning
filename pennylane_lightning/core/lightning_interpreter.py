# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains a class for executing plxpr using default qubit tools.
"""

import jax
from pennylane.capture import disable, enable
from pennylane.capture.base_interpreter import FlattenedHigherOrderPrimitives, PlxprInterpreter
from pennylane.capture.primitives import adjoint_transform_prim, ctrl_transform_prim, measure_prim
from pennylane.measurements import MidMeasureMP, Shots

from ._measurements_base import LightningBaseMeasurements
from ._state_vector_base import LightningBaseStateVector


class LightningInterpreter(PlxprInterpreter):
    """A class that can interpret pennylane variant JAXPR.

    Args:
        state (LightningBaseStateVector): the class containing the statevector
        measurement_class (type[LightningBaseMeasurement]): The type to use to perform
            measurements on the statevector
        shots (Shots): the number of shots to use. Shot vectors are not yet supported.

    .. code-block:: python

        import pennylane as qml
        import jax
        qml.capture.enable()

        statevector = LightningStateVector(num_wires=3, dtype=np.complex128)
        measurement_class = LightningMeasurements

        interpreter = LightningInterpreter(statevector, measurement_class, shots=Shots(None))

        def f(x):
            @qml.for_loop(3)
            def loop(i, y):
                qml.RX(y, i)
                return y + 0.5
            loop(x)
            return [qml.expval(qml.Z(i)) for i in range(3)]

        jaxpr = jax.make_jaxpr(f)(1.2)

        interpreter(f)(0.0), interpreter.eval(jaxpr.jaxpr, jaxpr.consts, 0.0)

    .. code-block::

        ([1.0, 0.8775825618903728, 0.5403023058681395], [1.0, 0.8775825618903728, 0.5403023058681395])
    """

    def __init__(
        self,
        state: LightningBaseStateVector,
        measurement_class: type[LightningBaseMeasurements],
        shots: Shots = Shots(None),
    ):
        self.state = state
        self.measurement_class = measurement_class
        self.shots = shots
        if self.shots.has_partitioned_shots:
            raise NotImplementedError("LightningInterpreter does not support partitioned shots.")
        self.reset = True
        super().__init__()

    def setup(self) -> None:
        """Reset the state if necessary."""
        if self.reset:
            self.state.reset_state()
            self.reset = False  # copies will have reset=False and wont reset state

    def cleanup(self) -> None:
        """Indicate that the state will need to be reset if this instance is reused."""
        self.reset = True

    def interpret_operation(self, op):
        """Apply an operation to the state."""
        self.state.apply_operations([op])

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret a given measurement equation."""
        if "mcm" == eqn.primitive.name[-3:]:
            raise NotImplementedError(
                "LightningInterpreter does not yet support postprocessing mcms"
            )
        return super().interpret_measurement_eqn(eqn)

    def interpret_measurement(self, measurement):
        """Apply a measurement to the state and return numerical results."""
        # measurements can sometimes create intermediary mps, but those intermediaries will not work with capture enabled
        disable()
        try:
            if self.shots:
                return self.measurement_class(self.state).measure_with_samples(
                    [measurement], self.shots
                )
            return self.measurement_class(self.state).measurement(measurement)
        finally:
            enable()


# pylint: disable=protected-access
LightningInterpreter._primitive_registrations.update(FlattenedHigherOrderPrimitives)


@LightningInterpreter.register_primitive(measure_prim)
def _(self, *invals, reset, postselect):
    mp = MidMeasureMP(invals, reset=reset, postselect=postselect)
    mcms = {}
    self.state.apply_operations([mp], mid_measurements=mcms)
    return mcms[mp]


# pylint: disable=unused-argument
@LightningInterpreter.register_primitive(adjoint_transform_prim)
def _(self, *invals, jaxpr, n_consts, lazy=True):
    # TODO: requires jaxpr -> list of ops first
    raise NotImplementedError


# pylint: disable=too-many-arguments
@LightningInterpreter.register_primitive(ctrl_transform_prim)
def _(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    # TODO: requires jaxpr -> list of ops first
    raise NotImplementedError
