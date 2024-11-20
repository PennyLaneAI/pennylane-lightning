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
from copy import copy
from typing import Union

import jax
import numpy as np

from pennylane.capture import disable, enable
from pennylane.capture.base_interpreter import PlxprInterpreter
from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    measure_prim,
    while_loop_prim,
)
from pennylane.measurements import MidMeasureMP, Shots

from ._measurements import LightningMeasurements
from ._state_vector import LightningStateVector

class LightningInterpreter(PlxprInterpreter):


    def __init__(
        self, num_wires: int, shots: int | None = None, c_dtype: Union[np.complex128, np.complex64] = np.complex128,
    ):
        self.num_wires = num_wires
        self.shots = Shots(shots)

        self.reset = True
        self.stateref : dict = {"state": LightningStateVector(num_wires=num_wires, dtype=c_dtype)}
        super().__init__()

    @property
    def state(self) -> LightningStateVector:
        """The current state of the system. None if not initialized."""
        return self.stateref["state"]

    def setup(self) -> None:
        if self.reset:
            self.state.reset_state()
            self.reset = False # copies will have reset=False and wont reset state

    def cleanup(self) -> None:
        self.reset = True

    def interpret_operation(self, op):
        self.state.apply_operations([op])

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        if "mcm" in eqn.primitive.name:
            raise NotImplementedError(
                "DefaultQubitInterpreter does not yet support postprocessing mcms"
            )
        return super().interpret_measurement_eqn(eqn)

    def interpret_measurement(self, measurement):
        # measurements can sometimes create intermediary mps, but those intermediaries will not work with capture enabled
        disable()
        try:
            if self.shots:
                return LightningMeasurements(self.state).measure_with_samples([measurement], self.shots)
            return LightningMeasurements(self.state).measurement(measurement)
        finally:
            enable()


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


# pylint: disable=too-many-arguments
@LightningInterpreter.register_primitive(for_loop_prim)
def _(self, start, stop, step, *invals, jaxpr_body_fn, consts_slice, args_slice):
    consts = invals[consts_slice]
    init_state = invals[args_slice]

    res = init_state
    for i in range(start, stop, step):
        res = copy(self).eval(jaxpr_body_fn, consts, i, *res)

    return res


# pylint: disable=too-many-arguments
@LightningInterpreter.register_primitive(while_loop_prim)
def _(self, *invals, jaxpr_body_fn, jaxpr_cond_fn, body_slice, cond_slice, args_slice):
    consts_body = invals[body_slice]
    consts_cond = invals[cond_slice]
    init_state = invals[args_slice]

    fn_res = init_state
    while copy(self).eval(jaxpr_cond_fn, consts_cond, *fn_res)[0]:
        fn_res = copy(self).eval(jaxpr_body_fn, consts_body, *fn_res)

    return fn_res


@LightningInterpreter.register_primitive(cond_prim)
def _(self, *invals, jaxpr_branches, consts_slices, args_slice):
    n_branches = len(jaxpr_branches)
    conditions = invals[:n_branches]
    args = invals[args_slice]

    for pred, jaxpr, const_slice in zip(conditions, jaxpr_branches, consts_slices):
        consts = invals[const_slice]
        if pred and jaxpr is not None:
            return copy(self).eval(jaxpr, consts, *args)
    return ()
