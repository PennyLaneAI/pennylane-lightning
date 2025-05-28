# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module contains utility functions for working with JAX Jaxprs in the context of adjoint differentiation."""

from typing import Sequence, Tuple

import jax
import numpy as np
import pennylane as qml
from pennylane.capture.primitives import AbstractMeasurement
from pennylane.typing import TensorLike


def validate_args_tangents(
    args: Sequence[TensorLike], tangents: Sequence[TensorLike]
) -> Tuple[TensorLike]:
    """Ensure args and tangents provided to the JAXPR are consistent and creates concrete
    arrays of zeros where the input tangent is an instance of `jax.interpreters.ad.Zero`.

    Args:
        args (Sequence[TensorLike]): the arguments to the JAXPR
        tangents (Sequence[TensorLike]): the tangents to the JAXPR

    Returns:
        Tuple[TensorLike]: the validated tangents. Entries that are an instance of  `jax.intepreters.ad.Zero`
        are replaced by a concrete array of zeros.
    """

    if len(args) != len(tangents):
        raise ValueError("The number of arguments and tangents must match")

    def _make_zero(tan, arg):
        return (
            jax.lax.zeros_like_array(arg).astype(tan.aval.dtype)
            if isinstance(tan, jax.interpreters.ad.Zero)
            else tan
        )

    tangents = tuple(map(_make_zero, tangents, args))

    for tan in tangents:
        if jax.numpy.issubdtype(jax.numpy.asarray(tan).dtype, jax.numpy.integer):
            # will be able to support ad.Zero integer tangents when can track trainable parameters
            raise ValueError("Tangents cannot be of integer type yet.")

    return tangents


def get_output_shapes(jaxpr: "jax.extend.core.Jaxpr", num_wires: int) -> Tuple:
    """
    Compute the output shapes and Jacobian shapes of a JAXPR.

    Args:
        jaxpr (jax.extend.core.Jaxpr): the JAXPR to analyze
        num_wires (int): the number of wires

    Returns:
        Tuple: a tuple containing the output shapes of the JAXPR and the Jacobian shapes of the JAXPR
    """

    # pylint: disable=no-member
    dtype_map = {
        float: (jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32),
        int: jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32,
        complex: (jax.numpy.complex128 if jax.config.jax_enable_x64 else jax.numpy.complex64),
    }

    def _get_shape(var):
        if isinstance(var.aval, AbstractMeasurement):
            # Shots-based measurements are not supported at this time
            s, dtype = var.aval.abstract_eval(num_device_wires=num_wires, shots=0)
            return jax.core.ShapedArray(s, dtype_map[dtype])
        raise NotImplementedError("The circuit should return a measurement")

    def _flatten_shaped_array(aval):
        if aval.shape == ():
            return [aval]
        num_elements = int(np.prod(aval.shape))
        return [jax.core.ShapedArray((), aval.dtype) for _ in range(num_elements)]

    def _get_jacobian_shape(shape_res):
        # We assume that all arguments to the JAXPR are trainable parameters
        train_args = jaxpr.invars + jaxpr.constvars
        flattened = [scalar for var in train_args for scalar in _flatten_shaped_array(var.aval)]
        if len(jaxpr.outvars) == 1:
            return flattened
        return [flattened for _ in shape_res]

    shapes_res = [_get_shape(var) for var in jaxpr.outvars]
    shapes_jac = _get_jacobian_shape(shapes_res)
    return shapes_res, shapes_jac


def convert_jaxpr_to_tape(
    jaxpr: "jax.extend.core.Jaxpr", args: Sequence[TensorLike]
) -> qml.tape.QuantumTape:
    """
    Convert a jaxpr to a PennyLane tape and ensure parameters match.

    Args:
        jaxpr (jax.extend.core.Jaxpr): the JAXPR to convert
        args (Sequence[TensorLike]): the arguments to the JAXPR

    Returns:
        qml.tape.QuantumTape: the tape created from the input JAXPR.
    """

    const_args = args[: len(jaxpr.constvars)]
    non_const_args = args[len(jaxpr.constvars) :]

    tape = qml.tape.plxpr_to_tape(jaxpr, const_args, *non_const_args)
    tape_params = tape.get_parameters()

    len_train_inputs = sum(jax.numpy.size(p) for p in args)
    if not qml.math.allclose(args, tape_params) or len_train_inputs != len(tape.trainable_params):
        raise NotImplementedError(
            "The provided arguments do not match the parameters of the jaxpr converted to quantum tape."
        )

    return tape
