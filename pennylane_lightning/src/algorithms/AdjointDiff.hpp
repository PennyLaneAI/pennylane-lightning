// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Represent the logic for the adjoint Jacobian method of arXiv:2009.02823
 */
#pragma once
#include "AlgUtil.hpp"
#include "Error.hpp"
#include "JacobianTape.hpp"
#include "LinearAlgebra.hpp"
#include "StateVectorManagedCPU.hpp"

#include <complex>
#include <numeric>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Pennylane::Algorithms {
/**
 * @brief Calculates the Jacobian for the statevector for the selected set
 * of parametric gates.
 *
 * For the statevector data associated with `psi` of length `num_elements`,
 * we make internal copies to a `%StateVectorManagedCPU<T>` object, with one
 * per required observable. The `operations` will be applied to the internal
 * statevector copies, with the operation indices participating in the
 * gradient calculations given in `trainableParams`, and the overall number
 * of parameters for the gradient calculation provided within `num_params`.
 * The resulting row-major ordered `jac` matrix representation will be of
 * size `jd.getSizeStateVec() * jd.getObservables().size()`. OpenMP is used
 * to enable independent operations to be offloaded to threads.
 *
 * @param jac Preallocated vector for Jacobian data results.
 * @param jd JacobianData represents the QuantumTape to differentiate.
 * @param apply_operations Indicate whether to apply operations to tape.psi
 * prior to calculation.
 */
template <typename T>
void adjointJacobian(std::span<T> jac, const JacobianData<T> &jd,
                     bool apply_operations = false) {
    const OpsData<T> &ops = jd.getOperations();
    const std::vector<std::string> &ops_name = ops.getOpsName();

    const auto &obs = jd.getObservables();
    const size_t num_observables = obs.size();

    // We can assume the trainable params are sorted (from Python)
    const std::vector<size_t> trainable_ops_indices = jd.getTrainableOpsIndcs();
    const size_t trainable_ops_size = trainable_ops_indices.size();

    if (!jd.hasTrainableOps()) {
        return;
    }

    for (const auto trainable_ops_idx : trainable_ops_indices) {
        PL_ABORT_IF_NOT(ops.getOpsParams()[trainable_ops_idx].size() == 1,
                        "Trainable operation must have a single parameter");
    }

    PL_ABORT_IF_NOT(jac.size() == trainable_ops_size * num_observables,
                    "The size of preallocated jacobian must be same as "
                    "the number of trainable parameters times the number of "
                    "observables provided.");

    // Track positions within par and non-par operations
    size_t trainable_ops_number = trainable_ops_size - 1;
    // Create $U_{1:p}\vert \lambda \rangle$
    StateVectorManagedCPU<T> lambda(jd.getPtrStateVec(), jd.getSizeStateVec());

    // Apply given operations to statevector if requested
    if (apply_operations) {
        applyOperations(lambda, ops);
    }

    // Create observable-applied state-vectors
    std::vector<StateVectorManagedCPU<T>> H_lambda(
        num_observables, StateVectorManagedCPU<T>{lambda.getNumQubits()});
    applyObservables(H_lambda, lambda, obs);

    StateVectorManagedCPU<T> mu(lambda.getNumQubits());

    for (int op_idx = std::ssize(ops_name) - 1; op_idx >= 0; op_idx--) {
        PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                    "The operation is not supported using the adjoint "
                    "differentiation method");
        if ((ops_name[op_idx] == "QubitStateVector") ||
            (ops_name[op_idx] == "BasisState")) {
            continue; // Ignore them
        }

        if (static_cast<std::size_t>(op_idx) < trainable_ops_indices[0]) {
            break; // All done
        }
        mu.updateData(lambda.getDataVector());
        applyOperation(lambda, ops, op_idx, true);

        if (ops.hasParams(op_idx) &&
            (std::find(trainable_ops_indices.begin(),
                       trainable_ops_indices.end(),
                       op_idx) != trainable_ops_indices.end())) {
            // if current parameter is a trainable parameter
            const T scalingFactor =
                mu.applyGenerator(ops_name[op_idx], ops.getOpsWires()[op_idx],
                                  !ops.getOpsInverses()[op_idx]) *
                (ops.getOpsInverses()[op_idx] ? -1 : 1);

            const size_t mat_row_idx = trainable_ops_number * num_observables;

            // clang-format off
            
            #if defined(_OPENMP)
            #pragma omp parallel for default(none)                         \
                shared(H_lambda, jac, mu, scalingFactor, mat_row_idx,      \
                        num_observables)
            #endif
            // clang-format on

            for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
                jac[mat_row_idx + obs_idx] =
                    -2 * scalingFactor *
                    std::imag(Util::innerProdC(
                        H_lambda[obs_idx].getDataVector(), mu.getDataVector()));
            }
            --trainable_ops_number;
        }
        applyOperationsAdj(H_lambda, ops, static_cast<size_t>(op_idx));
    }
    const auto jac_transpose = Util::Transpose(
        std::span<const T>{jac}, trainable_ops_size, num_observables);
    std::copy(std::begin(jac_transpose), std::end(jac_transpose),
              std::begin(jac));
}
} // namespace Pennylane::Algorithms
