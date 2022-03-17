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
 * Define Adjoint method for calculating Jacobian of an observable
 */
#pragma once
#include "AlgUtil.hpp"
#include "Error.hpp"
#include "JacobianTape.hpp"
#include "LinearAlgebra.hpp"
#include "StateVectorManaged.hpp"

#include <complex>
#include <numeric>
#include <type_traits>
#include <string>
#include <utility>
#include <vector>

/// @cond DEV
namespace {

using namespace Pennylane;
using namespace Pennylane::Util;

} // namespace
/// @endcond

namespace Pennylane::Algorithms {
/**
 * @brief Represent the logic for the adjoint Jacobian method of
 * arXiv:2009.02823
 *
 * @tparam T Floating-point precision.
 */
template <class T = double> class AdjointJacobian {
  private:
    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param sv1 Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param obs_index Observable index position of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    inline static void updateJacobian(const StateVectorManaged<T> &sv1,
                               const StateVectorManaged<T> &sv2,
                               std::vector<std::vector<T>> &jac,
                               T scaling_coeff, size_t obs_index,
                               size_t param_index) {
        jac[obs_index][param_index] =
            -2 * scaling_coeff *
            std::imag(innerProdC(sv1.getDataVector(), sv2.getDataVector()));
    }

    /**
     * @brief Inline utility to assist with getting the Jacobian index offset.
     *
     * @param obs_index
     * @param tp_index
     * @param tp_size
     * @return size_t
     */
    inline static auto getJacIndex(size_t obs_index, size_t tp_index, 
                                   size_t tp_size)
        -> size_t {
        return obs_index * tp_size + tp_index;
    }

    /**
     * @brief Copies complex data array into a `%vector` of the same dimension.
     *
     * @param input_state
     * @param state_length
     * @return std::vector<std::complex<T>>
     */
    inline static auto copyStateData(const std::complex<T> *input_state, size_t state_length)
        -> std::vector<std::complex<T>> {
        return {input_state, input_state + state_length};
    }

  public:
    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies to a `%StateVectorManaged<T>` object, with one
     * per required observable. The `operations` will be applied to the internal
     * statevector copies, with the operation indices participating in the
     * gradient calculations given in `trainableParams`, and the overall number
     * of parameters for the gradient calculation provided within `num_params`.
     * The resulting row-major ordered `jac` matrix representation will be of
     * size `jd.getSizeStateVec() * jd.getObservables().size()`. OpenMP is used
     * to enable independent operations to be offloaded to threads.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate
     * @param apply_operations Indicate whether to apply operations to tape.psi
     * prior to calculation.
     */
    static void adjointJacobian(std::vector<T> &jac, const JacobianData<T> &jd,
                         bool apply_operations = false) {
        PL_ABORT_IF(!jd.hasTrainableParams(),
                    "No trainable parameters provided.");

        const OpsData<T> &ops = jd.getOperations();
        const std::vector<std::string> &ops_name = ops.getOpsName();

        const std::vector<ObsDatum<T>> &obs = jd.getObservables();
        const size_t num_observables = obs.size();

        // We can assume the trainable params are sorted (from Python)
        const std::vector<size_t> &tp = jd.getTrainableParams();
        const size_t tp_size = tp.size();
        const size_t num_param_ops = ops.getNumParOps();

        // Track positions within par and non-par operations
        size_t trainableParamNumber = tp_size - 1;
        size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorManaged<T> lambda(jd.getPtrStateVec(), jd.getSizeStateVec());

        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda, ops);
        }

        const auto tp_rend = tp.rend();
        auto tp_it = tp.rbegin();

        // Create observable-applied state-vectors
        std::vector<StateVectorManaged<T>> H_lambda(
            num_observables, StateVectorManaged<T>{lambda.getNumQubits()});
        applyObservables(H_lambda, lambda, obs);

        StateVectorManaged<T> mu(lambda.getNumQubits());

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "QubitStateVector") ||
                (ops_name[op_idx] == "BasisState")) {
                continue; // Ignore them
            }

            if(tp_it == tp_rend) {
                break; // All done
            }
            mu.updateData(lambda.getDataVector());
            applyOperationAdj(lambda, ops, op_idx);

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    // if current parameter is a trainable parameter
                    const T scalingFactor =
                        mu.applyGenerator(ops_name[op_idx],
                                       ops.getOpsWires()[op_idx],
                                       !ops.getOpsInverses()[op_idx]) *
                        (ops.getOpsInverses()[op_idx] ? -1 : 1);

                    const size_t mat_row_idx =
                        trainableParamNumber * num_observables;

                    // clang-format off

                    #if defined(_OPENMP)
                        #pragma omp parallel for default(none)   \
                        shared(H_lambda, jac, mu, scalingFactor, \
                            mat_row_idx,        \
                            num_observables)
                    #endif

                    // clang-format on
                    for (size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        jac[mat_row_idx + obs_idx] =
                            -2 * scalingFactor *
                            std::imag(innerProdC(
                                H_lambda[obs_idx].getDataVector(),
                                mu.getDataVector()));
                    }
                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            applyOperationsAdj(H_lambda, ops, static_cast<size_t>(op_idx));
        }
        jac = Transpose(jac, jd.getNumParams(), num_observables);
    }
}; // class AdjointJacobian
} // namespace Pennylane::Algorithms
