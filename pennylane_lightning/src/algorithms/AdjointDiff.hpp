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
#pragma once

#include <complex>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "DynamicDispatcher.hpp"
#include "Error.hpp"
#include "JacobianTape.hpp"
#include "LinearAlgebra.hpp"
#include "StateVectorManaged.hpp"

#include <iostream>

/// @cond DEV
namespace {

using namespace Pennylane;
using namespace Pennylane::Util;

} // namespace
/// @endcond

namespace Pennylane::Algorithms {
/**
 * @brief Represent the logic for the adjoint Jacobian method of
 * arXiV:2009.02823
 *
 * @tparam T Floating-point precision.
 */
template <class T = double> class AdjointJacobian {
  private:
    using GeneratorFunc = void (*)(StateVectorManaged<T> &,
                                   const std::vector<size_t> &,
                                   const bool); // function pointer type

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
    inline void updateJacobian(const StateVectorManaged<T> &sv1,
                               const StateVectorManaged<T> &sv2,
                               std::vector<std::vector<T>> &jac,
                               T scaling_coeff, size_t obs_index,
                               size_t param_index) {
        jac[obs_index][param_index] =
            -2 * scaling_coeff *
            std::imag(innerProdC(sv1.getDataVector(), sv2.getDataVector()));
    }

    /**
     * @brief Utility method to apply all operations from given `%OpsData<T>`
     * object to `%StateVectorManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    inline void applyOperations(StateVectorManaged<T> &state,
                                const OpsData<T> &operations,
                                bool adj = false) {
        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 operations.getOpsInverses()[op_idx] ^ adj,
                                 operations.getOpsParams()[op_idx]);
        }
    }
    /**
     * @brief Utility method to apply the adjoint indexed operation from
     * `%OpsData<T>` object to `%StateVectorManaged<T>`.
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    inline void applyOperationAdj(StateVectorManaged<T> &state,
                                  const OpsData<T> &operations, size_t op_idx) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             !operations.getOpsInverses()[op_idx],
                             operations.getOpsParams()[op_idx]);
    }

    /**
     * @brief Utility method to apply a given operations from given
     * `%ObsDatum<T>` object to `%StateVectorManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param observable Observable to apply.
     */
    inline void applyObservable(StateVectorManaged<T> &state,
                                const ObsDatum<T> &observable) {
        using namespace Pennylane::Util;
        for (size_t j = 0; j < observable.getSize(); j++) {
            if (!observable.getObsParams().empty()) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        // Apply supported gate with given params
                        if constexpr (std::is_same_v<p_t, std::vector<T>>) {
                            state.applyOperation(observable.getObsName()[j],
                                                 observable.getObsWires()[j],
                                                 false, param);
                        }
                        // Apply provided matrix
                        else if constexpr (std::is_same_v<
                                               p_t,
                                               std::vector<std::complex<T>>>) {
                            state.applyMatrix(
                                param, observable.getObsWires()[j], false);
                        } else {
                            state.applyOperation(observable.getObsName()[j],
                                                 observable.getObsWires()[j],
                                                 false);
                        }
                    },
                    observable.getObsParams()[j]);
            } else { // Offloat to SV dispatcher if no parameters provided
                state.applyOperation(observable.getObsName()[j],
                                     observable.getObsWires()[j], false);
            }
        }
    }

    /**
     * @brief OpenMP accelerated application of observables to given
     * statevectors
     *
     * @param states Vector of statevector copies, one per observable.
     * @param reference_state Reference statevector
     * @param observables Vector of observables to apply to each statevector.
     */
    inline void applyObservables(std::vector<StateVectorManaged<T>> &states,
                                 const StateVectorManaged<T> &reference_state,
                                 const std::vector<ObsDatum<T>> &observables) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_observables = observables.size();
        #if defined(_OPENMP)
            #pragma omp parallel default(none)                                 \
            shared(states, reference_state, observables, ex, num_observables)
        {
            #pragma omp for
        #endif
            for (size_t h_i = 0; h_i < num_observables; h_i++) {
                try {
                    states[h_i].updateData(reference_state.getDataVector());
                    applyObservable(states[h_i], observables[h_i]);
                } catch (...) {
                    #if defined(_OPENMP)
                        #pragma omp critical
                    #endif
                    ex = std::current_exception();
                    #if defined(_OPENMP)
                        #pragma omp cancel for
                    #endif
                }
            }
        #if defined(_OPENMP)
            if (ex) {
                #pragma omp cancel parallel
            }
        }
        #endif
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief OpenMP accelerated application of adjoint operations to
     * statevectors.
     *
     * @param states Vector of all statevectors; 1 per observable
     * @param operations Operations list.
     * @param op_idx Index of given operation within operations list to take
     * adjoint of.
     */
    inline void applyOperationsAdj(std::vector<StateVectorManaged<T>> &states,
                                   const OpsData<T> &operations,
                                   size_t op_idx) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_states = states.size();
        #if defined(_OPENMP)
            #pragma omp parallel default(none)                                 \
                shared(states, operations, op_idx, ex, num_states)
        {
            #pragma omp for
        #endif
            for (size_t obs_idx = 0; obs_idx < num_states; obs_idx++) {
                try {
                    applyOperationAdj(states[obs_idx], operations, op_idx);
                } catch (...) {
                    #if defined(_OPENMP)
                        #pragma omp critical
                    #endif
                    ex = std::current_exception();
                    #if defined(_OPENMP)
                        #pragma omp cancel for
                    #endif
                }
            }
        #if defined(_OPENMP)
            if (ex) {
                #pragma omp cancel parallel
            }
        }
        #endif
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief Inline utility to assist with getting the Jacobian index offset.
     *
     * @param obs_index
     * @param tp_index
     * @param tp_size
     * @return size_t
     */
    inline auto getJacIndex(size_t obs_index, size_t tp_index, size_t tp_size)
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
    auto copyStateData(const std::complex<T> *input_state, size_t state_length)
        -> std::vector<std::complex<T>> {
        return {input_state, input_state + state_length};
    }

    /**
     * @brief Applies the gate generator for a given parameteric gate. Returns
     * the associated scaling coefficient.
     *
     * @param sv Statevector data to operate upon.
     * @param op_name Name of parametric gate.
     * @param wires Wires to operate upon.
     * @param adj Indicate whether to take the adjoint of the operation.
     * @return T Generator scaling coefficient.
     */
    template <class SVType>
    inline auto applyGenerator(StateVectorBase<T, SVType> &sv,
                               const std::string &op_name,
                               const std::vector<size_t> &wires, const bool adj)
        -> T {
        return sv.applyGenerator(op_name, wires, adj);
    }

  public:
    AdjointJacobian() = default;

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
    void adjointJacobian(std::vector<T> &jac, const JacobianData<T> &jd,
                         bool apply_operations = false) {
        PL_ABORT_IF(!jd.hasTrainableParams(),
                    "No trainable parameters provided.");

        const OpsData<T> &ops = jd.getOperations();
        const std::vector<std::string> &ops_name = ops.getOpsName();

        const std::vector<ObsDatum<T>> &obs = jd.getObservables();
        const size_t num_observables = obs.size();

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

        const auto tp_begin = tp.begin();
        auto tp_it = tp.end();

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
            if ((ops_name[op_idx] != "QubitStateVector") &&
                (ops_name[op_idx] != "BasisState")) {
                mu.updateData(lambda.getDataVector());
                applyOperationAdj(lambda, ops, op_idx);

                if (ops.hasParams(op_idx)) {
                    if ((current_param_idx == *(std::prev(tp_it))) ||
                        std::find(tp_begin, tp_it, current_param_idx) !=
                            tp_it) {
                        const T scalingFactor =
                            applyGenerator(mu, ops_name[op_idx],
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
                        std::advance(tp_it, -1);
                    }
                    current_param_idx--;
                }
                applyOperationsAdj(H_lambda, ops, static_cast<size_t>(op_idx));
            }
        }
        jac = Transpose(jac, jd.getNumParams(), num_observables);
    }
}; // class AdjointJacobian
} // namespace Pennylane::Algorithms
