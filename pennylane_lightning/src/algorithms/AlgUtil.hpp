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

#include "JacobianTape.hpp"
#include "LinearAlgebra.hpp"
#include "StateVectorManagedCPU.hpp"
#include "Util.hpp"

#include <stdexcept>
#include <variant>

namespace Pennylane::Algorithms {
/**
 * @brief Utility method to apply all operations from given `%OpsData<T>`
 * object to `%StateVectorManagedCPU<T>`
 *
 * @param state Statevector to be updated.
 * @param operations Operations to apply.
 * @param adj Take the adjoint of the given operations.
 */
template <typename T>
inline void applyOperations(StateVectorManagedCPU<T> &state,
                            const OpsData<T> &operations, bool adj = false) {
    for (size_t op_idx = 0; op_idx < operations.getOpsName().size(); op_idx++) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             operations.getOpsInverses()[op_idx] ^ adj,
                             operations.getOpsParams()[op_idx]);
    }
}
/**
 * @brief Utility method to apply the adjoint indexed operation from
 * `%OpsData<T>` object to `%StateVectorManagedCPU<T>`.
 *
 * @param state Statevector to be updated.
 * @param operations Operations to apply.
 * @param op_idx Adjointed operation index to apply.
 */
template <typename T>
inline void applyOperationAdj(StateVectorManagedCPU<T> &state,
                              const OpsData<T> &operations, size_t op_idx) {
    state.applyOperation(operations.getOpsName()[op_idx],
                         operations.getOpsWires()[op_idx],
                         !operations.getOpsInverses()[op_idx],
                         operations.getOpsParams()[op_idx]);
}

/**
 * @brief Utility method to apply a given operations from given
 * `%ObsDatum<T>` object to `%StateVectorManagedCPU<T>`
 *
 * @param state Statevector to be updated.
 * @param observable Observable to apply.
 */
template <typename T>
inline void applyObservable(StateVectorManagedCPU<T> &state,
                            Simulators::Observable<T> &observable) {
    observable.applyInPlace(state);
}

/**
 * @brief OpenMP accelerated application of observables to given
 * statevectors
 *
 * @param states Vector of statevector copies, one per observable.
 * @param reference_state Reference statevector
 * @param observables Vector of observables to apply to each statevector.
 */
template <typename T>
inline void
applyObservables(std::vector<StateVectorManagedCPU<T>> &states,
                 const StateVectorManagedCPU<T> &reference_state,
                 const std::vector<std::shared_ptr<Simulators::Observable<T>>>
                     &observables) {
    std::exception_ptr ex = nullptr;
    size_t num_observables = observables.size();

    if (num_observables > 1) {
        /* Globally scoped exception value to be captured within OpenMP block.
         * See the following for OpenMP design decisions:
         * https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
         * */
        // clang-format off

        #if defined(_OPENMP)
            #pragma omp parallel default(none)                                 \
            shared(states, reference_state, observables, ex, num_observables)
        {
            #pragma omp for
        #endif
            for (size_t h_i = 0; h_i < num_observables; h_i++) {
                try {
                    states[h_i].updateData(reference_state.getDataVector());
                    applyObservable(states[h_i], *observables[h_i]);
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
    } else {
        states[0].updateData(reference_state.getDataVector());
        applyObservable(states[0], *observables[0]);
    }
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
template <typename T>
inline void applyOperationsAdj(std::vector<StateVectorManagedCPU<T>> &states,
                               const OpsData<T> &operations, size_t op_idx) {
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
        for (size_t st_idx = 0; st_idx < num_states; st_idx++) {
            try {
                applyOperationAdj(states[st_idx], operations, op_idx);
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
} // namespace Pennylane::Algorithms
