// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
 * @file MeasurementsBase.hpp
 * Defines the Measurements CRTP base class.
 */
#pragma once

#include <vector>

#include "Observables.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Observables;
} // namespace
/// @endcond

namespace Pennylane::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class performs measurements in the state vector provided to its
 * constructor. Observables are defined by its operator(matrix), the observable
 * class, or through a string-based function dispatch.
 *
 * @tparam StateVectorT
 * @tparam Derived
 */
template <class StateVectorT, class Derived> class MeasurementsBase {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

  protected:
#ifdef _ENABLE_PLGPU
    StateVectorT &_statevector;
#else
    const StateVectorT &_statevector;
#endif

  public:
#ifdef _ENABLE_PLGPU
    explicit MeasurementsBase(StateVectorT &statevector)
        : _statevector{statevector} {};
#else
    explicit MeasurementsBase(const StateVectorT &statevector)
        : _statevector{statevector} {};
#endif

    /**
     * @brief Calculate the expectation value for a general Observable.
     *
     * @param obs Observable.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &obs) -> PrecisionT {
        return static_cast<Derived *>(this)->expval(obs);
    }

    /**
     * @brief Calculate the variance for a general Observable.
     *
     * @param obs Observable.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &obs) -> PrecisionT {
        return static_cast<Derived *>(this)->var(obs);
    }

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<PrecisionT> {
        return static_cast<Derived *>(this)->probs();
    };

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto probs(const std::vector<size_t> &wires) -> std::vector<PrecisionT> {
        return static_cast<Derived *>(this)->probs(wires);
    };

    /**
     * @brief  Generate samples
     *
     * @param num_samples Number of samples
     * @return 1-D vector of samples in binary with each sample
     * separated by a stride equal to the number of qubits.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {
        return static_cast<Derived *>(this)->generate_samples(num_samples);
    };
};

} // namespace Pennylane::Measures