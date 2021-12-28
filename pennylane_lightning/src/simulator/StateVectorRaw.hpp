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
 * Defines the class representation for quantum state vectors.
 */

#pragma once
#include <cmath>
#include <complex>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "Gates.hpp"
#include "StateVectorBase.hpp"
#include "Util.hpp"

#include <iostream>

namespace Pennylane {

/**
 * @brief State-vector operations class.
 *
 * This class binds to a given statevector data array, and defines all
 * operations to manipulate the statevector data for quantum circuit simulation.
 * We define gates as methods to allow direct manipulation of the bound data, as
 * well as through a string-based function dispatch. The bound data is assumed
 * to be complex, and is required to be in either 32-bit (64-bit
 * `complex<float>`) or 64-bit (128-bit `complex<double>`) floating point
 * representation.
 *
 * @tparam fp_t Floating point precision of underlying statevector data.
 */
template <class fp_t = double>
class StateVectorRaw : public StateVectorBase<fp_t, StateVectorRaw<fp_t>> {
  public:
    using CFP_t = std::complex<fp_t>;

  private:
    CFP_t *data_;
    size_t length_;

  public:
    StateVectorRaw(CFP_t *data, size_t length)
        : StateVectorBase<fp_t, StateVectorRaw<fp_t>>(
              Util::log2PerfectPower(length)),
          data_{data}, length_(length) {
        // check length is perfect power of 2
        if (!Util::isPerfectPowerOf2(length)) {
            PL_ABORT("The length of the array for StateVector must be "
                     "a perfect power of 2. But " +
                     std::to_string(length) +
                     " is given."); // TODO: change to std::format in C++20
        }
    }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return const CFP_t* Pointer to statevector data.
     */
    [[nodiscard]] auto getData() const -> CFP_t * { return data_; }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return CFP_t* Pointer to statevector data.
     */
    auto getData() -> CFP_t * { return data_; }

    /**
     * @brief Redefine statevector data pointer.
     *
     * @param data_ptr New data pointer.
     */
    void setData(CFP_t *data) { data_ = data; }

    /**
     * @brief Redefine the length of the statevector and number of qubits.
     *
     * @param length New number of elements in statevector.
     */
    void setLength(size_t length) {
        if (!Util::isPerfectPowerOf2(length)) {
            PL_ABORT("The length of the array for StateVector must be "
                     "a perfect power of 2. But " +
                     std::to_string(length) +
                     " is given."); // TODO: change to std::format in C++20
        }
        length_ = length;
        setNumQubits(Util::log2PerfectPower(length_));
    }
    /**
     * @brief Redefine the number of qubits in the statevector and number of
     * elements.
     *
     * @param qubits New number of qubits represented by statevector.
     */
    void setNumQubits(size_t num_qubits) {
        setNumQubits(num_qubits);
        length_ = Util::exp2(num_qubits);
    }

    /**
     * @brief Get the number of data elements in the statevector array.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getLength() const -> std::size_t { return length_; }
};
} // namespace Pennylane
