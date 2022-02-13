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
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "BitUtil.hpp"
#include "Error.hpp"
#include "StateVectorBase.hpp"

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
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 */
template <class PrecisionT = double>
class StateVectorRaw
    : public StateVectorBase<PrecisionT, StateVectorRaw<PrecisionT>> {
  public:
    using Base = StateVectorBase<PrecisionT, StateVectorRaw<PrecisionT>>;
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    ComplexPrecisionT *data_;
    size_t length_;

  public:
    /**
     * @brief Construct state-vector from a raw data pointer.
     *
     * @param data Raw data pointer.
     * @param length The size of the data, i.e. 2^(number of qubits).
     */
    StateVectorRaw(ComplexPrecisionT *data, size_t length)
        : StateVectorBase<PrecisionT, StateVectorRaw<PrecisionT>>(
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

    StateVectorRaw(const StateVectorRaw &) = default;
    StateVectorRaw(StateVectorRaw &&) noexcept = default;

    auto operator=(const StateVectorRaw &) -> StateVectorRaw & = default;
    auto operator=(StateVectorRaw &&) noexcept -> StateVectorRaw & = default;

    ~StateVectorRaw() = default;

    /**
     * @brief Get the underlying data pointer.
     *
     * @return const ComplexPrecisionT* Pointer to statevector data.
     */
    [[nodiscard]] auto getData() const -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return ComplexPrecisionT* Pointer to statevector data.
     */
    auto getData() -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Redefine statevector data pointer.
     *
     * @param data New raw data pointer.
     * @param length The size of the data, i.e. 2^(number of qubits).
     */
    void setData(ComplexPrecisionT *data, size_t length) {
        if (!Util::isPerfectPowerOf2(length)) {
            PL_ABORT("The length of the array for StateVector must be "
                     "a perfect power of 2. But " +
                     std::to_string(length) +
                     " is given."); // TODO: change to std::format in C++20
        }
        data_ = data;
        Base::setNumQubits(Util::log2PerfectPower(length));
        length_ = length;
    }

    /**
     * @brief Get the number of data elements in the statevector array.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getLength() const -> std::size_t { return length_; }
};
} // namespace Pennylane
