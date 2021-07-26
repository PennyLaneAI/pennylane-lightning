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
 * Contains uncategorised utility functions.
 */
#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace Pennylane {

namespace Util {

/**
 * Calculates 2^n -1 for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n -1
 */
inline size_t exp2(const size_t &n) { return static_cast<size_t>(1) << n; }

inline size_t log2(size_t value) {
    return static_cast<size_t>(std::log2(value));
}

/**
 * Calculates the decimal value for a qubit, assuming a big-endian convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline size_t maxDecimalForQubit(size_t qubitIndex, size_t qubits) {
    assert(qubitIndex < qubits);
    return exp2(qubits - qubitIndex - 1);
}

template <class DataPrecision>
inline static DataPrecision InvSqrt(DataPrecision value) {
    return 1 / std::sqrt(value);
}

template <class DataPrecision = double>
inline static constexpr DataPrecision ONE() {
    return std::complex<DataPrecision>{1, 0};
}
template <class DataPrecision = double>
inline static constexpr DataPrecision ZERO() {
    return std::complex<DataPrecision>{0, 0};
}
template <class DataPrecision = double>
inline static constexpr DataPrecision IMAG() {
    return std::complex<DataPrecision>{0, 1};
}

} // namespace Util
} // namespace Pennylane

// Helper similar to std::make_unique from c++14
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Exception for functions that aren't implemented
class NotImplementedException : public std::logic_error {
  public:
    explicit NotImplementedException(const std::string &fname)
        : std::logic_error(std::string("Function is not implemented. ") +
                           fname){};
};
