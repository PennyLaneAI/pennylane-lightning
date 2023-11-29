// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file ExpValFunctorsQubit.hpp
 * Define functors for in-place computation of expectation value of
 * Identity, PauliX, PauliY, PauliZ, and Hadamard operators.
 */

#pragma once

#include "BitUtil.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Functors {
/**
 * @brief Functor for computing expectation value of Identity operator.
 *
 * This functor provides an in-place implementation for computing the
 * expectation value of Identity operator for a given statevector.
 *
 * @tparam PrecisionT precision data type of the statevector.
 */
template <class PrecisionT> struct getExpectationValueIdentityFunctor {
  private:
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

  public:
    /**
     * @brief Get the Expectation Value Identity Functor object
     *
     * @param arr_ Pointer to the statevector
     * @param num_qubits_ Number of qubits
     * @param wires Holds index of the wire to apply the operation on
     */
    getExpectationValueIdentityFunctor(
        const std::complex<PrecisionT> *arr_,
        [[maybe_unused]] std::size_t num_qubits_,
        [[maybe_unused]] const std::vector<size_t> &wires)
        : arr(arr_), num_qubits(num_qubits_) {}

    inline PrecisionT operator()() const {
        size_t k;
        PrecisionT expval = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(num_qubits, arr) private(k)      \
    reduction(+ : expval)
#endif
        for (k = 0; k < exp2(num_qubits); k++) {
            expval += real(conj(arr[k]) * arr[k]);
        }

        return expval;
    }
};

/**
 * @brief Functor for computing expectation value of PauliX operator.
 *
 * This functor provides an in-place implementation for computing the
 * expectation value of PauliX operator for a given statevector.
 *
 * @tparam PrecisionT precision data type of the statevector.
 */
template <class PrecisionT> struct getExpectationValuePauliXFunctor {
  private:
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

  public:
    /**
     * @brief Get the Expectation Value PauliX Functor object
     *
     * @param arr_ Pointer to the statevector
     * @param num_qubits_ Number of qubits
     * @param wires Holds index of the wire to apply the operation on
     */
    getExpectationValuePauliXFunctor(const std::complex<PrecisionT> *arr_,
                                     std::size_t num_qubits_,
                                     const std::vector<size_t> &wires)
        : arr(arr_), num_qubits(num_qubits_),
          rev_wire(num_qubits - wires[0] - 1),
          rev_wire_shift((static_cast<size_t>(1U) << rev_wire)),
          wire_parity(fillTrailingOnes(rev_wire)),
          wire_parity_inv(fillLeadingOnes(rev_wire + 1)) {}

    inline PrecisionT operator()() const {
        size_t k;
        size_t i0;
        size_t i1;
        PrecisionT expval = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for default(none)                                         \
    shared(num_qubits, wire_parity_inv, wire_parity, rev_wire_shift,           \
               arr) private(k, i0, i1) reduction(+ : expval)
#endif
        for (k = 0; k < exp2(num_qubits - 1); k++) {
            i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            i1 = i0 | rev_wire_shift;

            expval +=
                real(conj(arr[i0]) * arr[i1]) + real(conj(arr[i1]) * arr[i0]);
        }

        return expval;
    }
};

/**
 * @brief Functor for computing expectation value of PauliY operator.
 *
 * This functor provides an in-place implementation for computing the
 * expectation value of PauliY operator for a given statevector.
 *
 * @tparam PrecisionT precision data type of the statevector.
 */
template <class PrecisionT> struct getExpectationValuePauliYFunctor {
  private:
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

  public:
    /**
     * @brief Get the Expectation Value PauliY Functor object
     *
     * @param arr_ Pointer to the statevector
     * @param num_qubits_ Number of qubits
     * @param wires Holds index of the wire to apply the operation on
     */
    getExpectationValuePauliYFunctor(const std::complex<PrecisionT> *arr_,
                                     std::size_t num_qubits_,
                                     const std::vector<size_t> &wires)
        : arr(arr_), num_qubits(num_qubits_),
          rev_wire(num_qubits - wires[0] - 1),
          rev_wire_shift((static_cast<size_t>(1U) << rev_wire)),
          wire_parity(fillTrailingOnes(rev_wire)),
          wire_parity_inv(fillLeadingOnes(rev_wire + 1)) {}

    inline PrecisionT operator()() const {
        size_t k;
        size_t i0;
        size_t i1;
        std::complex<PrecisionT> v0;
        std::complex<PrecisionT> v1;
        PrecisionT expval = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for default(none)                                         \
    shared(num_qubits, wire_parity_inv, wire_parity, rev_wire_shift,           \
               arr) private(k, i0, i1, v0, v1) reduction(+ : expval)
#endif
        for (k = 0; k < exp2(num_qubits - 1); k++) {
            i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            i1 = i0 | rev_wire_shift;
            v0 = arr[i0];
            v1 = arr[i1];

            expval += real(conj(arr[i0]) *
                           std::complex<PrecisionT>{imag(v1), -real(v1)}) +
                      real(conj(arr[i1]) *
                           std::complex<PrecisionT>{-imag(v0), real(v0)});
        }

        return expval;
    }
};

/**
 * @brief Functor for computing expectation value of PauliZ operator.
 *
 * This functor provides an in-place implementation for computing the
 * expectation value of PauliZ operator for a given statevector.
 *
 * @tparam PrecisionT precision data type of the statevector.
 */
template <class PrecisionT> struct getExpectationValuePauliZFunctor {
  private:
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

  public:
    /**
     * @brief Get the Expectation Value PauliZ Functor object
     *
     * @param arr_ Pointer to the statevector
     * @param num_qubits_ Number of qubits
     * @param wires Holds index of the wire to apply the operation on
     */
    getExpectationValuePauliZFunctor(const std::complex<PrecisionT> *arr_,
                                     std::size_t num_qubits_,
                                     const std::vector<size_t> &wires)
        : arr(arr_), num_qubits(num_qubits_),
          rev_wire(num_qubits - wires[0] - 1),
          rev_wire_shift((static_cast<size_t>(1U) << rev_wire)),
          wire_parity(fillTrailingOnes(rev_wire)),
          wire_parity_inv(fillLeadingOnes(rev_wire + 1)) {}

    inline PrecisionT operator()() const {
        size_t k;
        size_t i0;
        size_t i1;
        PrecisionT expval = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for default(none)                                         \
    shared(num_qubits, wire_parity_inv, wire_parity, rev_wire_shift,           \
               arr) private(k, i0, i1) reduction(+ : expval)
#endif
        for (k = 0; k < exp2(num_qubits - 1); k++) {
            i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            i1 = i0 | rev_wire_shift;

            expval += real(conj(arr[i1]) * (-arr[i1])) +
                      real(conj(arr[i0]) * (arr[i0]));
        }

        return expval;
    }
};

/**
 * @brief Functor for computing expectation value of Hadamard operator.
 *
 * This functor provides an in-place implementation for computing the
 * expectation value of Hadamard operator for a given statevector.
 *
 * @tparam PrecisionT precision data type of the statevector.
 */
template <class PrecisionT> struct getExpectationValueHadamardFunctor {
  private:
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();

  public:
    /**
     * @brief Get the Expectation Value Hadamard Functor object
     *
     * @param arr_ Pointer to the statevector
     * @param num_qubits_ Number of qubits
     * @param wires Holds index of the wire to apply the operation on
     */
    getExpectationValueHadamardFunctor(const std::complex<PrecisionT> *arr_,
                                       std::size_t num_qubits_,
                                       const std::vector<size_t> &wires)
        : arr(arr_), num_qubits(num_qubits_),
          rev_wire(num_qubits - wires[0] - 1),
          rev_wire_shift((static_cast<size_t>(1U) << rev_wire)),
          wire_parity(fillTrailingOnes(rev_wire)),
          wire_parity_inv(fillLeadingOnes(rev_wire + 1)) {}

    inline PrecisionT operator()() const {
        size_t k;
        size_t i0;
        size_t i1;
        std::complex<PrecisionT> v0;
        std::complex<PrecisionT> v1;
        PrecisionT expval = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for default(none)                                         \
    shared(num_qubits, wire_parity_inv, wire_parity, rev_wire_shift,           \
               arr) private(k, i0, i1, v0, v1) reduction(+ : expval)
#endif

        for (k = 0; k < exp2(num_qubits - 1); k++) {
            i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            i1 = i0 | rev_wire_shift;
            v0 = arr[i0];
            v1 = arr[i1];

            expval += real(isqrt2 * (conj(arr[i0]) * (v0 + v1) +
                                     conj(arr[i1]) * (v0 - v1)));
        }

        return expval;
    }
};
} // namespace Pennylane::LightningQubit::Functors