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
#pragma once

#include "BitUtil.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Functors {

template <class PrecisionT> struct getExpectationValueIdentityFunctor {
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    getExpectationValueIdentityFunctor(
        const std::complex<PrecisionT> *arr_,
        [[maybe_unused]] std::size_t num_qubits_,
        [[maybe_unused]] const std::vector<size_t> &wires) {
        arr = arr_;
        num_qubits = num_qubits_;
    }

    inline void operator()(PrecisionT &expval) const {
        size_t k;
#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(num_qubits, arr) private(k)      \
    reduction(+ : expval)
#endif
        for (k = 0; k < exp2(num_qubits); k++) {
            expval += real(conj(arr[k]) * arr[k]);
        }
    }
};

template <class PrecisionT> struct getExpectationValuePauliXFunctor {
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    getExpectationValuePauliXFunctor(const std::complex<PrecisionT> *arr_,
                                     std::size_t num_qubits_,
                                     const std::vector<size_t> &wires) {
        arr = arr_;
        num_qubits = num_qubits_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    inline void operator()(PrecisionT &expval) const {
        size_t k;
        size_t i0;
        size_t i1;
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
    }
};

template <class PrecisionT> struct getExpectationValuePauliYFunctor {
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    getExpectationValuePauliYFunctor(const std::complex<PrecisionT> *arr_,
                                     std::size_t num_qubits_,
                                     const std::vector<size_t> &wires) {
        arr = arr_;
        num_qubits = num_qubits_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    inline void operator()(PrecisionT &expval) const {
        size_t k;
        size_t i0;
        size_t i1;
        std::complex<PrecisionT> v0;
        std::complex<PrecisionT> v1;
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
    }
};

template <class PrecisionT> struct getExpectationValuePauliZFunctor {
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    getExpectationValuePauliZFunctor(const std::complex<PrecisionT> *arr_,
                                     std::size_t num_qubits_,
                                     const std::vector<size_t> &wires) {
        arr = arr_;
        num_qubits = num_qubits_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    inline void operator()(PrecisionT &expval) const {
        size_t k;
        size_t i0;
        size_t i1;
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
    }
};

template <class PrecisionT> struct getExpectationValueHadamardFunctor {
    const std::complex<PrecisionT> *arr;
    size_t num_qubits;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    PrecisionT isqrt2 = INVSQRT2<PrecisionT>();

    getExpectationValueHadamardFunctor(const std::complex<PrecisionT> *arr_,
                                       std::size_t num_qubits_,
                                       const std::vector<size_t> &wires) {
        arr = arr_;
        num_qubits = num_qubits_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    inline void operator()(PrecisionT &expval) const {
        size_t k;
        size_t i0;
        size_t i1;
        std::complex<PrecisionT> v0;
        std::complex<PrecisionT> v1;
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
    }
};
} // namespace Pennylane::LightningQubit::Functors