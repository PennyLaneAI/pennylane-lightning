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
 * Defines kernel functiosn with less memory (and fast)
 */
#pragma once

#include "Error.hpp"
#include "GateOperations.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "Util.hpp"

#include <cassert>
#include <climits>
#include <complex>
#include <vector>

namespace Pennylane {
/**
 * @brief Fill ones from LSB to rev_wire
 */
auto constexpr fillTrailingOnes(size_t pos) -> size_t {
    return (pos == 0) ? 0 : (~size_t(0) >> (CHAR_BIT * sizeof(size_t) - pos));
}
/**
 * @brief Fill ones from MSB to pos
 */
auto constexpr fillLeadingOnes(size_t pos) -> size_t {
    return (~size_t(0)) << pos;
}

/**
 * @brief
 */
template <class fp_t> class GateOperationsLM {
  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<fp_t>;
    
    constexpr static KernelType kernel_id = KernelType::LM;

    constexpr static std::array implemented_gates = {
        GateOperations::PauliX,
        GateOperations::PauliY,
        GateOperations::PauliZ,
        GateOperations::Hadamard,
        GateOperations::S,
        GateOperations::T,
        GateOperations::RX,
        GateOperations::RY,
        GateOperations::RZ,
        GateOperations::PhaseShift,
        GateOperations::Rot,
        GateOperations::CZ,
        GateOperations::CNOT
    };

  private:
    static void applySingleQubitOp(CFP_t *arr, size_t num_qubits,
                                   const CFP_t *op_matrix, size_t wire) {
        size_t rev_wire = num_qubits - wire - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t n = 0; n < Util::exp2(num_qubits - 1); ++n) {
            const size_t k = n;
            const size_t i0 = ((k << 1) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | (1 << rev_wire);
            const CFP_t v0 = arr[i0];
            const CFP_t v1 = arr[i1];
            arr[i0] = op_matrix[0B00] * v0 +
                      op_matrix[0B01] * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = op_matrix[0B10] * v0 +
                      op_matrix[0B11] * v1; // NOLINT(readability-magic-numbers)
        }
    }

  public:

    static void applyPauliX(CFP_t *arr, const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); ++k) {
            const size_t i0 = ((k << 1) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | (1 << rev_wire);
            std::swap(arr[i0], arr[i1]);
        }
    }

    static void applyPauliY(CFP_t *arr, const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        fp_t *data_z = reinterpret_cast<fp_t *>(arr);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); ++k) {
            const size_t i0 = ((k << 1) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | (1 << rev_wire);
            const auto v0_r = data_z[2 * i0];
            const auto v0_i = data_z[2 * i0 + 1];
            const auto v1_r = data_z[2 * i1];
            const auto v1_i = data_z[2 * i1 + 1];

            data_z[2 * i0] = v1_i;
            data_z[2 * i0 + 1] = -v1_r;
            data_z[2 * i1] = -v0_i;
            data_z[2 * i1 + 1] = v0_r;
        }
    }

    static void applyPauliZ(CFP_t *arr, const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); ++k) {
            const size_t i0 = ((k << 1) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | (1 << rev_wire);
            arr[i1] *= -1;
        }
    }

    static void applyHadamard(CFP_t *arr, const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        using Util::INVSQRT2;
        constexpr static std::array<CFP_t, 4> hadamardMat = {
            INVSQRT2<fp_t>(), INVSQRT2<fp_t>(), INVSQRT2<fp_t>(),
            -INVSQRT2<fp_t>()};
        applySingleQubitOp(arr, num_qubits, hadamardMat.data(), wires[0]);
    }

    static void applyS(CFP_t *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const CFP_t shift =
            (inverse) ? -Util::IMAG<fp_t>() : Util::IMAG<fp_t>();

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); ++k) {
            const size_t i0 = ((k << 1) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | (1 << rev_wire);
            arr[i1] *= shift;
        }
    }

    static void applyT(CFP_t *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const CFP_t shift =
            (inverse)
                ? std::conj(std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4))))
                : std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4)));

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); ++k) {
            const size_t i0 = ((k << 1) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | (1 << rev_wire);
            arr[i1] *= shift;
        }
    }

    template <class Param_t = fp_t>
    static void applyRX(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);

        const fp_t c = std::cos(angle / 2);
        const fp_t js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        const std::array<CFP_t, 4> RXMat = {c, Util::IMAG<fp_t>() * js,
                                            Util::IMAG<fp_t>() * js, c};
        applySingleQubitOp(arr, num_qubits, RXMat.data(), wires[0]);
    }

    template <class Param_t = fp_t>
    static void applyRY(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);

        const fp_t c = std::cos(angle / 2);
        const fp_t s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        const std::array<CFP_t, 4> RYMat = {c, -s, s, c};
        applySingleQubitOp(arr, num_qubits, RYMat.data(), wires[0]);
    }

    template <class Param_t = fp_t>
    static void applyRZ(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);

        const CFP_t first = CFP_t(std::cos(angle / 2), -std::sin(angle / 2));
        const CFP_t second = CFP_t(std::cos(angle / 2), std::sin(angle / 2));

        const std::array<CFP_t, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};
        const size_t rev_wire = num_qubits - wires[0] - 1;

        for (size_t k = 0; k < Util::exp2(num_qubits); ++k) {
            arr[k] *= shifts[(k >> rev_wire) & 1];
        }
    }

    template <class Param_t = fp_t>
    static void applyPhaseShift(CFP_t *arr, const size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                Param_t angle) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const CFP_t s =
            inverse ? std::exp(-CFP_t(0, angle)) : std::exp(CFP_t(0, angle));

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); ++k) {
            const size_t i0 = ((k << 1) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | (1 << rev_wire);
            arr[i1] *= s;
        }
    }

    template <class Param_t = fp_t>
    static void applyRot(CFP_t *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         Param_t phi, Param_t theta, Param_t omega) {
        assert(wires.size() == 1);

        const auto rotMat = (inverse)
                                ? Gates::getRot<fp_t>(-omega, -theta, -phi)
                                : Gates::getRot<fp_t>(phi, theta, omega);

        applySingleQubitOp(arr, num_qubits, rotMat.data(), wires[0]);
    }

    static void applyCNOT(CFP_t* arr, const size_t num_qubits, 
                          const std::vector<size_t>& wires, [[maybe_unused]] bool inverse) {
        assert (wires.size() == 2);
        
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Controll qubit

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);
        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle = fillLeadingOnes(rev_wire_min + 1) &
                                     fillTrailingOnes(rev_wire_max);
        /* This is faster than iterate over all indices */
        for (size_t k = 0; k < Util::exp2(num_qubits-2); ++k) {
            const size_t i00 = ((k << 2) & parity_high) | ((k << 1) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | (1U << rev_wire1);
            const size_t i11 = i00 | (1U << rev_wire1) | (1U << rev_wire0);

            std::swap(arr[i10], arr[i11]);
        }
    }

    static void applyCZ(CFP_t* arr, const size_t num_qubits, 
                          const std::vector<size_t>& wires, [[maybe_unused]] bool inverse) {
        assert (wires.size() == 2);
        
        const size_t rev_wire_min = std::min(num_qubits - wires[0] - 1, num_qubits - wires[1] - 1);
        const size_t rev_wire_max = std::max(num_qubits - wires[0] - 1, num_qubits - wires[1] - 1);
        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle = fillLeadingOnes(rev_wire_min + 1) &
                                     fillTrailingOnes(rev_wire_max);
        /* This is faster than iterate over all indices */
        for (size_t k = 0; k < Util::exp2(num_qubits-2); ++k) {
            const size_t i00 = ((k << 2) & parity_high) | ((k << 1) & parity_middle) | (k & parity_low);
            const size_t i11 = i00 | (1U << rev_wire_min) | (1U << rev_wire_max);
            arr[i11] *= -1; 
        }
    }
};

} // namespace Pennylane
