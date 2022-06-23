// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 * Defines kernel functions with AVX2
 */
#pragma once

// General implementations
#include "avx_common/AVX2Concept.hpp"

#include "avx_common/ApplyCNOT.hpp"
#include "avx_common/ApplyCZ.hpp"
#include "avx_common/ApplyHadamard.hpp"
#include "avx_common/ApplyIsingXX.hpp"
#include "avx_common/ApplyIsingYY.hpp"
#include "avx_common/ApplyIsingZZ.hpp"
#include "avx_common/ApplyPauliX.hpp"
#include "avx_common/ApplyPauliY.hpp"
#include "avx_common/ApplyPauliZ.hpp"
#include "avx_common/ApplyPhaseShift.hpp"
#include "avx_common/ApplyRX.hpp"
#include "avx_common/ApplyRY.hpp"
#include "avx_common/ApplyRZ.hpp"
#include "avx_common/ApplyS.hpp"
#include "avx_common/ApplySWAP.hpp"
#include "avx_common/ApplySingleQubitOp.hpp"
#include "avx_common/ApplyT.hpp"

#include "Error.hpp"
#include "GateImplementationsLM.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "Macros.hpp"

#include <immintrin.h>

#include <complex>
#include <vector>

namespace Pennylane::Gates {

class GateImplementationsAVX2 : public PauliGenerator<GateImplementationsAVX2> {
  public:
    constexpr static KernelType kernel_id = KernelType::AVX2;
    constexpr static std::string_view name = "AVX2";
    constexpr static uint32_t packed_bytes = 32;

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,     GateOperation::PauliY,
        GateOperation::PauliZ,     GateOperation::Hadamard,
        GateOperation::S,          GateOperation::T,
        GateOperation::PhaseShift, GateOperation::RX,
        GateOperation::RY,         GateOperation::RZ,
        GateOperation::Rot,        GateOperation::CZ,
        GateOperation::CNOT,       GateOperation::SWAP,
        GateOperation::IsingXX,    GateOperation::IsingYY,
        GateOperation::IsingZZ,
        /* CY, IsingXY, ControlledPhaseShift, CRX, CRY, CRZ, CRot */
    };

    constexpr static std::array implemented_generators = {
        GeneratorOperation::RX, GeneratorOperation::RY, GeneratorOperation::RZ};

    constexpr static std::array implemented_matrices = {
        MatrixOperation::SingleQubitOp,
    };

    template <typename PrecisionT>
    static void
    applySingleQubitOp(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::complex<PrecisionT> *matrix,
                       const std::vector<size_t> &wires, bool inverse = false) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;

        using SingleQubitOpProdAVX2 =
            AVX::ApplySingleQubitOp<PrecisionT,
                                    packed_bytes / sizeof(PrecisionT)>;

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applySingleQubitOp(arr, num_qubits, matrix,
                                                      wires, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            switch (rev_wire) {
            case 0:
                SingleQubitOpProdAVX2::template applyInternal<0>(
                    arr, num_qubits, matrix, inverse);
                return;
            case 1:
                SingleQubitOpProdAVX2::template applyInternal<1>(
                    arr, num_qubits, matrix, inverse);
                return;
            default:
                SingleQubitOpProdAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                     matrix, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (rev_wire == 0) {
                SingleQubitOpProdAVX2::template applyInternal<0>(
                    arr, num_qubits, matrix, inverse);
            } else {
                SingleQubitOpProdAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                     matrix, inverse);
            }
        }
    }

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliXAVX2 =
            AVX::ApplyPauliX<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        assert(wires.size() == 1);
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliXAVX2::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliXAVX2::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyPauliXAVX2::applyExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire == 0) {
                ApplyPauliXAVX2::template applyInternal<0>(arr, num_qubits);
            } else {
                ApplyPauliXAVX2::applyExternal(arr, num_qubits, rev_wire);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliYAVX2 =
            AVX::ApplyPauliY<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliY(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliYAVX2::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliYAVX2::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyPauliYAVX2::applyExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliY(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire == 0) {
                ApplyPauliYAVX2::template applyInternal<0>(arr, num_qubits);
            } else {
                ApplyPauliYAVX2::applyExternal(arr, num_qubits, rev_wire);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliZAVX2 =
            AVX::ApplyPauliZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliZ(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliZAVX2::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliZAVX2::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyPauliZAVX2::applyExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliZ(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire == 0) {
                ApplyPauliZAVX2::template applyInternal<0>(arr, num_qubits);
            } else {
                ApplyPauliZAVX2::applyExternal(arr, num_qubits, rev_wire);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        using ApplySAVX2 =
            AVX::ApplyS<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyS(arr, num_qubits, wires, inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplySAVX2::template applyInternal<0>(arr, num_qubits, inverse);
                return;
            case 1:
                ApplySAVX2::template applyInternal<1>(arr, num_qubits, inverse);
                return;
            default:
                ApplySAVX2::applyExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if (rev_wire == 0) {
                ApplySAVX2::template applyInternal<0>(arr, num_qubits, inverse);
            } else {
                ApplySAVX2::applyExternal(arr, num_qubits, rev_wire, inverse);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyT(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        using ApplyTAVX2 =
            AVX::ApplyT<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyT(arr, num_qubits, wires, inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyTAVX2::template applyInternal<0>(arr, num_qubits, inverse);
                return;
            case 1:
                ApplyTAVX2::template applyInternal<1>(arr, num_qubits, inverse);
                return;
            default:
                ApplyTAVX2::applyExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if (rev_wire == 0) {
                ApplyTAVX2::template applyInternal<0>(arr, num_qubits, inverse);
            } else {
                ApplyTAVX2::applyExternal(arr, num_qubits, rev_wire, inverse);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                const size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                ParamT angle) {
        using ApplyPhaseShiftAVX2 =
            AVX::ApplyPhaseShift<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPhaseShift(arr, num_qubits, wires,
                                                       inverse, angle);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPhaseShiftAVX2::template applyInternal<0>(arr, num_qubits,
                                                               inverse, angle);
                return;
            case 1:
                ApplyPhaseShiftAVX2::template applyInternal<1>(arr, num_qubits,
                                                               inverse, angle);
                return;
            default:
                ApplyPhaseShiftAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                   inverse, angle);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if (rev_wire == 0) {
                ApplyPhaseShiftAVX2::template applyInternal<0>(arr, num_qubits,
                                                               inverse, angle);
            } else {
                ApplyPhaseShiftAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                   inverse, angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        using ApplyHadamardAVX2 =
            AVX::ApplyHadamard<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyHadamard(arr, num_qubits, wires,
                                                     inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyHadamardAVX2::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyHadamardAVX2::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyHadamardAVX2::applyExternal(arr, num_qubits, rev_wire);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if (rev_wire == 0) {
                ApplyHadamardAVX2::template applyInternal<0>(arr, num_qubits);
            } else {
                ApplyHadamardAVX2::applyExternal(arr, num_qubits, rev_wire);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRXAVX2 =
            AVX::ApplyRX<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyRX(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRXAVX2::template applyInternal<0>(arr, num_qubits, inverse,
                                                       angle);
                return;
            case 1:
                ApplyRXAVX2::template applyInternal<1>(arr, num_qubits, inverse,
                                                       angle);
                return;
            default:
                ApplyRXAVX2::applyExternal(arr, num_qubits, rev_wire, inverse,
                                           angle);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if (rev_wire == 0) {
                ApplyRXAVX2::template applyInternal<0>(arr, num_qubits, inverse,
                                                       angle);
            } else {
                ApplyRXAVX2::applyExternal(arr, num_qubits, rev_wire, inverse,
                                           angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRYAVX2 =
            AVX::ApplyRY<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyRY(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRYAVX2::template applyInternal<0>(arr, num_qubits, inverse,
                                                       angle);
                return;
            case 1:
                ApplyRYAVX2::template applyInternal<1>(arr, num_qubits, inverse,
                                                       angle);
                return;
            default:
                ApplyRYAVX2::applyExternal(arr, num_qubits, rev_wire, inverse,
                                           angle);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if (rev_wire == 0) {
                ApplyRYAVX2::template applyInternal<0>(arr, num_qubits, inverse,
                                                       angle);
            } else {
                ApplyRYAVX2::applyExternal(arr, num_qubits, rev_wire, inverse,
                                           angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRZAVX2 =
            AVX::ApplyRZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRZAVX2::template applyInternal<0>(arr, num_qubits, inverse,
                                                       angle);
                return;
            case 1:
                ApplyRZAVX2::template applyInternal<1>(arr, num_qubits, inverse,
                                                       angle);
                return;
            default:
                ApplyRZAVX2::applyExternal(arr, num_qubits, rev_wire, inverse,
                                           angle);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if (rev_wire == 0) {
                ApplyRZAVX2::template applyInternal<0>(arr, num_qubits, inverse,
                                                       angle);
            } else {
                ApplyRZAVX2::applyExternal(arr, num_qubits, rev_wire, inverse,
                                           angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        assert(wires.size() == 1);

        const auto rotMat =
            (inverse) ? Gates::getRot<PrecisionT>(-omega, -theta, -phi)
                      : Gates::getRot<PrecisionT>(phi, theta, omega);

        applySingleQubitOp(arr, num_qubits, rotMat.data(), wires);
    }

    /* Two-qubit gates*/
    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        using ApplyCZAVX2 =
            AVX::ApplyCZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 2);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplyCZAVX2::applyInternalInternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                ApplyCZAVX2::applyInternalExternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1);
            } else {
                ApplyCZAVX2::applyExternalExternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (std::min(rev_wire0, rev_wire1) == 0) {
                ApplyCZAVX2::applyInternalExternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1);
            } else {
                ApplyCZAVX2::applyExternalExternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void
    applyCNOT(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        using ApplyCNOTAVX2 =
            AVX::ApplyCNOT<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t target = num_qubits - wires[1] - 1;
            const size_t control = num_qubits - wires[0] - 1;

            if (target < 2 && control < 2) {
                ApplyCNOTAVX2::template applyInternalInternal<0, 1>(arr,
                                                                    num_qubits);
                return;
            }
            if (control < 2) {
                if (control == 0) {
                    ApplyCNOTAVX2::template applyInternalExternal<0>(
                        arr, num_qubits, target);
                } else { // control == 1
                    ApplyCNOTAVX2::template applyInternalExternal<1>(
                        arr, num_qubits, target);
                }
                return;
            }
            if (target < 2) {
                if (target == 0) {
                    ApplyCNOTAVX2::template applyExternalInternal<0>(
                        arr, num_qubits, control);
                } else { // target == 1
                    ApplyCNOTAVX2::template applyExternalInternal<1>(
                        arr, num_qubits, control);
                }
                return;
            }
            ApplyCNOTAVX2::applyExternalExternal(arr, num_qubits, control,
                                                 target);
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t target = num_qubits - wires[1] - 1;
            const size_t control = num_qubits - wires[0] - 1;

            if (control == 0) {
                ApplyCNOTAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, target);
                return;
            }
            if (target == 0) {
                ApplyCNOTAVX2::template applyExternalInternal<0>(
                    arr, num_qubits, control);
                return;
            }
            ApplyCNOTAVX2::applyExternalExternal(arr, num_qubits, control,
                                                 target);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXX(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 2);

        using ApplyIsingXXAVX2 =
            AVX::ApplyIsingXX<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplyIsingXXAVX2::template applyInternalInternal<0, 1>(
                    arr, num_qubits, inverse, angle);
                return;
            }
            switch (min_rev_wire) {
            case 0:
                ApplyIsingXXAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            case 1:
                ApplyIsingXXAVX2::template applyInternalExternal<1>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            default:
                ApplyIsingXXAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (min_rev_wire == 0) {
                ApplyIsingXXAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
            } else {
                ApplyIsingXXAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingYY(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 2);

        using ApplyIsingYYAVX2 =
            AVX::ApplyIsingYY<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplyIsingYYAVX2::template applyInternalInternal<0, 1>(
                    arr, num_qubits, inverse, angle);
                return;
            }
            switch (min_rev_wire) {
            case 0:
                ApplyIsingYYAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            case 1:
                ApplyIsingYYAVX2::template applyInternalExternal<1>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            default:
                ApplyIsingYYAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (min_rev_wire == 0) {
                ApplyIsingYYAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
            } else {
                ApplyIsingYYAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 2);

        using ApplyIsingZZAVX2 =
            AVX::ApplyIsingZZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplyIsingZZAVX2::applyInternalInternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                ApplyIsingZZAVX2::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (std::min(rev_wire0, rev_wire1) == 0) {
                ApplyIsingZZAVX2::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void
    applySWAP(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        using ApplySWAPAVX2 =
            AVX::ApplySWAP<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplySWAPAVX2::template applyInternalInternal<0, 1>(arr,
                                                                    num_qubits);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                if (min_rev_wire == 0) {
                    ApplySWAPAVX2::template applyInternalExternal<0>(
                        arr, num_qubits, max_rev_wire);
                } else { // min_rev_wire0 == 1
                    ApplySWAPAVX2::template applyInternalExternal<1>(
                        arr, num_qubits, max_rev_wire);
                }
            } else {
                ApplySWAPAVX2::applyExternalExternal(arr, num_qubits, rev_wire0,
                                                     rev_wire1);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (min_rev_wire == 0) {
                ApplySWAPAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire);
            } else {
                ApplySWAPAVX2::applyExternalExternal(arr, num_qubits, rev_wire0,
                                                     rev_wire1);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }
};
} // namespace Pennylane::Gates
