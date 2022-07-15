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
 * Defines kernel functions with AVX512F and AVX512DQ
 */
#pragma once
#include "avx_common/AVX512Concept.hpp"

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
#include "avx_common/TwoQubitGateHelper.hpp"

#include "BitUtil.hpp"
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

class GateImplementationsAVX512
    : public PauliGenerator<GateImplementationsAVX512> {
  public:
    constexpr static KernelType kernel_id = KernelType::AVX512;
    constexpr static std::string_view name = "AVX512";
    template <typename PrecisionT>
    constexpr static size_t required_alignment = 64;
    constexpr static uint32_t packed_bytes = 64;

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,  GateOperation::PauliY,
        GateOperation::PauliZ,  GateOperation::Hadamard,
        GateOperation::S,       GateOperation::SWAP,
        GateOperation::T,       GateOperation::PhaseShift,
        GateOperation::RX,      GateOperation::RY,
        GateOperation::RZ,      GateOperation::Rot,
        GateOperation::CNOT,    GateOperation::CZ,
        GateOperation::IsingXX, GateOperation::IsingYY,
        GateOperation::IsingZZ,
        /* CRX, CRY, CRZ, CRot */
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

        using SingleQubitOpProdAVX512 =
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
                SingleQubitOpProdAVX512::template applyInternal<0>(
                    arr, num_qubits, matrix, inverse);
                return;
            case 1:
                SingleQubitOpProdAVX512::template applyInternal<1>(
                    arr, num_qubits, matrix, inverse);
                return;
            case 2:
                SingleQubitOpProdAVX512::template applyInternal<2>(
                    arr, num_qubits, matrix, inverse);
                return;
            default:
                SingleQubitOpProdAVX512::applyExternal(
                    arr, num_qubits, rev_wire, matrix, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            switch (rev_wire) {
            case 0:
                SingleQubitOpProdAVX512::template applyInternal<0>(
                    arr, num_qubits, matrix, inverse);
                return;
            case 1:
                SingleQubitOpProdAVX512::template applyInternal<1>(
                    arr, num_qubits, matrix, inverse);
                return;
            default:
                SingleQubitOpProdAVX512::applyExternal(
                    arr, num_qubits, rev_wire, matrix, inverse);
                return;
            }
        }
    }

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliXAVX512 =
            AVX::ApplyPauliX<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyPauliX(arr, num_qubits, wires, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliXAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliXAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            case 2:
                ApplyPauliXAVX512::template applyInternal<2>(arr, num_qubits);
                return;
            default:
                ApplyPauliXAVX512::applyExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliXAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliXAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyPauliXAVX512::applyExternal(arr, num_qubits, rev_wire);
                return;
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
        using ApplyPauliYAVX512 =
            AVX::ApplyPauliY<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyPauliY(arr, num_qubits, wires, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliYAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliYAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            case 2:
                ApplyPauliYAVX512::template applyInternal<2>(arr, num_qubits);
                return;
            default:
                ApplyPauliYAVX512::applyExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliYAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliYAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyPauliYAVX512::applyExternal(arr, num_qubits, rev_wire);
                return;
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
        using ApplyPauliZAVX512 =
            AVX::ApplyPauliZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyPauliZ(arr, num_qubits, wires, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliZAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliZAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            case 2:
                ApplyPauliZAVX512::template applyInternal<2>(arr, num_qubits);
                return;
            default:
                ApplyPauliZAVX512::applyExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliZAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliZAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyPauliZAVX512::applyExternal(arr, num_qubits, rev_wire);
                return;
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
        using ApplySAVX512 =
            AVX::ApplyS<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyS(arr, num_qubits, wires, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplySAVX512::template applyInternal<0>(arr, num_qubits,
                                                        inverse);
                return;
            case 1:
                ApplySAVX512::template applyInternal<1>(arr, num_qubits,
                                                        inverse);
                return;
            case 2:
                ApplySAVX512::template applyInternal<2>(arr, num_qubits,
                                                        inverse);
                return;
            default:
                ApplySAVX512::applyExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplySAVX512::template applyInternal<0>(arr, num_qubits,
                                                        inverse);
                return;
            case 1:
                ApplySAVX512::template applyInternal<1>(arr, num_qubits,
                                                        inverse);
                return;
            default:
                ApplySAVX512::applyExternal(arr, num_qubits, rev_wire, inverse);
                return;
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
        using ApplyTAVX512 =
            AVX::ApplyT<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyT(arr, num_qubits, wires, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyTAVX512::template applyInternal<0>(arr, num_qubits,
                                                        inverse);
                return;
            case 1:
                ApplyTAVX512::template applyInternal<1>(arr, num_qubits,
                                                        inverse);
                return;
            case 2:
                ApplyTAVX512::template applyInternal<2>(arr, num_qubits,
                                                        inverse);
                return;
            default:
                ApplyTAVX512::applyExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            switch (rev_wire) {
            case 0:
                ApplyTAVX512::template applyInternal<0>(arr, num_qubits,
                                                        inverse);
                return;
            case 1:
                ApplyTAVX512::template applyInternal<1>(arr, num_qubits,
                                                        inverse);
                return;
            default:
                ApplyTAVX512::applyExternal(arr, num_qubits, rev_wire, inverse);
                return;
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
        using ApplyPhaseShiftAVX512 =
            AVX::ApplyPhaseShift<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyPhaseShift(arr, num_qubits, wires,
                                                   inverse, angle);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPhaseShiftAVX512::template applyInternal<0>(
                    arr, num_qubits, inverse, angle);
                return;
            case 1:
                ApplyPhaseShiftAVX512::template applyInternal<1>(
                    arr, num_qubits, inverse, angle);
                return;
            case 2:
                ApplyPhaseShiftAVX512::template applyInternal<2>(
                    arr, num_qubits, inverse, angle);
                return;
            default:
                ApplyPhaseShiftAVX512::applyExternal(arr, num_qubits, rev_wire,
                                                     inverse, angle);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            switch (rev_wire) {
            case 0:
                ApplyPhaseShiftAVX512::template applyInternal<0>(
                    arr, num_qubits, inverse, angle);
                return;
            case 1:
                ApplyPhaseShiftAVX512::template applyInternal<1>(
                    arr, num_qubits, inverse, angle);
                return;
            default:
                ApplyPhaseShiftAVX512::applyExternal(arr, num_qubits, rev_wire,
                                                     inverse, angle);
                return;
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
        using ApplyHadamardAVX512 =
            AVX::ApplyHadamard<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyHadamard(arr, num_qubits, wires,
                                                 inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyHadamardAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyHadamardAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            case 2:
                ApplyHadamardAVX512::template applyInternal<2>(arr, num_qubits);
                return;
            default:
                ApplyHadamardAVX512::applyExternal(arr, num_qubits, rev_wire);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyHadamardAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyHadamardAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyHadamardAVX512::applyExternal(arr, num_qubits, rev_wire);
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
        using ApplyRXAVX512 =
            AVX::ApplyRX<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyRX(arr, num_qubits, wires, inverse,
                                           angle);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRXAVX512::template applyInternal<0>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 1:
                ApplyRXAVX512::template applyInternal<1>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 2:
                ApplyRXAVX512::template applyInternal<2>(arr, num_qubits,
                                                         inverse, angle);
                return;
            default:
                ApplyRXAVX512::applyExternal(arr, num_qubits, rev_wire, inverse,
                                             angle);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRXAVX512::template applyInternal<0>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 1:
                ApplyRXAVX512::template applyInternal<1>(arr, num_qubits,
                                                         inverse, angle);
                return;
            default:
                ApplyRXAVX512::applyExternal(arr, num_qubits, rev_wire, inverse,
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
        using ApplyRYAVX512 =
            AVX::ApplyRY<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyRY(arr, num_qubits, wires, inverse,
                                           angle);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRYAVX512::template applyInternal<0>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 1:
                ApplyRYAVX512::template applyInternal<1>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 2:
                ApplyRYAVX512::template applyInternal<2>(arr, num_qubits,
                                                         inverse, angle);
                return;
            default:
                ApplyRYAVX512::applyExternal(arr, num_qubits, rev_wire, inverse,
                                             angle);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRYAVX512::template applyInternal<0>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 1:
                ApplyRYAVX512::template applyInternal<1>(arr, num_qubits,
                                                         inverse, angle);
                return;
            default:
                ApplyRYAVX512::applyExternal(arr, num_qubits, rev_wire, inverse,
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
        using ApplyRZAVX512 =
            AVX::ApplyRZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 1);

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                           angle);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRZAVX512::template applyInternal<0>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 1:
                ApplyRZAVX512::template applyInternal<1>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 2:
                ApplyRZAVX512::template applyInternal<2>(arr, num_qubits,
                                                         inverse, angle);
                return;
            default:
                ApplyRZAVX512::applyExternal(arr, num_qubits, rev_wire, inverse,
                                             angle);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyRZAVX512::template applyInternal<0>(arr, num_qubits,
                                                         inverse, angle);
                return;
            case 1:
                ApplyRZAVX512::template applyInternal<1>(arr, num_qubits,
                                                         inverse, angle);
                return;
            default:
                ApplyRZAVX512::applyExternal(arr, num_qubits, rev_wire, inverse,
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
    static void
    applyCNOT(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        constexpr size_t packed_size = packed_bytes / sizeof(PrecisionT);
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyCNOT(arr, num_qubits, wires,
                                                 inverse);
            }
            AVX::applyCNOT<PrecisionT, packed_size>(arr, num_qubits, wires);
        } else if (std::is_same_v<PrecisionT, double>) {
            AVX::applyCNOT<PrecisionT, packed_size>(arr, num_qubits, wires);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }

    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        using ApplyCZAVX512 =
            AVX::ApplyCZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 2);

        const AVX::TwoQubitGateHelper<ApplyCZAVX512> gate_helper(
            &GateImplementationsLM::applyCZ<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applySWAP(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        using ApplySWAPAVX512 =
            AVX::ApplySWAP<PrecisionT, packed_bytes / sizeof(PrecisionT)>;
        assert(wires.size() == 2);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applySWAP(arr, num_qubits, wires,
                                                 inverse);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (rev_wire0 < 3 && rev_wire1 < 3) {
                switch (min_rev_wire ^ max_rev_wire) {
                case 1: // (0, 1)
                    ApplySWAPAVX512::template applyInternalInternal<0, 1>(
                        arr, num_qubits);
                    return;
                case 2: // (0, 2)
                    ApplySWAPAVX512::template applyInternalInternal<0, 2>(
                        arr, num_qubits);
                    return;
                case 3: // (1, 2)
                    ApplySWAPAVX512::template applyInternalInternal<1, 2>(
                        arr, num_qubits);
                    return;
                default:
                    PL_UNREACHABLE;
                }
            }
            switch (min_rev_wire) {
            case 0:
                ApplySWAPAVX512::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire);
                return;
            case 1:
                ApplySWAPAVX512::template applyInternalExternal<1>(
                    arr, num_qubits, max_rev_wire);
                return;
            case 2:
                ApplySWAPAVX512::template applyInternalExternal<2>(
                    arr, num_qubits, max_rev_wire);
                return;
            default:
                ApplySWAPAVX512::applyExternalExternal(arr, num_qubits,
                                                       rev_wire0, rev_wire1);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplySWAPAVX512::template applyInternalInternal<0, 1>(
                    arr, num_qubits);
                return;
            }
            switch (min_rev_wire) {
            case 0:
                ApplySWAPAVX512::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire);
                return;
            case 1:
                ApplySWAPAVX512::template applyInternalExternal<1>(
                    arr, num_qubits, max_rev_wire);
                return;
            default:
                ApplySWAPAVX512::applyExternalExternal(arr, num_qubits,
                                                       rev_wire0, rev_wire1);
                return;
            }
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
            if (num_qubits < 3) {
                GateImplementationsLM::applyIsingXX(arr, num_qubits, wires,
                                                    inverse, angle);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (rev_wire0 < 3 && rev_wire1 < 3) {
                switch (rev_wire0 ^ rev_wire1) {
                case 1: // (0, 1)
                    ApplyIsingXXAVX2::template applyInternalInternal<0, 1>(
                        arr, num_qubits, inverse, angle);
                    return;
                case 2: // (0, 2)
                    ApplyIsingXXAVX2::template applyInternalInternal<0, 2>(
                        arr, num_qubits, inverse, angle);
                    return;
                case 3: // (1,2)
                    ApplyIsingXXAVX2::template applyInternalInternal<1, 2>(
                        arr, num_qubits, inverse, angle);
                    return;
                default:
                    PL_UNREACHABLE;
                }
                return;
            } // else
            switch (min_rev_wire) {
            case 0:
                ApplyIsingXXAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            case 1:
                ApplyIsingXXAVX2::template applyInternalExternal<1>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            case 2:
                ApplyIsingXXAVX2::template applyInternalExternal<2>(
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
            if (num_qubits < 3) {
                GateImplementationsLM::applyIsingYY(arr, num_qubits, wires,
                                                    inverse, angle);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
            const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

            if (rev_wire0 < 3 && rev_wire1 < 3) {
                switch (rev_wire0 ^ rev_wire1) {
                case 1: // (0, 1)
                    ApplyIsingYYAVX2::template applyInternalInternal<0, 1>(
                        arr, num_qubits, inverse, angle);
                    return;
                case 2: // (0, 2)
                    ApplyIsingYYAVX2::template applyInternalInternal<0, 2>(
                        arr, num_qubits, inverse, angle);
                    return;
                case 3: // (1,2)
                    ApplyIsingYYAVX2::template applyInternalInternal<1, 2>(
                        arr, num_qubits, inverse, angle);
                    return;
                default:
                    PL_UNREACHABLE;
                }
                return;
            } // else
            switch (min_rev_wire) {
            case 0:
                ApplyIsingYYAVX2::template applyInternalExternal<0>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            case 1:
                ApplyIsingYYAVX2::template applyInternalExternal<1>(
                    arr, num_qubits, max_rev_wire, inverse, angle);
                return;
            case 2:
                ApplyIsingYYAVX2::template applyInternalExternal<2>(
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

        using ApplyIsingZZAVX512 =
            AVX::ApplyIsingZZ<PrecisionT, packed_bytes / sizeof(PrecisionT)>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyIsingZZ(arr, num_qubits, wires,
                                                    inverse, angle);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
            if (rev_wire0 < 3 && rev_wire1 < 3) {
                ApplyIsingZZAVX512::applyInternalInternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else if (std::min(rev_wire0, rev_wire1) < 3) {
                ApplyIsingZZAVX512::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX512::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplyIsingZZAVX512::applyInternalInternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                ApplyIsingZZAVX512::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX512::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }
};
} // namespace Pennylane::Gates
