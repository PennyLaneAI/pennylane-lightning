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
 * @file StateVectorKokkos.hpp
 */

#pragma once
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "BitUtil.hpp" // isPerfectPowerOf2
#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateFunctors.hpp"
#include "GateOperation.hpp"
#include "StateVectorBase.hpp"
#include "Util.hpp"

#include "CPUMemoryModel.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::LightningKokkos::Functors;
using Pennylane::Gates::GateOperation;
using Pennylane::Gates::GeneratorOperation;
using Pennylane::Util::array_contains;
using Pennylane::Util::exp2;
using Pennylane::Util::isPerfectPowerOf2;
using Pennylane::Util::log2;
using Pennylane::Util::reverse_lookup;
using std::size_t;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos {
/**
 * @brief  Kokkos state vector class
 *
 * @tparam fp_t Floating-point precision type.
 */
template <class fp_t = double>
class StateVectorKokkos final
    : public StateVectorBase<fp_t, StateVectorKokkos<fp_t>> {
  private:
    using BaseType = StateVectorBase<fp_t, StateVectorKokkos<fp_t>>;

  public:
    using PrecisionT = fp_t;
    using ComplexT = Kokkos::complex<fp_t>;
    using CFP_t = ComplexT;
    using DoubleLoopRank = Kokkos::Rank<2>;
    using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using KokkosVector = Kokkos::View<ComplexT *>;
    using KokkosSizeTVector = Kokkos::View<size_t *>;
    using UnmanagedComplexHostView =
        Kokkos::View<ComplexT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedSizeTHostView =
        Kokkos::View<size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstComplexHostView =
        Kokkos::View<const ComplexT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstSizeTHostView =
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedPrecisionHostView =
        Kokkos::View<PrecisionT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchViewComplex =
        Kokkos::View<ComplexT *, KokkosExecSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchViewSizeT =
        Kokkos::View<size_t *, KokkosExecSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using TeamPolicy = Kokkos::TeamPolicy<>;
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

    StateVectorKokkos() = delete;
    StateVectorKokkos(size_t num_qubits,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{num_qubits} {
        num_qubits_ = num_qubits;

        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize(kokkos_args);
            }
        }
        if (num_qubits > 0) {
            data_ = std::make_unique<KokkosVector>("data_", exp2(num_qubits));
            setBasisState(0U);
        }
    };

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() { Kokkos::deep_copy(getView(), ComplexT{0.0, 0.0}); }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const size_t index) {
        KokkosVector sv_view =
            getView(); // circumvent error capturing this with KOKKOS_LAMBDA
        Kokkos::parallel_for(
            sv_view.size(), KOKKOS_LAMBDA(const size_t i) {
                sv_view(i) =
                    (i == index) ? ComplexT{1.0, 0.0} : ComplexT{0.0, 0.0};
            });
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    void setStateVector(const std::vector<std::size_t> &indices,
                        const std::vector<ComplexT> &values) {
        initZeros();
        KokkosSizeTVector d_indices("d_indices", indices.size());
        KokkosVector d_values("d_values", values.size());
        Kokkos::deep_copy(d_indices, UnmanagedConstSizeTHostView(
                                         indices.data(), indices.size()));
        Kokkos::deep_copy(d_values, UnmanagedConstComplexHostView(
                                        values.data(), values.size()));
        KokkosVector sv_view =
            getView(); // circumvent error capturing this with KOKKOS_LAMBDA
        Kokkos::parallel_for(
            indices.size(), KOKKOS_LAMBDA(const std::size_t i) {
                sv_view(d_indices[i]) = d_values[i];
            });
    }

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state.
     *
     * @param num_qubits Number of qubits
     */
    void resetStateVector() {
        if (this->getLength() > 0) {
            setBasisState(0U);
        }
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(ComplexT *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        HostToDevice(hostdata_, length);
    }

    StateVectorKokkos(std::complex<PrecisionT> *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        HostToDevice(reinterpret_cast<ComplexT *>(hostdata_), length);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(const ComplexT *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        std::vector<ComplexT> hostdata_copy(hostdata_, hostdata_ + length);
        HostToDevice(hostdata_copy.data(), length);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(std::vector<ComplexT> hostdata_,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(hostdata_.data(), hostdata_.size(), kokkos_args) {}

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkos(const StateVectorKokkos &other,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(other.getNumQubits(), kokkos_args) {
        this->DeviceToDevice(other.getView());
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     *
     * @param other Another state vector
     */
    ~StateVectorKokkos() {
        data_.reset();
        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!is_exit_reg_) {
                is_exit_reg_ = true;
                std::atexit([]() {
                    if (!Kokkos::is_finalized()) {
                        Kokkos::finalize();
                    }
                });
            }
        }
    }

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<fp_t> &params = {},
                        const std::vector<ComplexT> &gate_matrix = {}) {
        if (opName == "Identity") {
            // No op
        } else if (opName == "C(GlobalPhase)") {
            if (inverse) {
                applyControlledGlobalPhase<true>(gate_matrix);
            } else {
                applyControlledGlobalPhase<false>(gate_matrix);
            }
        } else if (array_contains(gate_names, std::string_view{opName})) {
            applyNamedOperation(opName, wires, inverse, params);
        } else {
            PL_ABORT_IF(gate_matrix.size() == 0,
                        std::string("Operation does not exist for ") + opName +
                            std::string(" and no matrix provided."));
            KokkosVector matrix("gate_matrix", gate_matrix.size());
            Kokkos::deep_copy(
                matrix, UnmanagedConstComplexHostView(gate_matrix.data(),
                                                      gate_matrix.size()));
            return applyMultiQubitOp(matrix, wires, inverse);
        }
    }

    template <bool inverse = false>
    void applyControlledGlobalPhase(const std::vector<ComplexT> &diagonal) {
        KokkosVector diagonal_("diagonal_", diagonal.size());
        Kokkos::deep_copy(diagonal_, UnmanagedConstComplexHostView(
                                         diagonal.data(), diagonal.size()));
        auto two2N = BaseType::getLength();
        auto dataview = getView();
        Kokkos::parallel_for(
            two2N, KOKKOS_LAMBDA(const std::size_t i) {
                dataview(i) *= (inverse) ? conj(diagonal_(i)) : diagonal_(i);
            });
    }

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<fp_t> &params = {},
                        const std::vector<ComplexT> &gate_matrix = {}) {
        PL_ABORT_IF_NOT(controlled_wires.empty(),
                        "Controlled kernels not implemented.");
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        applyOperation(opName, wires, inverse, params, gate_matrix);
    }

    /**
     * @brief Apply a multi qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applyMultiQubitOp(const KokkosVector &matrix,
                           const std::vector<std::size_t> &wires,
                           bool inverse = false) {
        auto &&num_qubits = this->getNumQubits();
        std::size_t two2N = std::exp2(num_qubits - wires.size());
        std::size_t dim = std::exp2(wires.size());
        KokkosVector matrix_trans("matrix_trans", matrix.size());

        if (inverse) {
            Kokkos::MDRangePolicy<DoubleLoopRank> policy_2d({0, 0}, {dim, dim});
            Kokkos::parallel_for(
                policy_2d,
                KOKKOS_LAMBDA(const std::size_t i, const std::size_t j) {
                    matrix_trans(i + j * dim) = conj(matrix(i * dim + j));
                });
        } else {
            matrix_trans = matrix;
        }
        switch (wires.size()) {
        case 1:
            Kokkos::parallel_for(
                two2N, apply1QubitOpFunctor<fp_t>(*data_, num_qubits,
                                                  matrix_trans, wires));
            break;
        case 2:
            Kokkos::parallel_for(
                two2N, apply2QubitOpFunctor<fp_t>(*data_, num_qubits,
                                                  matrix_trans, wires));
            break;
        case 3:
            Kokkos::parallel_for(
                two2N, apply3QubitOpFunctor<fp_t>(*data_, num_qubits,
                                                  matrix_trans, wires));
            break;
        case 4:
            Kokkos::parallel_for(
                two2N, apply4QubitOpFunctor<fp_t>(*data_, num_qubits,
                                                  matrix_trans, wires));
            break;
        default:
            std::size_t scratch_size = ScratchViewComplex::shmem_size(dim) +
                                       ScratchViewSizeT::shmem_size(dim);
            Kokkos::parallel_for(
                "multiQubitOpFunctor",
                TeamPolicy(two2N, Kokkos::AUTO, dim)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
                multiQubitOpFunctor<PrecisionT>(*data_, num_qubits,
                                                matrix_trans, wires));
            break;
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(ComplexT *matrix, const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        size_t n = static_cast<std::size_t>(1U) << wires.size();
        KokkosVector matrix_(matrix, n * n);
        applyMultiQubitOp(matrix_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(std::vector<ComplexT> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(matrix.data(), wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const ComplexT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        size_t n = static_cast<std::size_t>(1U) << wires.size();
        size_t n2 = n * n;
        KokkosVector matrix_("matrix_", n2);
        Kokkos::deep_copy(matrix_, UnmanagedConstComplexHostView(matrix, n2));
        applyMultiQubitOp(matrix_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const std::vector<ComplexT> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(matrix.data(), wires, inverse);
    }

    void applyNamedOperation(const std::string &opName,
                             const std::vector<size_t> &wires,
                             bool inverse = false,
                             const std::vector<fp_t> &params = {}) {
        switch (reverse_lookup(gate_names, std::string_view{opName})) {
        case GateOperation::PauliX:
            applyGateFunctor<pauliXFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::PauliY:
            applyGateFunctor<pauliYFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::PauliZ:
            applyGateFunctor<pauliZFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::Hadamard:
            applyGateFunctor<hadamardFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::S:
            applyGateFunctor<sFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::T:
            applyGateFunctor<tFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::RX:
            applyGateFunctor<rxFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::RY:
            applyGateFunctor<ryFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::RZ:
            applyGateFunctor<rzFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::PhaseShift:
            applyGateFunctor<phaseShiftFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::Rot:
            applyGateFunctor<rotFunctor, 1>(wires, inverse, params);
            return;
        case GateOperation::CY:
            applyGateFunctor<cyFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::CZ:
            applyGateFunctor<czFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::CNOT:
            applyGateFunctor<cnotFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::SWAP:
            applyGateFunctor<swapFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::ControlledPhaseShift:
            applyGateFunctor<controlledPhaseShiftFunctor, 2>(wires, inverse,
                                                             params);
            return;
        case GateOperation::CRX:
            applyGateFunctor<crxFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::CRY:
            applyGateFunctor<cryFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::CRZ:
            applyGateFunctor<crzFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::CRot:
            applyGateFunctor<cRotFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::IsingXX:
            applyGateFunctor<isingXXFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::IsingXY:
            applyGateFunctor<isingXYFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::IsingYY:
            applyGateFunctor<isingYYFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::IsingZZ:
            applyGateFunctor<isingZZFunctor, 2>(wires, inverse, params);
            return;
        case GateOperation::SingleExcitation:
            applyGateFunctor<singleExcitationFunctor, 2>(wires, inverse,
                                                         params);
            return;
        case GateOperation::SingleExcitationMinus:
            applyGateFunctor<singleExcitationMinusFunctor, 2>(wires, inverse,
                                                              params);
            return;
        case GateOperation::SingleExcitationPlus:
            applyGateFunctor<singleExcitationPlusFunctor, 2>(wires, inverse,
                                                             params);
            return;
        case GateOperation::DoubleExcitation:
            applyGateFunctor<doubleExcitationFunctor, 4>(wires, inverse,
                                                         params);
            return;
        case GateOperation::DoubleExcitationMinus:
            applyGateFunctor<doubleExcitationMinusFunctor, 4>(wires, inverse,
                                                              params);
            return;
        case GateOperation::DoubleExcitationPlus:
            applyGateFunctor<doubleExcitationPlusFunctor, 4>(wires, inverse,
                                                             params);
            return;
        case GateOperation::MultiRZ:
            applyMultiRZ(wires, inverse, params);
            return;
        case GateOperation::GlobalPhase:
            applyGlobalPhase(wires, inverse, params);
            return;
        case GateOperation::CSWAP:
            applyGateFunctor<cSWAPFunctor, 3>(wires, inverse, params);
            return;
        case GateOperation::Toffoli:
            applyGateFunctor<toffoliFunctor, 3>(wires, inverse, params);
            return;
        default:
            PL_ABORT(std::string("Operation does not exist for ") + opName);
        }
    }

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<fp_t> &params = {}) -> fp_t {
        switch (reverse_lookup(generator_names, std::string_view{opName})) {
        case GeneratorOperation::RX:
            applyGateFunctor<pauliXFunctor, 1>(wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::RY:
            applyGateFunctor<pauliYFunctor, 1>(wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::RZ:
            applyGateFunctor<pauliZFunctor, 1>(wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::PhaseShift:
            applyGateFunctor<generatorPhaseShiftFunctor, 1>(wires, inverse,
                                                            params);
            return static_cast<fp_t>(1.0);
        case GeneratorOperation::IsingXX:
            applyGateFunctor<generatorIsingXXFunctor, 2>(wires, inverse,
                                                         params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::IsingXY:
            applyGateFunctor<generatorIsingXYFunctor, 2>(wires, inverse,
                                                         params);
            return static_cast<fp_t>(0.5);
        case GeneratorOperation::IsingYY:
            applyGateFunctor<generatorIsingYYFunctor, 2>(wires, inverse,
                                                         params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::IsingZZ:
            applyGateFunctor<generatorIsingZZFunctor, 2>(wires, inverse,
                                                         params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::SingleExcitation:
            applyGateFunctor<generatorSingleExcitationFunctor, 2>(
                wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::SingleExcitationMinus:
            applyGateFunctor<generatorSingleExcitationMinusFunctor, 2>(
                wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::SingleExcitationPlus:
            applyGateFunctor<generatorSingleExcitationPlusFunctor, 2>(
                wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::DoubleExcitation:
            applyGateFunctor<generatorDoubleExcitationFunctor, 4>(
                wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::DoubleExcitationMinus:
            applyGateFunctor<generatorDoubleExcitationMinusFunctor, 4>(
                wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::DoubleExcitationPlus:
            applyGateFunctor<generatorDoubleExcitationPlusFunctor, 4>(
                wires, inverse, params);
            return static_cast<fp_t>(0.5);
        case GeneratorOperation::ControlledPhaseShift:
            applyGateFunctor<generatorControlledPhaseShiftFunctor, 2>(
                wires, inverse, params);
            return static_cast<fp_t>(1);
        case GeneratorOperation::CRX:
            applyGateFunctor<generatorCRXFunctor, 2>(wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::CRY:
            applyGateFunctor<generatorCRYFunctor, 2>(wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::CRZ:
            applyGateFunctor<generatorCRZFunctor, 2>(wires, inverse, params);
            return -static_cast<fp_t>(0.5);
        case GeneratorOperation::MultiRZ:
            return applyGeneratorMultiRZ(wires, inverse, params);
        case GeneratorOperation::GlobalPhase:
            return static_cast<PrecisionT>(-1.0);
        /// LCOV_EXCL_START
        default:
            PL_ABORT(std::string("Generator does not exist for ") + opName);
            /// LCOV_EXCL_STOP
        }
    }

    /**
     * @brief Templated method that applies special n-qubit gates.
     *
     * @tparam functor_t Gate functor class for Kokkos dispatcher.
     * @tparam nqubits Number of qubits.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    template <template <class, bool> class functor_t, int nqubits>
    void applyGateFunctor(const std::vector<size_t> &wires,
                          bool inverse = false,
                          const std::vector<fp_t> &params = {}) {
        auto &&num_qubits = this->getNumQubits();
        PL_ASSERT(wires.size() == nqubits);
        PL_ASSERT(wires.size() <= num_qubits);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, exp2(num_qubits - nqubits)),
                functor_t<fp_t, false>(*data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, exp2(num_qubits - nqubits)),
                functor_t<fp_t, true>(*data_, num_qubits, wires, params));
        }
    }

    /**
     * @brief Apply a MultiRZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyMultiRZ(const std::vector<size_t> &wires, bool inverse = false,
                      const std::vector<fp_t> &params = {}) {
        auto &&num_qubits = this->getNumQubits();

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                multiRZFunctor<fp_t, false>(*data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                multiRZFunctor<fp_t, true>(*data_, num_qubits, wires, params));
        }
    }

    /**
     * @brief Apply a GlobalPhase operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyGlobalPhase(const std::vector<size_t> &wires,
                          bool inverse = false,
                          const std::vector<fp_t> &params = {}) {
        auto &&num_qubits = this->getNumQubits();

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                globalPhaseFunctor<fp_t, false>(*data_, num_qubits, wires,
                                                params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                globalPhaseFunctor<fp_t, true>(*data_, num_qubits, wires,
                                               params));
        }
    }

    /**
     * @brief Apply a MultiRZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorMultiRZ(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        auto &&num_qubits = this->getNumQubits();

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                generatorMultiRZFunctor<fp_t, false>(*data_, num_qubits,
                                                     wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                generatorMultiRZFunctor<fp_t, true>(*data_, num_qubits, wires));
        }
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Collapse the state vector after having measured one of the
     * qubits.
     *
     * The branch parameter imposes the measurement result on the given wire.
     *
     * @param wire Wire to collapse.
     * @param branch Branch 0 or 1.
     */
    void collapse(std::size_t wire, bool branch) {
        KokkosVector matrix("gate_matrix", 4);
        Kokkos::parallel_for(
            matrix.size(), KOKKOS_LAMBDA(std::size_t k) {
                matrix(k) = ((k == 0 && branch == 0) || (k == 3 && branch == 1))
                                ? ComplexT{1.0, 0.0}
                                : ComplexT{0.0, 0.0};
            });
        applyMultiQubitOp(matrix, {wire}, false);
        normalize();
    }

    /**
     * @brief Normalize vector (to have norm 1).
     */
    void normalize() {
        auto sv_view = getView();

        PrecisionT squaredNorm = 0.0;
        Kokkos::parallel_reduce(
            sv_view.size(),
            KOKKOS_LAMBDA(std::size_t i, PrecisionT & sum) {
                const PrecisionT norm = Kokkos::abs(sv_view(i));
                sum += norm * norm;
            },
            squaredNorm);

        PL_ABORT_IF(squaredNorm <
                        std::numeric_limits<PrecisionT>::epsilon() * 1e2,
                    "vector has norm close to zero and can't be normalized");

        const std::complex<PrecisionT> inv_norm =
            1. / Kokkos::sqrt(squaredNorm);
        Kokkos::parallel_for(
            sv_view.size(),
            KOKKOS_LAMBDA(std::size_t i) { sv_view(i) *= inv_norm; });
    }

    /**
     * @brief Update data of the class
     *
     * @param other Kokkos View
     */
    void updateData(const KokkosVector &other) {
        Kokkos::deep_copy(*data_, other);
    }

    /**
     * @brief Update data of the class
     *
     * @param other State vector
     */
    void updateData(const StateVectorKokkos<fp_t> &other) {
        updateData(other.getView());
    }

    /**
     * @brief Update data of the class
     *
     * @param new_data data pointer to new data.
     * @param new_size size of underlying data storage.
     */
    void updateData(ComplexT *new_data, std::size_t new_size) {
        updateData(KokkosVector(new_data, new_size));
    }

    /**
     * @brief Update data of the class
     *
     * @param other STL vector of type ComplexT
     */
    void updateData(std::vector<ComplexT> &other) {
        updateData(other.data(), other.size());
    }

    [[nodiscard]] auto getData() -> ComplexT * { return getView().data(); }

    [[nodiscard]] auto getData() const -> const ComplexT * {
        return getView().data();
    }

    /**
     * @brief Get the Kokkos data of the state vector.
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getView() const -> KokkosVector & { return *data_; }

    /**
     * @brief Get the Kokkos data of the state vector
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getView() -> KokkosVector & { return *data_; }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector() -> std::vector<ComplexT> {
        std::vector<ComplexT> data_(this->getLength());
        DeviceToHost(data_.data(), data_.size());
        return data_;
    }

    [[nodiscard]] auto getDataVector() const -> const std::vector<ComplexT> {
        std::vector<ComplexT> data_(this->getLength());
        DeviceToHost(data_.data(), data_.size());
        return data_;
    }

    /**
     * @brief Copy data from the host space to the device space.
     *
     */
    inline void HostToDevice(ComplexT *sv, std::size_t length) {
        Kokkos::deep_copy(*data_, UnmanagedComplexHostView(sv, length));
    }

    /**
     * @brief Copy data from the device space to the host space.
     *
     */
    inline void DeviceToHost(ComplexT *sv, std::size_t length) const {
        Kokkos::deep_copy(UnmanagedComplexHostView(sv, length), *data_);
    }

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        Kokkos::deep_copy(*data_, vector_to_copy);
    }

  private:
    size_t num_qubits_;
    std::mutex init_mutex_;
    std::unique_ptr<KokkosVector> data_;
    inline static bool is_exit_reg_ = false;
};

}; // namespace Pennylane::LightningKokkos
