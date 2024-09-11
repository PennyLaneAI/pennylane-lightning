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
#include "UtilKokkos.hpp"

#include "CPUMemoryModel.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::LightningKokkos::Functors;
using namespace Pennylane::LightningKokkos::Util;
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
    using KokkosSizeTVector = Kokkos::View<std::size_t *>;
    using UnmanagedComplexHostView =
        Kokkos::View<ComplexT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedSizeTHostView =
        Kokkos::View<std::size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstComplexHostView =
        Kokkos::View<const ComplexT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstSizeTHostView =
        Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedPrecisionHostView =
        Kokkos::View<PrecisionT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchViewComplex =
        Kokkos::View<ComplexT *, KokkosExecSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchViewSizeT =
        Kokkos::View<std::size_t *, KokkosExecSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using TeamPolicy = Kokkos::TeamPolicy<>;
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

    StateVectorKokkos() = delete;
    StateVectorKokkos(std::size_t num_qubits,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{num_qubits} {
        num_qubits_ = num_qubits;

        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize(kokkos_args);
            }
        }

#ifdef _WIN32
        PL_ABORT_IF_NOT(num_qubits,
                        "LightningKokkos zero-qubit device initialization is "
                        "not supported on Windows.");
#endif

        data_ = std::make_unique<KokkosVector>("data_", exp2(num_qubits));
        setBasisState(0U);
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
    void setBasisState(const std::size_t index) {
        KokkosVector sv_view =
            getView(); // circumvent error capturing this with KOKKOS_LAMBDA
        Kokkos::parallel_for(
            sv_view.size(), KOKKOS_LAMBDA(const std::size_t i) {
                sv_view(i) =
                    (i == index) ? ComplexT{1.0, 0.0} : ComplexT{0.0, 0.0};
            });
    }

    /**
     * @brief Prepares a single computational basis state.
     *
     * @param state Binary number representing the index
     * @param wires Wires.
     */
    void setBasisState(const std::vector<std::size_t> &state,
                       const std::vector<std::size_t> &wires) {
        PL_ABORT_IF_NOT(state.size() == wires.size(),
                        "state and wires must have equal dimensions.");
        const auto num_qubits = this->getNumQubits();
        PL_ABORT_IF_NOT(
            std::find_if(wires.begin(), wires.end(),
                         [&num_qubits](const auto i) {
                             return i >= num_qubits;
                         }) == wires.end(),
            "wires must take values lower than the number of qubits.");
        const auto n_wires = wires.size();
        std::size_t index{0U};
        for (std::size_t k = 0; k < n_wires; k++) {
            const auto bit = static_cast<std::size_t>(state[k]);
            index |= bit << (num_qubits - 1 - wires[k]);
        }
        setBasisState(index);
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
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    void setStateVector(const std::vector<std::size_t> &indices,
                        const std::vector<ComplexT> &values) {
        PL_ABORT_IF_NOT(indices.size() == values.size(),
                        "Inconsistent indices and values dimensions.")
        initZeros();
        auto d_indices = vector2view(indices);
        auto d_values = vector2view(values);
        KokkosVector sv_view =
            getView(); // circumvent error capturing this with KOKKOS_LAMBDA
        Kokkos::parallel_for(
            indices.size(), KOKKOS_LAMBDA(const std::size_t i) {
                sv_view(d_indices[i]) = d_values[i];
            });
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param state State.
     * @param wires Wires.
     */
    void setStateVector(const std::vector<ComplexT> &state,
                        const std::vector<std::size_t> &wires) {
        PL_ABORT_IF_NOT(state.size() == exp2(wires.size()),
                        "Inconsistent state and wires dimensions.");
        setStateVector(state.data(), wires);
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param state State.
     * @param wires Wires.
     */
    void setStateVector(const ComplexT *state,
                        const std::vector<std::size_t> &wires) {
        constexpr std::size_t one{1U};
        const auto num_qubits = this->getNumQubits();
        PL_ABORT_IF_NOT(
            std::find_if(wires.begin(), wires.end(),
                         [&num_qubits](const auto i) {
                             return i >= num_qubits;
                         }) == wires.end(),
            "wires must take values lower than the number of qubits.");
        const auto num_state = exp2(wires.size());
        auto d_sv = getView();
        auto d_state = pointer2view(state, num_state);
        auto d_wires = vector2view(wires);
        initZeros();
        Kokkos::parallel_for(
            num_state, KOKKOS_LAMBDA(const std::size_t i) {
                std::size_t index{0U};
                for (std::size_t w = 0; w < d_wires.size(); w++) {
                    const std::size_t bit = (i & (one << w)) >> w;
                    index |= bit << (num_qubits - 1 -
                                     d_wires(d_wires.size() - 1 - w));
                }
                d_sv(index) = d_state(i);
            });
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
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
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
            const std::size_t num_qubits = this->getNumQubits();
            const GateOperation gateop =
                reverse_lookup(gate_names, std::string_view{opName});
            applyNamedOperation<KokkosExecSpace>(gateop, *data_, num_qubits,
                                                 wires, inverse, params);
        } else {
            PL_ABORT_IF(gate_matrix.empty(),
                        std::string("Operation does not exist for ") + opName +
                            std::string(" and no matrix provided."));
            return applyMultiQubitOp(vector2view(gate_matrix), wires, inverse);
        }
    }

    /**
     * @brief Apply a PauliRot gate to the state-vector.
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Rotation angle.
     * @param word A Pauli word (e.g. "XYYX").
     */
    void applyPauliRot(const std::vector<std::size_t> &wires,
                       const bool inverse,
                       const std::vector<PrecisionT> &params,
                       const std::string &word) {
        PL_ABORT_IF_NOT(wires.size() == word.size(),
                        "wires and word have incompatible dimensions.");
        Pennylane::LightningKokkos::Functors::applyPauliRot<KokkosExecSpace,
                                                            PrecisionT>(
            getView(), this->getNumQubits(), wires, inverse, params[0], word);
    }

    template <bool inverse = false>
    void applyControlledGlobalPhase(const std::vector<ComplexT> &diagonal) {
        auto diagonal_ = vector2view(diagonal);
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
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
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
    void applyMultiQubitOp(const KokkosVector matrix,
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
    inline void applyMatrix(ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        std::size_t n = static_cast<std::size_t>(1U) << wires.size();
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
                            const std::vector<std::size_t> &wires,
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
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        std::size_t n = static_cast<std::size_t>(1U) << wires.size();
        std::size_t n2 = n * n;
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
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(matrix.data(), wires, inverse);
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
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<fp_t> &params = {}) -> fp_t {
        const std::size_t num_qubits = this->getNumQubits();
        const GeneratorOperation generator_op =
            reverse_lookup(generator_names, std::string_view{opName});
        return applyNamedGenerator<KokkosExecSpace>(
            generator_op, *data_, num_qubits, wires, inverse, params);
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
    void updateData(const KokkosVector other) {
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
        return view2vector(getView());
    }

    [[nodiscard]] auto getDataVector() const -> const std::vector<ComplexT> {
        return view2vector(getView());
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
    std::size_t num_qubits_;
    std::mutex init_mutex_;
    std::unique_ptr<KokkosVector> data_;
    inline static bool is_exit_reg_ = false;
};

}; // namespace Pennylane::LightningKokkos
