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
    using TeamPolicy = Util::TeamPolicy<>;
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

    StateVectorKokkos() = delete;

    /**
     * @brief Create a new state vector of the given number of qubits,
     * initialized to the |0...0> basis state, using Kokkos's own default
     * execution-space instance (lazily initializing Kokkos via
     * kokkos_args if this is the first Kokkos-related construction in the
     * process).
     *
     * @param num_qubits Number of qubits.
     * @param kokkos_args Arguments for Kokkos initialization. Ignored if
     * Kokkos is already initialized.
     */
    StateVectorKokkos(std::size_t num_qubits,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{num_qubits}, kokkos_init_guard_(kokkos_args), exec_() {
        // kokkos_init_guard_ (declared/constructed before exec_ below) has
        // already ensured Kokkos::initialize() has run by this point, so
        // exec_ default-constructing to KokkosExecSpace's own default
        // instance here is safe -- behavior-identical to the previous
        // implicit-default policy/view construction throughout this class (a
        // default-constructed instance IS what e.g. RangePolicy<Space>(0, N)
        // uses internally when no instance is given explicitly).  Real
        // per-device/per-stream binding only happens via the exec_space
        // constructor overload below.
        num_qubits_ = num_qubits;

#ifdef _WIN32
        PL_ABORT_IF_NOT(num_qubits,
                        "LightningKokkos zero-qubit device initialization is "
                        "not supported on Windows.");
#endif

        data_ = std::make_unique<KokkosVector>(
            Kokkos::view_alloc(exec_, "data_"), exp2(num_qubits));
        setBasisState(0U);
    };

    /**
     * @brief Create a new state vector bound to a specific, caller-owned
     * Kokkos execution-space instance (e.g. Kokkos::HIP(stream) built from a
     * stream on a specific physical GPU).
     *
     * Unlike the constructors above, this does NOT let Kokkos::initialize()
     * pick (or bias) which device this state vector's memory/kernels live
     * on: Kokkos::initialize() binds the WHOLE PROCESS to one device for its
     * entire lifetime, so it cannot be used to place different
     * StateVectorKokkos instances on different physical GPUs within one
     * process.  This constructor sidesteps that: `exec_space` is used
     * explicitly for every View allocation and every kernel dispatch this
     * instance performs (see e.g. applyOperation/applyMultiQubitOp/etc.),
     * so it works regardless of which device Kokkos::initialize() itself
     * happens to be bound to -- the caller is responsible for constructing
     * `exec_space` against the device/stream it wants (e.g.
     * hipSetDevice(N); hipStreamCreate(&stream); Kokkos::HIP exec(stream);)
     * and for that stream's lifetime outliving this state vector.
     *
     * @param num_qubits Number of qubits.
     * @param exec_space Execution-space instance to bind this state vector
     * to.
     */
    StateVectorKokkos(std::size_t num_qubits, const KokkosExecSpace &exec_space)
        : BaseType{num_qubits}, kokkos_init_guard_(), exec_(exec_space) {
        // kokkos_init_guard_ uses no device_id preference here on purpose:
        // with per-instance binding, Kokkos::initialize()'s own default
        // device is irrelevant to where THIS state vector's data/kernels
        // run -- exec_space (constructed by the caller against whichever
        // device/stream it wants) is what actually matters.
        num_qubits_ = num_qubits;

#ifdef _WIN32
        PL_ABORT_IF_NOT(num_qubits,
                        "LightningKokkos zero-qubit device initialization is "
                        "not supported on Windows.");
#endif

        data_ = std::make_unique<KokkosVector>(
            Kokkos::view_alloc(exec_, "data_"), exp2(num_qubits));
        setBasisState(0U);
    };

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() {
        Kokkos::deep_copy(exec_, getView(), ComplexT{0.0, 0.0});
    }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(std::size_t index) {
        KokkosVector sv_view =
            getView(); // circumvent error capturing this with KOKKOS_LAMBDA
        Kokkos::parallel_for(
            RangePolicy<KokkosExecSpace>(exec_, 0, sv_view.size()),
            KOKKOS_LAMBDA(std::size_t i) {
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
     */
    void resetStateVector() {
        if (this->getLength() > 0) {
            setBasisState(0U);
        }
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param indices Indices of the target elements.
     * @param values Values to be set for the target elements.
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
            RangePolicy<KokkosExecSpace>(exec_, 0, indices.size()),
            KOKKOS_LAMBDA(std::size_t i) {
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
            RangePolicy<KokkosExecSpace>(exec_, 0, num_state),
            KOKKOS_LAMBDA(std::size_t i) {
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
     * @param hostdata_ Host array for state vector
     * @param length Length of host array (must be power of 2)
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkos(ComplexT *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        HostToDevice(hostdata_, length);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param hostdata_ Host vector for state vector
     * @param length Length of host array (must be power of 2)
     * @param kokkos_args Arguments for Kokkos initialization
     */
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
     * @param hostdata_ Host array for state vector
     * @param length Length of host array (must be power of 2)
     * @param kokkos_args Arguments for Kokkos initialization
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
     * @param hostdata_ Host vector for state vector
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkos(std::vector<ComplexT> hostdata_,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(hostdata_.data(), hostdata_.size(), kokkos_args) {}

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param hostdata_ Host vector for state vector
     * @param kokkos_args Arguments for Kokkos initialization
     */
    template <class complex>
    StateVectorKokkos(std::vector<complex> hostdata_,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(reinterpret_cast<ComplexT *>(hostdata_.data()),
                            hostdata_.size(), kokkos_args) {}

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     * @param kokkos_args Arguments for Kokkos initialization (ignored: the
     * copy is placed on `other`'s own execution-space instance instead, so
     * it stays on the same device/stream as the state vector it's copying --
     * `kokkos_args` only matters the very first time Kokkos::initialize()
     * runs, which has already happened by the time `other` exists).
     */
    StateVectorKokkos(
        const StateVectorKokkos &other,
        [[maybe_unused]] const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(other.getNumQubits(), other.exec_) {
        this->DeviceToDevice(other.getView());
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     */
    ~StateVectorKokkos() {
        data_.reset();
        {
            const std::lock_guard<std::mutex> lock(KokkosInitGuard::mutex());
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
        } else if (array_contains(gate_names, std::string_view{opName})) {
            const std::size_t num_qubits = this->getNumQubits();
            const GateOperation gateop =
                reverse_lookup(gate_names, std::string_view{opName});
            applyNamedOperation<KokkosExecSpace>(
                exec_, gateop, *data_, num_qubits, wires, inverse, params);
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
    void applyPauliRot(const std::vector<std::size_t> &wires, bool inverse,
                       const std::vector<PrecisionT> &params,
                       const std::string &word) {
        PL_ABORT_IF_NOT(wires.size() == word.size(),
                        "wires and word have incompatible dimensions.");
        Pennylane::LightningKokkos::Functors::applyPauliRot<KokkosExecSpace,
                                                            PrecisionT>(
            exec_, getView(), this->getNumQubits(), wires, inverse, params[0],
            word);
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
        const std::size_t num_qubits = this->getNumQubits();
        const std::size_t two2N = exp2(num_qubits - wires.size());
        const std::size_t dim = exp2(wires.size());
        KokkosVector matrix_trans(Kokkos::view_alloc(exec_, "matrix_trans"),
                                  matrix.size());

        if (inverse) {
            // MDRangePolicy bounds are Kokkos::Array<int64_t, rank> regardless
            // of IndexType, so cast from size_t to silence -Wnarrowing.
            MDRangePolicy<2, KokkosExecSpace> policy_2d(
                exec_, {0, 0},
                {static_cast<std::int64_t>(dim),
                 static_cast<std::int64_t>(dim)});
            Kokkos::parallel_for(
                policy_2d, KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
                    matrix_trans(i + j * dim) =
                        Kokkos::conj(matrix(i * dim + j));
                });
        } else {
            matrix_trans = matrix;
        }
        switch (wires.size()) {
        case 1:
            Kokkos::parallel_for(
                RangePolicy<KokkosExecSpace>(exec_, 0, two2N),
                apply1QubitOpFunctor<fp_t>(exec_, *data_, num_qubits,
                                           matrix_trans, wires));
            break;
        case 2:
            Kokkos::parallel_for(
                RangePolicy<KokkosExecSpace>(exec_, 0, two2N),
                apply2QubitOpFunctor<fp_t>(exec_, *data_, num_qubits,
                                           matrix_trans, wires));
            break;
        case 3:
            Kokkos::parallel_for(
                RangePolicy<KokkosExecSpace>(exec_, 0, two2N),
                apply3QubitOpFunctor<fp_t>(exec_, *data_, num_qubits,
                                           matrix_trans, wires));
            break;
        case 4:
            Kokkos::parallel_for(
                RangePolicy<KokkosExecSpace>(exec_, 0, two2N),
                apply4QubitOpFunctor<fp_t>(exec_, *data_, num_qubits,
                                           matrix_trans, wires));
            break;
        default:
            // TODO: explore runtime determine L0 or L1 scratch level (for GPU
            // shared memory), or SIMD
            std::size_t scratch_size = ScratchViewComplex::shmem_size(dim) +
                                       ScratchViewSizeT::shmem_size(dim);
            // Kokkos stores TeamPolicy league_size as a 32-bit int on the
            // CUDA/HIP backends, so abort rather than silently truncate when
            // the number of teams would exceed the 2^31 league_size limit.
            PL_ABORT_IF(
                two2N >= (std::size_t{1} << 31),
                "Number of thread teams exceeds the Kokkos TeamPolicy "
                "league_size limit (2^31) on GPU backends for this gate.");
            Kokkos::parallel_for(
                "multiQubitOpFunctor",
                TeamPolicy(exec_, two2N, Kokkos::AUTO, dim)
                    .set_scratch_size(1, Kokkos::PerTeam(scratch_size)),
                multiQubitOpFunctor<PrecisionT>(exec_, *data_, num_qubits,
                                                matrix_trans, wires));
            break;
        }
    }

    /**
     * @brief Apply a controlled-single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate. (Default to
     * false)
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional unitary gate matrix if opName doesn't exist.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<fp_t> &params = {},
                        const std::vector<ComplexT> &gate_matrix = {}) {
        PL_ABORT_IF_NOT(
            areVecsDisjoint<std::size_t>(controlled_wires, wires),
            "`controlled_wires` and target wires must be disjoint.");
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");

        if (controlled_wires.empty()) {
            return applyOperation(opName, wires, inverse, params, gate_matrix);
        }
        if (array_contains(controlled_gate_names, std::string_view{opName})) {
            const std::size_t num_qubits = this->getNumQubits();
            const ControlledGateOperation gateop =
                reverse_lookup(controlled_gate_names, std::string_view{opName});
            applyNCNamedOperation<KokkosExecSpace>(
                exec_, gateop, *data_, num_qubits, controlled_wires,
                controlled_values, wires, inverse, params);
        } else {
            PL_ABORT_IF(gate_matrix.empty(),
                        std::string("Operation does not exist for ") + opName +
                            std::string(" and no matrix provided."));
            return applyNCMultiQubitOp(vector2view(gate_matrix),
                                       controlled_wires, controlled_values,
                                       wires, inverse);
        }
    }

    /**
     * @brief Apply a controlled-multi qubit operator to the state vector using
     * a matrix
     *
     * @param matrix Kokkos gate matrix in the device space.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (true or false).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate. (Default to
     * false)
     */
    void applyNCMultiQubitOp(const KokkosVector matrix,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             bool inverse = false) {
        const std::size_t num_qubits = this->getNumQubits();
        PL_ASSERT(num_qubits >= wires.size() + controlled_wires.size());
        const std::size_t two2N =
            exp2(num_qubits - wires.size() - controlled_wires.size());
        const std::size_t dim = exp2(wires.size());
        KokkosVector matrix_trans(Kokkos::view_alloc(exec_, "matrix_trans"),
                                  matrix.size());

        if (inverse) {
            MDRangePolicy<2, KokkosExecSpace> policy_2d(
                exec_, {0, 0},
                {static_cast<std::int64_t>(dim),
                 static_cast<std::int64_t>(dim)});
            Kokkos::parallel_for(
                policy_2d, KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
                    matrix_trans(i + j * dim) =
                        Kokkos::conj(matrix(i * dim + j));
                });
        } else {
            matrix_trans = matrix;
        }

        switch (wires.size()) {
        case 1:
            Kokkos::parallel_for(
                RangePolicy<KokkosExecSpace>(exec_, 0, two2N),
                applyNC1QubitOpFunctor<fp_t>(exec_, *data_, num_qubits,
                                             matrix_trans, controlled_wires,
                                             controlled_values, wires));
            break;
        case 2:
            Kokkos::parallel_for(
                RangePolicy<KokkosExecSpace>(exec_, 0, two2N),
                applyNC2QubitOpFunctor<fp_t>(exec_, *data_, num_qubits,
                                             matrix_trans, controlled_wires,
                                             controlled_values, wires));
            break;
        case 3:
            Kokkos::parallel_for(
                RangePolicy<KokkosExecSpace>(exec_, 0, two2N),
                applyNC3QubitOpFunctor<fp_t>(exec_, *data_, num_qubits,
                                             matrix_trans, controlled_wires,
                                             controlled_values, wires));
            break;
        default:
            // TODO: explore runtime determine L0 or L1 scratch level (for GPU
            // shared memory), or SIMD
            const std::size_t scratch_size =
                ScratchViewComplex::shmem_size(dim) +
                ScratchViewSizeT::shmem_size(dim);
            // Kokkos stores TeamPolicy league_size as a 32-bit int on the
            // CUDA/HIP backends, so abort rather than silently truncate when
            // the number of teams would exceed the 2^31 league_size limit.
            PL_ABORT_IF(
                two2N >= (std::size_t{1} << 31),
                "Number of thread teams exceeds the Kokkos TeamPolicy "
                "league_size limit (2^31) on GPU backends for this gate.");
            Kokkos::parallel_for(
                "multiNCQubitOpFunctor",
                TeamPolicy(exec_, two2N, Kokkos::AUTO, dim)
                    .set_scratch_size(1, Kokkos::PerTeam(scratch_size)),
                NCMultiQubitOpFunctor<PrecisionT>(
                    exec_, *data_, num_qubits, matrix_trans, controlled_wires,
                    controlled_values, wires));
            break;
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer on host memory.
     *
     * @param matrix Pointer to the array data on host memory (in row-major
     * format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken. (Default to
     * false)
     */
    inline void applyMatrix(ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::size_t n2 = exp2(wires.size() * 2);
        KokkosVector matrix_(Kokkos::view_alloc(exec_, "matrix_"), n2);

        // Note that when copying data between different memory spaces (host !=
        // device), Kokkos::View<Kokkos::complex*> cannot perform a deep copy of
        // unmanaged complex numbers during initialization via its constructor.
        // Thus, we need to explicitly deep-copy the matrix data using
        // Kokkos::deep_copy().
        Kokkos::deep_copy(exec_, matrix_, UnmanagedComplexHostView(matrix, n2));

        applyMultiQubitOp(matrix_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector on host memory.
     *
     * @param matrix Pointer to host matrix to apply to wires (in row-major
     * format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken. (Default to
     * false)
     */
    inline void applyMatrix(const ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::size_t n2 = exp2(wires.size() * 2);
        KokkosVector matrix_(Kokkos::view_alloc(exec_, "matrix_"), n2);
        Kokkos::deep_copy(exec_, matrix_,
                          UnmanagedConstComplexHostView(matrix, n2));
        applyMultiQubitOp(matrix_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix as a vector directly to the statevector.
     *
     * @param matrix Matrix data as a vector (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken. (Default to
     * false)
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
     * @brief Apply a given matrix for controlled operations directly to the
     * statevector using a raw matrix pointer on host memory.
     *
     * @param matrix  Pointer to host matrix to apply to wires (in row-major
     * format).
     * @param controlled_wires Controlled wires
     * @param controlled_values Controlled values (true or false)
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken. (Default to
     * false)
     */
    inline void applyControlledMatrix(
        ComplexT *matrix, const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::size_t n2 = exp2(wires.size() * 2);
        KokkosVector matrix_(Kokkos::view_alloc(exec_, "matrix_"), n2);
        Kokkos::deep_copy(exec_, matrix_, UnmanagedComplexHostView(matrix, n2));
        applyNCMultiQubitOp(matrix_, controlled_wires, controlled_values, wires,
                            inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer on host memory.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param controlled_wires Controlled wires
     * @param controlled_values Controlled values (true or false)
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken. (Default to
     * false)
     */
    inline void
    applyControlledMatrix(const ComplexT *matrix,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::size_t n2 = exp2(wires.size() * 2);
        KokkosVector matrix_(Kokkos::view_alloc(exec_, "matrix_"), n2);
        Kokkos::deep_copy(exec_, matrix_,
                          UnmanagedConstComplexHostView(matrix, n2));
        applyNCMultiQubitOp(matrix_, controlled_wires, controlled_values, wires,
                            inverse);
    }

    /**
     * @brief Apply a given controlled-matrix as a vector directly to the
     * statevector.
     *
     * @param matrix  Matrix data as a vector to apply to target wires (in
     * row-major format).
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken. (Default to
     * false)
     */
    inline void
    applyControlledMatrix(const std::vector<ComplexT> &matrix,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyControlledMatrix(matrix.data(), controlled_wires,
                              controlled_values, wires, inverse);
    }

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate. (Default to
     * false)
     * @return PrecisionT Generator scale prefactor
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false) -> PrecisionT {
        const std::size_t num_qubits = this->getNumQubits();
        const GeneratorOperation generator_op =
            reverse_lookup(generator_names, std::string_view{opName});
        return applyNamedGenerator<KokkosExecSpace>(exec_, generator_op, *data_,
                                                    num_qubits, wires, inverse);
    }

    /**
     * @brief Apply a single controlled generator to the state vector using the
     * given kernel.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (true or false).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate. (Default to
     * false)
     * @return PrecisionT Generator scale prefactor
     */
    auto
    applyControlledGenerator(const std::string &opName,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             bool inverse = false) -> PrecisionT {
        const std::size_t num_qubits = this->getNumQubits();
        const ControlledGeneratorOperation generator_op = reverse_lookup(
            controlled_generator_names, std::string_view{opName});
        return applyNCNamedGenerator<KokkosExecSpace>(
            exec_, generator_op, *data_, num_qubits, controlled_wires,
            controlled_values, wires, inverse);
    }

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (true or false).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate. (Default to
     * false)
     * @return PrecisionT Generator scale prefactor
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false) -> PrecisionT {
        PL_ABORT_IF_NOT(
            areVecsDisjoint<std::size_t>(controlled_wires, wires),
            "`controlled_wires` and `target wires` must be disjoint.");
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        if (controlled_wires.empty()) {
            return applyGenerator(opName, wires, inverse);
        }
        return applyControlledGenerator(opName, controlled_wires,
                                        controlled_values, wires, inverse);
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
        KokkosVector matrix(Kokkos::view_alloc(exec_, "gate_matrix"), 4);
        Kokkos::parallel_for(
            RangePolicy<KokkosExecSpace>(exec_, 0, matrix.size()),
            KOKKOS_LAMBDA(std::size_t k) {
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
            RangePolicy<KokkosExecSpace>(exec_, 0, sv_view.size()),
            KOKKOS_LAMBDA(std::size_t i, PrecisionT &sum) {
                const PrecisionT norm = Kokkos::abs(sv_view(i));
                sum += norm * norm;
            },
            squaredNorm);

        PL_ABORT_IF(squaredNorm <
                        std::numeric_limits<PrecisionT>::epsilon() * 1e2,
                    "Vector has norm close to zero and cannot be normalized");

        const std::complex<PrecisionT> inv_norm =
            1. / Kokkos::sqrt(squaredNorm);
        Kokkos::parallel_for(
            RangePolicy<KokkosExecSpace>(exec_, 0, sv_view.size()),
            KOKKOS_LAMBDA(std::size_t i) { sv_view(i) *= inv_norm; });
    }

    /**
     * @brief Update data of the class
     *
     * @param other Kokkos View
     */
    void updateData(const KokkosVector other) {
        Kokkos::deep_copy(exec_, *data_, other);
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

    /**
     * @brief Get underlying Kokkos view data on the device
     *
     * @return ComplexT *
     */
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
     * @brief Get the execution-space instance this state vector's data and
     * kernels are bound to (see the exec_space constructor overload above).
     *
     * @return The execution-space instance.
     */
    [[nodiscard]] auto exec() const -> const KokkosExecSpace & { return exec_; }

    /**
     * @brief Get the vector-converted Kokkos view
     *
     * @return std::vector<ComplexT>
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
        Kokkos::deep_copy(exec_, *data_, UnmanagedComplexHostView(sv, length));
    }

    /**
     * @brief Copy data from the device space to the host space.
     *
     */
    inline void DeviceToHost(ComplexT *sv, std::size_t length) const {
        Kokkos::deep_copy(exec_, UnmanagedComplexHostView(sv, length), *data_);
    }

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        Kokkos::deep_copy(exec_, *data_, vector_to_copy);
    }

  private:
    std::size_t num_qubits_;

    /**
     * @brief Ensures Kokkos::initialize() has run, exactly once per
     * process, before any KokkosExecSpace instance is constructed.
     *
     * Kokkos::initialize() is a process-wide singleton operation, so the
     * mutex guarding it (see mutex() below) is deliberately *static*
     * (shared by every instance) rather than a per-instance member: a
     * per-instance mutex cannot serialize concurrent construction of
     * multiple StateVectorKokkos instances from different threads, which is
     * exactly the scenario the exec_space constructor overload above exists
     * to support.
     */
    struct KokkosInitGuard {
        /**
         * @brief Run Kokkos::initialize(kokkos_args) if Kokkos has not
         * already been initialized in this process.
         *
         * @param kokkos_args Arguments for Kokkos initialization. Ignored
         * if Kokkos is already initialized (Kokkos::initialize() only ever
         * takes effect the first time it is called in a process).
         */
        explicit KokkosInitGuard(
            const Kokkos::InitializationSettings &kokkos_args = {}) {
            const std::lock_guard<std::mutex> lock(mutex());
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize(kokkos_args);
            }
        }

        /**
         * @brief Process-wide mutex serializing the
         * Kokkos::is_initialized()/Kokkos::initialize() check-then-act
         * above, and the analogous is_exit_reg_ check in the destructor,
         * since both guard process-wide singleton state.
         *
         * @return Reference to the shared mutex instance.
         */
        static std::mutex &mutex() {
            static std::mutex m;
            return m;
        }
    };
    // kokkos_init_guard_ MUST be declared (and therefore constructed)
    // before exec_ below: member initialization always runs in declaration
    // order regardless of the order written in a constructor's initializer
    // list, and default-constructing exec_ (e.g. Kokkos::HIP{}) requires
    // Kokkos::initialize() to have already run.
    KokkosInitGuard kokkos_init_guard_;
    // exec_ MUST be declared (and therefore destroyed) after data_ is
    // destroyed: C++ destroys members in REVERSE declaration order, and
    // data_'s View must be freed while exec_ (and the stream/device it may
    // wrap) is still alive.  Declaring exec_ before data_ here means exec_ is
    // constructed first / destroyed last, which is the order we need.
    KokkosExecSpace exec_;
    std::unique_ptr<KokkosVector> data_;
    inline static bool is_exit_reg_ = false;
};

}; // namespace Pennylane::LightningKokkos
