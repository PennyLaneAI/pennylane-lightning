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

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "BitUtil.hpp" // isPerfectPowerOf2
#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateFunctors.hpp"
#include "GateOperation.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Functors;
using Pennylane::Gates::GateOperation;
using Pennylane::Gates::GeneratorOperation;
using Pennylane::Util::array_contains;
using Pennylane::Util::exp2;
using Pennylane::Util::isPerfectPowerOf2;
using Pennylane::Util::log2;
using Pennylane::Util::reverse_lookup;
using std::size_t;

// LCOV_EXCL_START
inline void errhandler(int errcode, const char *str) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    fprintf(stderr, "%s: %s\n", str, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
// LCOV_EXCL_STOP

#define PL_MPI_IS_SUCCESS(fn)                                                  \
    {                                                                          \
        int errcode;                                                           \
        errcode = (fn);                                                        \
        if (errcode != MPI_SUCCESS)                                            \
            errhandler(errcode, #fn);                                          \
    }

} // namespace
/// @endcond

namespace Pennylane::LightningKokkos {
/**
 * @brief  Kokkos state vector class
 *
 * @tparam PrecisionT Floating-point precision type.
 */
template <class PrecisionT = double>
class StateVectorKokkosMPI final
    : public StateVectorBase<PrecisionT, StateVectorKokkosMPI<PrecisionT>> {

  private:
    using BaseType =
        StateVectorBase<PrecisionT, StateVectorKokkosMPI<PrecisionT>>;

  public:
    using ComplexT = Kokkos::complex<PrecisionT>;
    using SVK = StateVectorKokkos<PrecisionT>;
    using KokkosVector = SVK::KokkosVector;
    // using CFP_t = ComplexT;
    // using DoubleLoopRank = Kokkos::Rank<2>;
    // using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
    // using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    // using KokkosVector = Kokkos::View<ComplexT *>;
    // using KokkosSizeTVector = Kokkos::View<size_t *>;
    // using UnmanagedComplexHostView =
    //     Kokkos::View<ComplexT *, Kokkos::HostSpace,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // using UnmanagedSizeTHostView =
    //     Kokkos::View<size_t *, Kokkos::HostSpace,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // using UnmanagedConstComplexHostView =
    //     Kokkos::View<const ComplexT *, Kokkos::HostSpace,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // using UnmanagedConstSizeTHostView =
    //     Kokkos::View<const std::size_t *, Kokkos::HostSpace,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // using UnmanagedPrecisionHostView =
    //     Kokkos::View<PrecisionT *, Kokkos::HostSpace,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // using ScratchViewComplex =
    //     Kokkos::View<ComplexT *, KokkosExecSpace::scratch_memory_space,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // using ScratchViewSizeT =
    //     Kokkos::View<size_t *, KokkosExecSpace::scratch_memory_space,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // using TeamPolicy = Kokkos::TeamPolicy<>;
    // using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

    StateVectorKokkosMPI() = delete;
    StateVectorKokkosMPI(std::size_t num_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{num_qubits} {
        // Init MPI
        int status = 0;
        MPI_Initialized(&status);
        if (!status) {
            PL_MPI_IS_SUCCESS(MPI_Init(nullptr, nullptr));
        }
        communicator_ = MPI_COMM_WORLD;
        // Init Kokkos
        // {
        //     const std::lock_guard<std::mutex> lock(init_mutex_);
        //     if (!Kokkos::is_initialized()) {
        //         Kokkos::initialize(kokkos_args);
        //     }
        // }
        // Init attrs
        num_qubits_ = num_qubits;
        if (num_qubits > 0) {
            sv_ = std::make_unique<SVK>(get_num_local_qubits(), kokkos_args);
            // recvbuf_ =
            //     std::make_unique<KokkosVector>("recvbuf_", get_blk_size());
            // sendbuf_ =
            //     std::make_unique<KokkosVector>("sendbuf_", get_blk_size());
            setBasisState(0U);
        }
    };

    std::size_t get_mpi_rank() {
        int rank;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communicator_, &rank));
        return static_cast<std::size_t>(rank);
    }
    std::size_t get_mpi_size() {
        int size;
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communicator_, &size));
        return static_cast<std::size_t>(size);
    }
    void mpi_barrier() { PL_MPI_IS_SUCCESS(MPI_Barrier(communicator_)); }

    std::size_t get_num_global_qubits() { return log2(get_mpi_size()); }
    std::size_t get_num_local_qubits() {
        return num_qubits_ - get_num_global_qubits();
    }
    std::size_t get_blk_size() { return exp2(get_num_local_qubits()); }
    std::pair<std::size_t, std::size_t>
    global_2_local_index(const std::size_t index) {
        auto blk = get_blk_size();
        return std::pair<std::size_t, std::size_t>{index / blk, index % blk};
    }

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() { Kokkos::deep_copy(getView(), ComplexT{0.0, 0.0}); }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const std::size_t global_index) {
        KokkosVector sv_view = getView(); // circumvent error capturing this
                                          // with KOKKOS_LAMBDA
        auto index = global_2_local_index(global_index);
        auto rank = get_mpi_rank();
        Kokkos::parallel_for(
            sv_view.size(), KOKKOS_LAMBDA(const std::size_t i) {
                sv_view(i) = (index.first == rank && index.second == i)
                                 ? ComplexT{1.0, 0.0}
                                 : ComplexT{0.0, 0.0};
            });
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    // void setStateVector(const std::vector<std::size_t> &indices,
    //                     const std::vector<ComplexT> &values) {
    //     initZeros();
    //     KokkosSizeTVector d_indices("d_indices", indices.size());
    //     KokkosVector d_values("d_values", values.size());
    //     Kokkos::deep_copy(d_indices, UnmanagedConstSizeTHostView(
    //                                      indices.data(), indices.size()));
    //     Kokkos::deep_copy(d_values, UnmanagedConstComplexHostView(
    //                                     values.data(), values.size()));
    //     KokkosVector sv_view =
    //         getView(); // circumvent error capturing this with
    //                              // KOKKOS_LAMBDA
    //     Kokkos::parallel_for(
    //         indices.size(), KOKKOS_LAMBDA(const std::size_t i) {
    //             sv_view(d_indices[i]) = d_values[i];
    //         });
    // }

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
    StateVectorKokkosMPI(ComplexT *hostdata_, std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        (*sv_).HostToDevice(hostdata_, length);
    }

    StateVectorKokkosMPI(std::complex<PrecisionT> *hostdata_,
                         std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        (*sv_).HostToDevice(reinterpret_cast<ComplexT *>(hostdata_), length);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkosMPI(const ComplexT *hostdata_, std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        std::vector<ComplexT> hostdata_copy(hostdata_, hostdata_ + length);
        (*sv_).HostToDevice(hostdata_copy.data(), length);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    template <class complex>
    StateVectorKokkosMPI(std::vector<complex> hostdata_,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(hostdata_.data(), hostdata_.size(),
                               kokkos_args) {}

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkosMPI(const StateVectorKokkosMPI &other,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(other.getNumQubits(), kokkos_args) {
        (*sv_).DeviceToDevice(other.getView());
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     *
     * @param other Another state vector
     */
    ~StateVectorKokkosMPI() {
        // int initflag;
        // int finflag;
        // PL_MPI_IS_SUCCESS(MPI_Initialized(&initflag));
        // PL_MPI_IS_SUCCESS(MPI_Finalized(&finflag));
        // if (initflag && !finflag) {
        //     // finalize Kokkos first
        //     PL_MPI_IS_SUCCESS(MPI_Finalize());
        // }
    }

    /**
     * @brief Get the Kokkos data of the state vector.
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getView() const -> KokkosVector & {
        return (*sv_).getView();
    }

    /**
     * @brief Get the Kokkos data of the state vector
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getView() -> KokkosVector & { return (*sv_).getView(); }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector() -> std::vector<ComplexT> {
        std::vector<ComplexT> data_((*sv_).getLength());
        (*sv_).DeviceToHost(data_.data(), data_.size());
        return data_;
    }

  private:
    std::size_t num_qubits_;
    std::unique_ptr<SVK> sv_;
    // std::unique_ptr<KokkosVector> recvbuf_;
    // std::unique_ptr<KokkosVector> sendbuf_;
    MPI_Comm communicator_;
};

}; // namespace Pennylane::LightningKokkos
