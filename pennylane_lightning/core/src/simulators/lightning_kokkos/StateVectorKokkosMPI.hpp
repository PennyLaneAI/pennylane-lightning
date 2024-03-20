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
class StateVectorKokkosMPI final : public StateVectorKokkos<PrecisionT> {

  private:
    using BaseType = StateVectorKokkos<PrecisionT>;

  public:
    using ComplexT = Kokkos::complex<PrecisionT>;
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
        Kokkos::View<const std::size_t *, Kokkos::HostSpace,
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

    StateVectorKokkosMPI() = delete;
    StateVectorKokkosMPI(std::size_t num_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{0, kokkos_args, false} {
printf("SVKMPI::l125\n");        // Init MPI
printf("SVKMPI::l126\n");        int status = 0;
printf("SVKMPI::l127\n");        MPI_Initialized(&status);
printf("SVKMPI::l128\n");        if (!status) {
printf("SVKMPI::l129\n");            PL_MPI_IS_SUCCESS(MPI_Init(nullptr, nullptr));
printf("SVKMPI::l130\n");        }
printf("SVKMPI::l131\n");        communicator_ = MPI_COMM_WORLD;
printf("SVKMPI::l132\n");        // Init Kokkos
printf("SVKMPI::l133\n");        {
printf("SVKMPI::l134\n");            const std::lock_guard<std::mutex> lock(init_mutex_);
printf("SVKMPI::l135\n");            if (!Kokkos::is_initialized()) {
printf("SVKMPI::l136\n");                Kokkos::initialize(kokkos_args);
printf("SVKMPI::l137\n");            }
printf("SVKMPI::l138\n");        }
printf("SVKMPI::l139\n");        // Init attrs
printf("SVKMPI::l140\n");        num_qubits_ = num_qubits;
printf("SVKMPI::l141\n");        if (num_qubits > 0) {
printf("SVKMPI::l142\n");            data_ = std::make_unique<KokkosVector>("data_", get_blk_size());
printf("SVKMPI::l144\n");            recvbuf_ =
                std::make_unique<KokkosVector>("recvbuf_", get_blk_size());
printf("SVKMPI::l145\n");            sendbuf_ =
                std::make_unique<KokkosVector>("sendbuf_", get_blk_size());
printf("SVKMPI::l147\n");            setBasisState(0U);
        }
printf("SVKMPI::l149\n");
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

    std::size_t get_blk_size() { return exp2(num_qubits_ - get_mpi_size()); }

    std::pair<std::size_t, std::size_t>
    global_2_local_index(const std::size_t index) {
        auto blk = get_blk_size();
        return std::pair<std::size_t, std::size_t>{index / blk, index % blk};
    }

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() {
        Kokkos::deep_copy(BaseType::getView(), ComplexT{0.0, 0.0});
    }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const std::size_t global_index) {
printf("SVKMPI::line186\n");KokkosVector sv_view =
            BaseType::getView(); // circumvent error capturing this with
                                 // KOKKOS_LAMBDA
printf("SVKMPI::line189\n");        auto index = global_2_local_index(global_index);
printf("SVKMPI::line190\n");        auto rank = get_mpi_rank();
printf("SVKMPI::line191\n");        Kokkos::parallel_for(
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
            BaseType::getView(); // circumvent error capturing this with
                                 // KOKKOS_LAMBDA
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
    StateVectorKokkosMPI(ComplexT *hostdata_, std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        BaseType::HostToDevice(hostdata_, length);
    }

    StateVectorKokkosMPI(std::complex<PrecisionT> *hostdata_,
                         std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        BaseType::HostToDevice(reinterpret_cast<ComplexT *>(hostdata_), length);
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
        BaseType::HostToDevice(hostdata_copy.data(), length);
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
        BaseType::DeviceToDevice(other.getView());
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     *
     * @param other Another state vector
     */
    ~StateVectorKokkosMPI() {
        int initflag;
        int finflag;
        PL_MPI_IS_SUCCESS(MPI_Initialized(&initflag));
        PL_MPI_IS_SUCCESS(MPI_Finalized(&finflag));
        if (initflag && !finflag) {
            // finalize Kokkos first
            BaseType::StateVectorKokkos::~StateVectorKokkos();
            PL_MPI_IS_SUCCESS(MPI_Finalize());
        }
    }

  private:
    std::size_t num_qubits_;
    std::mutex init_mutex_;
    std::unique_ptr<KokkosVector> data_;
    std::unique_ptr<KokkosVector> recvbuf_;
    std::unique_ptr<KokkosVector> sendbuf_;
    inline static bool is_exit_reg_ = false;
    MPI_Comm communicator_;
};

}; // namespace Pennylane::LightningKokkos
