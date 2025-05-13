// Copyright 2025 Xanadu Quantum Technologies Inc.

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
 * @file StateVectorKokkosMPI.hpp
 */

#pragma once
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>

#include "BitUtil.hpp" // isPerfectPowerOf2
#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateFunctors.hpp"
#include "GateOperation.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorKokkos.hpp"
#include "Util.hpp"
#include "UtilKokkos.hpp"

#include "CPUMemoryModel.hpp"
// #include <roctracer/roctx.h>

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

template <class T> MPI_Datatype getMPIType() {
    PL_ABORT("No corresponding MPI type.");
}
template <> MPI_Datatype getMPIType<float>() { return MPI_FLOAT; }
template <> MPI_Datatype getMPIType<double>() { return MPI_DOUBLE; }
template <> MPI_Datatype getMPIType<Kokkos::complex<float>>() {
    return MPI_C_FLOAT_COMPLEX;
}
template <> MPI_Datatype getMPIType<Kokkos::complex<double>>() {
    return MPI_C_DOUBLE_COMPLEX;
}

} // namespace
/// @endcond

namespace Pennylane::LightningKokkos {
/**
 * @brief  Kokkos state vector class
 *
 * @tparam fp_t Floating-point precision type.
 */
template <class fp_t = double>
class StateVectorKokkosMPI final
    : public StateVectorBase<fp_t, StateVectorKokkosMPI<fp_t>> {
  private:
    using BaseType = StateVectorBase<fp_t, StateVectorKokkosMPI<fp_t>>;

  public:
    using PrecisionT = fp_t;
    using SVK = StateVectorKokkos<PrecisionT>;
    using ComplexT = typename SVK::ComplexT;
    using CFP_t = typename SVK::CFP_t;
    using KokkosVector = typename SVK::KokkosVector;
    using UnmanagedComplexHostView = typename SVK::UnmanagedComplexHostView;
    using UnmanagedConstComplexHostView =
        typename SVK::UnmanagedConstComplexHostView;
    using KokkosSizeTVector = typename SVK::KokkosSizeTVector;
    using UnmanagedSizeTHostView = typename SVK::UnmanagedSizeTHostView;
    using UnmanagedConstSizeTHostView =
        typename SVK::UnmanagedConstSizeTHostView;
    using UnmanagedPrecisionHostView = typename SVK::UnmanagedPrecisionHostView;
    using KokkosExecSpace = typename SVK::KokkosExecSpace;
    using HostExecSpace = typename SVK::HostExecSpace;

    StateVectorKokkosMPI() = delete;
    StateVectorKokkosMPI(std::size_t num_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : BaseType{num_qubits} {

        // TODO: move MPI initialization to Python
        int status = 0;
        MPI_Initialized(&status);
        if (!status) {
            PL_MPI_IS_SUCCESS(MPI_Init(nullptr, nullptr));
        }
        communicator_ = communicator;
        Kokkos::InitializationSettings settings = kokkos_args;
        num_qubits_ = num_qubits;

        // TODO: FIX! with srun, each rank sees GPU ID 0
        // Or maybe we don't need to set it
        // settings.set_device_id(getMPIRank());
        settings.set_device_id(0);
        global_wires_.resize(log2(
            static_cast<std::size_t>(getMPISize()))); // set to constructor line
        local_wires_.resize(getNumLocalWires());
        mpi_rank_to_global_index_map_.resize(getMPISize());

        resetIndices();

        if (num_qubits > 0) {
            sv_ = std::make_unique<SVK>(getNumLocalWires(), settings);
            setBasisState(0U);
        }
        // std::cout << "Initialized LK_MPI SV!" << std::endl;
    };

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    template <class complex>
    StateVectorKokkosMPI(complex *hostdata_, const std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : StateVectorKokkosMPI(log2(length), kokkos_args, communicator) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        const std::size_t blk{getLocalBlockSize()};
        const std::size_t offset{blk * getMPIRank()};
        (*sv_).HostToDevice(reinterpret_cast<ComplexT *>(hostdata_ + offset),
                            blk);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkosMPI(const ComplexT *hostdata_, const std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : StateVectorKokkosMPI(log2(length), kokkos_args, communicator) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        const std::size_t blk{getLocalBlockSize()};
        const std::size_t offset{blk * getMPIRank()};
        std::vector<ComplexT> hostdata_copy(hostdata_ + offset,
                                            hostdata_ + offset + blk);
        (*sv_).HostToDevice(hostdata_copy.data(), hostdata_copy.size());
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    template <class complex>
    StateVectorKokkosMPI(std::vector<complex> hostdata_,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : StateVectorKokkosMPI(hostdata_.data(), hostdata_.size(), kokkos_args,
                               communicator) {}

    /******************
    MPI-related methods
    ******************/

    /**
     * @brief  Returns the MPI-process rank.
     */
    int getMPIRank() {
        int rank;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communicator_, &rank));
        return rank;
    }

    /**
     * @brief  Returns the number of MPI processes.
     */
    int getMPISize() {
        int size;
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communicator_, &size));
        return size;
    }

    /**
     * @brief  Calls all barriers.
     */
    void barrier() {
        Kokkos::fence();
        mpi_barrier();
    }

    /**
     * @brief  Calls MPI_Barrier.
     */
    void mpi_barrier() { PL_MPI_IS_SUCCESS(MPI_Barrier(communicator_)); }

    template <typename T> T allReduceSum(const T &data) const {
        T sum;
        MPI_Allreduce(&data, &sum, 1, getMPIType<T>(), MPI_SUM, communicator_);
        return sum;
    }

    void mpi_sendrecv(const std::size_t send_rank, const std::size_t recv_rank,
                      const std::size_t size, const int tag) {
        PL_MPI_IS_SUCCESS(MPI_Sendrecv(
            (*sendbuf_).data(), size, getMPIType<ComplexT>(), send_rank, tag,
            (*recvbuf_).data(), size, getMPIType<ComplexT>(), recv_rank, tag,
            communicator_, MPI_STATUS_IGNORE));
    }

    void allocateBuffers() {
        PL_ABORT_IF_NOT(
            getNumLocalWires() > 0,
            "State vector must be initialized before allocating buffers.");
        if (!sendbuf_) {
            sendbuf_ = std::make_shared<KokkosVector>(
                "sendbuf_", exp2(getNumLocalWires() - 1));
        }
        if (!recvbuf_) {
            recvbuf_ = std::make_shared<KokkosVector>(
                "recvbuf_", exp2(getNumLocalWires() - 1));
        }
    }
    /********************
    Wires-related methods
    ********************/

    /**
     * @brief  Returns the MPI-distribution block size, or the size of the local
     * state vector data.
     */
    std::size_t getLocalBlockSize() const { return exp2(getNumLocalWires()); }

    template <typename T>
    bool isElementInVector(const std::vector<T> &vec, const T &element) const {
        return findElementInVector(vec, element) != vec.end();
    }

    template <typename T>
    auto findElementInVector(const std::vector<T> &vec,
                             const T &element) const {
        return std::find(vec.begin(), vec.end(), element);
    }

    template <typename T>
    std::size_t getElementIndexInVector(const std::vector<T> &vec,
                                        const T &element) const {
        auto it = findElementInVector(vec, element);
        if (it != vec.end()) {
            return std::distance(vec.begin(), it);
        } else {
            PL_ABORT("Element not in vector");
        }
    }

    std::size_t getRevWireIndex(const std::vector<std::size_t> &wires,
                                std::size_t element_index) const {
        return wires.size() - 1 - element_index;
    }

    std::size_t getRevLocalWireIndex(const std::size_t wire) const {
        return getRevWireIndex(local_wires_, getLocalWireIndex(wire));
    }

    std::size_t getRevGlobalWireIndex(const std::size_t wire) const {
        return getRevWireIndex(global_wires_, getGlobalWireIndex(wire));
    }
    /**
     * @brief  Returns the number of global wires.
     */
    std::size_t getNumGlobalWires() const { return global_wires_.size(); }

    /**
     * @brief  Returns the number of local wires.
     */
    std::size_t getNumLocalWires() const {
        return num_qubits_ - getNumGlobalWires();
    }

    SVK &getLocalSV() { return *sv_; }

    const std::vector<std::size_t> &getMPIRankToGlobalIndexMap() const {
        return mpi_rank_to_global_index_map_;
    }

    const std::vector<std::size_t> &getGlobalWires() const { return global_wires_; }
    const std::vector<std::size_t> &getLocalWires() const { return local_wires_; }

    std::vector<std::size_t>
    getLocalWireIndices(const std::vector<std::size_t> &wires) const {
        std::vector<std::size_t> local_wires_indices;
        for (const auto &wire : wires) {
            local_wires_indices.push_back(getLocalWireIndex(wire));
        }
        return local_wires_indices;
    }

    size_t getLocalWireIndex(const std::size_t wire) const {
        return getElementIndexInVector(local_wires_, wire);
    }

    std::vector<std::size_t>
    getGlobalWiresIndices(const std::vector<std::size_t> &wires) const {
        std::vector<std::size_t> global_wires_indices;
        for (const auto &wire : wires) {
            global_wires_indices.push_back(getGlobalWireIndex(wire));
        }
        return global_wires_indices;
    }

    size_t getGlobalWireIndex(const std::size_t wire) const {
        return getElementIndexInVector(global_wires_, wire);
    }

    std::vector<std::size_t>
    findGlobalWires(const std::vector<std::size_t> &wires) const {
        std::vector<std::size_t> global_wires;
        for (const auto &wire : wires) {
            if (isWiresGlobal({wire})) {
                global_wires.push_back(wire);
            }
        }
        return global_wires;
    }

    /**
     * @brief  Converts a global state vector index to a local one.
     *
     * @param index Global index.
     */
    std::pair<std::size_t, std::size_t>
    global2localIndex(const std::size_t index) const {
        auto blk = getLocalBlockSize();
        return std::pair<std::size_t, std::size_t>{index / blk, index % blk};
    }

    std::size_t getGlobalIndexFromMPIRank(const std::size_t mpi_rank) const {
        return mpi_rank_to_global_index_map_[mpi_rank];
    }

    std::size_t
    getMPIRankFromGlobalIndex(const std::size_t global_index) const {
        return getElementIndexInVector(mpi_rank_to_global_index_map_,
                                       global_index);
    }

    void resetIndices() {
        std::iota(global_wires_.begin(), global_wires_.end(), 0);
        std::iota(local_wires_.begin(), local_wires_.end(),
                  getNumGlobalWires());
        std::iota(mpi_rank_to_global_index_map_.begin(),
                  mpi_rank_to_global_index_map_.end(), 0);
    }

    bool isWiresLocal(const std::vector<std::size_t> &wires) const {
        return std::all_of(wires.begin(), wires.end(), [this](const auto i) {
            return isElementInVector(local_wires_, i);
        });
    }

    bool isWiresGlobal(const std::vector<std::size_t> &wires) const {
        return std::all_of(wires.begin(), wires.end(), [this](const auto i) {
            return isElementInVector(global_wires_, i);
        });
    }

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() {
        resetIndices();
        (*sv_).initZeros();
    }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(std::size_t global_index) {
        const auto index = global2localIndex(global_index);
        resetIndices();
        const auto rank = static_cast<std::size_t>(getMPIRank());
        if (index.first == rank) {
            (*sv_).setBasisState(index.second);
        } else {
            (*sv_).initZeros();
        }
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
        resetIndices();
        const std::size_t blk{getLocalBlockSize()};
        const std::size_t offset{blk * getMPIRank()};
        initZeros();
        std::vector<std::size_t> d_indices(blk);
        std::vector<ComplexT> d_values(blk);
        std::copy(indices.data() + offset, indices.data() + offset + blk,
                  d_indices.begin());
        std::copy(values.data() + offset, values.data() + offset + blk,
                  d_values.begin());
        (*sv_).setStateVector(d_indices, d_values);
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
        PL_ABORT_IF(wires.size() != (getNumGlobalWires() + getNumLocalWires()),
                    "Setting sub-statevector not implemented yet.");
        PL_ABORT("Not yet implemented.");
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param hostdata_ Host array for state vector
     * @param length Length of host array (must be power of 2)
     * @param kokkos_args Arguments for Kokkos initialization
     */
    /* StateVectorKokkos(ComplexT *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        HostToDevice(hostdata_, length);
    } */

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param hostdata_ Host vector for state vector
     * @param length Length of host array (must be power of 2)
     * @param kokkos_args Arguments for Kokkos initialization
     */
    /* StateVectorKokkos(std::complex<PrecisionT> *hostdata_, std::size_t
    length, const Kokkos::InitializationSettings &kokkos_args = {}) :
    StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        HostToDevice(reinterpret_cast<ComplexT *>(hostdata_), length);
    } */

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param hostdata_ Host array for state vector
     * @param length Length of host array (must be power of 2)
     * @param kokkos_args Arguments for Kokkos initialization
     */
    /* StateVectorKokkos(const ComplexT *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        std::vector<ComplexT> hostdata_copy(hostdata_, hostdata_ + length);
        HostToDevice(hostdata_copy.data(), length);
    } */

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param hostdata_ Host vector for state vector
     * @param kokkos_args Arguments for Kokkos initialization
     */
    /* StateVectorKokkos(std::vector<ComplexT> hostdata_,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(hostdata_.data(), hostdata_.size(), kokkos_args) {}
     */

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkosMPI(const StateVectorKokkosMPI &other,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : StateVectorKokkosMPI(other.getNumQubits(), kokkos_args,
                               communicator) {
        (*sv_).DeviceToDevice(other.getView());
        global_wires_ = other.global_wires_;
        local_wires_ = other.local_wires_;
        mpi_rank_to_global_index_map_ = other.mpi_rank_to_global_index_map_;
        sendbuf_ = other.sendbuf_;
        recvbuf_ = other.recvbuf_;
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     */

    ~StateVectorKokkosMPI() {}

    std::vector<std::size_t>
    localWiresSubsetToSwap(const std::vector<std::size_t> &global_wires,
                           const std::vector<std::size_t> &wires) {
        PL_ABORT_IF(global_wires.size() > local_wires_.size(),
                    "global_wires must be smaller than local_wires.");
        std::vector<std::size_t> local_wires;
        int j = 0;

        while (local_wires.size() != global_wires.size()) {
            if (!isElementInVector(wires, local_wires_[j])) {
                local_wires.push_back(local_wires_[j]);
            }
            j++;
        }
        // TODO: FIX ME with better algorithm based on memory pattern
        return local_wires;
    }

    /**
     * @brief
     *
     * @param global_wires_to_swap
     * @param local_wires_to_swap
     *
     * Example:
     * For Global|Local wires = 0 | 1 2 and swapping global_wires {0} and
     * local_wires {2}, the elements {0|01, 0|11} in global_index 0 will be swap
     * with elements {1|00, 1|10} in global_index 1. This will be done when
     * batch_index = 1:
     * - receiving_global_index = local_global_index ^ batch_index
     * - the (relevant, i.e. to be swapped) local_wire = (relevant) local_wire ^
     * batch_index (wire 2 in example)
     * - the (irrelevant, not swapped) local_wires are looped over and copied
     * to/from the buffers to send/recv
     *
     * Note: the batch index loops over all the global wires (including those
     * that are not swapped). W e then check whether the specific batch
     * number requires communication
     *
     * For each batch, a single pairwise MPI_Sendrecv is performed.
     * The number of batches = 2^(num_swapping_wires) - 1 .
     * The number of elements sent in each batch is 1/2^(num_swapping_wires) *
     * size_of_subSV.
     *
     */
    void
    swapGlobalLocalWires(const std::vector<std::size_t> &global_wires_to_swap,
                         const std::vector<std::size_t> &local_wires_to_swap) {
        PL_ABORT_IF_NOT(global_wires_to_swap.size() ==
                            local_wires_to_swap.size(),
                        "global_wires_to_swap and local_wires_to_swap must "
                        "have equal dimensions.");
        PL_ABORT_IF_NOT(isWiresGlobal(global_wires_to_swap),
                        "global_wires_to_swap must be global wires.");
        PL_ABORT_IF_NOT(isWiresLocal(local_wires_to_swap),
                        "local_wires_to_swap must be local wires.");
        // std::sort(global_wires_to_swap.begin(), global_wires_to_swap.end());
        // std::sort(local_wires_to_swap.begin(), local_wires_to_swap.end());

        // #ifdef LKMPI_DEBUG
        // roctxMark("ROCTX-MARK: Start of swapGlobalLocalWires");
        //   A little debug message:
        // if (getMPIRank() == 0) {
        //    std::cout << "Swapping global wires: ";
        //    for (const auto &wire : global_wires_to_swap) {
        //        std::cout << wire << " ";
        //    }
        //    std::cout << "with local wires: ";
        //    for (const auto &wire : local_wires_to_swap) {
        //        std::cout << wire << " ";
        //    }
        //    std::cout << std::endl;
        //}
        // #endif

        std::vector<std::size_t> rev_global_wires_index_to_swap;
        std::vector<std::size_t> rev_local_wires_index_to_swap;
        std::vector<std::size_t> rev_local_wires_index_not_swapping;
        for (std::size_t i = 0; i < getNumLocalWires(); i++) {
            if (!isElementInVector(local_wires_to_swap, local_wires_[i])) {
                rev_local_wires_index_not_swapping.push_back(
                    getRevLocalWireIndex(local_wires_[i]));
            }
        }
        std::sort(rev_local_wires_index_not_swapping.begin(),
                  rev_local_wires_index_not_swapping.end());
        for (std::size_t i = 0; i < local_wires_to_swap.size(); i++) {
            rev_local_wires_index_to_swap.push_back(
                getRevLocalWireIndex(local_wires_to_swap[i]));
        }
        for (std::size_t i = 0; i < global_wires_to_swap.size(); i++) {
            rev_global_wires_index_to_swap.push_back(
                getRevGlobalWireIndex(global_wires_to_swap[i]));
        }

        std::size_t global_index = getGlobalIndexFromMPIRank(getMPIRank());
        allocateBuffers();
        for (std::size_t batch_index = 1;
             batch_index < exp2(getNumGlobalWires()); batch_index++) {
            // We loop over all the global indices (ranks) and check if the
            // batch index actually requires swapping

            bool send = true;
            for (std::size_t digits = 0; digits < global_wires_.size();
                 digits++) {
                bool is_global_wire_in_swap = isElementInVector(
                    global_wires_to_swap,
                    global_wires_[getNumGlobalWires() - 1 - digits]);
                bool batch_index_digit = (batch_index >> digits) & 1;
                send = send && (!batch_index_digit || is_global_wire_in_swap);
            }
            if (send) {
#ifdef LKMPI_DEBUG
                // A little debug message:
                std::cout << "I am rank " << getMPIRank()
                          << " batch index = " << batch_index << std::endl;
#endif
                std::size_t swap_wire_mask = 0;
                for (std::size_t i = 0; i < local_wires_to_swap.size(); i++) {
                    swap_wire_mask |= ((((batch_index ^ global_index) >>
                                         rev_global_wires_index_to_swap[i]) &
                                        1)
                                       << rev_local_wires_index_to_swap[i]);
                }

#ifdef LKMPI_DEBUG
                // A little debug message:
                std::cout << "I am rank " << getMPIRank()
                          << " and swap_wire_mask = " << swap_wire_mask
                          << std::endl;
#endif

                // These are defined since on AMD compiler it's more strict what
                // host functions can be included in the KOKKOS_LAMBDA - e.g.
                // dereferencing, size are all not allowed
                const std::size_t not_swapping_local_wire_size =
                    rev_local_wires_index_not_swapping.size();
                auto rev_local_wires_index_not_swapping_view =
                    vector2view(rev_local_wires_index_not_swapping);

                auto sendbuf_view = (*sendbuf_);
                auto recvbuf_view = (*recvbuf_);
                auto sv_view = (*sv_).getView();
                std::size_t send_size =
                    exp2((getNumLocalWires() - local_wires_to_swap.size()));

                // roctxMark("ROCTX-MARK: Start of copy_sendbuf");
                Kokkos::parallel_for(
                    "copy_sendbuf", send_size,
                    KOKKOS_LAMBDA(std::size_t buffer_index) {
                        std::size_t SV_index = swap_wire_mask;
                        for (std::size_t i = 0;
                             i < not_swapping_local_wire_size; i++) {
                            SV_index |=
                                (((buffer_index >> i) & 1)
                                 << rev_local_wires_index_not_swapping_view(i));
                        }
                        sendbuf_view(buffer_index) = sv_view(SV_index);
                    });
                Kokkos::fence();

                // roctxMark("ROCTX-MARK: End of copy_sendbuf");
                std::size_t other_global_index = batch_index ^ global_index;
                std::size_t other_mpi_rank =
                    getMPIRankFromGlobalIndex(other_global_index);

#ifdef LKMPI_DEBUG
                // A little debug message:
                std::cout << "I am rank " << getMPIRank()
                          << " and I am sending to rank " << other_mpi_rank
                          << " with tag " << batch_index
                          << " and this number of elements " << send_size
                          << std::endl;
#endif

                // roctxMark("ROCTX-MARK: Start of sendrecv");
                mpi_sendrecv(other_mpi_rank, other_mpi_rank, send_size,
                             batch_index);
                // roctxMark("ROCTX-MARK: End of sendrecv");

                // roctxMark("ROCTX-MARK: Start of copy_recvbuf");
                Kokkos::parallel_for(
                    "copy_recvbuf", send_size,
                    KOKKOS_LAMBDA(std::size_t buffer_index) {
                        std::size_t SV_index = swap_wire_mask;

                        for (std::size_t i = 0;
                             i < not_swapping_local_wire_size; i++) {
                            SV_index |=
                                (((buffer_index >> i) & 1)
                                 << rev_local_wires_index_not_swapping_view(i));
                        }

                        sv_view(SV_index) = recvbuf_view(buffer_index);
                    });
                Kokkos::fence();
                // roctxMark("ROCTX-MARK: End of copy_recvbuf");
            }
        }

        // Swap global and local wires labels
        for (size_t i = 0; i < global_wires_to_swap.size(); ++i) {
            std::size_t global_wire_idx =
                getGlobalWireIndex(global_wires_to_swap[i]);
            std::size_t local_wire_idx =
                getLocalWireIndex(local_wires_to_swap[i]);
            std::swap(global_wires_[global_wire_idx],
                      local_wires_[local_wire_idx]);
        }
    }

    void matchWires(const StateVectorKokkosMPI &other_sv) {
        // Swap global-local wires if necessary

        if (global_wires_ != other_sv.global_wires_) {
            std::vector<std::size_t> global_wires_to_swap;
            std::vector<std::size_t> local_wires_to_swap;
            for (std::size_t i = 0; i < global_wires_.size(); ++i) {
                if (!isElementInVector(other_sv.global_wires_,
                                       global_wires_[i])) {
                    global_wires_to_swap.push_back(global_wires_[i]);
                }
            }
            for (std::size_t i = 0; i < local_wires_.size(); ++i) {
                if (!isElementInVector(other_sv.local_wires_,
                                       local_wires_[i])) {
                    local_wires_to_swap.push_back(local_wires_[i]);
                }
            }
            swapGlobalLocalWires(global_wires_to_swap, local_wires_to_swap);
        }
        // Swap global-global wires if necessary
        if ((global_wires_ != other_sv.global_wires_) ||
            (mpi_rank_to_global_index_map_ !=
             other_sv.mpi_rank_to_global_index_map_)) {
            matchGlobalWiresAndIndex(other_sv);
        }

        // Swap local-local wires if necessary
        if (local_wires_ != other_sv.local_wires_) {
            matchLocalWires(other_sv.local_wires_);
        }
    }

    void matchLocalWires(const std::vector<std::size_t> &other_local_wires) {
        for (std::size_t i = 0; i < local_wires_.size(); ++i) {
            if (local_wires_[i] != other_local_wires[i]) {
                applyOperation("SWAP", {local_wires_[i], other_local_wires[i]},
                               false);
                local_wires_[getElementIndexInVector(
                    local_wires_, other_local_wires[i])] = local_wires_[i];
                local_wires_[i] = other_local_wires[i];
            }
        }
        PL_ABORT_IF(local_wires_ != other_local_wires,
                    "Local wires do not match after swap.");
    }

    void matchGlobalWiresAndIndex(const StateVectorKokkosMPI &other_sv) {
        const auto other_global_wires = other_sv.getGlobalWires();
        const auto other_mpi_rank_to_global_index_map =
            other_sv.getMPIRankToGlobalIndexMap();
        matchGlobalWiresAndIndex(other_global_wires, other_mpi_rank_to_global_index_map);
    }

    void matchGlobalWiresAndIndex(const std::vector<std::size_t> &global_wires_target,
                                  const std::vector<std::size_t> &mpi_rank_to_global_index_map_target) {
        std::size_t my_global_index = getGlobalIndexFromMPIRank(getMPIRank());
        std::size_t dest_global_index = 0;
        for (std::size_t i = 0; i < global_wires_.size(); ++i) {
            dest_global_index |=
                (((my_global_index >> i) & 1)
                 << (global_wires_.size() - 1 -
                     getElementIndexInVector(
                         global_wires_target,
                         global_wires_[global_wires_.size() - i - 1])));
        }

        std::size_t dest_mpi_rank = getElementIndexInVector(mpi_rank_to_global_index_map_target,
                                       dest_global_index);

        allocateBuffers();
        std::size_t send_size = exp2(getNumLocalWires() - 1);
        auto sendbuf_view = (*sendbuf_);
        auto recvbuf_view = (*recvbuf_);
        auto sv_view = (*sv_).getView();
        // COPY to buffer
        Kokkos::parallel_for(
            "copy_sendbuf", send_size, KOKKOS_LAMBDA(std::size_t buffer_index) {
                sendbuf_view(buffer_index) = sv_view(buffer_index);
            });
        Kokkos::fence();
        // SENDRECV
        mpi_sendrecv(dest_mpi_rank, dest_mpi_rank, send_size, 0);
        // COPY FROM BUFFER

        Kokkos::parallel_for(
            "copy_recvbuf", send_size, KOKKOS_LAMBDA(std::size_t buffer_index) {
                sv_view(buffer_index) = recvbuf_view(buffer_index);
            });

        // Repeat

        // COPY to buffer
        Kokkos::parallel_for(
            "copy_sendbuf", send_size, KOKKOS_LAMBDA(std::size_t buffer_index) {
                sendbuf_view(buffer_index) = sv_view(buffer_index + send_size);
            });
        Kokkos::fence();
        // SENDRECV
        mpi_sendrecv(dest_mpi_rank, dest_mpi_rank, send_size, 0);
        // COPY FROM BUFFER

        Kokkos::parallel_for(
            "copy_recvbuf", send_size, KOKKOS_LAMBDA(std::size_t buffer_index) {
                sv_view(buffer_index + send_size) = recvbuf_view(buffer_index);
            });

        // copy target index map and global wires to local index map and global
        // wires

        global_wires_ = global_wires_target;
        mpi_rank_to_global_index_map_ = mpi_rank_to_global_index_map_target;
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
            return;
        }

        if (opName == "GlobalPhase"){
            (*sv_).applyOperation("GlobalPhase", {}, inverse, {params});
            return;
        }

        if (isWiresGlobal(wires)) {

            if (opName == "PauliX") {
                std::size_t rev_distance = getRevGlobalWireIndex(wires[0]);
                for (auto &global_index : mpi_rank_to_global_index_map_) {
                    global_index = global_index ^ (1U << rev_distance);
                }
                return;
            } else if (opName == "PauliY") {
                std::size_t rev_distance = getRevGlobalWireIndex(wires[0]);
                std::size_t global_index =
                    getGlobalIndexFromMPIRank(getMPIRank());
                fp_t phase = ((global_index >> rev_distance) & 1)
                                 ? (M_PI_2)
                                 : (-1.0) * M_PI_2;
                (*sv_).applyOperation("GlobalPhase", {}, false, {phase});
                for (auto &global_index : mpi_rank_to_global_index_map_) {
                    global_index = global_index ^ (1U << rev_distance);
                }

                return;
            } else if (opName == "PauliZ") {
                std::size_t rev_distance = getRevGlobalWireIndex(wires[0]);
                std::size_t global_index =
                    getGlobalIndexFromMPIRank(getMPIRank());
                if ((global_index >> rev_distance) & 1) {
                    (*sv_).applyOperation("GlobalPhase", {}, false, {M_PI});
                }
                return;
            } else if (opName == "CNOT") {

                std::size_t rev_distance_0 = getRevGlobalWireIndex(wires[0]);
                std::size_t rev_distance_1 = getRevGlobalWireIndex(wires[1]);
                for (auto &global_index : mpi_rank_to_global_index_map_) {
                    global_index = ((global_index >> rev_distance_0) & 1)
                                       ? global_index ^ (1U << rev_distance_1)
                                       : global_index;
                }
                return;
            }
            // TODO: Add similar for CY, CZ, SWAP, PhaseShift, GlobalPhase
            // etc.
        }

        if (!isWiresLocal(wires)) {
            auto global_wires_to_swap = findGlobalWires(wires);
            auto local_wires_to_swap =
                localWiresSubsetToSwap(global_wires_to_swap, wires);
            // if (getMPIRank() == 0) {
            //     std::cout << "global_wires =";
            //     for (const auto &wire : global_wires_) {
            //         std::cout << wire << " ";
            //     }
            //     std::cout << "local_wires = ";
            //     for (const auto &wire : local_wires_) {
            //         std::cout << wire << " ";
            //     }
            //     std::cout << "global_wires_to_swap = ";
            //     for (const auto &wire : global_wires_to_swap) {
            //         std::cout << wire << " ";
            //     }
            //     std::cout << "local_wires_to_swap = ";
            //     for (const auto &wire : local_wires_to_swap) {
            //         std::cout << wire << " ";
            //     }
            // }
            swapGlobalLocalWires(global_wires_to_swap, local_wires_to_swap);
        }

#ifdef LKMPI_DEBUG
        if (getMPIRank() == 0) {
            std::cout << "I am rank " << getMPIRank()
                      << " and I am applying the operation " << opName
                      << " to the wires " << wires[0] << wires[1]
                      << "and converted to " << getLocalWireIndices(wires)[0]
                      << getLocalWireIndices(wires)[1] << std::endl;
        }
#endif
        (*sv_).applyOperation(opName, getLocalWireIndices(wires), inverse,
                              params, gate_matrix);
    }

    /**
     * @brief Apply a PauliRot gate to the state-vector.
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Rotation angle.
     * @param word A Pauli word (e.g. "XYYX").
     */
    /* void applyPauliRot(const std::vector<std::size_t> &wires, bool inverse,
                       const std::vector<PrecisionT> &params,
                       const std::string &word) {
        PL_ABORT_IF_NOT(wires.size() == word.size(),
                        "wires and word have incompatible dimensions.");
        Pennylane::LightningKokkos::Functors::applyPauliRot<KokkosExecSpace,
                                                            PrecisionT>(
            getView(), this->getNumQubits(), wires, inverse, params[0], word);
    } */

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

        if (opName == "Identity") {
            // No op
            return;
        }

        // Swap target wires to all local
        if (!isWiresLocal(wires)) {
            auto global_wires_to_swap = findGlobalWires(wires);
            auto local_wires_to_swap =
                localWiresSubsetToSwap(global_wires_to_swap, wires);
            swapGlobalLocalWires(global_wires_to_swap, local_wires_to_swap);
            barrier();
        }

        std::vector<std::size_t> global_control_wires;
        std::vector<bool> global_control_values;
        std::vector<std::size_t> local_control_wires;
        std::vector<bool> local_control_values;

        for (std::size_t i = 0; i < controlled_wires.size(); i++) {
            if (isWiresLocal({controlled_wires[i]})) {
                local_control_wires.push_back(controlled_wires[i]);
                local_control_values.push_back(controlled_values[i]);
            } else {
                global_control_wires.push_back(controlled_wires[i]);
                global_control_values.push_back(controlled_values[i]);
            }
        }

#ifdef LKMPI_DEBUG
        for (auto wire : global_control_wires) {
            std::cout << "I am rank " << getMPIRank()
                      << " applying to the global control wire " << wire
                      << std::endl;
        }
#endif

        std::size_t global_index = getGlobalIndexFromMPIRank(getMPIRank());
        bool operate = true;
        for (std::size_t i = 0; i < global_control_wires.size(); i++) {
            operate =
                operate && (((global_index >>
                              getRevGlobalWireIndex(global_control_wires[i])) &
                             1) == global_control_values[i]);
        }
        if (operate) {
#ifdef LKMPI_DEBUG
            // A little debug message:
            for (auto lcw : local_control_wires) {
                std::cout << "I am rank " << getMPIRank()
                          << " applying to the local control wire " << lcw
                          << "and local index = " << getLocalWireIndex(lcw)
                          << std::endl;
            }
            for (auto w : wires) {
                std::cout << "I am rank " << getMPIRank()
                          << " applying to the target wire " << w
                          << " and local index = " << getLocalWireIndex(w)
                          << std::endl;
            }
#endif

            (*sv_).applyOperation(
                opName, getLocalWireIndices(local_control_wires),
                local_control_values, getLocalWireIndices(wires), inverse,
                params, gate_matrix);
        }
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
        size_t n = static_cast<std::size_t>(1U) << wires.size();
        const std::vector<ComplexT> matrix_(matrix, matrix + n * n);
        applyOperation("Matrix", wires, inverse, {}, matrix_);
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
        applyOperation("Matrix", wires, inverse, {}, matrix);
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
        size_t n = static_cast<std::size_t>(1U) << wires.size();
        const std::vector<ComplexT> matrix_(matrix, matrix + n * n);
        applyOperation("Matrix", controlled_wires, controlled_values, wires,
                       inverse, {}, matrix_);
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
        applyOperation("Matrix", controlled_wires, controlled_values, wires,
                       inverse, {}, matrix);
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

        if (!isWiresLocal(wires)) {
            auto global_wires = findGlobalWires(wires);
            auto local_wires = localWiresSubsetToSwap(global_wires, wires);
            swapGlobalLocalWires(global_wires, local_wires);
        }
        return (*sv_).applyGenerator(opName, getLocalWireIndices(wires),
                                     inverse);
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

        std::vector<std::size_t> all_wires = controlled_wires;
        all_wires.insert(all_wires.end(), wires.begin(), wires.end());
        if (!isWiresLocal(all_wires)) {
            auto global_wires = findGlobalWires(all_wires);
            auto local_wires = localWiresSubsetToSwap(global_wires, all_wires);
            swapGlobalLocalWires(global_wires, local_wires);
        }
        return (*sv_).applyControlledGenerator(
            opName, getLocalWireIndices(controlled_wires), controlled_values,
            getLocalWireIndices(wires), inverse);
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
    void collapse([[maybe_unused]] std::size_t wire,
                  [[maybe_unused]] bool branch) {
        /* KokkosVector matrix("gate_matrix", 4);
        Kokkos::parallel_for(
            matrix.size(), KOKKOS_LAMBDA(std::size_t k) {
                matrix(k) = ((k == 0 && branch == 0) || (k == 3 && branch == 1))
                                ? ComplexT{1.0, 0.0}
                                : ComplexT{0.0, 0.0};
            });
        applyMultiQubitOp(matrix, {wire}, false);
        normalize(); */
        PL_ABORT("Not implemented yet.");
    }

    /**
     * @brief Update data of the class
     *
     * @param other Kokkos View
     */
    void updateData(const KokkosVector other) { (*sv_).updateData(other); }

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

    void updateData(const StateVectorKokkosMPI<PrecisionT> &other) {
        updateData(other.getView());
        global_wires_ = other.global_wires_;
        local_wires_ = other.local_wires_;
        mpi_rank_to_global_index_map_ = other.mpi_rank_to_global_index_map_;
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
        (*sv_).HostToDevice(sv, length);
    }

    /**
     * @brief Copy data from the device space to the host space.
     *
     */
    inline void DeviceToHost(ComplexT *sv, std::size_t length) const {
        (*sv_).DeviceToHost(sv, length);
    }

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        (*sv_).DeviceToDevice(vector_to_copy);
    }

    void reorderLocalWires() {
        PL_ABORT_IF_NOT(std::all_of(local_wires_.begin(), local_wires_.end(),
                                    [this](const auto i) {
                                        return (getNumGlobalWires() <= i) &&
                                               (i < num_qubits_);
                                    }),
                        "local wires must be least significant indices. Run "
                        "reorder_global_wires first.");

        for (std::size_t i = 0; i < getNumLocalWires(); ++i) {
            std::size_t wire_i = i + getNumGlobalWires();
            if (local_wires_[i] > wire_i) {
                std::cout << "I am rank " << getMPIRank()
                          << " and I am swapping " << local_wires_[i]
                          << " with " << wire_i << std::endl;
                applyOperation("SWAP", {local_wires_[i], wire_i}, false);
            }
        }

        std::iota(local_wires_.begin(), local_wires_.end(),
                  getNumGlobalWires());
    }

    void reorderGlobalLocalWires() {
        std::vector<std::size_t> global_wires;
        std::vector<std::size_t> local_wires;
        for (const auto &wire : global_wires_) {
            if (wire >= getNumGlobalWires()) {
                global_wires.push_back(wire);
            }
        }

        for (const auto &wire : local_wires_) {
            if (wire < getNumGlobalWires()) {
                local_wires.push_back(wire);
            }
        }
        if (!global_wires.empty()) {
            swapGlobalLocalWires(global_wires, local_wires);
        }
    }

    void reorderAllWires(){
        reorderGlobalLocalWires();
        std::vector<std::size_t> global_wires_target(getNumGlobalWires());
        std::vector<std::size_t> mpi_rank_to_global_index_map_target(getMPISize());

        std::iota(global_wires_target.begin(), global_wires_target.end(), 0);  
        std::iota(mpi_rank_to_global_index_map_target.begin(),
                  mpi_rank_to_global_index_map_target.end(), 0);

        matchGlobalWiresAndIndex(global_wires_target, mpi_rank_to_global_index_map_target);
        reorderLocalWires();
    
    }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector(const int root = 0)
        -> std::vector<ComplexT> {
        reorderGlobalLocalWires();
        reorderLocalWires();
        std::vector<ComplexT> data_((getMPIRank() == root) ? this->getLength()
                                                           : 0);
        std::vector<ComplexT> local_((*sv_).getLength());
        (*sv_).DeviceToHost(local_.data(), local_.size());
        std::vector<int> recvcount(getMPISize(), local_.size());
        std::vector<int> displacements(getMPISize(), 0);

        for (std::size_t rank = 0; rank < getMPISize(); rank++) {
            for (std::size_t i = 0; i < getNumGlobalWires(); i++) {
                std::size_t temp =
                    ((getGlobalIndexFromMPIRank(rank) >>
                      (getNumGlobalWires() - 1 - i)) &
                     1)
                    << (getNumGlobalWires() - 1 - global_wires_[i]);
                displacements[rank] += temp;
            }
            displacements[rank] *= local_.size();
        }

#ifdef LKMPI_DEBUG
        // A little debug message:
        if (getMPIRank() == root) {
            for (std::size_t rank = 0; rank < getMPISize(); rank++) {
                std::cout << "Rank: " << rank
                          << ", Displacement: " << displacements[rank]
                          << ", Recvcount: " << recvcount[rank] << std::endl;
            }
        }
#endif

        PL_MPI_IS_SUCCESS(
            MPI_Gatherv(local_.data(), local_.size(), getMPIType<ComplexT>(),
                        data_.data(), recvcount.data(), displacements.data(),
                        getMPIType<ComplexT>(), root, communicator_));
        return data_;
    }

  private:
    std::size_t num_qubits_;
    std::unique_ptr<SVK> sv_;
    std::shared_ptr<KokkosVector> recvbuf_;
    std::shared_ptr<KokkosVector> sendbuf_;
    MPI_Comm communicator_;

    std::vector<std::size_t> mpi_rank_to_global_index_map_;
    std::vector<std::size_t> global_wires_;
    std::vector<std::size_t> local_wires_;
};
}; // namespace Pennylane::LightningKokkos
