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
#include "MPIManagerKokkos.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorKokkos.hpp"
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
using Pennylane::Util::findElementInVector;
using Pennylane::Util::getElementIndexInVector;
using Pennylane::Util::getRevWireIndex;
using Pennylane::Util::isElementInVector;
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
class StateVectorKokkosMPI final
    : public StateVectorBase<fp_t, StateVectorKokkosMPI<fp_t>> {
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

    using BaseType = StateVectorBase<fp_t, StateVectorKokkosMPI<fp_t>>;

  private:
    std::unique_ptr<SVK> sv_;
    std::shared_ptr<KokkosVector> recvbuf_;
    std::shared_ptr<KokkosVector> sendbuf_;
    MPIManagerKokkos mpi_manager_;

    std::size_t num_qubits_;
    std::size_t numGlobalQubits_;
    std::size_t numLocalQubits_;
    std::vector<std::size_t> mpi_rank_to_global_index_map_;
    std::vector<std::size_t> global_wires_;
    std::vector<std::size_t> local_wires_;

  public:
    StateVectorKokkosMPI() = delete;

    /**
     * @brief Create a new state vector with mpi_manager.
     *
     * @param mpi_manager Kokkos MPIManager
     * @param num_global_qubits Number of global qubits
     * @param num_local_qubits Number of local qubits
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkosMPI(MPIManagerKokkos mpi_manager,
                         std::size_t num_global_qubits,
                         std::size_t num_local_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{num_global_qubits + num_local_qubits},
          mpi_manager_(mpi_manager),
          num_qubits_(num_global_qubits + num_local_qubits),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits) {
        Kokkos::InitializationSettings settings = kokkos_args;

        global_wires_.resize(numGlobalQubits_); // set to constructor line
        local_wires_.resize(num_local_qubits);
        mpi_rank_to_global_index_map_.resize(mpi_manager_.getSize());

        resetIndices();

        if (num_local_qubits > 0) {
            sv_ = std::make_unique<SVK>(num_local_qubits, settings);
            setBasisState(0U);
        }
        allocateBuffers();
    }

    /**
     * @brief Create a new state vector with mpi_manager.
     *
     * @param mpi_manager Kokkos MPIManager
     * @param total_num_qubits Number of qubits
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkosMPI(MPIManagerKokkos mpi_manager,
                         std::size_t total_num_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(mpi_manager, log2(mpi_manager.getSize()),
                               total_num_qubits - log2(mpi_manager.getSize()),
                               kokkos_args) {}

    /**
     * @brief Create a new state vector with MPI Comm.
     *
     * @param mpi_communicator MPI_Comm communicator
     * @param total_num_qubits Number of qubits
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkosMPI(MPI_Comm mpi_communicator,
                         std::size_t total_num_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(MPIManagerKokkos(mpi_communicator),
                               total_num_qubits, kokkos_args) {}

    /**
     * @brief Create a new state vector with default MPI_COMM_WORLD
     *
     * @param total_num_qubits Number of qubits
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkosMPI(std::size_t total_num_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(MPIManagerKokkos(MPI_COMM_WORLD),
                               total_num_qubits, kokkos_args) {}

    /**
     * @brief Create a new state vector with MPI Comm.
     *
     * @param mpi_communicator MPI_Comm communicator
     * @param num_global_qubits Number of global qubits
     * @param num_local_qubits Number of local qubits
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkosMPI(MPI_Comm mpi_communicator,
                         std::size_t num_global_qubits,
                         std::size_t num_local_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(MPIManagerKokkos(mpi_communicator),
                               num_global_qubits, num_local_qubits,
                               kokkos_args) {};

    /**
     * @brief Create a new state vector with default MPI_COMM_WORLD
     *
     * @param num_global_qubits Number of global qubits
     * @param num_local_qubits Number of local qubits
     * @param kokkos_args Arguments for Kokkos initialization
     */
    StateVectorKokkosMPI(std::size_t num_global_qubits,
                         std::size_t num_local_qubits,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(MPIManagerKokkos(MPI_COMM_WORLD),
                               num_global_qubits, num_local_qubits,
                               kokkos_args) {}

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_global_qubits Number of global qubits
     * @param num_local_qubits Number of local qubits
     * @param hostdata_ Host array for state vector
     * @param length Length of host array (must be equal to exp2(total qubits))
     * @param kokkos_args Arguments for Kokkos initialization
     * @param communicator MPI Communicator
     */
    template <class complex>
    StateVectorKokkosMPI(std::size_t num_global_qubits,
                         std::size_t num_local_qubits, complex *hostdata_,
                         const std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : StateVectorKokkosMPI(num_global_qubits, num_local_qubits,
                               communicator) {
        PL_ABORT_IF_NOT(
            exp2(num_qubits_) == length,
            "length of complex data does not match the number of qubits");
        const std::size_t blk{getLocalBlockSize()};
        const std::size_t offset{blk * mpi_manager_.getRank()};
        (*sv_).HostToDevice(reinterpret_cast<ComplexT *>(hostdata_ + offset),
                            blk);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_global_qubits Number of global qubits
     * @param num_local_qubits Number of local qubits
     * @param hostdata_ Host array for state vector
     * @param length Length of host array (must be equal to exp2(total qubits))
     * @param kokkos_args Arguments for Kokkos initialization
     * @param communicator MPI Communicator
     */
    StateVectorKokkosMPI(std::size_t num_global_qubits,
                         std::size_t num_local_qubits,
                         const ComplexT *hostdata_, const std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : StateVectorKokkosMPI(communicator, num_global_qubits,
                               num_local_qubits, kokkos_args) {
        PL_ABORT_IF_NOT(
            exp2(num_qubits_) == length,
            "length of complex data does not match the number of qubits");
        const std::size_t blk{getLocalBlockSize()};
        const std::size_t offset{blk * mpi_manager_.getRank()};
        std::vector<ComplexT> hostdata_copy(hostdata_ + offset,
                                            hostdata_ + offset + blk);
        (*sv_).HostToDevice(hostdata_copy.data(), hostdata_copy.size());
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_global_qubits Number of global qubits
     * @param num_local_qubits Number of local qubits
     * @param hostdata_ Host array for state vector
     * @param kokkos_args Arguments for Kokkos initialization
     * @param communicator MPI Communicator
     */
    template <class complex>
    StateVectorKokkosMPI(std::size_t num_global_qubits,
                         std::size_t num_local_qubits,
                         std::vector<complex> hostdata_,
                         const Kokkos::InitializationSettings &kokkos_args = {},
                         const MPI_Comm &communicator = MPI_COMM_WORLD)
        : StateVectorKokkosMPI(num_global_qubits, num_local_qubits,
                               hostdata_.data(), hostdata_.size(), kokkos_args,
                               communicator) {}

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkosMPI(const StateVectorKokkosMPI &other,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(other.getMPIManager(), other.getNumGlobalWires(),
                               other.getNumLocalWires(), kokkos_args) {
        global_wires_ = other.getGlobalWires();
        local_wires_ = other.getLocalWires();
        mpi_rank_to_global_index_map_ = other.getMPIRankToGlobalIndexMap();
        sendbuf_ = other.sendbuf_;
        recvbuf_ = other.recvbuf_;
        (*sv_).DeviceToDevice(other.getView());
    }

    ~StateVectorKokkosMPI() = default;

    SVK &getLocalSV() { return *sv_; }

    auto getMPIManager() const { return mpi_manager_; }

    auto getSendBuffer() const { return sendbuf_; };
    auto getRecvBuffer() const { return recvbuf_; };

    std::size_t getNumGlobalWires() const { return numGlobalQubits_; }

    std::size_t getNumLocalWires() const { return numLocalQubits_; }

    /**
     * @brief Reset the indices of the global_wires_, local_wires_ and
     *       the mpi_rank_to_global_index_map_.
     *
     */
    void resetIndices() {
        std::iota(global_wires_.begin(), global_wires_.end(), 0);
        std::iota(local_wires_.begin(), local_wires_.end(),
                  getNumGlobalWires());
        std::iota(mpi_rank_to_global_index_map_.begin(),
                  mpi_rank_to_global_index_map_.end(), 0);
    }

    /**
     * @brief Set to initial zero state for the state-vector on device.
     */
    void initZeros() {
        resetIndices();
        (*sv_).initZeros();
    }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param global_index Index of the target element.
     */
    void setBasisState(std::size_t global_index) {
        const auto index = global2localIndex(global_index);
        resetIndices();
        const auto rank = mpi_manager_.getRank();
        if (index.first == rank) {
            (*sv_).setBasisState(index.second);
        } else {
            (*sv_).initZeros();
        }
    }

    /**
     * @brief Allocate send and recv buffers for MPI communication.
     */
    void allocateBuffers() {
        if (!sendbuf_) {
            sendbuf_ = std::make_shared<KokkosVector>(
                "sendbuf_", exp2(getNumLocalWires() - 1));
        }
        if (!recvbuf_) {
            recvbuf_ = std::make_shared<KokkosVector>(
                "recvbuf_", exp2(getNumLocalWires() - 1));
        }
    }

    /******************
    MPI-related methods
    ******************/
    void barrier() {
        Kokkos::fence();
        mpi_manager_.Barrier();
    }

    /**
     * @brief Perform an all-reduce operation with the sum operation
     *
     * @param data Data to be reduced
     */
    template <typename T> T allReduceSum(const T &data) const {
        return mpi_manager_.allreduce(data, std::string{"sum"});
    }

    /**
     * @brief Perform a send-receive operation between two ranks using the
     * sendbuf_ and recvbuf_
     *
     * @param send_rank Rank to send data to
     * @param recv_rank Rank to receive data from
     * @param size Size of the data to send/receive
     * @param tag Tag for the MPI message
     */
    void sendrecvBuffers(const std::size_t send_rank,
                         const std::size_t recv_rank, const std::size_t size,
                         const std::size_t tag) {
        mpi_manager_.Sendrecv(*sendbuf_, send_rank, *recvbuf_, recv_rank, size,
                              tag);
    }
    /********************
    Wires-related methods
    ********************/

    /**
     * @brief  Returns the MPI-distribution block size, or the size of the local
     * state vector data.
     */
    std::size_t getLocalBlockSize() const { return exp2(getNumLocalWires()); }

    std::size_t getRevLocalWireIndex(const std::size_t wire) const {
        return getRevWireIndex(local_wires_, getLocalWireIndex(wire));
    }

    std::size_t getRevGlobalWireIndex(const std::size_t wire) const {
        return getRevWireIndex(global_wires_, getGlobalWireIndex(wire));
    }

    const std::vector<std::size_t> &getMPIRankToGlobalIndexMap() const {
        return mpi_rank_to_global_index_map_;
    }

    const std::vector<std::size_t> &getGlobalWires() const {
        return global_wires_;
    }
    const std::vector<std::size_t> &getLocalWires() const {
        return local_wires_;
    }

    std::vector<std::size_t>
    getLocalWireIndices(const std::vector<std::size_t> &wires) const {
        std::vector<std::size_t> local_wires_indices;
        local_wires_indices.reserve(wires.size());
        std::transform(
            wires.begin(), wires.end(), std::back_inserter(local_wires_indices),
            [&](std::size_t wire) { return getLocalWireIndex(wire); });
        return local_wires_indices;
    }

    size_t getLocalWireIndex(const std::size_t wire) const {
        return getElementIndexInVector(local_wires_, wire);
    }

    std::vector<std::size_t>
    getGlobalWiresIndices(const std::vector<std::size_t> &wires) const {
        std::vector<std::size_t> global_wires_indices;
        global_wires_indices.reserve(wires.size());
        std::transform(
            wires.begin(), wires.end(),
            std::back_inserter(global_wires_indices),
            [&](std::size_t wire) { return getGlobalWireIndex(wire); });
        return global_wires_indices;
    }

    size_t getGlobalWireIndex(const std::size_t wire) const {
        return getElementIndexInVector(global_wires_, wire);
    }

    std::vector<std::size_t>
    findGlobalWires(const std::vector<std::size_t> &wires) const {
        std::vector<std::size_t> global_wires;
        std::copy_if(wires.begin(), wires.end(),
                     std::back_inserter(global_wires),
                     [&](std::size_t wire) { return isWiresGlobal({wire}); });
        return global_wires;
    }

    std::vector<std::size_t>
    findLocalWires(const std::vector<std::size_t> &wires) const {
        std::vector<std::size_t> local_wires;
        std::copy_if(wires.begin(), wires.end(),
                     std::back_inserter(local_wires),
                     [&](std::size_t wire) { return isWiresLocal({wire}); });
        return local_wires;
    }

    /**
     * @brief  Converts a global state vector index to a local one.
     *
     * @param index Pair containing {global index/rank location, local index}
     */
    std::pair<std::size_t, std::size_t>
    global2localIndex(const std::size_t index) const {
        PL_ABORT_IF_NOT(index < exp2(this->getNumQubits()),
                        "Index out of bounds.");
        auto blk = getLocalBlockSize();
        return std::pair<std::size_t, std::size_t>{index / blk, index % blk};
    }

    std::size_t getGlobalIndexFromMPIRank(const int mpi_rank) const {
        return mpi_rank_to_global_index_map_[mpi_rank];
    }

    std::size_t
    getMPIRankFromGlobalIndex(const std::size_t global_index) const {
        return getElementIndexInVector(mpi_rank_to_global_index_map_,
                                       global_index);
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
     * @brief Normalize the state vector.
     */
    void normalize() {
        auto sv_view = getView();

        PrecisionT squaredLocalNorm = 0.0;
        Kokkos::parallel_reduce(
            sv_view.size(),
            KOKKOS_LAMBDA(std::size_t i, PrecisionT &sum) {
                const PrecisionT norm = Kokkos::abs(sv_view(i));
                sum += norm * norm;
            },
            squaredLocalNorm);

        PrecisionT squaredNorm = allReduceSum(squaredLocalNorm);

        PL_ABORT_IF(squaredNorm <
                        std::numeric_limits<PrecisionT>::epsilon() * 1e2,
                    "Vector has norm close to zero and cannot be normalized");

        const std::complex<PrecisionT> inv_norm =
            1. / Kokkos::sqrt(squaredNorm);
        Kokkos::parallel_for(
            sv_view.size(),
            KOKKOS_LAMBDA(std::size_t i) { sv_view(i) *= inv_norm; });
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
        // This implementation can be improved for most/least significant bits
        // setStateVector, similar to LGPU
        const auto num_qubits = this->getNumQubits();
        PL_ABORT_IF_NOT(
            std::find_if(wires.begin(), wires.end(),
                         [&num_qubits](const auto i) {
                             return i >= num_qubits;
                         }) == wires.end(),
            "wires must take values lower than the number of qubits.");
        initZeros();
        auto global_wires = findGlobalWires(wires);
        auto local_wires = findLocalWires(wires);

        std::size_t global_index = getMPIManager().getRank();

        std::size_t global_mask = 0U;
        for (std::size_t i = 0; i < global_wires.size(); i++) {
            global_mask |=
                ((global_index >> getRevGlobalWireIndex(global_wires[i])) & 1U)
                << getRevWireIndex(
                       wires, getElementIndexInVector(wires, global_wires[i]));
        }
        std::vector<ComplexT> local_state(exp2(local_wires.size()));
        for (std::size_t i = 0; i < exp2(local_wires.size()); i++) {
            std::size_t index = global_mask;
            for (std::size_t j = 0; j < local_wires.size(); j++) {
                index |=
                    (((i >> (local_wires.size() - 1 - j)) & 1U)
                     << getRevWireIndex(wires, getElementIndexInVector(
                                                   wires, local_wires[j])));
            }
            local_state[i] = state[index];
        }

        bool set = true;
        for (std::size_t i = 0; i < getNumGlobalWires(); i++) {
            if (!isElementInVector(global_wires, global_wires_[i])) {
                if ((global_index >> getRevGlobalWireIndex(global_wires_[i])) &
                    1U) {
                    set = false;
                    break;
                }
            }
        }
        if (set) {
            (*sv_).setStateVector(local_state,
                                  getLocalWireIndices(local_wires));
        }
    }

    /**
     * @brief Get the local wires that could be used for global/local swaps.
     * It will return a set of local wires of the same size as the global wires
     * to be swapped. The local wires will also not be in wires, i.e. wires used
     * in an operation.
     *
     * @param global_wires Global wires to swap, obtained from findGlobalWires()
     * @param wires Wires used in an operation to be excluded in returned wires
     *
     */
    std::vector<std::size_t>
    localWiresSubsetToSwap(const std::vector<std::size_t> &global_wires,
                           const std::vector<std::size_t> &wires) {
        PL_ABORT_IF(
            global_wires.size() > local_wires_.size(),
            "global_wires to swap must be have less wires than local_wires.");
        std::vector<std::size_t> local_wires;
        for (std::size_t i = 0; i < local_wires_.size(); i++) {
            if (local_wires.size() == global_wires.size()) {
                break;
            }
            if (!isElementInVector(wires, local_wires_[i])) {
                local_wires.push_back(local_wires_[i]);
            }
        }
        PL_ABORT_IF(local_wires.size() != global_wires.size(),
                    "Not enough local wires to swap with global wires.");
        // TODO: improve with algorithm based on memory pattern
        return local_wires;
    }

    /**
     * @brief Perform a swap between global and local wires.
     *
     * @param global_wires_to_swap wire indices for global wires to swap
     * @param local_wires_to_swap wire indices for local wires to swap
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
     * For each batch, a single pairwise sendrecv is performed.
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

        std::vector<std::size_t> rev_global_wires_index_to_swap;
        rev_global_wires_index_to_swap.reserve(global_wires_to_swap.size());
        std::vector<std::size_t> rev_local_wires_index_to_swap;
        rev_local_wires_index_to_swap.reserve(local_wires_to_swap.size());

        std::transform(
            local_wires_to_swap.begin(), local_wires_to_swap.end(),
            std::back_inserter(rev_local_wires_index_to_swap),
            [this](std::size_t wire) { return getRevLocalWireIndex(wire); });

        std::transform(
            global_wires_to_swap.begin(), global_wires_to_swap.end(),
            std::back_inserter(rev_global_wires_index_to_swap),
            [this](std::size_t wire) { return getRevGlobalWireIndex(wire); });

        std::vector<std::size_t> rev_local_wires_index_not_swapping;
        rev_local_wires_index_not_swapping.reserve(getNumLocalWires() -
                                                   local_wires_to_swap.size());
        for (std::size_t i = 0; i < getNumLocalWires(); i++) {
            if (!isElementInVector(rev_local_wires_index_to_swap, i)) {
                rev_local_wires_index_not_swapping.push_back(i);
            }
        }
        std::size_t global_index =
            getGlobalIndexFromMPIRank(mpi_manager_.getRank());

        // To get the relevant batch_index, we loop a compressed_batch_index
        // over 1 to 2^(num_swapping_wires) - 1 then map to the batch_index (we
        // start from 1 since the global index does not need to swap with
        // itself). For example, if we have global_wires_={0,1,2} and
        // global_wires_to_swap={0,2}, the compressed_batch_index will be {01,
        // 10, 11} and the batch_index will be {001, 100, 101}, where a 0 is
        // inserted in the middle to wire 1 which is not being swapped.
        for (std::size_t compressed_batch_index = 1;
             compressed_batch_index < exp2(global_wires_to_swap.size());
             compressed_batch_index++) {
            std::size_t batch_index = 0;
            for (std::size_t i = 0; i < global_wires_to_swap.size(); i++) {
                batch_index |= ((compressed_batch_index >> i) & 1)
                               << rev_global_wires_index_to_swap[i];
            }

            std::size_t swap_wire_mask = 0;
            for (std::size_t i = 0; i < local_wires_to_swap.size(); i++) {
                swap_wire_mask |= ((((batch_index ^ global_index) >>
                                     rev_global_wires_index_to_swap[i]) &
                                    1)
                                   << rev_local_wires_index_to_swap[i]);
            }

            // These are defined since on AMD compiler it's more strict what
            // host functions can be included in the KOKKOS_LAMBDA - e.g.
            // dereferencing, size() are all not allowed
            const std::size_t not_swapping_local_wire_size =
                rev_local_wires_index_not_swapping.size();
            auto rev_local_wires_index_not_swapping_view =
                vector2view(rev_local_wires_index_not_swapping);

            auto sendbuf_view = (*sendbuf_);
            auto recvbuf_view = (*recvbuf_);
            auto sv_view = (*sv_).getView();
            std::size_t send_size =
                exp2((getNumLocalWires() - local_wires_to_swap.size()));

            // Copy to send buffer
            Kokkos::parallel_for(
                "copy_sendbuf", send_size,
                KOKKOS_LAMBDA(std::size_t buffer_index) {
                    std::size_t SV_index = swap_wire_mask;
                    for (std::size_t i = 0; i < not_swapping_local_wire_size;
                         i++) {
                        SV_index |=
                            (((buffer_index >> i) & 1)
                             << rev_local_wires_index_not_swapping_view(i));
                    }
                    sendbuf_view(buffer_index) = sv_view(SV_index);
                });
            Kokkos::fence();

            // MPI Sendrecv
            std::size_t other_global_index = batch_index ^ global_index;
            std::size_t other_mpi_rank =
                getMPIRankFromGlobalIndex(other_global_index);

            sendrecvBuffers(other_mpi_rank, other_mpi_rank, send_size,
                            batch_index);

            // Copy from recv buffer
            Kokkos::parallel_for(
                "copy_recvbuf", send_size,
                KOKKOS_LAMBDA(std::size_t buffer_index) {
                    std::size_t SV_index = swap_wire_mask;

                    for (std::size_t i = 0; i < not_swapping_local_wire_size;
                         i++) {
                        SV_index |=
                            (((buffer_index >> i) & 1)
                             << rev_local_wires_index_not_swapping_view(i));
                    }

                    sv_view(SV_index) = recvbuf_view(buffer_index);
                });
            Kokkos::fence();
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

    /**
     * @brief Match the global/local wires and map from another state vector
     * The following steps are performed (if necessary):
     * Swap Global Local Wires (so that the first n-wires are global, and last
     * wires a local, e.g. G={2,0,1}, L={3,5,4}) Swap Global Global Wires (make
     * sure the global wires are in order, so move G to {0, 1, 2} Swap Local
     * Local Wires (make sure the local wires are in order, so move L to {3, 4,
     * 5}
     *
     * @param other_sv State vector to match
     */
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
        matchGlobalWiresAndIndex(other_global_wires,
                                 other_mpi_rank_to_global_index_map);
    }

    void matchGlobalWiresAndIndex(
        const std::vector<std::size_t> &global_wires_target,
        const std::vector<std::size_t> &mpi_rank_to_global_index_map_target) {
        std::size_t my_global_index =
            getGlobalIndexFromMPIRank(mpi_manager_.getRank());
        std::size_t dest_global_index = 0;
        for (std::size_t i = 0; i < global_wires_.size(); ++i) {
            std::size_t dest_global_wire_index = getElementIndexInVector(
                global_wires_target,
                global_wires_[global_wires_.size() - i - 1]);
            dest_global_index |=
                (((my_global_index >> i) & 1)
                 << getRevWireIndex(global_wires_, dest_global_wire_index));
        }

        std::size_t dest_mpi_rank = getElementIndexInVector(
            mpi_rank_to_global_index_map_target, dest_global_index);

        std::size_t send_size = exp2(getNumLocalWires() - 1);
        auto sendbuf_view = (*sendbuf_);
        auto recvbuf_view = (*recvbuf_);
        auto sv_view = (*sv_).getView();

        // Since the buffer is half the size of the state vector, we need to
        // do two copies
        for (std::size_t i = 0; i < 2; i++) {
            std::size_t offset = i * send_size;
            // COPY to buffer
            Kokkos::parallel_for(
                "copy_sendbuf", send_size,
                KOKKOS_LAMBDA(std::size_t buffer_index) {
                    sendbuf_view(buffer_index) = sv_view(buffer_index + offset);
                });
            Kokkos::fence();
            // SENDRECV
            sendrecvBuffers(dest_mpi_rank, dest_mpi_rank, send_size, 0);
            // COPY FROM BUFFER

            Kokkos::parallel_for(
                "copy_recvbuf", send_size,
                KOKKOS_LAMBDA(std::size_t buffer_index) {
                    sv_view(buffer_index + offset) = recvbuf_view(buffer_index);
                });
        }

        // copy global_wires_target and mpi_rank_to_global_index_map_target
        global_wires_ = global_wires_target;
        mpi_rank_to_global_index_map_ = mpi_rank_to_global_index_map_target;
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

        PL_ABORT_IF(wires.size() > getNumLocalWires(),
                    "Number of wires must be smaller than or equal to the "
                    "number of local wires.");
        if (!isWiresLocal(wires)) {
            auto global_wires_to_swap = findGlobalWires(wires);
            auto local_wires_to_swap =
                localWiresSubsetToSwap(global_wires_to_swap, wires);
            swapGlobalLocalWires(global_wires_to_swap, local_wires_to_swap);
        }
        (*sv_).applyPauliRot(getLocalWireIndices(wires), inverse, params, word);
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

        if (opName == "GlobalPhase") {
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
                    getGlobalIndexFromMPIRank(mpi_manager_.getRank());
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
                    getGlobalIndexFromMPIRank(mpi_manager_.getRank());
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

        PL_ABORT_IF(wires.size() > getNumLocalWires(),
                    "Number of wires must be smaller than or equal to the "
                    "number of local wires.");
        if (!isWiresLocal(wires)) {
            auto global_wires_to_swap = findGlobalWires(wires);
            auto local_wires_to_swap =
                localWiresSubsetToSwap(global_wires_to_swap, wires);
            swapGlobalLocalWires(global_wires_to_swap, local_wires_to_swap);
        }

        (*sv_).applyOperation(opName, getLocalWireIndices(wires), inverse,
                              params, gate_matrix);
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
        std::size_t global_index =
            getGlobalIndexFromMPIRank(mpi_manager_.getRank());
        bool operate = true;
        for (std::size_t i = 0; i < global_control_wires.size(); i++) {
            operate =
                operate && (((global_index >>
                              getRevGlobalWireIndex(global_control_wires[i])) &
                             1) == global_control_values[i]);
        }
        if (operate) {
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
    void collapse(std::size_t wire, bool branch) {
        // Swap target wire to all local
        if (!isWiresLocal({wire})) {
            auto global_wires_to_swap = findGlobalWires({wire});
            auto local_wires_to_swap =
                localWiresSubsetToSwap(global_wires_to_swap, {wire});
            swapGlobalLocalWires(global_wires_to_swap, local_wires_to_swap);
            barrier();
        }

        KokkosVector matrix("gate_matrix", 4);
        Kokkos::parallel_for(
            matrix.size(), KOKKOS_LAMBDA(std::size_t k) {
                matrix(k) = ((k == 0 && branch == 0) || (k == 3 && branch == 1))
                                ? ComplexT{1.0, 0.0}
                                : ComplexT{0.0, 0.0};
            });
        (*sv_).applyMultiQubitOp(matrix, {getLocalWireIndices({wire})[0]},
                                 false);
        normalize();
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
     * @brief Get the local vector-converted Kokkos view
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

    /**
     * @brief Make sure local wires are in ascending order
     *        Must be used after reorderGlobalLocalWires()
     */
    void reorderLocalWires() {
        PL_ABORT_IF_NOT(
            std::all_of(local_wires_.begin(), local_wires_.end(),
                        [this](const auto i) {
                            return (getNumGlobalWires() <= i) &&
                                   (i < num_qubits_);
                        }),
            "local wires must only contain least significant indices. Run "
            "reorder_global_wires first.");

        for (std::size_t i = 0; i < getNumLocalWires(); ++i) {
            std::size_t wire_i = i + getNumGlobalWires();
            if (local_wires_[i] > wire_i) {
                applyOperation("SWAP", {local_wires_[i], wire_i}, false);
            }
        }
        std::iota(local_wires_.begin(), local_wires_.end(),
                  getNumGlobalWires());
    }

    /**
     * @brief Make sure global wires are lowest numbered wires and in ascending
     *        order
     */
    void reorderGlobalLocalWires() {
        std::vector<std::size_t> global_wires;
        std::vector<std::size_t> local_wires;

        std::copy_if(global_wires_.begin(), global_wires_.end(),
                     std::back_inserter(global_wires), [&](std::size_t wire) {
                         return wire >= numGlobalQubits_;
                     });

        std::copy_if(local_wires_.begin(), local_wires_.end(),
                     std::back_inserter(local_wires),
                     [&](std::size_t wire) { return wire < numGlobalQubits_; });
        if (!global_wires.empty()) {
            swapGlobalLocalWires(global_wires, local_wires);
        }
    }

    void reorderAllWires() {
        reorderGlobalLocalWires();
        std::vector<std::size_t> global_wires_target(getNumGlobalWires());
        std::vector<std::size_t> mpi_rank_to_global_index_map_target(
            mpi_manager_.getSize());

        std::iota(global_wires_target.begin(), global_wires_target.end(), 0);
        std::iota(mpi_rank_to_global_index_map_target.begin(),
                  mpi_rank_to_global_index_map_target.end(), 0);

        matchGlobalWiresAndIndex(global_wires_target,
                                 mpi_rank_to_global_index_map_target);
        reorderLocalWires();
    }
};
}; // namespace Pennylane::LightningKokkos
