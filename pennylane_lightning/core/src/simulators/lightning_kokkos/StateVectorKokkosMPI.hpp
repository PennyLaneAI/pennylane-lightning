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

template <class T> [[maybe_unused]] MPI_Datatype get_mpi_type() {
    PL_ABORT("No corresponding MPI type.");
}
template <> [[maybe_unused]] MPI_Datatype get_mpi_type<float>() {
    return MPI_FLOAT;
}
template <> [[maybe_unused]] MPI_Datatype get_mpi_type<double>() {
    return MPI_DOUBLE;
}
template <>
[[maybe_unused]] MPI_Datatype get_mpi_type<Kokkos::complex<float>>() {
    return MPI_C_FLOAT_COMPLEX;
}
template <>
[[maybe_unused]] MPI_Datatype get_mpi_type<Kokkos::complex<double>>() {
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

        // Init MPI
        int status = 0;
        MPI_Initialized(&status);
        if (!status) {
            PL_MPI_IS_SUCCESS(MPI_Init(nullptr, nullptr));
        }
        communicator_ = communicator;
        Kokkos::InitializationSettings settings = kokkos_args;
        num_qubits_ = num_qubits;

        settings.set_device_id(get_mpi_rank());
        global_wires_.resize(log2(static_cast<std::size_t>(
            get_mpi_size()))); // set to constructor line
        local_wires_.resize(get_num_local_wires());
        mpi_rank_to_global_index_map_.resize(get_mpi_size());

        reset_indices_();

        if (num_qubits > 0) {
            sv_ = std::make_unique<SVK>(get_num_local_wires(), settings);
            setBasisState(0U);
            recvbuf_ = std::make_unique<SVK>(
                get_num_local_wires(),
                settings); // This could be smaller, even dynamic!
            sendbuf_ = std::make_unique<SVK>(
                get_num_local_wires(),
                settings); // This could be smaller, even dynamic!
        }
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
        const std::size_t blk{get_blk_size()};
        const std::size_t offset{blk * get_mpi_rank()};
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
        const std::size_t blk{get_blk_size()};
        const std::size_t offset{blk * get_mpi_rank()};
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
    int get_mpi_rank() {
        int rank;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communicator_, &rank));
        return rank;
    }

    /**
     * @brief  Returns the number of MPI processes.
     */
    int get_mpi_size() {
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

    /**
     * @brief  Receives the local state vector of another MPI-process.
     *
     * @param  source Destination MPI rank.
     * @param  request MPI request allowing to validate the transfer has been
     * completed.
     */
    /* void mpi_irecv(const std::size_t source, const std::size_t size,
    MPI_Request &request, int tag) { KokkosVector sv_view =
    (*recvbuf_).getView(); PL_MPI_IS_SUCCESS(MPI_Irecv(reinterpret_cast<void
    *>(sv_view.data()), size, get_mpi_type<ComplexT>(),
                                    static_cast<int>(source), tag,
    communicator_, &request));
    } */

    /**
     * @brief  Sends local state vector to another MPI-process.
     *
     * @param  dest Destination MPI rank.
     * @param  request MPI request allowing to validate the transfer has been
     * completed.
     * @param  copy If true, copy the state vector data in sendbuf_ before
     * sending.
     */
    /* void mpi_isend(const std::size_t dest, const std::size_t size,
    MPI_Request &request, int tag) { KokkosVector sd_view =
    (*sendbuf_).getView(); mpi_wait(request);
        PL_MPI_IS_SUCCESS(MPI_Isend(reinterpret_cast<void *>(sd_view.data()),
                                    size, get_mpi_type<ComplexT>(),
                                    static_cast<int>(dest), tag, communicator_,
                                    &request));
    } */

    /**
     * @brief  Waits for an MPI transfer completion.
     *
     * @param  request MPI request allowing to validate the transfer has been
     * completed.
     */
    void mpi_wait(MPI_Request &request) {
        MPI_Status status;
        PL_MPI_IS_SUCCESS(MPI_Wait(&request, &status));
    }

    /**
     * @brief  Returns the MPI-distribution block size, or the size of the local
     * state vector data.
     */
    std::size_t get_blk_size() { return exp2(get_num_local_wires()); }

    template <typename T> T all_reduce_sum(const T &data) const {
        T sum;
        MPI_Allreduce(&data, &sum, 1, get_mpi_type<T>(), MPI_SUM,
                      communicator_);
        return sum;
    }

    /********************
Wires-related methods
********************/

    /**
     * @brief  Returns the number of global wires.
     */
    std::size_t get_num_global_wires() { return global_wires_.size(); }

    /**
     * @brief  Returns the number of local wires.
     */
    std::size_t get_num_local_wires() {
        return num_qubits_ - get_num_global_wires();
    }

    /**
     * @brief  Converts a global state vector index to a local one.
     *
     * @param index Global index.
     */
    std::pair<std::size_t, std::size_t>
    global_2_local_index(const std::size_t index) {
        auto blk = get_blk_size();
        return std::pair<std::size_t, std::size_t>{index / blk, index % blk};
    }

    std::size_t get_global_index_from_mpi_rank(const std::size_t mpi_rank) {
        return mpi_rank_to_global_index_map_[mpi_rank];
    }

    std::size_t get_mpi_rank_from_global_index(const std::size_t global_index) {
        return std::find(mpi_rank_to_global_index_map_.begin(),
                         mpi_rank_to_global_index_map_.end(), global_index) -
               mpi_rank_to_global_index_map_.begin();
    }

    void reset_indices_() {
        std::iota(global_wires_.begin(), global_wires_.end(), 0);
        std::iota(local_wires_.begin(), local_wires_.end(),
                  get_num_global_wires());
        std::iota(mpi_rank_to_global_index_map_.begin(),
                  mpi_rank_to_global_index_map_.end(), 0);
    }

    bool is_wires_local(const std::vector<std::size_t> &wires) {
        return std::all_of(wires.begin(), wires.end(), [this](const auto i) {
            return std::find(local_wires_.begin(), local_wires_.end(), i) !=
                   local_wires_.end();
        });
    }

    bool is_wires_global(const std::vector<std::size_t> &wires) {
        return std::all_of(wires.begin(), wires.end(), [this](const auto i) {
            return std::find(global_wires_.begin(), global_wires_.end(), i) !=
                   global_wires_.end();
        });
    }

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() {
        reset_indices_();
        (*sv_).initZeros();
    }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(std::size_t global_index) {
        const auto index = global_2_local_index(global_index);
        reset_indices_();
        const auto rank = static_cast<std::size_t>(get_mpi_rank());
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
        reset_indices_();
        const std::size_t blk{get_blk_size()};
        const std::size_t offset{blk * get_mpi_rank()};
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
        PL_ABORT("Not implemented yet.");
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
        PL_ABORT("Not implemented yet.");
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
            num_state, KOKKOS_LAMBDA(std::size_t i) {
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
        // TODO: need to copy global/local wires_ too
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     */

    ~StateVectorKokkosMPI() {}

    std::vector<std::size_t>
    find_global_wires(const std::vector<std::size_t> &wires) {
        std::vector<std::size_t> global_wires;
        for (const auto &wire : wires) {
            if (std::find(global_wires_.begin(), global_wires_.end(), wire) !=
                global_wires_.end()) {
                global_wires.push_back(wire);
            }
        }
        return global_wires;
    }

    std::vector<std::size_t>
    local_wires_subset_to_swap(const std::vector<std::size_t> &global_wires,
                               const std::vector<std::size_t> &wires) {
        PL_ABORT_IF(global_wires.size() > local_wires_.size(),
                    "global_wires must be smaller than local_wires.");
        std::vector<std::size_t> local_wires;
        int j = 0;

        while (local_wires.size() != global_wires.size()) {
            if (std::find(wires.begin(), wires.end(), local_wires_[j]) ==
                wires.end()) {
                local_wires.push_back(local_wires_[j]);
            }
            j++;
        }
        // TODO: FIX ME with better algorithm based on memory pattern
        return local_wires;
    }

    bool is_generalized_permutation_matrix(
        [[maybe_unused]] const std::string &opName) {
        return false; // TODO: implement me
    }

    void swap_global_local_wires(std::vector<std::size_t> &global_wires,
                                 std::vector<std::size_t> &local_wires) {
        PL_ABORT_IF_NOT(
            global_wires.size() == local_wires.size(),
            "global_wires and local_wires must have equal dimensions.");
        std::sort(global_wires.begin(), global_wires.end());
        std::sort(local_wires.begin(), local_wires.end());

        // Map local wires to actual local wire indices

        auto local_wires_indices = get_local_wires_indices(local_wires);
        std::size_t global_index =
            get_global_index_from_mpi_rank(get_mpi_rank());

        // Actually swap memory
        // TODO: improve me, or at least parallelize me
        for (std::size_t batch_index = 1;
             batch_index < (1 << get_num_global_wires()); batch_index++) {
            bool send = true;
            for (std::size_t digits = 0; digits < global_wires_.size();
                 digits++) {
                bool is_global_wire_in_swap =
                    std::find(
                        global_wires.begin(), global_wires.end(),
                        global_wires_[get_num_global_wires() - digits - 1]) !=
                    global_wires.end();
                bool batch_index_digit = (batch_index >> digits) & 1;
                send = send && (!batch_index_digit || is_global_wire_in_swap);
            }
            if (send) {
                barrier();
                std::size_t j = 0;
                for (std::size_t i = 0; i < (*sv_).getView().size(); i++) {
                    bool relevant = true;
                    for (std::size_t k = 0; k < local_wires_indices.size();
                         k++) {
                        auto it =
                            std::find(global_wires_.begin(),
                                      global_wires_.end(), global_wires[k]);
                        std::size_t global_index_to_shift =
                            global_wires_.size() - 1 -
                            std::distance(global_wires_.begin(), it);
                        std::size_t local_index_to_shift =
                            local_wires_.size() - 1 - local_wires_indices[k];
                        relevant &= (i >> local_index_to_shift & 1) ==
                                    (((batch_index ^ global_index) >>
                                      global_index_to_shift) &
                                     1);
                    }
                    if (relevant) {
                        (*sendbuf_).getView()(j) = (*sv_).getView()(i);
                        j++;
                    }
                }

                barrier();
                std::size_t other_global_index = batch_index ^ global_index;
                std::size_t other_mpi_rank =
                    get_mpi_rank_from_global_index(other_global_index);
                std::cout << "I am rank " << get_mpi_rank()
                          << " and I am sending to rank " << other_mpi_rank
                          << " with tag " << batch_index
                          << " and this number of elements "
                          << (1 << (get_num_local_wires() - local_wires.size()))
                          << std::endl;
                MPI_Sendrecv((*sendbuf_).getView().data(),
                             1 << (get_num_local_wires() - local_wires.size()),
                             get_mpi_type<ComplexT>(), other_mpi_rank,
                             batch_index, (*recvbuf_).getView().data(),
                             1 << (get_num_local_wires() - local_wires.size()),
                             get_mpi_type<ComplexT>(), other_mpi_rank,
                             batch_index, communicator_, MPI_STATUS_IGNORE);

                barrier();
                j = 0;
                for (std::size_t i = 0; i < (*sv_).getView().size(); i++) {
                    bool relevant = true;
                    for (std::size_t k = 0; k < local_wires_indices.size();
                         k++) {
                        auto it =
                            std::find(global_wires_.begin(),
                                      global_wires_.end(), global_wires[k]);
                        std::size_t global_index_to_shift =
                            global_wires_.size() - 1 -
                            std::distance(global_wires_.begin(), it);
                        std::size_t local_index_to_shift =
                            local_wires_.size() - 1 - local_wires_indices[k];
                        relevant &= (i >> local_index_to_shift & 1) ==
                                    (((batch_index ^ global_index) >>
                                      global_index_to_shift) &
                                     1);
                    }
                    if (relevant) {
                        (*sv_).getView()(i) = (*recvbuf_).getView()(j);
                        j++;
                    }
                }
                barrier();
            }
        }

        // Swap global and local wires labels
        std::unordered_map<int, size_t> global_wires_positions;
        std::unordered_map<int, size_t> local_wires_positions;
        for (size_t i = 0; i < global_wires.size(); ++i) {
            auto it_g = std::find(global_wires_.begin(), global_wires_.end(),
                                  global_wires[i]);
            if (it_g == global_wires_.end()) {
                PL_ABORT("Error");
            }
            global_wires_positions[global_wires[i]] =
                std::distance(global_wires_.begin(), it_g);

            auto it_l = std::find(local_wires_.begin(), local_wires_.end(),
                                  local_wires[i]);
            if (it_l == local_wires_.end()) {
                PL_ABORT("Error");
            }
            local_wires_positions[local_wires[i]] =
                std::distance(local_wires_.begin(), it_l);
        }

        for (size_t i = 0; i < global_wires.size(); ++i) {
            std::swap(global_wires_[global_wires_positions[global_wires[i]]],
                      local_wires_[local_wires_positions[local_wires[i]]]);
        }
        // barrier();
    }

    std::vector<std::size_t>
    get_local_wires_indices(const std::vector<std::size_t> &wires) {
        std::vector<std::size_t> local_wires_indices;
        for (const auto &wire : wires) {
            auto it = std::find(local_wires_.begin(), local_wires_.end(), wire);
            if (it != local_wires_.end()) {
                local_wires_indices.push_back(
                    std::distance(local_wires_.begin(), it));
            }
        }
        return local_wires_indices;
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
    void applyOperation(
        [[maybe_unused]] const std::string &opName,
        [[maybe_unused]] const std::vector<std::size_t> &wires,
        [[maybe_unused]] bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {},
        [[maybe_unused]] const std::vector<ComplexT> &gate_matrix = {}) {
        // if (opName == "Identity") {
        //     // No op
        //     return;
        // }

        if (is_wires_global(wires) &&
            is_generalized_permutation_matrix(opName)) {
            PL_ABORT("Not implemented this optimization yet.");
            return;
        }

        if (!is_wires_local(wires)) {
            auto global_wires = find_global_wires(wires);
            auto local_wires = local_wires_subset_to_swap(global_wires, wires);
            swap_global_local_wires(global_wires, local_wires);
        }
        barrier();
        if (get_mpi_rank() == 0) {
            std::cout << "I am rank " << get_mpi_rank()
                      << " and I am applying the operation " << opName
                      << " to the wires " << wires[0] << wires[1]
                      << "and converted to "
                      << get_local_wires_indices(wires)[0]
                      << get_local_wires_indices(wires)[1] << std::endl;
        }
        (*sv_).applyOperation(opName, get_local_wires_indices(wires), inverse,
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

        // if (is_wires_local(wires) && is_wires_global(controlled_wires)) {
        //     PL_ABORT("Optimization Not implemented yet.");
        //     return;
        // }

        // TODO: FIX ME - need to make sure we don't swap out control or target
        // wires to global when swapping
        if (!is_wires_local(wires)) {
            auto global_wires = find_global_wires(wires);
            auto local_wires = local_wires_subset_to_swap(global_wires, wires);
            swap_global_local_wires(global_wires, local_wires);
        }

        if (!is_wires_local(controlled_wires)) {
            auto global_wires = find_global_wires(controlled_wires);
            auto local_wires = local_wires_subset_to_swap(global_wires, wires);
            swap_global_local_wires(global_wires, local_wires);
        }

        (*sv_).applyOperation(opName, get_local_wires_indices(controlled_wires),
                              controlled_values, get_local_wires_indices(wires),
                              inverse, params, gate_matrix);
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

        if (!is_wires_local(wires)) {
            auto global_wires = find_global_wires(wires);
            auto local_wires = local_wires_subset_to_swap(global_wires, wires);
            swap_global_local_wires(global_wires, local_wires);
        }
        return (*sv_).applyGenerator(opName, get_local_wires_indices(wires),
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
        if (is_wires_local(wires) && is_wires_local(controlled_wires)) {
            return (*sv_).applyControlledGenerator(
                opName, get_local_wires_indices(controlled_wires),
                controlled_values, get_local_wires_indices(wires), inverse);
        } else {
            PL_ABORT("Not implemented yet.");
        }
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
     * @brief Normalize vector (to have norm 1).
     */
    void normalize() {
        // TODO: To update
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
        (*sv_).HostToDevice(sv, length);
    }

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        (*sv_).DeviceToDevice(vector_to_copy);
    }

    void reorder_local_wires() {
        PL_ABORT_IF_NOT(std::all_of(local_wires_.begin(), local_wires_.end(),
                                    [this](const auto i) {
                                        return (get_num_global_wires() <= i) &&
                                               (i < num_qubits_);
                                    }),
                        "local wires must be least significant indices. Run "
                        "reorder_global_wires first.");

        for (std::size_t i = 0; i < get_num_local_wires(); ++i) {
            std::size_t wire_i = i + get_num_global_wires();
            if (local_wires_[i] > wire_i) {
                std::cout << "I am rank " << get_mpi_rank()
                          << " and I am swapping " << local_wires_[i]
                          << " with " << wire_i << std::endl;
                applyOperation("SWAP", {local_wires_[i], wire_i}, false);
            }
        }

        std::iota(local_wires_.begin(), local_wires_.end(),
                  get_num_global_wires());
    }

    void reorder_global_wires() {
        std::vector<std::size_t> global_wires;
        std::vector<std::size_t> local_wires;
        for (const auto &wire : global_wires_) {
            if (wire >= get_num_global_wires()) {
                global_wires.push_back(wire);
            }
        }

        for (const auto &wire : local_wires_) {
            if (wire < get_num_global_wires()) {
                local_wires.push_back(wire);
            }
        }
        swap_global_local_wires(global_wires, local_wires);
    }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector(const int root = 0)
        -> std::vector<ComplexT> {
        // reorder_local_wires();
        std::vector<ComplexT> data_((get_mpi_rank() == root) ? this->getLength()
                                                             : 0);
        std::vector<ComplexT> local_((*sv_).getLength());
        (*sv_).DeviceToHost(local_.data(), local_.size());
        std::vector<int> recvcount(get_mpi_size(), local_.size());
        std::vector<int> displacements(get_mpi_size(), 0);
        for (std::size_t rank = 0; rank < get_mpi_size(); rank++) {
            for (std::size_t i = 0; i < get_num_global_wires(); i++) {
                std::size_t temp =
                    ((get_global_index_from_mpi_rank(rank) >>
                      (get_num_global_wires() - 1 - i)) &
                     1) *
                    (1 << (get_num_global_wires() - 1 - global_wires_[i]));
                displacements[rank] += temp;
            }
            displacements[rank] *= local_.size();
        }

        if (get_mpi_rank() == root) {
            for (std::size_t rank = 0; rank < get_mpi_size(); rank++) {
                std::cout << "Rank: " << rank
                          << ", Displacement: " << displacements[rank]
                          << ", Recvcount: " << recvcount[rank] << std::endl;
            }
        }

        PL_MPI_IS_SUCCESS(MPI_Gatherv(
            local_.data(), local_.size(), get_mpi_type<ComplexT>(),
            data_.data(), recvcount.data(), displacements.data(),
            get_mpi_type<ComplexT>(), root,
            communicator_)); // TODO: change to Gatherv to reorder result!
        return data_;
    }

  private:
    std::size_t num_qubits_;
    std::unique_ptr<SVK> sv_;
    std::unique_ptr<SVK> recvbuf_;
    std::unique_ptr<SVK> sendbuf_;
    MPI_Comm communicator_;
    std::vector<std::size_t> mpi_rank_to_global_index_map_;

  public: // TODO: make private
    std::vector<std::size_t> global_wires_;
    std::vector<std::size_t> local_wires_;
};
}; // namespace Pennylane::LightningKokkos
