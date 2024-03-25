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
#include <iostream>

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
#include "Gates.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorKokkos.hpp"
#include "UtilLinearAlg.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Functors;
using namespace Pennylane::Util;
using Pennylane::Gates::GateOperation;
using Pennylane::Gates::GeneratorOperation;
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
    using UnmanagedComplexHostView = SVK::UnmanagedComplexHostView;
    using UnmanagedConstComplexHostView = SVK::UnmanagedConstComplexHostView;
    // using UnmanagedSizeTHostView =
    //     Kokkos::View<size_t *, Kokkos::HostSpace,
    //                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
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
            sv_ = std::make_unique<SVK>(get_num_local_wires(), kokkos_args);
            recvbuf_ =
                std::make_unique<SVK>(get_num_local_wires(), kokkos_args);
            sendbuf_ = KokkosVector("sendbuf_", get_blk_size());
            setBasisState(0U);
        }
    };

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
    void mpi_irecv(const std::size_t source, MPI_Request &request) {
        KokkosVector sv_view = (*recvbuf_).getView();
        PL_MPI_IS_SUCCESS(
            MPI_Irecv(sv_view.data(), sv_view.size(), get_mpi_type<ComplexT>(),
                      static_cast<int>(source), 0, communicator_, &request));
    }

    /**
     * @brief  Sends local state vector to another MPI-process.
     *
     * @param  dest Destination MPI rank.
     * @param  request MPI request allowing to validate the transfer has been
     * completed.
     * @param  copy If true, copy the state vector data in sendbuf_ before
     * sending.
     */
    void mpi_isend(const std::size_t dest, MPI_Request &request,
                   const bool copy = false) {
        if (copy) {
            KokkosVector sv_view =
                getView(); // circumvent error capturing this with KOKKOS_LAMBDA
            Kokkos::parallel_for(
                sv_view.size(), KOKKOS_LAMBDA(const std::size_t i) {
                    sendbuf_(i) = sv_view(i);
                });
            Kokkos::fence();
        }
        PL_MPI_IS_SUCCESS(MPI_Isend(
            sendbuf_.data(), sendbuf_.size(), get_mpi_type<ComplexT>(),
            static_cast<int>(dest), 0, communicator_, &request));
    }

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

    /**
     * @brief  Returns the matrix-column/row corresponding to a given
     * MPI-process.
     *
     * @param rank MPI-process rank.
     * @param wires Wire indices.
     */
    std::size_t rank_2_matrix_index(const std::size_t rank,
                                    const std::vector<std::size_t> &wires) {
        constexpr std::size_t one{1U};
        auto n = wires.size();
        std::size_t index{0U};
        for (std::size_t i = 0; i < wires.size(); i++) {
            index = index | (((rank & (one << wires[i])) >> wires[i])
                             << (n - 1 - i)); // select ith bit
        }
        return index;
    }

    /**
     * @brief  Returns the matrix-columns/rows corresponding to a given
     * MPI-process.
     *
     * @param rank MPI-process rank.
     * @param wires Wire indices.
     * @param local_wires Mask indicating local wires.
     */
    std::vector<std::size_t>
    rank_2_matrix_indices(const std::size_t rank,
                          const std::vector<std::size_t> &wires,
                          const std::vector<bool> &local_wires) {
        constexpr std::size_t one{1U};
        PL_ABORT_IF_NOT(wires.size() == local_wires.size(), "")
        auto n = wires.size();
        std::size_t nlw =
            std::count(local_wires.begin(), local_wires.end(), true);
        std::vector<std::size_t> indices;
        indices.reserve(nlw);
        std::size_t index{0U};
        for (std::size_t i = 0; i < wires.size(); i++) {
            if (local_wires[i]) {
                continue;
            }
            index = index | (((rank & (one << wires[i])) >> wires[i])
                             << (n - 1 - i)); // select ith bit
        }
        indices.push_back(index);
        for (std::size_t i = 0; i < wires.size(); i++) {
            if (local_wires[i]) {
                auto ub = indices.size();
                for (std::size_t j = 0; j < ub; j++) {
                    indices.push_back(indices[j] ^ (one << (n - 1 - i)));
                }
            }
        }
        return indices;
    }

    /********************
    Wires-related methods
    ********************/

    /**
     * @brief  Returns the number of global wires.
     */
    std::size_t get_num_global_wires() {
        return log2(static_cast<std::size_t>(get_mpi_size()));
    }

    /**
     * @brief  Returns the number of local wires.
     */
    std::size_t get_num_local_wires() {
        return num_qubits_ - get_num_global_wires();
    }

    /**
     * @brief  Returns the number of local wires.
     */
    std::size_t get_rev_wire(const std::size_t wire) {
        return num_qubits_ - 1 - wire;
    }

    /**
     * @brief  Shifts wire indices by the number of global wires to yield local
     * wire indices (global wires have a negative index).
     *
     * @param wires Wire indices.
     */
    std::vector<std::size_t>
    get_local_wires_indices(const std::vector<std::size_t> &wires) {
        std::vector<std::size_t> local_wires(wires.size());
        auto n_global{get_num_global_wires()};
        std::transform(wires.begin(), wires.end(), local_wires.begin(),
                       [=](const std::size_t wire) { return wire - n_global; });
        return local_wires;
    }

    /**
     * @brief  Returns an array containing solely valid local wire indices
     * (global wires are pruned).
     *
     * @param wires Wire indices.
     */
    std::vector<std::size_t>
    get_local_wires(const std::vector<std::size_t> &wires) {
        return get_local_wires_indices(prune_global_wires(wires));
    }

    /**
     * @brief  Prunes global wires from an array of global wire indices.
     *
     * @param wires Wire indices.
     */
    std::vector<std::size_t>
    prune_global_wires(const std::vector<std::size_t> &wires) {
        std::vector<std::size_t> local_wires;
        local_wires.reserve(wires.size());
        auto n_global{get_num_global_wires()};
        for (auto e : wires) {
            if (e >= n_global) {
                local_wires.push_back(e);
            }
        }
        return local_wires;
    }

    /**
     * @brief  Returns true if all wires are local and false otherwise.
     *
     * @param wires Wire indices.
     */
    bool is_wires_local(const std::vector<std::size_t> &wires) {
        auto n_local{get_num_local_wires()};
        return std::find_if(wires.begin(), wires.end(),
                            [=, this](const std::size_t wire) {
                                return get_rev_wire(wire) > (n_local - 1);
                            }) == wires.end();
    }

    /**
     * @brief  Returns true if any wire is local and false otherwise.
     *
     * @param wires Wire indices.
     */
    bool has_local_wires(const std::vector<std::size_t> &wires) {
        auto n_local{get_num_local_wires()};
        return std::find_if(wires.begin(), wires.end(),
                            [=, this](const std::size_t wire) {
                                return get_rev_wire(wire) <= (n_local - 1);
                            }) != wires.end();
    }

    /**
     * @brief  Flip the endianness of global wires.
     *
     * @param wires Wire indices.
     */
    std::vector<std::size_t>
    get_global_rev_wires(const std::vector<std::size_t> &wires) {
        std::vector<std::size_t> rev_wires(wires.size());
        for (std::size_t i = 0; i < wires.size(); i++) {
            rev_wires[i] = get_num_global_wires() - 1 - wires[i];
        }
        return rev_wires;
    }

    /**
     * @brief  Returns a mask indicating local wires.
     *
     * @param wires Wire indices.
     */
    std::vector<bool>
    get_local_wire_mask(const std::vector<std::size_t> &wires) {
        std::vector<bool> local_wire_mask(wires.size());
        auto n_global{get_num_global_wires()};
        std::transform(
            wires.begin(), wires.end(), local_wire_mask.begin(),
            [=](const std::size_t wire) { return wire >= n_global; });
        return local_wire_mask;
    }

    /**
     * @brief  Returns the sub-matrix corresponding to given rows and columns.
     *
     * @param matrix The matrix.
     * @param rows Row indices.
     * @param cols Column indices.
     */
    template <typename T>
    std::vector<T> select_sub_matrix(const std::vector<T> &matrix,
                                     const std::vector<std::size_t> &rows,
                                     const std::vector<std::size_t> &cols) {
        auto n = std::sqrt(matrix.size());
        auto nsc = cols.size();
        auto nsr = rows.size();
        std::vector<T> sub_matrix(nsr * nsc);
        for (std::size_t i = 0; i < nsr; i++) {
            for (std::size_t j = 0; j < nsc; j++) {
                sub_matrix[i * nsr + j] = matrix[rows[i] * n + cols[j]];
            }
        }
        return sub_matrix;
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
        auto rank = static_cast<std::size_t>(get_mpi_rank());
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
    template <class complex>
    StateVectorKokkosMPI(complex *hostdata_, const std::size_t length,
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(log2(length), kokkos_args) {
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
                         const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkosMPI(log2(length), kokkos_args) {
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
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires,
                        const bool inverse = false,
                        const std::vector<PrecisionT> &params = {},
                        const std::vector<ComplexT> &gate_matrix = {}) {
        if (opName == "Identity") {
            return;
        }
        if (is_wires_local(wires)) {
            (*sv_).applyOperation(opName, get_local_wires_indices(wires),
                                  inverse, params, gate_matrix);
            return;
        }
        if (wires.size() == 1) {
            apply1QOperation(opName, wires, inverse, params, gate_matrix);
            return;
        }
        if (wires.size() == 2) {
            apply2QOperation(opName, wires, inverse, params, gate_matrix);
            return;
        }
        PL_ABORT("applyOperation is not implemented on many global wires.");
    }

    /**
     * @brief Apply a single 1-qubit gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void apply1QOperation(const std::string &opName,
                          const std::vector<size_t> &wires,
                          const bool inverse = false,
                          const std::vector<PrecisionT> &params = {},
                          const std::vector<ComplexT> &gate_matrix = {}) {
        constexpr std::size_t one{1U};
        PL_ABORT_IF_NOT(wires.size() == one,
                        "Wires must contain a single wire index.")
        PL_ABORT_IF(has_local_wires(wires), "Target wires must be global.")
        std::vector<ComplexT> matrix(exp2(wires.size() * 2));
        if (array_contains(gate_names, std::string_view{opName})) {
            auto gate_op = reverse_lookup(gate_names, std::string_view{opName});
            matrix = Pennylane::Gates::getMatrix<Kokkos::complex, PrecisionT>(
                gate_op, params, inverse);
        } else {
            PL_ABORT_IF_NOT(
                gate_matrix.size() == matrix.size(),
                std::string("Operation does not exist for ") + opName +
                    std::string(" and/or incorrect matrix provided."));
            matrix = (inverse) ? transpose(gate_matrix, true) : gate_matrix;
        }
        const auto ncol = exp2(wires.size());
        const auto myrank = static_cast<std::size_t>(get_mpi_rank());
        const auto rev_wires = get_global_rev_wires(wires);
        const auto myrow = rank_2_matrix_index(myrank, rev_wires);
        auto rank = myrank ^ (one << rev_wires[0]); // toggle global bit

        // Initiate data transfer
        MPI_Request send_req;
        MPI_Request recv_req;
        mpi_isend(rank, send_req, true);
        mpi_irecv(rank, recv_req);
        auto col = myrow;
        (*sv_).rescale(matrix[col + myrow * ncol]);

        // Data has arrived, accumulate dot product
        mpi_wait(recv_req);
        col = rank_2_matrix_index(rank, rev_wires);
        (*sv_).axpby(matrix[col + myrow * ncol], (*recvbuf_).getView());
        mpi_wait(send_req);
    }

    /**
     * @brief Apply a single 2-qubit gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void apply2QOperation(const std::string &opName,
                          const std::vector<size_t> &wires,
                          const bool inverse = false,
                          const std::vector<PrecisionT> &params = {},
                          const std::vector<ComplexT> &gate_matrix = {}) {
        constexpr std::size_t two{2U};
        PL_ABORT_IF_NOT(wires.size() == two,
                        "Wires must contain a single wire index.")
        if (has_local_wires(wires)) {
            applySemiLocal2QOperation(opName, wires, inverse, params,
                                      gate_matrix);
        } else {
            applyGlobal2QOperation(opName, wires, inverse, params, gate_matrix);
        }
    }
    /**
     * @brief Apply a single 2-qubit gate with semi-local indices to the state
     * vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applySemiLocal2QOperation(
        const std::string &opName, const std::vector<size_t> &wires,
        const bool inverse = false, const std::vector<PrecisionT> &params = {},
        const std::vector<ComplexT> &gate_matrix = {}) {
        constexpr std::size_t one{1U};
        constexpr std::size_t two{2U};
        PL_ABORT_IF_NOT(wires.size() == two,
                        "Wires must contain a single wire index.")
        std::vector<ComplexT> matrix(exp2(wires.size() * 2));
        if (array_contains(gate_names, std::string_view{opName})) {
            auto gate_op = reverse_lookup(gate_names, std::string_view{opName});
            matrix = Pennylane::Gates::getMatrix<Kokkos::complex, PrecisionT>(
                gate_op, params, inverse);
        } else {
            PL_ABORT_IF_NOT(
                gate_matrix.size() == matrix.size(),
                std::string("Operation does not exist for ") + opName +
                    std::string(" and/or incorrect matrix provided."));
            matrix = (inverse) ? transpose(gate_matrix, true) : gate_matrix;
        }
        const auto local_wires = prune_global_wires(wires);
        const auto rev_wires = get_global_rev_wires(wires);
        const auto loc_wire_mask = get_local_wire_mask(wires);
        const auto myrank = static_cast<std::size_t>(get_mpi_rank());
        const auto myrows =
            rank_2_matrix_indices(myrank, rev_wires, loc_wire_mask);
        auto cols = myrows;
        auto rank = myrank ^ (one << ((is_wires_local({wires[1]}))
                                          ? rev_wires[0]
                                          : rev_wires[1])); // toggle global bit
        // Initiate data transfer
        MPI_Request send_req;
        MPI_Request recv_req;
        mpi_isend(rank, send_req, true);
        mpi_irecv(rank, recv_req);
        auto sub_matrix = select_sub_matrix(matrix, myrows, cols);
        applyOperation("Matrix", local_wires, false, {}, sub_matrix);

        // Data has arrived, accumulate dot product
        cols = rank_2_matrix_indices(rank, rev_wires, loc_wire_mask);
        sub_matrix = select_sub_matrix(matrix, myrows, cols);
        mpi_wait(recv_req);
        (*recvbuf_).applyOperation("Matrix", get_local_wires(wires), false, {},
                                   sub_matrix);
        (*sv_).axpby(Kokkos::complex{1.0, 0.0}, (*recvbuf_).getView());
        mpi_wait(send_req);
    }

    /**
     * @brief Apply a single 2-qubit gate with global indices to the state
     * vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applyGlobal2QOperation(const std::string &opName,
                                const std::vector<size_t> &wires,
                                const bool inverse = false,
                                const std::vector<PrecisionT> &params = {},
                                const std::vector<ComplexT> &gate_matrix = {}) {
        constexpr std::size_t one{1U};
        constexpr std::size_t two{2U};
        PL_ABORT_IF_NOT(wires.size() == two,
                        "Wires must contain a single wire index.")
        PL_ABORT_IF(has_local_wires(wires), "Target wires must be global.")
        std::vector<ComplexT> matrix(exp2(wires.size() * 2));
        if (array_contains(gate_names, std::string_view{opName})) {
            auto gate_op = reverse_lookup(gate_names, std::string_view{opName});
            matrix = Pennylane::Gates::getMatrix<Kokkos::complex, PrecisionT>(
                gate_op, params, inverse);
        } else {
            PL_ABORT_IF_NOT(
                gate_matrix.size() == matrix.size(),
                std::string("Operation does not exist for ") + opName +
                    std::string(" and/or incorrect matrix provided."));
            matrix = (inverse) ? transpose(gate_matrix, true) : gate_matrix;
        }
        const auto ncol = exp2(wires.size());
        const auto myrank = static_cast<std::size_t>(get_mpi_rank());
        const auto rev_wires = get_global_rev_wires(wires);
        const auto myrow = rank_2_matrix_index(myrank, rev_wires);
        auto rank = myrank ^ (one << rev_wires[0]); // toggle 1st global bit

        // Initiate data transfer
        MPI_Request send_req;
        MPI_Request recv_req;
        mpi_isend(rank, send_req, true);
        mpi_irecv(rank, recv_req);
        auto col = myrow;
        (*sv_).rescale(matrix[col + myrow * ncol]);

        // Data has arrived, accumulate dot product
        mpi_wait(recv_req);
        col = rank_2_matrix_index(rank, rev_wires);
        (*sv_).axpby(matrix[col + myrow * ncol], (*recvbuf_).getView());
        rank = myrank ^ (one << rev_wires[1]); // toggle 2nd global bit
        mpi_irecv(rank, recv_req);
        mpi_wait(send_req);
        mpi_isend(rank, send_req);

        // Data has arrived, accumulate dot product
        mpi_wait(recv_req);
        col = rank_2_matrix_index(rank, rev_wires);
        (*sv_).axpby(matrix[col + myrow * ncol], (*recvbuf_).getView());
        rank = myrank ^ (one << rev_wires[1]) ^
               (one << rev_wires[0]); // toggle both global bit
        mpi_irecv(rank, recv_req);
        mpi_wait(send_req);
        mpi_isend(rank, send_req);

        // Data has arrived, accumulate dot product
        mpi_wait(recv_req);
        col = rank_2_matrix_index(rank, rev_wires);
        (*sv_).axpby(matrix[col + myrow * ncol], (*recvbuf_).getView());
        mpi_wait(send_req);
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
    [[nodiscard]] auto getDataVector(const int root = 0)
        -> std::vector<ComplexT> {
        std::vector<ComplexT> data_((get_mpi_rank() == root) ? this->getLength()
                                                             : 0);
        std::vector<ComplexT> local_((*sv_).getLength());
        (*sv_).DeviceToHost(local_.data(), local_.size());
        PL_MPI_IS_SUCCESS(MPI_Gather(local_.data(), local_.size(),
                                     get_mpi_type<ComplexT>(), data_.data(),
                                     local_.size(), get_mpi_type<ComplexT>(),
                                     root, communicator_));
        return data_;
    }

  private:
    std::size_t num_qubits_;
    std::unique_ptr<SVK> sv_;
    std::unique_ptr<SVK> recvbuf_;
    KokkosVector sendbuf_;
    MPI_Comm communicator_;
};

}; // namespace Pennylane::LightningKokkos
