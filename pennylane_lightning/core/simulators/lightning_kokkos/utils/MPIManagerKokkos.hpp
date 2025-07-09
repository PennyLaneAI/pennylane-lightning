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

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <bit>
#include <complex>
#include <cstring>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "Error.hpp"
#include "MPIManager.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Util {
/**
 * @brief MPI operation class for Lightning Kokkos. Maintains MPI related
 * operations.
 */
class MPIManagerKokkos final : public MPIManager {
    /**
     * @brief Map of std::string and MPI_Datatype.
     */
    std::unordered_map<std::string, MPI_Datatype> cpp_mpi_type_map_with_kokkos =
        {
            {cppTypeToString<char>(), MPI_CHAR},
            {cppTypeToString<signed char>(), MPI_SIGNED_CHAR},
            {cppTypeToString<unsigned char>(), MPI_UNSIGNED_CHAR},
            {cppTypeToString<wchar_t>(), MPI_WCHAR},
            {cppTypeToString<short>(), MPI_SHORT},
            {cppTypeToString<unsigned short>(), MPI_UNSIGNED_SHORT},
            {cppTypeToString<int>(), MPI_INT},
            {cppTypeToString<unsigned int>(), MPI_UNSIGNED},
            {cppTypeToString<long>(), MPI_LONG},
            {cppTypeToString<unsigned long>(), MPI_UNSIGNED_LONG},
            {cppTypeToString<long long>(), MPI_LONG_LONG_INT},
            {cppTypeToString<float>(), MPI_FLOAT},
            {cppTypeToString<double>(), MPI_DOUBLE},
            {cppTypeToString<long double>(), MPI_LONG_DOUBLE},
            {cppTypeToString<int8_t>(), MPI_INT8_T},
            {cppTypeToString<int16_t>(), MPI_INT16_T},
            {cppTypeToString<int32_t>(), MPI_INT32_T},
            {cppTypeToString<int64_t>(), MPI_INT64_T},
            {cppTypeToString<uint8_t>(), MPI_UINT8_T},
            {cppTypeToString<uint16_t>(), MPI_UINT16_T},
            {cppTypeToString<uint32_t>(), MPI_UINT32_T},
            {cppTypeToString<uint64_t>(), MPI_UINT64_T},
            {cppTypeToString<bool>(), MPI_C_BOOL},
            {cppTypeToString<std::complex<float>>(), MPI_C_FLOAT_COMPLEX},
            {cppTypeToString<std::complex<double>>(), MPI_C_DOUBLE_COMPLEX},
            {cppTypeToString<std::complex<long double>>(),
             MPI_C_LONG_DOUBLE_COMPLEX},
            {cppTypeToString<Kokkos::complex<float>>(), MPI_C_FLOAT_COMPLEX},
            {cppTypeToString<Kokkos::complex<double>>(), MPI_C_DOUBLE_COMPLEX},
        };

    auto get_cpp_mpi_type_map() const
        -> const std::unordered_map<std::string, MPI_Datatype> & override {
        return cpp_mpi_type_map_with_kokkos;
    }

  public:
    MPIManagerKokkos(MPI_Comm communicator = MPI_COMM_WORLD)
        : MPIManager(communicator) {}

    MPIManagerKokkos(int argc, char **argv) : MPIManager(argc, argv) {}

    using MPIManager::Bcast;

    /**
     * @brief MPI_Sendrecv wrapper for Kokkos::Views.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer Kokkos::View.
     * @param dest Rank of destination.
     * @param recvBuf Receive buffer Kokkos::View.
     * @param source Rank of source.
     * @param size Number of elements of the data to send/receive.
     * @param tag Tag for the MPI message.
     */
    template <typename T>
    void Sendrecv(Kokkos::View<T *> &sendBuf, std::size_t dest,
                  Kokkos::View<T *> &recvBuf, std::size_t source,
                  std::size_t size, std::size_t tag = 0) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Status status;
        int sendtag = static_cast<int>(tag);
        int recvtag = static_cast<int>(tag);
        int destInt = static_cast<int>(dest);
        int sourceInt = static_cast<int>(source);
        int sizeInt = static_cast<int>(size);
        PL_MPI_IS_SUCCESS(MPI_Sendrecv(
            sendBuf.data(), sizeInt, datatype, destInt, sendtag, recvBuf.data(),
            sizeInt, datatype, sourceInt, recvtag, this->getComm(), &status));
    }

    /**
     * @brief MPI_AllGatherV wrapper for Kokkos::Views.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer Kokkos::View.
     * @param recvBuf Receive buffer Kokkos::View.
     * @param recvCounts Number of elements received from each rank.
     * @param displacements Elements shifted from each rank for gather.
     */
    template <typename T>
    void AllGatherV(Kokkos::View<T *> &sendBuf, Kokkos::View<T *> &recvBuf,
                    std::vector<int> &recvCounts,
                    std::vector<int> &displacements) {
        MPI_Datatype datatype = getMPIDatatype<T>();

        PL_MPI_IS_SUCCESS(
            MPI_Allgatherv(sendBuf.data(), sendBuf.size(), datatype,
                           recvBuf.data(), recvCounts.data(),
                           displacements.data(), datatype, this->getComm()));
    }

    /**
     * @brief MPI_Bcast wrapper for Kokkos::Views.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer Kokkos::View.
     * @param root Rank of broadcast root.
     */
    template <typename T>
    void Bcast(Kokkos::View<T *> &sendBuf, std::size_t root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        int rootInt = static_cast<int>(root);
        PL_MPI_IS_SUCCESS(MPI_Bcast(sendBuf.data(), sendBuf.size(), datatype,
                                    rootInt, this->getComm()));
    }
};
} // namespace Pennylane::LightningKokkos::Util
