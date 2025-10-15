// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <custatevec.h>

#include "DataBuffer.hpp"
#include "Error.hpp"
#include "MPIManager.hpp" // cppTypeToString

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Util {
/**
 * @brief MPI operation class for Lightning GPU. Maintains MPI related
 * operations.
 */
class MPIManagerGPU final : public MPIManager {
  private:
    /**
     * @brief Map of std::string and MPI_Datatype.
     */
    std::unordered_map<std::string, MPI_Datatype> cpp_mpi_type_map_with_cuda = {
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
        {cppTypeToString<float2>(), MPI_C_FLOAT_COMPLEX},
        {cppTypeToString<cuComplex>(), MPI_C_FLOAT_COMPLEX},
        {cppTypeToString<cuFloatComplex>(), MPI_C_FLOAT_COMPLEX},
        {cppTypeToString<double2>(), MPI_C_DOUBLE_COMPLEX},
        {cppTypeToString<cuDoubleComplex>(), MPI_C_DOUBLE_COMPLEX},
        {cppTypeToString<custatevecIndex_t>(), MPI_INT64_T},
        // cuda related types
        {cppTypeToString<cudaIpcMemHandle_t>(), MPI_UINT8_T},
        {cppTypeToString<cudaIpcEventHandle_t>(), MPI_UINT8_T}};

  public:
    MPIManagerGPU(MPI_Comm communicator = MPI_COMM_WORLD)
        : MPIManager(communicator) {}

    MPIManagerGPU(int argc, char **argv) : MPIManager(argc, argv) {}

    auto get_cpp_mpi_type_map() const
        -> const std::unordered_map<std::string, MPI_Datatype> & override {
        return cpp_mpi_type_map_with_cuda;
    }

    using MPIManager::Allgather;
    using MPIManager::Reduce;
    /**
     * @brief MPI_Allgather wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param recvBuf Receive buffer vector.
     * @param sendCount Number of elements received from any process.
     */
    template <typename T>
    void Allgather(T &sendBuf, std::vector<T> &recvBuf,
                   std::size_t sendCount = 1) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        if (sendCount != 1) {
            if (cppTypeToString<T>() != cppTypeToString<cudaIpcMemHandle_t>() &&
                cppTypeToString<T>() !=
                    cppTypeToString<cudaIpcEventHandle_t>()) {
                PL_ABORT("Unsupported MPI DataType implementation.\n");
            }
        }
        PL_ABORT_IF(recvBuf.size() != this->getSize(),
                    "Incompatible size of sendBuf and recvBuf.");

        int sendCountInt = static_cast<int>(sendCount);
        PL_MPI_IS_SUCCESS(MPI_Allgather(&sendBuf, sendCountInt, datatype,
                                        recvBuf.data(), sendCountInt, datatype,
                                        this->getComm()));
    }

    /**
     * @brief MPI_Reduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer (DataBuffer type).
     * @param recvBuf Receive buffer (DataBuffer type).
     * @param root Rank of root process.
     * @param op_str String of MPI_Op.
     */
    template <typename T>
    void Reduce(DataBuffer<T> &sendBuf, DataBuffer<T> &recvBuf,
                std::size_t length, std::size_t root,
                const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Reduce(sendBuf.getData(), recvBuf.getData(),
                                     length, datatype, op, root,
                                     this->getComm()));
    }

    /**
     * @brief Creates new MPIManager based on colors and keys.
     *
     * @param color Processes with the same color are in the same new
     * communicator.
     * @param key Rank assignment control.
     * @return new MPIManager object.
     */
    auto split(std::size_t color, std::size_t key) -> MPIManagerGPU {
        MPI_Comm newcomm;
        int colorInt = static_cast<int>(color);
        int keyInt = static_cast<int>(key);
        PL_MPI_IS_SUCCESS(
            MPI_Comm_split(this->getComm(), colorInt, keyInt, &newcomm));
        return MPIManagerGPU(newcomm);
    }
};
} // namespace Pennylane::LightningGPU::Util
