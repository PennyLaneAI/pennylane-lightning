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

#ifdef _ENABLE_PLGPU
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <custatevec.h>
#endif

#include "MPIManager.hpp"
#include "DataBuffer.hpp"
#include "Error.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Util {
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

template <typename T> auto cppTypeToString() -> const std::string {
    const std::string typestr = std::type_index(typeid(T)).name();
    return typestr;
}

/**
 * @brief MPI operation class. Maintains MPI related operations.
 */
class MPIManagerGPU final : public MPIManager {
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
#ifdef _ENABLE_PLGPU
        {cppTypeToString<custatevecIndex_t>(), MPI_INT64_T},
#endif
        // cuda related types
        {cppTypeToString<cudaIpcMemHandle_t>(), MPI_UINT8_T},
        {cppTypeToString<cudaIpcEventHandle_t>(), MPI_UINT8_T}};

    /**
     * @brief Find C++ data type's corresponding MPI data type.
     *
     * @tparam T C++ data type.
     */
    template <typename T> auto getMPIDatatype() -> MPI_Datatype {
        auto it = cpp_mpi_type_map_with_cuda.find(cppTypeToString<T>());
        if (it != cpp_mpi_type_map_with_cuda.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Type not supported");
        }
    }

    
    public:
    MPIManagerGPU(MPI_Comm communicator = MPI_COMM_WORLD) : MPIManager(communicator) {}

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
                throw std::runtime_error(
                    "Unsupported MPI DataType implementation.\n");
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

};
} // namespace Pennylane::LightningGPU::Util
