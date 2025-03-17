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

// Adapted from JET: https://github.com/XanaduAI/jet.git
// and from Lightning: https://github.com/PennylaneAI/pennylane-lightning.git
/**
 * @file cuError.hpp
 */

#pragma once
#include <string>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cusparse_v2.h>

#include "Error.hpp"
#include "Util.hpp"
// LCOV_EXCL_START

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

/// @cond DEV
#ifndef CUDA_UNSAFE

/**
 * @brief Macro that throws Exception from CUDA failure error codes.
 *
 * @param err CUDA function error-code.
 */
#define PL_CUDA_IS_SUCCESS(err)                                                \
    PL_ABORT_IF_NOT(err == cudaSuccess, cudaGetErrorString(err))

#define PL_CUBLAS_IS_SUCCESS(err)                                              \
    PL_ABORT_IF_NOT(err == CUBLAS_STATUS_SUCCESS, GetCuBlasErrorString(err))

#define PL_CUSPARSE_IS_SUCCESS(err)                                            \
    PL_ABORT_IF_NOT(err == CUSPARSE_STATUS_SUCCESS, GetCuSparseErrorString(err))

#else
#define PL_CUDA_IS_SUCCESS(err)                                                \
    { static_cast<void>(err); }
#define PL_CUBLAS_IS_SUCCESS(err)                                              \
    { static_cast<void>(err); }
#define PL_CUSPARSE_IS_SUCCESS(err)                                            \
    { static_cast<void>(err); }
#endif
/// @endcond
namespace Pennylane::LightningGPU::Util {
static const std::string GetCuBlasErrorString(const cublasStatus_t &err) {
    std::string result;
    switch (err) {
    case CUBLAS_STATUS_SUCCESS:
        result = "No errors";
        break;
    case CUBLAS_STATUS_NOT_INITIALIZED:
        result = "cuBLAS library was not initialized";
        break;
    case CUBLAS_STATUS_ALLOC_FAILED:
        result = "cuBLAS memory allocation failed";
        break;
    case CUBLAS_STATUS_INVALID_VALUE:
        result = "cuBLAS invalid value";
        break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
        result = "cuBLAS CUDA device architecture mismatch";
        break;
    case CUBLAS_STATUS_MAPPING_ERROR:
        result = "cuBLAS mapping error";
        break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
        result = "cuBLAS internal error";
        break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
        result = "cuBLAS Unsupported operation/device";
        break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
        result = "cuBLAS GPU program failed to execute";
        break;
    case CUBLAS_STATUS_LICENSE_ERROR:
        result = "cuBLAS license error";
        break;
    default:
        result = "cuBLAS status not found. Error code=" + std::to_string(err);
    }
    return result;
}

static const std::string GetCuSparseErrorString(const cusparseStatus_t &err) {
    std::string result;
    switch (err) {
    case CUSPARSE_STATUS_SUCCESS:
        result = "No errors";
        break;
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        result = "cuSparse library was not initialized";
        break;
    case CUSPARSE_STATUS_ALLOC_FAILED:
        result = "cuSparse memory allocation failed";
        break;
    case CUSPARSE_STATUS_INVALID_VALUE:
        result = "Invalid value";
        break;
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        result = "CUDA device architecture mismatch";
        break;
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        result = "The matrix type is not supported by cuSparse";
        break;
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        result = "Internal cuBLAS error";
        break;
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        result = "Unsupported operation/device";
        break;
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        result = "GPU program failed to execute";
        break;
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        result = "The resources are not sufficient to complete the operation.";
        break;
    default:
        result = "cuSPARSE status not found. Error code=" + std::to_string(err);
    }
    return result;
}

} // namespace Pennylane::LightningGPU::Util
  // LCOV_EXCL_STOP
