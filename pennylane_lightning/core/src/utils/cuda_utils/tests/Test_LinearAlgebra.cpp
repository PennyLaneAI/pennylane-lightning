// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <complex>
#include <cstdio>
#include <vector>

#include <catch2/catch.hpp>

#include "DataBuffer.hpp"
#include "LinearAlg.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp" // exp2
#include "cuda_helpers.hpp"

/**
 * @file
 *  Tests linear algebra functionality defined for the class
 * StateVectorCudaManaged.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Linear Algebra::SparseMV", "[Linear Algebra]", float,
                   double) {
    using ComplexT = std::complex<TestType>;
    using IdxT = typename std::conditional<std::is_same<TestType, float>::value,
                                           int32_t, int64_t>::type;

    using CFP_t =
        typename std::conditional<std::is_same<TestType, float>::value,
                                  cuFloatComplex, cuDoubleComplex>::type;

    std::size_t num_qubits = 3;
    std::size_t data_size = exp2(num_qubits);

    std::vector<ComplexT> vectors = {{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                     {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                     {0.3, 0.4}, {0.4, 0.5}};

    std::vector<CFP_t> vectors_cu;

    std::transform(vectors.begin(), vectors.end(),
                   std::back_inserter(vectors_cu),
                   [](ComplexT x) { return complexToCu<ComplexT>(x); });

    const std::vector<ComplexT> result_refs = {
        {0.2, -0.1}, {-0.1, 0.2}, {0.2, 0.1}, {0.1, 0.2},
        {0.7, -0.2}, {-0.1, 0.6}, {0.6, 0.1}, {0.2, 0.7}};

    std::vector<IdxT> indptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    std::vector<IdxT> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                 4, 7, 5, 6, 5, 6, 4, 7};
    std::vector<ComplexT> values = {
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0},
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0}};

    DataBuffer<CFP_t> sv_x(data_size);
    DataBuffer<CFP_t> sv_y(data_size);

    sv_x.CopyHostDataToGpu(vectors_cu.data(), vectors_cu.size());

    SECTION("Testing sparse matrix vector product:") {
        std::vector<CFP_t> result(data_size);
        auto cusparsehandle = make_shared_cusparse_handle();

        SparseMV_cuSparse<IdxT, TestType, CFP_t>(
            indptr.data(), static_cast<int64_t>(indptr.size()), indices.data(),
            values.data(), static_cast<int64_t>(values.size()), sv_x.getData(),
            sv_y.getData(), sv_x.getDevice(), sv_x.getStream(),
            cusparsehandle.get());

        sv_y.CopyGpuDataToHost(result.data(), result.size());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(result[j].x == Approx(real(result_refs[j])));
            CHECK(result[j].y == Approx(imag(result_refs[j])));
        }
    }
}

TEMPLATE_TEST_CASE("Linear Algebra::square_matrix_CUDA_device",
                   "[Linear Algebra]", float, double) {
    using ComplexT = std::complex<TestType>;

    using CFP_t =
        typename std::conditional<std::is_same<TestType, float>::value,
                                  cuFloatComplex, cuDoubleComplex>::type;

    std::size_t row_size = 2;

    std::vector<ComplexT> matrix = {{0.2, 0.2},
                                    {0.3, 0.3},
                                    {0.3, 0.4},
                                    {0.4, 0.5}}; // from numpy calculation

    std::vector<CFP_t> matrix_cu;

    std::transform(matrix.begin(), matrix.end(), std::back_inserter(matrix_cu),
                   [](ComplexT x) { return complexToCu<ComplexT>(x); });

    const std::vector<ComplexT> result_refs = {
        {-0.03, 0.29}, {-0.03, 0.39}, {-0.1, 0.45}, {-0.12, 0.61}};

    DataBuffer<CFP_t> mat(matrix_cu.size());

    mat.CopyHostDataToGpu(matrix_cu.data(), matrix_cu.size());

    SECTION("Testing square matrix multiplication:") {
        std::vector<CFP_t> result(matrix_cu.size());
        auto cublas_caller = make_shared_cublas_caller();

        square_matrix_CUDA_device<CFP_t>(mat.getData(), row_size, row_size,
                                         mat.getDevice(), mat.getStream(),
                                         *cublas_caller);

        mat.CopyGpuDataToHost(result.data(), result.size());

        for (std::size_t j = 0; j < matrix_cu.size(); j++) {
            CHECK(result[j].x == Approx(real(result_refs[j])));
            CHECK(result[j].y == Approx(imag(result_refs[j])));
        }
    }

    SECTION("Throwing exception for non-square matrix multiplication:") {
        std::vector<CFP_t> result(matrix_cu.size());
        auto cublas_caller = make_shared_cublas_caller();

        CHECK_THROWS_WITH(square_matrix_CUDA_device<CFP_t>(
                              mat.getData(), row_size, row_size + 1,
                              mat.getDevice(), mat.getStream(), *cublas_caller),
                          Catch::Contains("Matrix must be square."));
    }
}
