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
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "CSRMatrix.hpp"
#include "MPIManagerGPU.hpp"

using namespace Pennylane;
using namespace Pennylane::LightningGPU::MPI;

TEMPLATE_TEST_CASE("CSRMatrix", "[Sparse Matrix]", float, double) {
    SECTION("Default constructibility") {
        REQUIRE(std::is_constructible_v<CSRMatrix<TestType, std::size_t>>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<CSRMatrix<TestType, std::size_t>,
                                        std::size_t, std::size_t>);
    }

    SECTION("Constructibility - optional parameters") {
        REQUIRE(
            std::is_constructible_v<CSRMatrix<TestType, std::size_t>,
                                    std::size_t, std::size_t, std::size_t *,
                                    std::size_t *, std::complex<TestType> *>);
        REQUIRE(std::is_constructible_v<CSRMatrix<TestType, int32_t>,
                                        std::size_t, std::size_t, int32_t *,
                                        int32_t *, std::complex<TestType> *>);
    }
}

TEMPLATE_TEST_CASE("CRSMatrix::Split", "[CRSMatrix]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    using IndexT =
        typename std::conditional<std::is_same<TestType, float>::value, int32_t,
                                  int64_t>::type;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    IndexT csrOffsets[9] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    IndexT columns[16] = {0, 3, 1, 2, 1, 2, 0, 3, 4, 7, 5, 6, 5, 6, 4, 7};

    cp_t values[16] = {{1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
                       {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0},
                       {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
                       {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0}};

    IndexT num_csrOffsets = 9;
    IndexT num_rows = num_csrOffsets - 1;

    SECTION("Apply split") {
        if (rank == 0) {
            auto CSRMatVector = splitCSRMatrix<TestType, IndexT>(
                mpi_manager, num_rows, csrOffsets, columns, values);

            std::vector<IndexT> localcsrOffsets = {0, 2, 4, 6, 8};
            std::vector<IndexT> local_indices = {0, 3, 1, 2, 1, 2, 0, 3};

            for (std::size_t i = 0; i < localcsrOffsets.size(); i++) {
                CHECK(CSRMatVector[0][0].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
                CHECK(CSRMatVector[1][1].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
            }

            for (std::size_t i = 0; i < local_indices.size(); i++) {
                CHECK(CSRMatVector[0][0].getColumns()[i] == local_indices[i]);
                CHECK(CSRMatVector[1][1].getColumns()[i] == local_indices[i]);
            }

            for (std::size_t i = 0; i < 8; i++) {
                CHECK(CSRMatVector[0][0].getValues()[i] == values[i]);
                CHECK(CSRMatVector[1][1].getValues()[i] == values[i + 8]);
            }

            CHECK(CSRMatVector[0][1].getValues().size() == 0);
            CHECK(CSRMatVector[1][0].getValues().size() == 0);
        }
    }

    SECTION("Apply SparseMatrix scatter") {
        std::vector<std::vector<CSRMatrix<TestType, IndexT>>> csrmatrix_blocks;

        if (rank == 0) {
            csrmatrix_blocks = splitCSRMatrix<TestType, IndexT>(
                mpi_manager, num_rows, csrOffsets, columns, values);
        }

        std::size_t local_num_rows = num_rows / size;

        std::vector<CSRMatrix<TestType, IndexT>> localCSRMatVector;
        for (std::size_t i = 0; i < mpi_manager.getSize(); i++) {
            auto localCSRMat = scatterCSRMatrix<TestType, IndexT>(
                mpi_manager, csrmatrix_blocks[i], local_num_rows, 0);
            localCSRMatVector.push_back(localCSRMat);
        }

        std::vector<IndexT> localcsrOffsets = {0, 2, 4, 6, 8};
        std::vector<IndexT> local_indices = {0, 3, 1, 2, 1, 2, 0, 3};

        if (rank == 0) {
            for (std::size_t i = 0; i < localcsrOffsets.size(); i++) {
                CHECK(localCSRMatVector[0].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
            }

            for (std::size_t i = 0; i < local_indices.size(); i++) {
                CHECK(localCSRMatVector[0].getColumns()[i] == local_indices[i]);
            }

            for (std::size_t i = 0; i < 8; i++) {
                CHECK(localCSRMatVector[0].getValues()[i] == values[i]);
            }

            CHECK(localCSRMatVector[1].getValues().size() == 0);
        } else {
            for (std::size_t i = 0; i < localcsrOffsets.size(); i++) {
                CHECK(localCSRMatVector[1].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
            }

            for (std::size_t i = 0; i < local_indices.size(); i++) {
                CHECK(localCSRMatVector[1].getColumns()[i] == local_indices[i]);
            }

            for (std::size_t i = 0; i < 8; i++) {
                CHECK(localCSRMatVector[1].getValues()[i] == values[i + 8]);
            }

            CHECK(localCSRMatVector[0].getValues().size() == 0);
        }
    }
}
