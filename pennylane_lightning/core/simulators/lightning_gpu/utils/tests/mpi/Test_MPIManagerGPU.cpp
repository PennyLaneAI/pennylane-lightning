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
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MPIManagerGPU.hpp"

using namespace Pennylane;
using namespace Pennylane::LightningGPU::Util;

TEST_CASE("MPIManagerGPU ctor", "[MPIManagerGPU]") {
    SECTION("Default constructibility") {
        REQUIRE(std::is_constructible_v<MPIManagerGPU>);
    }

    SECTION("Construct with MPI_Comm") {
        REQUIRE(std::is_constructible_v<MPIManagerGPU, MPI_Comm>);
    }

    SECTION("Construct with args") {
        REQUIRE(std::is_constructible_v<MPIManagerGPU, int, char **>);
    }

    SECTION("MPIManagerGPU {MPIManagerGPU&}") {
        REQUIRE(std::is_copy_constructible_v<MPIManagerGPU>);
    }
}

TEST_CASE("MPIManagerGPU::getMPIDatatype", "[MPIManagerGPU]") {
    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);

    SECTION("Test valid type") {
        MPI_Datatype datatype = mpi_manager.getMPIDatatype<char>();
        REQUIRE(datatype == MPI_CHAR);
    }

    SECTION("Test invalid type") {
        // This should throw an exception
        REQUIRE_THROWS_WITH(mpi_manager.getMPIDatatype<std::string>(),
                            Catch::Matchers::Contains("Type not supported"));
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Scatter int", "[MPIManagerGPU]", std::size_t,
                   int) {
    using PrecisionT = TestType;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();
    int root = 0;

    SECTION("Apply scatter") {
        std::vector<PrecisionT> sendBuf(size);
        PrecisionT result(2 * rank);
        if (rank == root) {
            for (std::size_t i = 0; i < sendBuf.size(); i++) {
                PrecisionT data(2 * i);
                sendBuf[i] = data;
            }
        }

        auto recvBuf = mpi_manager.scatter<PrecisionT>(sendBuf, root);
        CHECK(recvBuf[0] == result);
    }

    SECTION("Apply Scatter") {
        std::vector<PrecisionT> sendBuf(size);
        std::vector<PrecisionT> recvBuf(1);
        PrecisionT result(2 * rank);
        if (rank == root) {
            for (std::size_t i = 0; i < sendBuf.size(); i++) {
                PrecisionT data(2 * i);
                sendBuf[i] = data;
            }
        }

        mpi_manager.Scatter<PrecisionT>(sendBuf, recvBuf, root);
        CHECK(recvBuf[0] == result);
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Scatter Complex", "[MPIManagerGPU]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();
    int root = 0;

    SECTION("Apply Scatter") {
        std::vector<cp_t> sendBuf(size);
        cp_t result(2.0 * rank, 2.0 * rank + 1);
        if (rank == root) {
            for (std::size_t i = 0; i < sendBuf.size(); i++) {
                cp_t data(2.0 * i, 2.0 * i + 1);
                sendBuf[i] = data;
            }
        }

        auto recvBuf = mpi_manager.scatter<cp_t>(sendBuf, root);
        CHECK(recvBuf[0].real() == result.real());
        CHECK(recvBuf[0].imag() == result.imag());
    }

    SECTION("Apply Scatter") {
        std::vector<cp_t> sendBuf(size);
        std::vector<cp_t> recvBuf(1);
        cp_t result(2.0 * rank, 2.0 * rank + 1);
        if (rank == root) {
            for (std::size_t i = 0; i < sendBuf.size(); i++) {
                cp_t data(2.0 * i, 2.0 * i + 1);
                sendBuf[i] = data;
            }
        }

        mpi_manager.Scatter<cp_t>(sendBuf, recvBuf, root);
        CHECK(recvBuf[0].real() == result.real());
        CHECK(recvBuf[0].imag() == result.imag());
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Allgather", "[MPIManagerGPU]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Allgather scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        std::vector<cp_t> recvBuf(size);

        mpi_manager.Allgather<cp_t>(sendBuf, recvBuf);

        for (std::size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply Allgather vector") {
        std::vector<cp_t> sendBuf(1,
                                  {(static_cast<PrecisionT>(rank) + 1) * 2, 0});
        std::vector<cp_t> recvBuf(mpi_manager.getSize());

        mpi_manager.Allgather<cp_t>(sendBuf, recvBuf);

        for (std::size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>((i + 1) * 2));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply allgather scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};

        auto recvBuf = mpi_manager.allgather<cp_t>(sendBuf);
        for (std::size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply allgather vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        auto recvBuf = mpi_manager.allgather<cp_t>(sendBuf);

        for (std::size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Reduce", "[MPIManagerGPU]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Reduce vector - sum") {
        std::vector<cp_t> sendBuf(1,
                                  {(static_cast<PrecisionT>(rank) + 1) * 2, 0});
        std::vector<cp_t> recvBuf(1, {0, 0});

        mpi_manager.Reduce<cp_t>(sendBuf, recvBuf, 0, "sum");

        if (mpi_manager.getRank() == 0) {
            CHECK(recvBuf[0].real() ==
                  static_cast<PrecisionT>((size + 1) * size));
            CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply Reduce vector - prod") {
        std::vector<cp_t> sendBuf(1, {2, 0});
        std::vector<cp_t> recvBuf(1, {0, 0});

        mpi_manager.Reduce<cp_t>(sendBuf, recvBuf, 0, "prod");

        if (mpi_manager.getRank() == 0) {
            CHECK(recvBuf[0].real() == exp2(size));
            CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Catch failures caused by unsupported ops") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        std::vector<cp_t> recvBuf(1, {0, 0});
        REQUIRE_THROWS_WITH(
            mpi_manager.Reduce<cp_t>(sendBuf, recvBuf, 0, "SUM"),
            Catch::Matchers::Contains("Op not supported"));
    }

    SECTION("Catch failures caused by unsupported ops") {
        std::vector<std::string> sendBuf(1, "test");
        std::vector<std::string> recvBuf(1, "test");
        REQUIRE_THROWS_WITH(
            mpi_manager.Reduce<std::string>(sendBuf, recvBuf, 0, "SUM"),
            Catch::Matchers::Contains("Type not supported"));
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Allreduce Sum", "[MPIManagerGPU]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Allreduce scalar") {
        cp_t sendBuf = {(static_cast<PrecisionT>(rank) + 1) * 2, 0};
        cp_t recvBuf;

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "sum");
        CHECK(recvBuf.real() == static_cast<PrecisionT>((size + 1) * size));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce scalar") {
        cp_t sendBuf = {(static_cast<PrecisionT>(rank) + 1) * 2, 0};
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "sum");

        CHECK(recvBuf.real() == static_cast<PrecisionT>((size + 1) * size));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Allreduce vector") {
        std::vector<cp_t> sendBuf(1,
                                  {(static_cast<PrecisionT>(rank) + 1) * 2, 0});
        std::vector<cp_t> recvBuf(1);

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "sum");

        CHECK(recvBuf[0].real() == static_cast<PrecisionT>((size + 1) * size));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce vector") {
        std::vector<cp_t> sendBuf(1,
                                  {(static_cast<PrecisionT>(rank) + 1) * 2, 0});
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "sum");

        CHECK(recvBuf[0].real() == static_cast<PrecisionT>((size + 1) * size));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Allreduce Prod", "[MPIManagerGPU]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int size = mpi_manager.getSize();

    SECTION("Apply Allreduce scalar") {
        cp_t sendBuf = {2, 0};
        cp_t recvBuf;

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "prod");
        CHECK(recvBuf.real() == exp2(size));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce scalar") {
        cp_t sendBuf = {2, 0};
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "prod");

        CHECK(recvBuf.real() == exp2(size));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Allreduce vector") {
        std::vector<cp_t> sendBuf(1, {2, 0});
        std::vector<cp_t> recvBuf(1);

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "prod");

        CHECK(recvBuf[0].real() == exp2(size));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce vector") {
        std::vector<cp_t> sendBuf(1, {2, 0});
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "prod");

        CHECK(recvBuf[0].real() == exp2(size));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Bcast", "[MPIManagerGPU]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();

    SECTION("Apply Bcast scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        mpi_manager.Bcast<cp_t>(sendBuf, 0);
        CHECK(sendBuf.real() == static_cast<PrecisionT>(0));
        CHECK(sendBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Bcast vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        mpi_manager.Bcast<cp_t>(sendBuf, 0);
        CHECK(sendBuf[0].real() == static_cast<PrecisionT>(0));
        CHECK(sendBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEMPLATE_TEST_CASE("MPIManagerGPU::Sendrecv", "[MPIManagerGPU]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    SECTION("Apply Sendrecv scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0.0};
        cp_t recvBuf = {-1.0, -1.0};

        mpi_manager.Sendrecv<cp_t>(sendBuf, dest, recvBuf, source);

        CHECK(recvBuf.real() == static_cast<PrecisionT>(source));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Sendrecv vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0.0});
        std::vector<cp_t> recvBuf(1, {-1.0, -1.0});
        mpi_manager.Sendrecv<cp_t>(sendBuf, dest, recvBuf, source);
        CHECK(recvBuf[0].real() == static_cast<PrecisionT>(source));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEST_CASE("MPIManagerGPU::split") {
    MPIManagerGPU mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int color = rank % 2;
    int key = rank;
    auto newComm = mpi_manager.split(color, key);
    CHECK(newComm.getSize() * 2 == mpi_manager.getSize());
}
