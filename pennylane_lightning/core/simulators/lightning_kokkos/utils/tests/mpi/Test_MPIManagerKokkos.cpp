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

#include "MPIManagerKokkos.hpp"

using namespace Pennylane;
using namespace Pennylane::LightningKokkos::Util;

TEST_CASE("MPIManagerKokkos ctor", "[MPIManagerKokkos]") {
    SECTION("Default constructibility") {
        REQUIRE(std::is_constructible_v<MPIManagerKokkos>);
    }

    SECTION("Construct with MPI_Comm") {
        REQUIRE(std::is_constructible_v<MPIManagerKokkos, MPI_Comm>);
    }

    SECTION("Construct with args") {
        REQUIRE(std::is_constructible_v<MPIManagerKokkos, int, char **>);
    }

    SECTION("MPIManagerKokkos {MPIManagerKokkos&}") {
        REQUIRE(std::is_copy_constructible_v<MPIManagerKokkos>);
    }
}

TEST_CASE("MPIManagerKokkos::getMPIDatatype", "[MPIManagerKokkos]") {
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);

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
TEMPLATE_TEST_CASE("MPIManager::Sendrecv", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = Kokkos::complex<PrecisionT>;

    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    std::size_t mpi_rank = mpi_manager.getRank();
    std::size_t mpi_size = mpi_manager.getSize();

    std::size_t message_size = 3;

    SECTION("Sendrecv cyclic") {
        std::size_t dest = (mpi_rank + 1) % mpi_size;
        std::size_t source = (mpi_rank - 1 + mpi_size) % mpi_size;
        Kokkos::View<cp_t *> sendBuf("sendBuf", message_size);
        Kokkos::View<cp_t *> recvBuf("recvBuf", message_size);
        for (std::size_t i = 0; i < message_size; ++i) {
            sendBuf(i) = static_cast<PrecisionT>(mpi_rank + i);
        }
        mpi_manager.Sendrecv(sendBuf, dest, recvBuf, source, message_size);
        for (std::size_t i = 0; i < message_size; ++i) {
            CHECK(recvBuf(i).real() == static_cast<PrecisionT>(source + i));
            CHECK(recvBuf(i).imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Sendrecv 0-1 2-3") {
        std::size_t dest = mpi_rank ^ 1U;
        std::size_t source = dest;
        Kokkos::View<cp_t *> sendBuf("sendBuf", message_size);
        Kokkos::View<cp_t *> recvBuf("recvBuf", message_size);
        for (std::size_t i = 0; i < message_size; ++i) {
            sendBuf(i) = static_cast<PrecisionT>(mpi_rank + i);
        }
        mpi_manager.Sendrecv(sendBuf, dest, recvBuf, source, message_size);
        for (std::size_t i = 0; i < message_size; ++i) {
            CHECK(recvBuf(i).real() ==
                  static_cast<PrecisionT>((mpi_rank ^ 1U) + i));
            CHECK(recvBuf(i).imag() == static_cast<PrecisionT>(0));
        }
    }
}
