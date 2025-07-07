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
#include "UtilKokkos.hpp"

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

TEMPLATE_TEST_CASE("MPIManagerKokkos::Sendrecv", "[MPIManagerKokkos]", float,
                   double) {
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

        std::vector<cp_t> h_sendBuf(message_size);
        for (std::size_t i = 0; i < message_size; ++i) {
            h_sendBuf[i] = static_cast<PrecisionT>(mpi_rank + i);
        }
        Kokkos::View<cp_t *> sendBuf = vector2view(h_sendBuf);
        Kokkos::View<cp_t *> recvBuf("recvBuf", message_size);
        mpi_manager.Sendrecv(sendBuf, dest, recvBuf, source, message_size);
        auto h_recvBuf = view2vector(recvBuf);

        for (std::size_t i = 0; i < message_size; ++i) {
            CHECK(h_recvBuf[i].real() == static_cast<PrecisionT>(source + i));
            CHECK(h_recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Sendrecv 0-1 2-3") {
        std::size_t dest = mpi_rank ^ 1U;
        std::size_t source = dest;

        std::vector<cp_t> h_sendBuf(message_size);
        for (std::size_t i = 0; i < message_size; ++i) {
            h_sendBuf[i] = static_cast<PrecisionT>(mpi_rank + i);
        }
        Kokkos::View<cp_t *> sendBuf = vector2view(h_sendBuf);
        Kokkos::View<cp_t *> recvBuf("recvBuf", message_size);
        mpi_manager.Sendrecv(sendBuf, dest, recvBuf, source, message_size);
        auto h_recvBuf = view2vector(recvBuf);

        for (std::size_t i = 0; i < message_size; ++i) {
            CHECK(h_recvBuf[i].real() == static_cast<PrecisionT>(source + i));
            CHECK(h_recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }
}

TEMPLATE_TEST_CASE("MPIManagerKokkos::AllGatherV", "[MPIManagerKokkos]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = Kokkos::complex<PrecisionT>;

    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    std::size_t mpi_rank = mpi_manager.getRank();
    std::size_t mpi_size = mpi_manager.getSize();

    std::size_t message_size = 3;

    std::vector<cp_t> h_sendBuf(message_size);
    for (std::size_t i = 0; i < message_size; ++i) {
        h_sendBuf[i] = static_cast<PrecisionT>(mpi_rank + i);
    }
    Kokkos::View<cp_t *> sendBuf = vector2view(h_sendBuf);
    Kokkos::View<cp_t *> recvBuf("recvBuf", message_size * mpi_size);

    std::vector<int> recv_counts(mpi_size, int(message_size));
    std::vector<int> displacements{0, 3, 6, 9};
    mpi_manager.AllGatherV(sendBuf, recvBuf, recv_counts, displacements);
    auto h_recvBuf = view2vector(recvBuf);

    std::vector<PrecisionT> expected(message_size * mpi_size);
    for (std::size_t i = 0; i < mpi_size; ++i) {
        for (std::size_t j = 0; j < message_size; ++j) {
            expected[i * message_size + j] = static_cast<PrecisionT>(i + j);
        }
    }
    for (std::size_t i = 0; i < message_size * mpi_size; ++i) {
        CHECK(h_recvBuf[i].real() == expected[i]);
    }
}

TEMPLATE_TEST_CASE("MPIManagerKokkos::Bcast", "[MPIManagerKokkos]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = Kokkos::complex<PrecisionT>;

    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    std::size_t mpi_rank = mpi_manager.getRank();

    std::size_t message_size = 3;

    std::vector<cp_t> h_sendBuf(message_size);
    if (mpi_rank == 0) {
        for (std::size_t i = 0; i < message_size; ++i) {
            h_sendBuf[i] = static_cast<PrecisionT>(i);
        }
    } else {
        for (std::size_t i = 0; i < message_size; ++i) {
            h_sendBuf[i] = static_cast<PrecisionT>(0);
        }
    }
    Kokkos::View<cp_t *> sendBuf = vector2view(h_sendBuf);
    mpi_manager.Bcast(sendBuf, 0);
    h_sendBuf = view2vector(sendBuf);

    for (std::size_t i = 0; i < message_size; ++i) {
        CHECK(h_sendBuf[i].real() == static_cast<PrecisionT>(i));
        CHECK(h_sendBuf[i].imag() == 0.0);
    }
}
