// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "MPSCutn.hpp"
#include "cuGateTensorCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

/**
 * @brief Tests the constructability of the MPSCutn class.
 *
 */
TEMPLATE_TEST_CASE("MPSCutn::MPSCutn", "[MPSCutn_Nonparam]", float, double) {
    SECTION("MPSCutn<TestType> {std::size_t, std::size_t, std::vector<size_t>, "
            "DevTag<int>}") {
        REQUIRE(std::is_constructible<MPSCutn<TestType>, std::size_t,
                                      std::size_t, std::vector<std::size_t>,
                                      DevTag<int>>::value);
    }

    SECTION("MPSCutn<TestType> (&other)") {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        size_t index = 1;
        sv.setBasisState(index);

        MPSCutn<TestType> sv_copy(sv);

        std::vector<std::complex<TestType>> expected_state(
            1 << num_qubits, std::complex<TestType>({0.0, 0.0}));

        expected_state[index] = {1.0, 0.0};

        CHECK(expected_state ==
              Pennylane::Util::approx(sv_copy.getDataVector()));
    }
}

TEMPLATE_TEST_CASE("MPSCutn::applyHadamard", "[MPSCutn_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

                CHECK(sv.getDataVector()[0] == cp_t{1, 0});
                sv.applyOperation("Hadamard", {index}, inverse);
                cp_t expected(1.0 / std::sqrt(2), 0);

                CHECK(expected.real() == Approx(sv.getDataVector()[0].real()));
                CHECK(expected.imag() == Approx(sv.getDataVector()[0].imag()));

                CHECK(
                    expected.real() ==
                    Approx(sv.getDataVector()[0b1 << ((num_qubits - index - 1))]
                               .real()));
                CHECK(
                    expected.imag() ==
                    Approx(sv.getDataVector()[0b1 << ((num_qubits - index - 1))]
                               .imag()));
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSCutn::applyPauliX", "[MPSCutn_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};
                CHECK(sv.getDataVector()[0] ==
                      cuUtil::ONE<std::complex<TestType>>());
                sv.applyOperation("PauliX", {index}, inverse);
                CHECK(sv.getDataVector()[0] ==
                      cuUtil::ZERO<std::complex<TestType>>());
                CHECK(sv.getDataVector()[0b1 << (num_qubits - index - 1)] ==
                      cuUtil::ONE<std::complex<TestType>>());
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyPauliY",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = false; // GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;

        const cp_t p = cuUtil::ConstMult(
            std::complex<TestType>(0.5, 0.0),
            cuUtil::ConstMult(cuUtil::INVSQRT2<std::complex<TestType>>(),
                              cuUtil::IMAG<std::complex<TestType>>()));
        const cp_t m = cuUtil::ConstMult(std::complex<TestType>(-1, 0), p);

        const std::vector<std::vector<cp_t>> expected_results = {
            {m, m, m, m, p, p, p, p},
            {m, m, p, p, m, m, p, p},
            {m, p, m, p, m, p, m, p}};

        SECTION("Apply using dispatcher") {
            std::size_t num_qubits = 3;
            std::size_t maxExtent = 2;
            std::vector<size_t> qubitDims = {2, 2, 2};
            DevTag<int> dev_tag{0, 0};

            for (std::size_t index = 0; index < num_qubits; index++) {
                MPSCutn<TestType> sv_init{num_qubits, maxExtent, qubitDims,
                                          dev_tag};

                sv_init.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}, {"PauliY"}},
                    {{0}, {1}, {2}, {index}}, {false, false, false, inverse});

                CHECK(sv_init.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSCutn::applyPauliZ", "[MPSCutn_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        using cp_t = std::complex<TestType>;
        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        // Test using |+++> state
        sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                           {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const cp_t p(static_cast<TestType>(0.5) *
                     cuUtil::INVSQRT2<std::complex<TestType>>());
        const cp_t m(cuUtil::ConstMult(cp_t{-1.0, 0.0}, p));

        const std::vector<std::vector<cp_t>> expected_results = {
            {p, p, p, p, m, m, m, m},
            {p, p, m, m, p, p, m, m},
            {p, m, p, m, p, m, p, m}};

        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                MPSCutn<TestType> sv_dispatch(sv);
                CHECK(sv_dispatch.getDataVector() == init_state);
                sv_dispatch.applyOperation("PauliZ", {index}, inverse);
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSCutn::applyS", "[MPSCutn_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        // Test using |+++> state
        sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                           {{0}, {1}, {2}}, {{false}, {false}, {false}});

        cp_t r(std::complex<TestType>(0.5, 0.0) *
               cuUtil::INVSQRT2<std::complex<TestType>>());
        cp_t i(cuUtil::ConstMult(r, cuUtil::IMAG<std::complex<TestType>>()));

        if (inverse) {
            i = conj(i);
        }

        const std::vector<std::vector<cp_t>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                MPSCutn<TestType> sv_dispatch(sv);
                CHECK(sv_dispatch.getDataVector() == init_state);
                sv_dispatch.applyOperation("S", {index}, inverse);
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSCutn::applyT", "[MPSCutn_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};
        // Test using |+++> state
        sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                           {{0}, {1}, {2}}, {{false}, {false}, {false}});

        cp_t r(1.0 / (2.0 * std::sqrt(2)), 0);
        cp_t i(1.0 / 4, 1.0 / 4);

        if (inverse) {
            i = conj(i);
        }

        const std::vector<std::vector<cp_t>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                MPSCutn<TestType> sv_dispatch(sv);
                CHECK(sv_dispatch.getDataVector() == init_state);
                sv_dispatch.applyOperation("T", {index}, inverse);
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSCutn::applyCNOT", "[MPSCutn_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};
        // Test using |+00> state to generate 3-qubit GHZ state
        sv.applyOperation("Hadamard", {0});
        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            MPSCutn<TestType> sv_dispatch(sv);

            for (std::size_t index = 1; index < num_qubits; index++) {
                sv_dispatch.applyOperation("CNOT", {index - 1, index}, inverse);
            }
            CHECK(sv_dispatch.getDataVector().front() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(sv_dispatch.getDataVector().back() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("MPSCutn::applySWAP", "[MPSCutn_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            SECTION("SWAP0,1 |+10> -> |1+0>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>()};

                MPSCutn<TestType> sv01(sv);
                MPSCutn<TestType> sv10(sv);

                sv01.applyOperation("SWAP", {0, 1}, inverse);
                sv10.applyOperation("SWAP", {1, 0}, inverse);

                CHECK(sv01.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv10.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }

            SECTION("SWAP0,2 |+10> -> |01+>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>()};

                MPSCutn<TestType> sv02(sv);
                MPSCutn<TestType> sv20(sv);

                sv02.applyOperation("SWAP", {0, 2}, inverse);
                sv20.applyOperation("SWAP", {2, 0}, inverse);

                CHECK(sv02.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv20.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("SWAP1,2 |+10> -> |+01>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>()};

                MPSCutn<TestType> sv12(sv);
                MPSCutn<TestType> sv21(sv);

                sv12.applyOperation("SWAP", {1, 2}, inverse);
                sv21.applyOperation("SWAP", {2, 1}, inverse);

                CHECK(sv12.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv21.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("MPSCutn::applyCY", "[MPSCutn_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            SECTION("CY0,1") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(0, -1 / sqrt(2)),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>()};
                MPSCutn<TestType> sv01(sv);
                sv01.applyOperation("CY", {0, 1}, inverse);

                CHECK(sv01.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("CY1,0") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(0, -1.0 / sqrt(2)),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(0, 1 / sqrt(2)),
                    cuUtil::ZERO<std::complex<TestType>>()};
                MPSCutn<TestType> sv10(sv);
                sv10.applyOperation("CY", {1, 0}, inverse);

                CHECK(sv10.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("CY0,2") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0.0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(0.0, 1.0 / sqrt(2))};
                MPSCutn<TestType> sv02(sv);
                sv02.applyOperation("CY", {0, 2}, inverse);

                CHECK(sv02.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("CY2,0") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0.0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0.0),
                    cuUtil::ZERO<std::complex<TestType>>()};
                MPSCutn<TestType> sv20(sv);
                sv20.applyOperation("CY", {2, 0}, inverse);

                CHECK(sv20.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("MPSCutn::applyCZ", "[MPSCutn_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            SECTION("CZ0,1 |+10> -> |1+0>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(-1 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>()};

                MPSCutn<TestType> sv01(sv);
                MPSCutn<TestType> sv10(sv);

                sv01.applyOperation("CZ", {0, 1}, inverse);
                sv10.applyOperation("CZ", {1, 0}, inverse);

                CHECK(sv01.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv10.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("MPSCutn::applyToffoli", "[MPSCutn_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply using dispatcher") {
            SECTION("Toffoli [0,1,2], [1,0,2] |+10> -> |+1+>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0)};

                MPSCutn<TestType> sv012(sv);
                MPSCutn<TestType> sv102(sv);

                sv012.applyOperation("Toffoli", {0, 1, 2}, inverse);
                sv102.applyOperation("Toffoli", {1, 0, 2}, inverse);

                CHECK(sv012.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv102.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("MPSCutn::applyCSWAP", "[MPSCutn_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        SECTION("Apply using dispatcher") {
            SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>()};
                // MPSCutn<TestType> sv012(sv);

                std::size_t num_qubits = 3;
                std::size_t maxExtent = 2;
                std::vector<size_t> qubitDims = {2, 2, 2};
                DevTag<int> dev_tag{0, 0};

                MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};
                // Test using |+10> state
                sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                   {false, false});

                sv.applyOperation("CSWAP", {0, 1, 2}, inverse);
                CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSCutn::SetIthStates", "[MPSCutn_Nonparam]", float,
                   double) {
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    std::vector<size_t> qubitDims = {2, 2, 2};
    DevTag<int> dev_tag{0, 0};

    SECTION(
        "Set Ith element of the state state on device with data on the host") {

        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        size_t index = 3;
        sv.setBasisState(index);

        std::vector<std::complex<TestType>> expected_state(
            1 << num_qubits, std::complex<TestType>({0.0, 0.0}));

        expected_state[index] = {1.0, 0.0};

        CHECK(expected_state == Pennylane::Util::approx(sv.getDataVector()));
    }
}
