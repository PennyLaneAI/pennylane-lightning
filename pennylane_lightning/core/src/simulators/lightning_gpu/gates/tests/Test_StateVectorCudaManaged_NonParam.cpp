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
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorCudaManaged.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::Util;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

/**
 * @brief Tests the constructability of the StateVectorCudaManaged class.
 *
 */
TEMPLATE_TEST_CASE("StateVectorCudaManaged::StateVectorCudaManaged",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    SECTION("StateVectorCudaManaged<TestType> {std::complex<TestType>, "
            "std::size_t}") {
        REQUIRE(std::is_constructible<StateVectorCudaManaged<TestType>,
                                      std::complex<TestType> *,
                                      std::size_t>::value);
    }
    SECTION("StateVectorCudaManaged<TestType> cross types") {
        if constexpr (!std::is_same_v<TestType, double>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVectorCudaManaged<TestType>,
                                      std::complex<double> *,
                                      std::size_t>::value);
            REQUIRE_FALSE(std::is_constructible<StateVectorCudaManaged<double>,
                                                std::complex<TestType> *,
                                                std::size_t>::value);
        } else if constexpr (!std::is_same_v<TestType, float>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVectorCudaManaged<TestType>,
                                      std::complex<float> *,
                                      std::size_t>::value);
            REQUIRE_FALSE(std::is_constructible<StateVectorCudaManaged<float>,
                                                std::complex<TestType> *,
                                                std::size_t>::value);
        }
    }
    SECTION("StateVectorCudaManaged<TestType> transfers") {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        const std::vector<cp_t> init_state{{1, 0}, {0, 0}, {0, 0}, {0, 0},
                                           {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        SECTION("GPU <-> host data: std::complex") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            std::vector<cp_t> out_data(Pennylane::Util::exp2(num_qubits),
                                       {0.5, 0.5});
            std::vector<cp_t> ref_data(Pennylane::Util::exp2(num_qubits),
                                       {0.0, 0.0});
            sv.CopyGpuDataToHost(out_data.data(), out_data.size());
            CHECK(out_data == init_state);

            sv.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                               {{0}, {1}, {2}}, {false, false, false});
            sv.CopyHostDataToGpu(out_data);
            sv.CopyGpuDataToHost(ref_data.data(), ref_data.size());
            CHECK(out_data == Pennylane::Util::approx(ref_data));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyHadamard",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv{num_qubits};
                CHECK(sv.getDataVector()[0] == cp_t{1, 0});
                sv.applyHadamard({index}, inverse);
                CAPTURE(sv.getDataVector());

                cp_t expected(1 / std::sqrt(2), 0);
                CHECK(expected.real() == Approx(sv.getDataVector()[0].real()));
                CHECK(expected.imag() == Approx(sv.getDataVector()[0].imag()));

                CHECK(expected.real() ==
                      Approx(sv.getDataVector()[0b1 << (num_qubits - index - 1)]
                                 .real()));
                CHECK(expected.imag() ==
                      Approx(sv.getDataVector()[0b1 << (num_qubits - index - 1)]
                                 .imag()));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv{num_qubits};

                CHECK(sv.getDataVector()[0] == cp_t{1, 0});
                sv.applyOperation("Hadamard", {index}, inverse);
                cp_t expected(1.0 / std::sqrt(2), 0);

                CHECK(expected.real() == Approx(sv.getDataVector()[0].real()));
                CHECK(expected.imag() == Approx(sv.getDataVector()[0].imag()));

                CHECK(expected.real() ==
                      Approx(sv.getDataVector()[0b1 << (num_qubits - index - 1)]
                                 .real()));
                CHECK(expected.imag() ==
                      Approx(sv.getDataVector()[0b1 << (num_qubits - index - 1)]
                                 .imag()));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyPauliX",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        const std::size_t num_qubits = 3;
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv{num_qubits};
                CHECK(sv.getDataVector()[0] ==
                      cuUtil::ONE<std::complex<TestType>>());
                sv.applyPauliX({index}, inverse);
                CHECK(sv.getDataVector()[0] ==
                      cuUtil::ZERO<std::complex<TestType>>());
                CHECK(sv.getDataVector()[0b1 << (num_qubits - index - 1)] ==
                      cuUtil::ONE<std::complex<TestType>>());
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv{num_qubits};
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
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};
        // Test using |+++> state
        sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                           {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const cp_t p = cuUtil::ConstMult(
            std::complex<TestType>(0.5, 0.0),
            cuUtil::ConstMult(cuUtil::INVSQRT2<std::complex<TestType>>(),
                              cuUtil::IMAG<std::complex<TestType>>()));
        const cp_t m = cuUtil::ConstMult(std::complex<TestType>(-1, 0), p);

        const std::vector<std::vector<cp_t>> expected_results = {
            {m, m, m, m, p, p, p, p},
            {m, m, p, p, m, m, p, p},
            {m, p, m, p, m, p, m, p}};

        const auto init_state = sv.getDataVector();
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                CHECK(sv_direct.getDataVector() == init_state);
                sv_direct.applyPauliY({index}, inverse);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                CHECK(sv_dispatch.getDataVector() == init_state);
                sv_dispatch.applyOperation("PauliY", {index}, inverse);
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyPauliZ",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};
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
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                CHECK(sv_direct.getDataVector() == init_state);
                sv_direct.applyPauliZ({index}, inverse);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                CHECK(sv_dispatch.getDataVector() == init_state);
                sv_dispatch.applyOperation("PauliZ", {index}, inverse);
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyS",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};
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
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                CHECK(sv_direct.getDataVector() == init_state);
                sv_direct.applyS({index}, inverse);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                CHECK(sv_dispatch.getDataVector() == init_state);
                sv_dispatch.applyOperation("S", {index}, inverse);
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyT",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};
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
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                CHECK(sv_direct.getDataVector() == init_state);
                sv_direct.applyT({index}, inverse);
                CAPTURE(sv_direct.getDataVector());
                CAPTURE(expected_results[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                CHECK(sv_dispatch.getDataVector() == init_state);
                sv_dispatch.applyOperation("T", {index}, inverse);
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyCNOT",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};

        // Test using |+00> state to generate 3-qubit GHZ state
        sv.applyOperation("Hadamard", {0});
        const auto init_state = sv.getDataVector();

        SECTION("Apply directly") {
            StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                       init_state.size()};

            for (std::size_t index = 1; index < num_qubits; index++) {
                sv_direct.applyCNOT({index - 1, index}, inverse);
            }
            CHECK(sv_direct.getDataVector().front() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(sv_direct.getDataVector().back() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
        }

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                         init_state.size()};

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
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applySWAP",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply directly") {
            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(std::vector<cp_t>{
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::INVSQRT2<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::INVSQRT2<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>()}));

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

                StateVectorCudaManaged<TestType> sv01{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv10{init_state.data(),
                                                      init_state.size()};

                sv01.applySWAP({0, 1}, inverse);
                sv10.applySWAP({1, 0}, inverse);

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

                StateVectorCudaManaged<TestType> sv02{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv20{init_state.data(),
                                                      init_state.size()};

                sv02.applySWAP({0, 2}, inverse);
                sv20.applySWAP({2, 0}, inverse);

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

                StateVectorCudaManaged<TestType> sv12{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv21{init_state.data(),
                                                      init_state.size()};

                sv12.applySWAP({1, 2}, inverse);
                sv21.applySWAP({2, 1}, inverse);

                CHECK(sv12.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv21.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
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

                StateVectorCudaManaged<TestType> sv01{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv10{init_state.data(),
                                                      init_state.size()};

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

                StateVectorCudaManaged<TestType> sv02{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv20{init_state.data(),
                                                      init_state.size()};

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

                StateVectorCudaManaged<TestType> sv12{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv21{init_state.data(),
                                                      init_state.size()};

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
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyCY",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply directly") {
            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(std::vector<cp_t>{
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      std::complex<TestType>(1.0 / sqrt(2), 0),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      std::complex<TestType>(1.0 / sqrt(2), 0),
                      cuUtil::ZERO<std::complex<TestType>>()}));
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
                StateVectorCudaManaged<TestType> sv01{init_state.data(),
                                                      init_state.size()};
                sv01.applyCY({0, 1}, inverse);

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
                StateVectorCudaManaged<TestType> sv10{init_state.data(),
                                                      init_state.size()};
                sv10.applyCY({1, 0}, inverse);

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
                StateVectorCudaManaged<TestType> sv02{init_state.data(),
                                                      init_state.size()};
                sv02.applyCY({0, 2}, inverse);

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
                StateVectorCudaManaged<TestType> sv20{init_state.data(),
                                                      init_state.size()};
                sv20.applyCY({2, 0}, inverse);

                CHECK(sv20.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }

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
                StateVectorCudaManaged<TestType> sv01{init_state.data(),
                                                      init_state.size()};
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
                StateVectorCudaManaged<TestType> sv10{init_state.data(),
                                                      init_state.size()};
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
                StateVectorCudaManaged<TestType> sv02{init_state.data(),
                                                      init_state.size()};
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
                StateVectorCudaManaged<TestType> sv20{init_state.data(),
                                                      init_state.size()};
                sv20.applyOperation("CY", {2, 0}, inverse);

                CHECK(sv20.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyCZ",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply directly") {
            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(std::vector<cp_t>{
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      std::complex<TestType>(1.0 / sqrt(2), 0),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      cuUtil::ZERO<std::complex<TestType>>(),
                      std::complex<TestType>(1.0 / sqrt(2), 0),
                      cuUtil::ZERO<std::complex<TestType>>()}));

            SECTION("CZ0,1 |+10> -> |-10>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(-1 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>()};

                StateVectorCudaManaged<TestType> sv01{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv10{init_state.data(),
                                                      init_state.size()};

                sv01.applyCZ({0, 1}, inverse);
                sv10.applyCZ({1, 0}, inverse);

                CHECK(sv01.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv10.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }

            SECTION("CZ0,2 |+10> -> |+10>") {
                auto &&expected = init_state;

                StateVectorCudaManaged<TestType> sv02{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv20{init_state.data(),
                                                      init_state.size()};

                sv02.applyCZ({0, 2}, inverse);
                sv20.applyCZ({2, 0}, inverse);

                CHECK(sv02.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv20.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("CZ1,2 |+10> -> |+10>") {
                auto &&expected = init_state;

                StateVectorCudaManaged<TestType> sv12{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv21{init_state.data(),
                                                      init_state.size()};

                sv12.applyCZ({1, 2}, inverse);
                sv21.applyCZ({2, 1}, inverse);

                CHECK(sv12.getDataVector() ==
                      Pennylane::Util::approx(expected));
                CHECK(sv21.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
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

                StateVectorCudaManaged<TestType> sv01{init_state.data(),
                                                      init_state.size()};
                StateVectorCudaManaged<TestType> sv10{init_state.data(),
                                                      init_state.size()};

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
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyToffoli",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply directly") {
            SECTION("Toffoli 0,1,2 |+10> -> |010> + |111>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::INVSQRT2<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::INVSQRT2<std::complex<TestType>>()};

                StateVectorCudaManaged<TestType> sv012{init_state.data(),
                                                       init_state.size()};

                sv012.applyToffoli({0, 1, 2}, inverse);

                CHECK(sv012.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }

            SECTION("Toffoli 1,0,2 |+10> -> |010> + |111>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0)};

                StateVectorCudaManaged<TestType> sv102{init_state.data(),
                                                       init_state.size()};

                sv102.applyToffoli({1, 0, 2}, inverse);

                CHECK(sv102.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("Toffoli 0,2,1 |+10> -> |+10>") {
                auto &&expected = init_state;

                StateVectorCudaManaged<TestType> sv021{init_state.data(),
                                                       init_state.size()};

                sv021.applyToffoli({0, 2, 1}, inverse);

                CHECK(sv021.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("Toffoli 1,2,0 |+10> -> |+10>") {
                auto &&expected = init_state;

                StateVectorCudaManaged<TestType> sv120{init_state.data(),
                                                       init_state.size()};

                sv120.applyToffoli({1, 2, 0}, inverse);

                CHECK(sv120.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
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

                StateVectorCudaManaged<TestType> sv012{init_state.data(),
                                                       init_state.size()};
                StateVectorCudaManaged<TestType> sv102{init_state.data(),
                                                       init_state.size()};

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
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyCSWAP",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        StateVectorCudaManaged<TestType> sv{num_qubits};

        // Test using |+10> state
        sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                           {false, false});
        const auto init_state = sv.getDataVector();

        SECTION("Apply directly") {
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

                StateVectorCudaManaged<TestType> sv012{init_state.data(),
                                                       init_state.size()};

                sv012.applyCSWAP({0, 1, 2}, inverse);

                CHECK(sv012.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }

            SECTION("CSWAP 1,0,2 |+10> -> |01+>") {
                std::vector<cp_t> expected{
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    std::complex<TestType>(1.0 / sqrt(2), 0),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>(),
                    cuUtil::ZERO<std::complex<TestType>>()};

                StateVectorCudaManaged<TestType> sv102{init_state.data(),
                                                       init_state.size()};

                sv102.applyCSWAP({1, 0, 2}, inverse);

                CHECK(sv102.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
            SECTION("CSWAP 2,1,0 |+10> -> |+10>") {
                auto &&expected = init_state;

                StateVectorCudaManaged<TestType> sv021{init_state.data(),
                                                       init_state.size()};

                sv021.applyCSWAP({2, 1, 0}, inverse);
                CHECK(sv021.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
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
                StateVectorCudaManaged<TestType> sv012{init_state.data(),
                                                       init_state.size()};

                sv012.applyOperation("CSWAP", {0, 1, 2}, inverse);
                CHECK(sv012.getDataVector() ==
                      Pennylane::Util::approx(expected));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::SetStateVector",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using PrecisionT = TestType;
    const std::size_t num_qubits = 3;
    std::mt19937 re{1337};

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        auto init_state =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        auto expected_state = init_state;

        for (std::size_t i = 0; i < Pennylane::Util::exp2(num_qubits - 1);
             i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
        }

        StateVectorCudaManaged<TestType> sv{num_qubits};

        std::vector<std::complex<PrecisionT>> values(init_state.begin(),
                                                     init_state.end());

        sv.setStateVector(values.data(), values.size(),
                          std::vector<std::size_t>{0, 1, 2});
        CHECK(init_state == Pennylane::Util::approx(sv.getDataVector()));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::SetIthStates",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using PrecisionT = TestType;
    const std::size_t num_qubits = 3;
    std::mt19937 re{1337};

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        auto init_state =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        std::vector<std::complex<PrecisionT>> expected_state(init_state.size(),
                                                             {0, 0});

        expected_state[expected_state.size() - 1] = {1.0, 0};

        StateVectorCudaManaged<TestType> sv{num_qubits};
        sv.CopyHostDataToGpu(init_state.data(), init_state.size());

        std::vector<std::size_t> state(num_qubits, 1);
        std::vector<std::size_t> wires(num_qubits, 0);
        std::iota(wires.begin(), wires.end(), 0);

        sv.setBasisState(state, wires, false);

        CHECK(expected_state == Pennylane::Util::approx(sv.getDataVector()));
    }
}