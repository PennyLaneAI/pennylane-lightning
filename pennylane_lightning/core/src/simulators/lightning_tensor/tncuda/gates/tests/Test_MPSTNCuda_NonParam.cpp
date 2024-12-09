// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <complex>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "MPSTNCuda.hpp"
#include "TNCudaGateCache.hpp"

#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::LightningTensor::TNCuda::Gates;
using namespace Pennylane::Util;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::Identity", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperation("Hadamard", {index}, inverse);

            mps_state.applyOperation("Identity", {index}, inverse);
            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = mps_state.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::Hadamard", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.append_mps_final_state();

            mps_state.applyOperation("Hadamard", {index}, inverse);

            mps_state.append_mps_final_state();

            mps_state.applyOperation("Identity", {index}, inverse);

            // Test for multiple final states appendings
            mps_state.append_mps_final_state();

            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = mps_state.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::PauliX", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperation("PauliX", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results[0] == cuUtil::ZERO<std::complex<TestType>>());
            CHECK(results[0b1 << (num_qubits - index - 1)] ==
                  cuUtil::ONE<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::applyOperation-gatematrix",
                   "[MPSTNCuda_Nonparam]", float, double) {
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);
        MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

        std::vector<std::complex<TestType>> gate_matrix = {
            cuUtil::ZERO<std::complex<TestType>>(),
            cuUtil::ONE<std::complex<TestType>>(),
            cuUtil::ONE<std::complex<TestType>>(),
            cuUtil::ZERO<std::complex<TestType>>()};

        mps_state.applyOperation("applyMatrix", {index}, false, {},
                                 gate_matrix);

        auto results = mps_state.getDataVector();

        CHECK(results[0] == cuUtil::ZERO<std::complex<TestType>>());
        CHECK(results[0b1 << (num_qubits - index - 1)] ==
              cuUtil::ONE<std::complex<TestType>>());
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::PauliY", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const cp_t p = cuUtil::ConstMult(
            std::complex<TestType>(0.5, 0.0),
            cuUtil::ConstMult(cuUtil::INVSQRT2<std::complex<TestType>>(),
                              cuUtil::IMAG<std::complex<TestType>>()));
        const cp_t m = cuUtil::ConstMult(std::complex<TestType>(-1, 0), p);

        const std::vector<std::vector<cp_t>> expected_results = {
            {m, m, m, m, p, p, p, p},
            {m, m, p, p, m, m, p, p},
            {m, p, m, p, m, p, m, p}};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("PauliY", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::PauliZ", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const cp_t p(static_cast<TestType>(0.5) *
                     cuUtil::INVSQRT2<std::complex<TestType>>());
        const cp_t m(cuUtil::ConstMult(cp_t{-1.0, 0.0}, p));

        const std::vector<std::vector<cp_t>> expected_results = {
            {p, p, p, p, m, m, m, m},
            {p, p, m, m, p, p, m, m},
            {p, m, p, m, p, m, p, m}};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("PauliZ", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::S", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

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

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("S", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::T", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        cp_t r(1.0 / (2.0 * std::sqrt(2)), 0);
        cp_t i(1.0 / 4, 1.0 / 4);

        if (inverse) {
            i = conj(i);
        }

        const std::vector<std::vector<cp_t>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("T", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::CNOT", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "CNOT", "CNOT"},
                                      {{0}, {0, 1}, {1, 2}},
                                      {false, inverse, inverse});

            auto results = mps_state.getDataVector();

            CHECK(results.front() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results.back() == cuUtil::INVSQRT2<std::complex<TestType>>());
        }

        SECTION("Apply non-adjacent wire indices") {
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperation("Hadamard", {0}, false);
            mps_state.applyOperation("CNOT", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results[0] == cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results[5] == cuUtil::INVSQRT2<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::SWAP", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("SWAP", {0, 1}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("SWAP", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::CY", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(0, -1 / sqrt(2)),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CY", {0, 1}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0.0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(0.0, 1.0 / sqrt(2))};

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CY", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::CZ", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(-1 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CZ", {0, 1}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CZ", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Non_Param_Gates::2+_wires",
                   "[MPSTNCuda_Nonparam]", float, double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Toffoli gate") {
            std::size_t num_qubits = 3;

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            REQUIRE_THROWS_AS(
                mps_state.applyOperation("Toffoli", {0, 1, 2}, inverse),
                LightningException);
        }

        SECTION("CSWAP gate") {
            std::size_t num_qubits = 4;

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};
            REQUIRE_THROWS_AS(
                mps_state.applyOperation("CSWAP", {0, 1, 2}, inverse),
                LightningException);
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyControlledOperation non-param "
                   "one-qubit with controls",
                   "[MPSTNCuda]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<PrecisionT>;
    const int num_qubits = 4;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    const auto margin = PrecisionT{1e-5};
    const std::size_t control = GENERATE(0, 1, 2, 3);
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    MPSTNCuda<PrecisionT> mps_state0{num_qubits, maxExtent, dev_tag};
    MPSTNCuda<PrecisionT> mps_state1{num_qubits, maxExtent, dev_tag};

    DYNAMIC_SECTION("Controlled gates with base operation - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            mps_state0.applyControlledOperation(
                "PauliX", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire});

            mps_state1.applyOperation(
                "CNOT", std::vector<std::size_t>{control, wire}, false);

            REQUIRE(mps_state0.getDataVector() ==
                    approx(mps_state1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("Controlled gates with a target matrix - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            std::vector<ComplexT> gate_matrix = {
                ComplexT{0.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                ComplexT{0.0, 0.0}};
            mps_state0.applyControlledOperation(
                "applyControlledGates", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire}, false,
                {}, gate_matrix);

            mps_state1.applyOperation(
                "CNOT", std::vector<std::size_t>{control, wire}, false);

            REQUIRE(mps_state0.getDataVector() ==
                    approx(mps_state1.getDataVector()).margin(margin));
        }
    }

    SECTION("Throw exception for 1+ target wires gates") {
        REQUIRE_THROWS_AS(mps_state0.applyControlledOperation(
                              "CSWAP", {0}, {true, true}, {1, 2}),
                          LightningException);
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyMPO::2+_wires", "[MPSTNCuda_NonParam]",
                   float, double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t maxExtent = 2;
        std::size_t max_mpo_bond = 16;
        DevTag<int> dev_tag{0, 0};

        std::vector<std::vector<cp_t>> mpo_cnot(
            2, std::vector<cp_t>(16, {0.0, 0.0}));

        // in-order decomposition of the cnot operator
        // data from scipy decompose in the lightning.tensor python layer
        mpo_cnot[0][0] = {1.0, 0.0};
        mpo_cnot[0][3] = {-1.0, 0.0};
        mpo_cnot[0][9] = {1.0, 0.0};
        mpo_cnot[0][10] = {-1.0, 0.0};

        mpo_cnot[1][0] = {1.0, 0.0};
        mpo_cnot[1][7] = {-1.0, 0.0};
        mpo_cnot[1][10] = {1.0, 0.0};
        mpo_cnot[1][13] = {-1.0, 0.0};

        std::vector<std::vector<cp_t>> mpo_cswap;
        mpo_cswap.emplace_back(std::vector<cp_t>(16, {0.0, 0.0}));
        mpo_cswap.emplace_back(std::vector<cp_t>(64, {0.0, 0.0}));
        mpo_cswap.emplace_back(std::vector<cp_t>(16, {0.0, 0.0}));

        mpo_cswap[0][0] = {-1.5811388300841898, 0.0};
        mpo_cswap[0][2] = {0.7071067811865475, 0.0};
        mpo_cswap[0][5] = {-1.0, 0.0};
        mpo_cswap[0][9] = mpo_cswap[0][0];
        mpo_cswap[0][11] = -mpo_cswap[0][2];
        mpo_cswap[0][14] = {1.0, 0.0};

        mpo_cswap[1][0] = {-0.413452607315265, 0.0};
        mpo_cswap[1][1] = {0.6979762349196628, 0.0};
        mpo_cswap[1][7] = {0.9870874576374964, 0.0};
        mpo_cswap[1][8] = {0.5736348503222318, 0.0};
        mpo_cswap[1][9] = {0.11326595025589799, 0.0};
        mpo_cswap[1][15] = {0.16018224300696726, 0.0};
        mpo_cswap[1][34] = -mpo_cswap[1][7];
        mpo_cswap[1][36] = mpo_cswap[1][0];
        mpo_cswap[1][37] = -mpo_cswap[1][1];
        mpo_cswap[1][42] = -mpo_cswap[1][15];
        mpo_cswap[1][44] = mpo_cswap[1][8];
        mpo_cswap[1][45] = -mpo_cswap[1][9];

        mpo_cswap[2][0] = mpo_cswap[1][15];
        mpo_cswap[2][1] = -mpo_cswap[1][7];
        mpo_cswap[2][7] = {1.0, 0.0};
        mpo_cswap[2][10] = {-1.0, 0.0};
        mpo_cswap[2][12] = -mpo_cswap[2][1];
        mpo_cswap[2][13] = mpo_cswap[2][0];

        SECTION("Target at wire indices") {
            std::size_t num_qubits = 3;

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            MPSTNCuda<TestType> mps_state_mpo{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                          {{0}, {1}, {2}},
                                          {false, false, false});

            mps_state.applyOperation("CNOT", {0, 1}, inverse);

            mps_state_mpo.applyMPOOperation(mpo_cnot, {0, 1}, max_mpo_bond);

            auto ref = mps_state.getDataVector();
            auto res = mps_state_mpo.getDataVector();

            CHECK(res == Pennylane::Util::approx(ref));
        }

        SECTION("Target at non-adjacent wire indices") {
            std::size_t num_qubits = 3;

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            MPSTNCuda<TestType> mps_state_mpo{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                          {{0}, {1}, {2}},
                                          {false, false, false});

            mps_state.applyOperation("CNOT", {0, 2}, inverse);

            mps_state_mpo.applyMPOOperation(mpo_cnot, {0, 2}, max_mpo_bond);

            auto ref = mps_state.getDataVector();
            auto res = mps_state_mpo.getDataVector();

            CHECK(res == Pennylane::Util::approx(ref));
        }

        SECTION("Tests for 3-wire MPOs") {
            std::size_t num_qubits = 3;

            MPSTNCuda<TestType> mps_state_mpo{num_qubits, maxExtent, dev_tag};
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                          {{0}, {1}, {2}},
                                          {false, false, false});
            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state_mpo.applyMPOOperation(mpo_cswap, {0, 1, 2}, max_mpo_bond);

            auto res = mps_state_mpo.getDataVector();
            auto ref = mps_state.getDataVector();

            CHECK(res == Pennylane::Util::approx(ref));
        }
    }
}
