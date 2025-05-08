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
#include "TestHelpersTNCuda.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::LightningTensor::TNCuda::Gates;
using namespace Pennylane::Util;
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
/// @endcond

TEMPLATE_LIST_TEST_CASE("TNCuda::applyMPO::2+_wires", "[TNCuda_Nonparam]",
                        TestMPSBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using PrecisionT = typename TNDevice_T::PrecisionT;

    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;
    constexpr std::size_t max_mpo_bond = 16;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    std::vector<std::vector<cp_t>> mpo_cnot(2,
                                            std::vector<cp_t>(16, {0.0, 0.0}));

    // in-order decomposition of the cnot operator
    // data from scipy decompose in the lightning.tensor python layer
    // TODO: this is a temporary solution, it will be removed once SVD
    // decomposition is implemented in the C++ layer
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
        MPSTNCuda<PrecisionT> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("CNOT", {0, 1}, inverse);

        mps_state_mpo.applyMPOOperation(mpo_cnot, {0, 1}, max_mpo_bond);

        auto ref = tn_state->getDataVector();
        auto res = mps_state_mpo.getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }

    SECTION("Target at non-adjacent wire indices") {
        MPSTNCuda<PrecisionT> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("CNOT", {0, 2}, inverse);

        mps_state_mpo.applyMPOOperation(mpo_cnot, {0, 2}, max_mpo_bond);

        auto ref = tn_state->getDataVector();
        auto res = mps_state_mpo.getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }

    SECTION("Tests for 3-wire MPOs") {
        MPSTNCuda<PrecisionT> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyMPOOperation(mpo_cswap, {0, 1, 2}, max_mpo_bond);

        auto res = mps_state_mpo.getDataVector();
        auto ref = tn_state->getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::applyMPO::SingleExcitation", "[TNCuda_Param]",
                        TestMPSBackends) {
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;
    constexpr std::size_t max_mpo_bond = 4;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    std::vector<std::vector<cp_t>> mpo_single_excitation(
        2, std::vector<cp_t>(16, {0.0, 0.0}));

    // in-order decomposition of the cnot operator
    // data from scipy decompose in the lightning.tensor python layer
    // TODO: this is a temporary solution, it will be removed once SVD
    // decomposition is implemented in the C++ layer

    mpo_single_excitation[0][0] = {-1.40627352, 0.0};
    mpo_single_excitation[0][3] = {-0.14943813, 0.0};
    mpo_single_excitation[0][6] = {0.00794005, 0.0};
    mpo_single_excitation[0][9] = {-1.40627352, 0.0};
    mpo_single_excitation[0][12] = {-0.14943813, 0.0};
    mpo_single_excitation[0][15] = {-0.00794005, 0.0};

    mpo_single_excitation[1][0] = {-0.707106781, 0.0};
    mpo_single_excitation[1][3] = {0.707106781, 0.0};
    mpo_single_excitation[1][6] = {1.0, 0.0};
    mpo_single_excitation[1][9] = {-1.0, 0.0};
    mpo_single_excitation[1][12] = {-0.707106781, 0.0};
    mpo_single_excitation[1][15] = {-0.707106781, 0.0};

    SECTION("Target at wire indices") {
        MPSTNCuda<Precision_T> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("SingleExcitation", {0, 1}, false, {0.3});

        mps_state_mpo.applyMPOOperation(mpo_single_excitation, {0, 1},
                                        max_mpo_bond);

        auto ref = tn_state->getDataVector();
        auto res = mps_state_mpo.getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }

    SECTION("Target at non-adjacent wire indices") {
        MPSTNCuda<Precision_T> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("SingleExcitation", {0, 2}, false, {0.3});

        mps_state_mpo.applyMPOOperation(mpo_single_excitation, {0, 2},
                                        max_mpo_bond);

        auto ref = tn_state->getDataVector();
        auto res = mps_state_mpo.getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }
}
