// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "JacobianTape.hpp"

#include <catch2/catch.hpp>

using Pennylane::Algorithms::OpsData;

TEST_CASE("Algorithms::OpsData works correctly", "[Algorithms]") {
    SECTION("Test two instances are equal") {
        auto ops1 = OpsData<double>({"RX", "CNOT"}, {{0.312}, {}}, {{0}, {2}},
                                    {false, true});
        auto ops2 = OpsData<double>({"RX", "CNOT"}, {{0.312}, {}}, {{0}, {2}},
                                    {false, true});

        REQUIRE(ops1 == ops2);
    }
    SECTION("Test two instances are not equal") {
        auto ops1 = OpsData<double>({"RX", "CNOT"}, {{0.312}, {}}, {{0}, {2}},
                                    {false, true});
        auto ops2 = OpsData<double>({"RX", "CZ"}, {{0.312}, {}}, {{0}, {2}},
                                    {false, true});
        auto ops3 = OpsData<double>({"RX", "CNOT"}, {{0.128}, {}}, {{0}, {2}},
                                    {false, true});
        auto ops4 = OpsData<double>({"RX", "CNOT"}, {{0.312}, {}}, {{1}, {2}},
                                    {false, true});
        auto ops5 = OpsData<double>({"RX", "CNOT"}, {{0.312}, {}}, {{0}, {2}},
                                    {true, true});

        REQUIRE(ops1 != ops2);
        REQUIRE(ops1 != ops3);
        REQUIRE(ops1 != ops4);
        REQUIRE(ops1 != ops5);
        REQUIRE(ops2 != ops3);
        REQUIRE(ops2 != ops4);
        REQUIRE(ops2 != ops5);
        REQUIRE(ops3 != ops4);
        REQUIRE(ops3 != ops5);
        REQUIRE(ops4 != ops5);
    }
}
