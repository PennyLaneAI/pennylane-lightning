// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Program.hpp"

namespace {
using namespace Pennylane::LightningCliffish::Program;
using Catch::Matchers::ContainsSubstring;
} // namespace

TEST_CASE("Program allocates qubit ids that are never reused", "[Program]") {
    Program program;
    CHECK(program.allocate(3) == std::vector<QubitId>{0, 1, 2});
    CHECK(program.numQubits() == 3);

    program.release(1);
    CHECK_FALSE(program.isLive(1));
    CHECK(program.numQubits() == 2);

    CHECK(program.allocate(1) == std::vector<QubitId>{3});
    CHECK(program.numQubits() == 3);
}

TEST_CASE("Program holds ops as received until a flush takes them",
          "[Program]") {
    Program program;
    program.allocate(2);
    program.append(Op{.kind = OpKind::Gate, .name = "CNOT", .wires = {0, 1}});
    const ResultId first =
        program.appendMeasurement(Op{.kind = OpKind::Measure, .wires = {1}});
    const ResultId second = program.appendMeasurement(
        Op{.kind = OpKind::PauliMeasure, .wires = {0, 1}, .pauli_word = "XX"});

    CHECK(first != result_true);
    CHECK(second == first + 1);
    CHECK(program.isResult(first));
    CHECK(program.isResult(second));
    CHECK_FALSE(program.isResult(second + 1));
    CHECK_FALSE(program.isResult(result_true));

    const auto &window = program.window();
    REQUIRE(window.size() == 4);
    CHECK(window[0].kind == OpKind::Allocate);
    CHECK(window[1].name == "CNOT");
    CHECK(window[2].result == first);
    CHECK(window[3].pauli_word == "XX");

    const auto taken = program.takeWindow();
    CHECK(taken.size() == 4);
    CHECK(program.window().empty());

    // Ids carry over to the next window.
    CHECK(program.appendMeasurement(
              Op{.kind = OpKind::Measure, .wires = {0}}) == second + 1);
}

TEST_CASE("Program rejects invalid ops", "[Program]") {
    Program program;
    program.allocate(2);
    program.release(0);

    REQUIRE_THROWS_WITH(
        program.append(Op{.kind = OpKind::Gate, .name = "H", .wires = {0}}),
        ContainsSubstring("not allocated"));
    REQUIRE_THROWS_WITH(
        program.append(Op{.kind = OpKind::Gate, .name = "H", .wires = {5}}),
        ContainsSubstring("not allocated"));
    REQUIRE_THROWS_WITH(
        program.append(
            Op{.kind = OpKind::Gate, .name = "CNOT", .wires = {1, 1}}),
        ContainsSubstring("same qubit twice"));
    REQUIRE_THROWS_WITH(
        program.append(Op{.kind = OpKind::Measure, .wires = {1}}),
        ContainsSubstring("appendMeasurement"));
    REQUIRE_THROWS_WITH(
        program.appendMeasurement(Op{.kind = OpKind::Gate, .wires = {1}}),
        ContainsSubstring("not a measurement"));
    REQUIRE_THROWS_WITH(program.append(Op{.kind = OpKind::PauliRot,
                                          .params = {0.5},
                                          .wires = {1},
                                          .pauli_word = "XX"}),
                        ContainsSubstring("one character per wire"));
    REQUIRE_THROWS_WITH(program.release(0), ContainsSubstring("not allocated"));
}
