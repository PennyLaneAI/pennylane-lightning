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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "LightningCliffishSimulator.hpp"

#ifdef LIGHTNING_CLIFFISH_DEVICE_EXTENSIONS

namespace {
using namespace Catalyst::Runtime::Simulator;
using Catch::Matchers::ContainsSubstring;
using LCSimulator = LightningCliffishSimulator;

/// The device's tables, called the way Catalyst's runtime calls them.
struct Tables {
    explicit Tables(LCSimulator &device)
        : delegated{*static_cast<const CatalystDelegatedTable *>(
              device.QueryInterface("Delegated", 1))},
          gadget{*static_cast<const CatalystGadgetTable *>(
              device.QueryInterface("Gadget", 1))} {}

    const CatalystDelegatedTable &delegated;
    const CatalystGadgetTable &gadget;

    [[nodiscard]] auto self() const -> void * { return delegated.header.self; }

    auto ppm(const std::string &word, std::vector<QubitIdType> wires,
             bool negated = false) const -> RESULT * {
        return delegated.pauli_measure(self(), word.c_str(), wires.data(),
                                       wires.size(), negated,
                                       CATALYST_NO_POSTSELECT);
    }
};

auto expval(LCSimulator &device, ObsId pauli, QubitIdType wire) -> double {
    return device.Expval(device.Observable(pauli, {}, {wire}));
}

/**
 * @brief A pi/8 gadget for exp(-i theta/2 Z) on `data`, theta = +-pi/4, in the
 * runtime calls Catalyst emits for the pauli-corrected method.
 *
 * With `wrong_correction`, the final Pauli correction is a pi/4 rotation, as
 * in Catalyst's clifford-corrected bug for negative angles.
 */
void piOverEightGadget(LCSimulator &device, const Tables &tables,
                       QubitIdType data, bool negative,
                       bool wrong_correction = false,
                       uint32_t verdict = CATALYST_GADGET_VERIFIED) {
    auto &table = tables.delegated;
    void *self = tables.self();
    const double theta = negative ? -M_PI / 4 : M_PI / 4;
    std::vector<QubitIdType> wires{data};
    tables.gadget.begin_gadget(self, "Z", wires.data(), 1, theta, verdict);

    // pbc.fabricate magic, lowered to H then T.
    const QubitIdType magic = device.AllocateQubit();
    device.NamedOperation("Hadamard", {}, {magic});
    device.NamedOperation("T", {}, {magic});
    RESULT *m0 = tables.ppm("ZZ", {data, magic}, negative);

    // select.ppm (m0 ? Y : X) on the magic qubit.
    table.push_predicate(self, m0);
    RESULT *my = tables.ppm("Y", {magic});
    table.pop_predicate(self);
    RESULT *not_m0 = table.result_not(self, m0);
    table.push_predicate(self, not_m0);
    RESULT *mx = tables.ppm("X", {magic});
    table.pop_predicate(self);
    table.release_result(self, not_m0);
    RESULT *m1 = table.result_select(self, m0, my, mx);
    table.release_result(self, my);
    table.release_result(self, mx);
    table.release_result(self, m0);

    // pbc.ppr Z(2) cond(m1): exp(-i pi/2 Z).
    table.push_predicate(self, m1);
    device.NamedOperation("PauliRot", {wrong_correction ? M_PI / 2 : M_PI},
                          {data}, false, {}, {}, {"Z"});
    table.pop_predicate(self);
    table.release_result(self, m1);
    device.ReleaseQubit(magic);
    tables.gadget.end_gadget(self);
}
} // namespace

TEST_CASE("A delegated pi/8 gadget collapses to its rotation", "[delegated]") {
    LCSimulator device;
    device.SetDeviceShots(0);
    const Tables tables(device);
    const QubitIdType q = device.AllocateQubit();
    tables.delegated.begin_program(tables.self(),
                                   CATALYST_POSTSELECT_FILL_SHOTS);

    // T on |+>, then T^dagger twice: the state is exp(+i pi/8 Z)|+>.
    device.NamedOperation("Hadamard", {}, {q});
    piOverEightGadget(device, tables, q, false);
    piOverEightGadget(device, tables, q, true);
    piOverEightGadget(device, tables, q, true);

    CHECK(std::abs(expval(device, ObsId::PauliX, q) - std::cos(M_PI / 4)) <
          1e-12);
    CHECK(std::abs(expval(device, ObsId::PauliY, q) + std::sin(M_PI / 4)) <
          1e-12);
    const auto &report = device.gadgetReport();
    CHECK(report.gadgets == 3);
    CHECK(report.collapsed == 3);
    // The two signs are two shapes.
    CHECK(report.shapes == 2);
}

TEST_CASE("A wide gadget is checked on one data qubit", "[delegated]") {
    LCSimulator device;
    device.SetDeviceShots(0);
    const Tables tables(device);
    auto qubits = device.AllocateQubits(5);
    tables.delegated.begin_program(tables.self(),
                                   CATALYST_POSTSELECT_FILL_SHOTS);
    auto &table = tables.delegated;
    void *self = tables.self();

    const std::string word = "XYZZY";
    tables.gadget.begin_gadget(self, word.c_str(), qubits.data(), qubits.size(),
                               M_PI / 4, CATALYST_GADGET_VERIFIED);
    const QubitIdType magic = device.AllocateQubit();
    device.NamedOperation("Hadamard", {}, {magic});
    device.NamedOperation("T", {}, {magic});
    std::vector<QubitIdType> wide(qubits);
    wide.push_back(magic);
    RESULT *m0 = tables.ppm(word + "Z", wide);
    table.push_predicate(self, m0);
    RESULT *my = tables.ppm("Y", {magic});
    table.pop_predicate(self);
    RESULT *not_m0 = table.result_not(self, m0);
    table.push_predicate(self, not_m0);
    RESULT *mx = tables.ppm("X", {magic});
    table.pop_predicate(self);
    RESULT *m1 = table.result_select(self, m0, my, mx);
    table.push_predicate(self, m1);
    device.NamedOperation("PauliRot", {M_PI}, qubits, false, {}, {}, {word});
    table.pop_predicate(self);
    device.ReleaseQubit(magic);
    tables.gadget.end_gadget(self);

    // exp(-i pi/8 P) on |0...0>: <P'> for P' anticommuting with P.
    const double z0 = expval(device, ObsId::PauliZ, qubits[0]);
    CHECK(std::abs(z0 - std::cos(M_PI / 4)) < 1e-12);
    CHECK(device.gadgetReport().collapsed == 1);
}

TEST_CASE("A gadget that does not implement its source fails loudly",
          "[delegated]") {
    SECTION("measured") {
        LCSimulator device;
        device.SetDeviceShots(0);
        const Tables tables(device);
        const QubitIdType q = device.AllocateQubit();
        tables.delegated.begin_program(tables.self(), 0);
        piOverEightGadget(device, tables, q, true, /*wrong_correction=*/true);
        REQUIRE_THROWS_WITH(expval(device, ObsId::PauliZ, q),
                            ContainsSubstring("does not merge") ||
                                ContainsSubstring("branch-and-merge"));
        CHECK(device.gadgetReport().collapsed == 0);
        CHECK_THAT(device.gadgetReport().uncollapsed.front(),
                   ContainsSubstring("do not merge"));
    }
    SECTION("source") {
        LCSimulator device("{'gadget_collapse': 'source'}");
        device.SetDeviceShots(0);
        const Tables tables(device);
        const QubitIdType q = device.AllocateQubit();
        tables.delegated.begin_program(tables.self(), 0);
        device.NamedOperation("Hadamard", {}, {q});
        piOverEightGadget(device, tables, q, true, /*wrong_correction=*/true);
        // The source rotation is applied, and the substitution recorded.
        CHECK(std::abs(expval(device, ObsId::PauliY, q) + std::sin(M_PI / 4)) <
              1e-12);
        CHECK(device.gadgetReport().substitutions == 1);
    }
    SECTION("off") {
        LCSimulator device("{'gadget_collapse': 'off'}");
        device.SetDeviceShots(0);
        const Tables tables(device);
        const QubitIdType q = device.AllocateQubit();
        tables.delegated.begin_program(tables.self(), 0);
        piOverEightGadget(device, tables, q, false);
        REQUIRE_THROWS_WITH(expval(device, ObsId::PauliZ, q),
                            ContainsSubstring("branch-and-merge"));
    }
}

TEST_CASE("A correct gadget with a 'mismatch' verdict fails loudly",
          "[delegated]") {
    LCSimulator device;
    device.SetDeviceShots(0);
    const Tables tables(device);
    const QubitIdType q = device.AllocateQubit();
    tables.delegated.begin_program(tables.self(), 0);
    piOverEightGadget(device, tables, q, false, false,
                      CATALYST_GADGET_MISMATCH);
    REQUIRE_THROWS_WITH(expval(device, ObsId::PauliZ, q),
                        ContainsSubstring("verdict is 'mismatch'"));
}

TEST_CASE("A delegated measurement outside a gadget fails loudly",
          "[delegated]") {
    LCSimulator device;
    device.SetDeviceShots(100);
    const Tables tables(device);
    const QubitIdType q = device.AllocateQubit();
    tables.delegated.begin_program(tables.self(), 0);
    device.NamedOperation("Hadamard", {}, {q});
    RESULT *m = device.Measure(q);
    tables.delegated.release_result(tables.self(), m);
    REQUIRE_THROWS_WITH(expval(device, ObsId::PauliZ, q),
                        ContainsSubstring("mcm_method='one-shot'"));
}

TEST_CASE("Without a delegated program the tables measure eagerly",
          "[delegated]") {
    LCSimulator device;
    device.SetDeviceShots(0);
    const Tables tables(device);
    const QubitIdType q = device.AllocateQubit();
    // Gadget brackets are ignored, and a negated Pauli measurement of |0>
    // reads 1.
    std::vector<QubitIdType> wires{q};
    tables.gadget.begin_gadget(tables.self(), "Z", wires.data(), 1, M_PI, 0);
    CHECK(*tables.ppm("Z", {q}, /*negated=*/true));
    tables.gadget.end_gadget(tables.self());
    RESULT *kept = tables.delegated.pauli_measure(
        tables.self(), "X", wires.data(), 1, false, /*postselect=*/1);
    CHECK(*kept);
    CHECK(std::abs(expval(device, ObsId::PauliX, q) + 1.0) < 1e-12);
    CHECK(device.gadgetReport().gadgets == 0);
}

TEST_CASE("The device provides version 1 of its tables", "[delegated]") {
    LCSimulator device;
    CHECK(device.QueryInterface("Delegated", 1) != nullptr);
    CHECK(device.QueryInterface("Gadget", 1) != nullptr);
    CHECK(device.QueryInterface("Delegated", 2) == nullptr);
    CHECK(device.QueryInterface("Records", 1) == nullptr);
    REQUIRE_THROWS_WITH(LCSimulator("{'gadget_collapse': 'maybe'}"),
                        ContainsSubstring("Unknown gadget_collapse mode"));
}

#endif
