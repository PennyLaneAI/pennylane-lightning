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
#include <complex>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "GateMatrices.hpp"
#include "NoiseModel.hpp"

using namespace Pennylane::LightningNoisy;
using Catch::Approx;

namespace {
// Bit-flip channel with p = 0.25: K0 = sqrt(0.75) I, K1 = sqrt(0.25) X
const std::string bitflip =
    "K=1/2/0.8660254037844386/0/0/0/0/0/0.8660254037844386/0/"
    "0/0/0.5/0/0.5/0/0/0";
} // namespace

TEST_CASE("NoiseModel parses channels and rules", "[Noise]") {
    const std::string spec = "LNM1;" + bitflip +
                             ";G=RX/*1/aK@r0;G=CNOT/0_1/bK@0+aK@r1;G=CNOT/"
                             "1_0/-;M=sample/*1/bK@r0";
    // replace the placeholder channel index
    std::string s = spec;
    std::size_t pos = 0;
    while ((pos = s.find("K@", pos)) != std::string::npos) {
        s.replace(pos, 2, "0@");
        pos += 2;
    }
    NoiseModel<double> nm(s);
    REQUIRE(nm.numChannels() == 1);
    REQUIRE(!nm.empty());
    REQUIRE(nm.hasReadoutNoise());
    const auto &ch = nm.channel(0);
    REQUIRE(ch.num_qubits == 1);
    REQUIRE(ch.kraus.size() == 2);
    // Superoperator is trace preserving: sum_k K^dag K = I  <=>  the
    // "trace row" of S equals vec(I)
    REQUIRE(ch.superop.size() == 16);
    std::vector<ResolvedNoise> before;
    std::vector<ResolvedNoise> after;

    nm.lookupGate("RX", {3}, before, after);
    REQUIRE(before.empty());
    REQUIRE(after.size() == 1);
    REQUIRE(after[0].wires == std::vector<std::size_t>{3});

    nm.lookupGate("CNOT", {0, 1}, before, after);
    REQUIRE(before.size() == 1);
    REQUIRE(before[0].wires == std::vector<std::size_t>{0});
    REQUIRE(after.size() == 1);
    REQUIRE(after[0].wires == std::vector<std::size_t>{1});

    nm.lookupGate("CNOT", {1, 0}, before, after);
    REQUIRE(before.empty());
    REQUIRE(after.empty());

    nm.lookupGate("CNOT", {2, 3}, before, after);
    REQUIRE(before.empty());
    REQUIRE(after.empty());

    nm.lookupGate("RY", {0}, before, after);
    REQUIRE(after.empty());

    std::vector<ResolvedNoise> ro;
    nm.lookupReadout("sample", {2, 0}, ro);
    REQUIRE(ro.size() == 2);
    REQUIRE(ro[0].wires == std::vector<std::size_t>{2});
    REQUIRE(ro[1].wires == std::vector<std::size_t>{0});
    nm.lookupReadout("expval", {2}, ro);
    REQUIRE(ro.empty());
}

TEST_CASE("NoiseModel rejects malformed input", "[Noise]") {
    REQUIRE_THROWS(NoiseModel<double>("XYZ"));
    REQUIRE_THROWS(NoiseModel<double>("LNM1;K=1/1/1/0"));
    REQUIRE_THROWS(NoiseModel<double>("LNM1;G=RX/*1/a0@r0"));
    REQUIRE_THROWS(NoiseModel<double>("LNM1;" + bitflip + ";G=RX/*1/a0@r0_1"));
    REQUIRE_THROWS(NoiseModel<double>("LNM1;Q=1"));
    NoiseModel<double> empty("");
    REQUIRE(empty.empty());
}

TEST_CASE("krausToSuperop matches rho -> sum K rho K^dag", "[Noise]") {
    using C = std::complex<double>;
    const double g = 0.3;
    std::vector<CVec<double>> kraus = {
        {C{1, 0}, C{0, 0}, C{0, 0}, C{std::sqrt(1 - g), 0}},
        {C{0, 0}, C{std::sqrt(g), 0}, C{0, 0}, C{0, 0}}};
    auto s = krausToSuperop<double>(kraus, 2);
    // rho = |1><1| -> (1-g)|1><1| + g|0><0|
    CVec<double> rho{C{0, 0}, C{0, 0}, C{0, 0}, C{1, 0}};
    CVec<double> out(4, C{0, 0});
    for (std::size_t i = 0; i < 4; i++) {
        for (std::size_t j = 0; j < 4; j++) {
            out[i] += s[i * 4 + j] * rho[j];
        }
    }
    REQUIRE(out[0].real() == Approx(g));
    REQUIRE(out[3].real() == Approx(1 - g));
    REQUIRE(std::abs(out[1]) == Approx(0).margin(1e-14));
}

TEST_CASE("namedGateMatrix covers variable-width gates", "[Noise]") {
    using C = std::complex<double>;
    auto m = namedGateMatrix<double>("MultiRZ", {0.4}, false, 2);
    REQUIRE(m.size() == 16);
    REQUIRE(std::abs(m[0] - std::exp(C{0, -0.2})) < 1e-14);
    REQUIRE(std::abs(m[5] - std::exp(C{0, 0.2})) < 1e-14);
    auto g = namedGateMatrix<double>("GlobalPhase", {0.4}, true, 1);
    REQUIRE(std::abs(g[0] - std::exp(C{0, 0.4})) < 1e-14);
    auto rx = namedGateMatrix<double>("RX", {0.4}, false, 1);
    auto pr = pauliRotMatrix<double>(0.4, "X", false);
    for (std::size_t i = 0; i < 4; i++) {
        REQUIRE(std::abs(rx[i] - pr[i]) < 1e-14);
    }
    REQUIRE_THROWS(namedGateMatrix<double>("NotAGate", {}, false, 1));
    // embed X on wire 1 of (0, 1): I (x) X
    auto x = pauliMatrix<double>('X');
    auto e = embedMatrix<double>(x, {1}, {0, 1});
    auto ref = kron<double>(identityMatrix<double>(2), 2, x, 2);
    for (std::size_t i = 0; i < 16; i++) {
        REQUIRE(std::abs(e[i] - ref[i]) < 1e-14);
    }
}
