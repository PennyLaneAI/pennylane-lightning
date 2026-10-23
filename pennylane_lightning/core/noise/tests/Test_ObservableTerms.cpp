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
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ObservableTerms.hpp"

using namespace Pennylane::LightningNoisy;
using Catch::Approx;
using C = std::complex<double>;

TEST_CASE("Pauli algebra of observable terms", "[Observables]") {
    auto x0 = namedObservable<double>("PauliX", {0});
    auto y0 = namedObservable<double>("PauliY", {0});
    auto z1 = namedObservable<double>("PauliZ", {1});

    // X0 Y0 = i Z0
    auto xy = productObservable<double>({x0, y0});
    REQUIRE(xy.size() == 1);
    REQUIRE(xy[0].paulis == "Z");
    REQUIRE(xy[0].wires == std::vector<std::size_t>{0});
    REQUIRE(std::abs(xy[0].coeff - C{0, 1}) < 1e-15);

    // Y0 X0 = -i Z0
    auto yx = productObservable<double>({y0, x0});
    REQUIRE(std::abs(yx[0].coeff - C{0, -1}) < 1e-15);

    // X0 X0 = I
    auto xx = productObservable<double>({x0, x0});
    REQUIRE(xx.size() == 1);
    REQUIRE(xx[0].wires.empty());

    // (X0 + Z1)^2 = 2 I + X0 Z1 + Z1 X0 = 2 I + 2 X0 Z1
    auto h = linearCombination<double>({1.0, 1.0}, {x0, z1});
    auto h2 = squareObservable<double>(h);
    REQUIRE(h2.size() == 2);
    for (const auto &t : h2) {
        if (t.wires.empty()) {
            REQUIRE(t.coeff.real() == Approx(2.0));
        } else {
            REQUIRE(t.paulis == "XZ");
            REQUIRE(t.coeff.real() == Approx(2.0));
        }
    }
}

TEST_CASE("Hadamard and Hermitian observables", "[Observables]") {
    auto h = namedObservable<double>("Hadamard", {2});
    REQUIRE(h.size() == 2);
    // H^2 = I
    auto h2 = squareObservable<double>(h);
    REQUIRE(h2.size() == 1);
    REQUIRE(h2[0].wires.empty());
    REQUIRE(h2[0].coeff.real() == Approx(1.0));

    CVec<double> herm{C{1, 0}, C{0, -1}, C{0, 1}, C{2, 0}};
    auto obs = namedObservable<double>("Hermitian", {1}, herm);
    auto z0 = namedObservable<double>("PauliZ", {0});
    auto prod = productObservable<double>({z0, obs});
    REQUIRE(prod.size() == 1);
    REQUIRE(prod[0].dense);
    REQUIRE(prod[0].wires == std::vector<std::size_t>{0, 1});
    auto ref = kron<double>(pauliMatrix<double>('Z'), 2, herm, 2);
    for (std::size_t i = 0; i < 16; i++) {
        REQUIRE(std::abs(prod[0].matrix[i] - ref[i]) < 1e-14);
    }
    REQUIRE(observableWires(prod) == std::vector<std::size_t>{0, 1});
    REQUIRE_THROWS(namedObservable<double>("Hermitian", {0, 1}, herm));
    REQUIRE_THROWS(namedObservable<double>("Foo", {0}));
}
