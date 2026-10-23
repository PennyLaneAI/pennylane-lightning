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
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "GateMatrices.hpp"
#include "HermitianEigen.hpp"

using namespace Pennylane::LightningNoisy;
using C = std::complex<double>;

namespace {
void checkDecomposition(const CVec<double> &h, std::size_t d) {
    auto [w, v] = hermitianEigen<double>(h, d);
    REQUIRE(w.size() == d);
    // V unitary
    auto vv = matmul<double>(adjoint<double>(v, d), v, d);
    auto id = identityMatrix<double>(d);
    for (std::size_t i = 0; i < d * d; i++) {
        REQUIRE(std::abs(vv[i] - id[i]) < 1e-12);
    }
    // V diag(w) V^dag == H
    CVec<double> dw(d * d, C{0, 0});
    for (std::size_t i = 0; i < d; i++) {
        dw[i * d + i] = w[i];
    }
    auto rec =
        matmul<double>(matmul<double>(v, dw, d), adjoint<double>(v, d), d);
    for (std::size_t i = 0; i < d * d; i++) {
        INFO("d=" << d << " index " << i);
        REQUIRE(std::abs(rec[i] - h[i]) < 1e-11);
    }
}
} // namespace

TEST_CASE("hermitianEigen diagonalises random Hermitian matrices",
          "[Observables]") {
    std::mt19937 g(1234);
    std::normal_distribution<double> nd(0, 1);
    for (std::size_t d : {1U, 2U, 3U, 4U, 8U, 16U}) {
        for (int rep = 0; rep < 5; rep++) {
            CVec<double> a(d * d);
            for (auto &x : a) {
                x = {nd(g), nd(g)};
            }
            auto h = a;
            for (std::size_t i = 0; i < d; i++) {
                for (std::size_t j = 0; j < d; j++) {
                    h[i * d + j] =
                        0.5 * (a[i * d + j] + std::conj(a[j * d + i]));
                }
            }
            checkDecomposition(h, d);
        }
    }
}

TEST_CASE("hermitianEigen handles Paulis and degenerate spectra",
          "[Observables]") {
    for (char p : {'X', 'Y', 'Z'}) {
        checkDecomposition(pauliMatrix<double>(p), 2);
    }
    // X (x) X has a doubly degenerate spectrum
    auto xx =
        kron<double>(pauliMatrix<double>('X'), 2, pauliMatrix<double>('X'), 2);
    checkDecomposition(xx, 4);
    checkDecomposition(identityMatrix<double>(4), 4);
    // Hadamard observable
    const double s = 1.0 / std::sqrt(2.0);
    checkDecomposition({C{s, 0}, C{s, 0}, C{s, 0}, C{-s, 0}}, 2);
    REQUIRE_THROWS(hermitianEigen<double>(CVec<double>(3), 2));
}
