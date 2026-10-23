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
#include <tuple>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "MPSKokkos.hpp"
#include "MPSLinAlg.hpp"

using namespace Pennylane::LightningTensornet;
using Catch::Approx;

namespace {
template <class T> T tol() {
    return std::is_same_v<T, float> ? T{5e-4} : T{1e-10};
}

template <class T>
std::vector<std::complex<T>> randomMatrix(std::size_t m, std::size_t n,
                                          std::mt19937 &g,
                                          std::size_t rank = 0) {
    std::normal_distribution<double> nd(0, 1);
    auto rnd = [&](std::size_t r, std::size_t c) {
        std::vector<std::complex<double>> a(r * c);
        for (auto &x : a) {
            x = {nd(g), nd(g)};
        }
        return a;
    };
    std::vector<std::complex<double>> a;
    if (rank == 0) {
        a = rnd(m, n);
    } else {
        const auto u = rnd(m, rank);
        const auto v = rnd(rank, n);
        a.assign(m * n, {0, 0});
        for (std::size_t i = 0; i < m; i++) {
            for (std::size_t k = 0; k < rank; k++) {
                for (std::size_t j = 0; j < n; j++) {
                    a[i * n + j] += u[i * rank + k] * v[k * n + j];
                }
            }
        }
    }
    std::vector<std::complex<T>> out(a.size());
    for (std::size_t i = 0; i < a.size(); i++) {
        out[i] = {static_cast<T>(a[i].real()), static_cast<T>(a[i].imag())};
    }
    return out;
}

/// Check M = U S Vh, U^dag U = I, Vh Vh^dag = I, S descending
template <class T>
void checkSVD(const std::vector<std::complex<T>> &M, std::size_t m,
              std::size_t n, const SVDResult<T> &r) {
    const std::size_t k = std::min(m, n);
    REQUIRE(r.S.size() == k);
    REQUIRE(r.U.size() == m * k);
    REQUIRE(r.Vh.size() == k * n);
    T scale = 1;
    for (const auto &x : M) {
        scale = std::max(scale, std::abs(x));
    }
    for (std::size_t i = 0; i + 1 < k; i++) {
        CHECK(r.S[i] + tol<T>() * scale >= r.S[i + 1]);
    }
    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t j = 0; j < n; j++) {
            std::complex<T> s{0, 0};
            for (std::size_t q = 0; q < k; q++) {
                s += r.U[i * k + q] * r.S[q] * r.Vh[q * n + j];
            }
            CHECK(std::abs(s - M[i * n + j]) <= tol<T>() * scale * 10);
        }
    }
    // orthonormality of the columns of U / rows of Vh for non-zero S
    for (std::size_t a = 0; a < k; a++) {
        for (std::size_t b = 0; b < k; b++) {
            if (r.S[a] <= tol<T>() * scale || r.S[b] <= tol<T>() * scale) {
                continue;
            }
            std::complex<T> u{0, 0};
            std::complex<T> v{0, 0};
            for (std::size_t i = 0; i < m; i++) {
                u += std::conj(r.U[i * k + a]) * r.U[i * k + b];
            }
            for (std::size_t j = 0; j < n; j++) {
                v += r.Vh[a * n + j] * std::conj(r.Vh[b * n + j]);
            }
            const T target = a == b ? T{1} : T{0};
            CHECK(std::abs(u - target) <= tol<T>() * 100);
            CHECK(std::abs(v - target) <= tol<T>() * 100);
        }
    }
}
} // namespace

TEMPLATE_TEST_CASE("MPSLinAlg: batched GEMM matches a naive product",
                   "[LightningTensornet]", float, double) {
    using T = TestType;
    // make sure Kokkos is initialised
    const MPSKokkos<T> init(1);
    std::mt19937 g(7);
    const std::size_t batch = 3;
    const std::size_t m = 4;
    const std::size_t n = 5;
    const std::size_t k = 6;
    for (int ta = 0; ta < 3; ta++) {
        for (int tb = 0; tb < 3; tb++) {
            const auto A = randomMatrix<T>(batch, m * k, g);
            const auto B = randomMatrix<T>(batch, k * n, g);
            const auto dA = toDevice<T>(A);
            const auto dB = toDevice<T>(B);
            KVector<T> dC("C", batch * m * n);
            const auto opA = static_cast<Trans>(ta);
            const auto opB = static_cast<Trans>(tb);
            const std::size_t lda = opA == Trans::N ? k : m;
            const std::size_t ldb = opB == Trans::N ? n : k;
            batchedGemm<T>(opA, opB, batch, m, n, k, dA, 0, lda, m * k, dB, 0,
                           ldb, k * n, dC, 0, n, m * n);
            const auto C = toHost<T>(dC);
            for (std::size_t b = 0; b < batch; b++) {
                for (std::size_t i = 0; i < m; i++) {
                    for (std::size_t j = 0; j < n; j++) {
                        std::complex<T> s{0, 0};
                        for (std::size_t q = 0; q < k; q++) {
                            auto x = opA == Trans::N
                                         ? A[b * m * k + i * lda + q]
                                         : A[b * m * k + q * lda + i];
                            auto y = opB == Trans::N
                                         ? B[b * k * n + q * ldb + j]
                                         : B[b * k * n + j * ldb + q];
                            if (opA == Trans::C) {
                                x = std::conj(x);
                            }
                            if (opB == Trans::C) {
                                y = std::conj(y);
                            }
                            s += x * y;
                        }
                        CHECK(std::abs(s - C[b * m * n + i * n + j]) <=
                              tol<T>() * 50);
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSLinAlg: Jacobi and LAPACK SVD", "[LightningTensornet]",
                   float, double) {
    using T = TestType;
    std::mt19937 g(11);
    const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>
        shapes = {{1, 1, 0}, {1, 5, 0},   {5, 1, 0},   {4, 4, 0},  {8, 3, 0},
                  {3, 8, 0}, {16, 16, 3}, {12, 20, 2}, {20, 12, 5}};
    for (const auto &[m, n, rank] : shapes) {
        const auto M = randomMatrix<T>(m, n, g, rank);
        const auto rj = jacobiSVD<T>(M, m, n);
        checkSVD<T>(M, m, n, rj);
        const auto rs = svd<T>(M, m, n);
        checkSVD<T>(M, m, n, rs);
        for (std::size_t i = 0; i < rj.S.size(); i++) {
            CHECK(rj.S[i] == Approx(rs.S[i]).margin(tol<T>() * 100));
        }
        if (rank != 0) {
            // numerical rank
            const T thr = rs.S[0] * T{1e-4};
            std::size_t r = 0;
            for (auto s : rs.S) {
                r += s > thr ? 1U : 0U;
            }
            CHECK(r == rank);
        }
    }
    // zero matrix
    const std::vector<std::complex<T>> Z(6, {0, 0});
    const auto rz = svd<T>(Z, 2, 3);
    CHECK(rz.S[0] == Approx(0).margin(tol<T>()));
#ifdef SCIPY_OPENBLAS_ENABLED
    CHECK(svdUsesLapack());
#else
    CHECK_FALSE(svdUsesLapack());
#endif
}
