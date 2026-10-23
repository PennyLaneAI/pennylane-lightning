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
/**
 * @file HermitianEigen.hpp
 * Small dense Hermitian eigensolver (cyclic complex Jacobi). Used on the host
 * to diagonalise observables for finite-shot measurements. Dimensions are
 * those of an observable (at most a few qubits), so O(d^3) per sweep is fine.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "GateMatrices.hpp"

namespace Pennylane::LightningNoisy {

/**
 * @brief Eigen-decomposition H = V diag(w) V^dagger of a Hermitian matrix.
 *
 * @param h Row-major d x d Hermitian matrix.
 * @param d Dimension.
 * @return (eigenvalues, V) with the eigenvectors stored as the columns of the
 * row-major matrix V.
 */
template <class T>
std::pair<std::vector<T>, CVec<T>> hermitianEigen(const CVec<T> &h,
                                                  std::size_t d) {
    using C = std::complex<T>;
    PL_ABORT_IF_NOT(h.size() == d * d, "hermitianEigen: size mismatch");
    CVec<T> a = h;
    CVec<T> v = identityMatrix<T>(d);
    auto offNorm = [&]() {
        T s = 0;
        for (std::size_t i = 0; i < d; i++) {
            for (std::size_t j = 0; j < d; j++) {
                if (i != j) {
                    s += std::norm(a[i * d + j]);
                }
            }
        }
        return s;
    };
    T scale = 0;
    for (const auto &x : a) {
        scale += std::norm(x);
    }
    const T eps = std::numeric_limits<T>::epsilon();
    const T tol = eps * eps * std::max<T>(scale, T{1}) * static_cast<T>(d * d);
    for (int sweep = 0; sweep < 100 && offNorm() > tol; sweep++) {
        for (std::size_t p = 0; p + 1 < d; p++) {
            for (std::size_t q = p + 1; q < d; q++) {
                const C apq = a[p * d + q];
                const T mag = std::abs(apq);
                if (mag <= eps * eps) {
                    continue;
                }
                const C ph = apq / mag;      // e^{i phi}
                const C phc = std::conj(ph); // e^{-i phi}
                const T app = a[p * d + p].real();
                const T aqq = a[q * d + q].real();
                const T theta = T{0.5} * std::atan2(T{2} * mag, aqq - app);
                const T c = std::cos(theta);
                const T s = std::sin(theta);
                // G = [[c, s], [-s e^{-i phi}, c e^{-i phi}]] on (p, q)
                // columns: A <- A G, V <- V G
                for (std::size_t i = 0; i < d; i++) {
                    const C aip = a[i * d + p];
                    const C aiq = a[i * d + q];
                    a[i * d + p] = c * aip - s * phc * aiq;
                    a[i * d + q] = s * aip + c * phc * aiq;
                    const C vip = v[i * d + p];
                    const C viq = v[i * d + q];
                    v[i * d + p] = c * vip - s * phc * viq;
                    v[i * d + q] = s * vip + c * phc * viq;
                }
                // rows: A <- G^dagger A
                for (std::size_t j = 0; j < d; j++) {
                    const C apj = a[p * d + j];
                    const C aqj = a[q * d + j];
                    a[p * d + j] = c * apj - s * ph * aqj;
                    a[q * d + j] = s * apj + c * ph * aqj;
                }
                a[p * d + q] = C{0, 0};
                a[q * d + p] = C{0, 0};
            }
        }
    }
    std::vector<T> w(d);
    for (std::size_t i = 0; i < d; i++) {
        w[i] = a[i * d + i].real();
    }
    return {w, v};
}

} // namespace Pennylane::LightningNoisy
