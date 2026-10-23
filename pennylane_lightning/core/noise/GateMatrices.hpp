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
 * @file GateMatrices.hpp
 * Dense gate matrices used by the noisy Kokkos backends
 * (lightning.tensornet).
 *
 * Convention: row-major matrices, the first wire of an operation is the most
 * significant bit of the local basis index (same as PennyLane / Lightning).
 */
#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"

namespace Pennylane::LightningNoisy {

template <class T> using CVec = std::vector<std::complex<T>>;

inline constexpr std::size_t pow2(std::size_t n) { return std::size_t{1} << n; }

/**
 * @brief Identity matrix of dimension dim.
 */
template <class T> CVec<T> identityMatrix(std::size_t dim) {
    CVec<T> m(dim * dim, {0, 0});
    for (std::size_t i = 0; i < dim; i++) {
        m[i * dim + i] = {1, 0};
    }
    return m;
}

/**
 * @brief Matrix product C = A B for square matrices of dimension dim.
 */
template <class T>
CVec<T> matmul(const CVec<T> &a, const CVec<T> &b, std::size_t dim) {
    CVec<T> c(dim * dim, {0, 0});
    for (std::size_t i = 0; i < dim; i++) {
        for (std::size_t k = 0; k < dim; k++) {
            const auto aik = a[i * dim + k];
            if (aik == std::complex<T>{0, 0}) {
                continue;
            }
            for (std::size_t j = 0; j < dim; j++) {
                c[i * dim + j] += aik * b[k * dim + j];
            }
        }
    }
    return c;
}

/**
 * @brief Kronecker product of two square matrices.
 */
template <class T>
CVec<T> kron(const CVec<T> &a, std::size_t da, const CVec<T> &b,
             std::size_t db) {
    const std::size_t d = da * db;
    CVec<T> c(d * d);
    for (std::size_t i = 0; i < da; i++) {
        for (std::size_t j = 0; j < da; j++) {
            const auto aij = a[i * da + j];
            for (std::size_t k = 0; k < db; k++) {
                for (std::size_t l = 0; l < db; l++) {
                    c[(i * db + k) * d + (j * db + l)] = aij * b[k * db + l];
                }
            }
        }
    }
    return c;
}

template <class T> CVec<T> conjugate(const CVec<T> &a) {
    CVec<T> c(a.size());
    for (std::size_t i = 0; i < a.size(); i++) {
        c[i] = std::conj(a[i]);
    }
    return c;
}

template <class T> CVec<T> adjoint(const CVec<T> &a, std::size_t dim) {
    CVec<T> c(a.size());
    for (std::size_t i = 0; i < dim; i++) {
        for (std::size_t j = 0; j < dim; j++) {
            c[j * dim + i] = std::conj(a[i * dim + j]);
        }
    }
    return c;
}

/**
 * @brief 2x2 Pauli matrix for a character in {I, X, Y, Z}.
 */
template <class T> CVec<T> pauliMatrix(char p) {
    using C = std::complex<T>;
    switch (p) {
    case 'I':
        return {C{1, 0}, C{0, 0}, C{0, 0}, C{1, 0}};
    case 'X':
        return {C{0, 0}, C{1, 0}, C{1, 0}, C{0, 0}};
    case 'Y':
        return {C{0, 0}, C{0, -1}, C{0, 1}, C{0, 0}};
    case 'Z':
        return {C{1, 0}, C{0, 0}, C{0, 0}, C{-1, 0}};
    default:
        PL_ABORT(std::string("Invalid Pauli character: ") + p);
    }
    return {};
}

/**
 * @brief Dense matrix of a Pauli word, e.g. "XYZ".
 */
template <class T> CVec<T> pauliWordMatrix(const std::string &word) {
    CVec<T> m{{1, 0}};
    std::size_t dim = 1;
    for (char p : word) {
        m = kron<T>(m, dim, pauliMatrix<T>(p), 2);
        dim *= 2;
    }
    return m;
}

/**
 * @brief PauliRot(theta, word) = exp(-i theta/2 P).
 */
template <class T>
CVec<T> pauliRotMatrix(T theta, const std::string &word, bool inverse) {
    const T th = inverse ? -theta : theta;
    const std::size_t dim = pow2(word.size());
    auto p = pauliWordMatrix<T>(word);
    CVec<T> m(dim * dim);
    const std::complex<T> c{std::cos(th / 2), 0};
    const std::complex<T> s{0, -std::sin(th / 2)};
    for (std::size_t i = 0; i < dim * dim; i++) {
        m[i] = s * p[i];
    }
    for (std::size_t i = 0; i < dim; i++) {
        m[i * dim + i] += c;
    }
    return m;
}

/**
 * @brief Dense matrix of a named PennyLane gate.
 *
 * @param name Gate name (Lightning naming, e.g. "RX", "CNOT", "MultiRZ").
 * @param params Gate parameters.
 * @param inverse Whether to return the adjoint.
 * @param num_wires Number of target wires (needed by variable-width gates).
 */
template <class T>
CVec<T> namedGateMatrix(const std::string &name, const std::vector<T> &params,
                        bool inverse, std::size_t num_wires) {
    using namespace Pennylane::Gates;
    using Pennylane::Util::array_contains;
    using Pennylane::Util::reverse_lookup;
    if (name == "Identity") {
        return identityMatrix<T>(pow2(num_wires));
    }
    if (name == "MultiRZ") {
        PL_ABORT_IF(params.empty(), "MultiRZ needs one parameter");
        const T th = inverse ? -params[0] : params[0];
        const std::size_t dim = pow2(num_wires);
        CVec<T> m(dim * dim, {0, 0});
        for (std::size_t i = 0; i < dim; i++) {
            const T sign =
                (__builtin_popcountll(static_cast<unsigned long long>(i)) % 2)
                    ? T{-1}
                    : T{1};
            m[i * dim + i] = std::exp(std::complex<T>{0, -th / 2 * sign});
        }
        return m;
    }
    if (name == "GlobalPhase") {
        PL_ABORT_IF(params.empty(), "GlobalPhase needs one parameter");
        const T ph = inverse ? -params[0] : params[0];
        const std::size_t dim = pow2(num_wires);
        CVec<T> m(dim * dim, {0, 0});
        const auto v = std::exp(std::complex<T>{0, -ph});
        for (std::size_t i = 0; i < dim; i++) {
            m[i * dim + i] = v;
        }
        return m;
    }
    if (name == "PCPhase") {
        PL_ABORT_IF(params.size() < 2,
                    "PCPhase needs two parameters (phase, dimension)");
        const T ph = inverse ? -params[0] : params[0];
        const auto d = static_cast<std::size_t>(params[1]);
        const std::size_t dim = pow2(num_wires);
        CVec<T> m(dim * dim, {0, 0});
        for (std::size_t i = 0; i < dim; i++) {
            m[i * dim + i] = std::exp(std::complex<T>{0, (i < d) ? ph : -ph});
        }
        return m;
    }
    PL_ABORT_IF_NOT(
        array_contains(Constant::gate_names, std::string_view{name}),
        std::string("No dense matrix available for gate ") + name);
    const auto op =
        reverse_lookup(Constant::gate_names, std::string_view{name});
    return getMatrix<std::complex, T>(op, params, inverse);
}

/**
 * @brief Embed a matrix acting on `op_wires` into the space of
 * `target_wires` (which must contain every wire of op_wires).
 */
template <class T>
CVec<T> embedMatrix(const CVec<T> &mat,
                    const std::vector<std::size_t> &op_wires,
                    const std::vector<std::size_t> &target_wires) {
    const std::size_t n = target_wires.size();
    const std::size_t k = op_wires.size();
    const std::size_t dim = pow2(n);
    const std::size_t dk = pow2(k);
    // bit position (from the MSB side) of each op wire inside target_wires
    std::vector<std::size_t> pos(k);
    std::size_t op_mask = 0;
    for (std::size_t i = 0; i < k; i++) {
        bool found = false;
        for (std::size_t j = 0; j < n; j++) {
            if (target_wires[j] == op_wires[i]) {
                pos[i] = n - 1 - j;
                found = true;
            }
        }
        PL_ABORT_IF_NOT(found, "embedMatrix: wire not in target wires.");
        op_mask |= std::size_t{1} << pos[i];
    }
    auto local = [&](std::size_t idx) {
        std::size_t r = 0;
        for (std::size_t i = 0; i < k; i++) {
            r |= ((idx >> pos[i]) & 1U) << (k - 1 - i);
        }
        return r;
    };
    CVec<T> out(dim * dim, {0, 0});
    for (std::size_t r = 0; r < dim; r++) {
        for (std::size_t c = 0; c < dim; c++) {
            if ((r & ~op_mask) != (c & ~op_mask)) {
                continue;
            }
            out[r * dim + c] = mat[local(r) * dk + local(c)];
        }
    }
    return out;
}

/**
 * @brief Superoperator S = sum_k K_k (x) conj(K_k), acting on the vectorised
 * density matrix |rho>> = sum rho[r, c] |r>|c> (row index most significant).
 */
template <class T>
CVec<T> krausToSuperop(const std::vector<CVec<T>> &kraus, std::size_t dim) {
    CVec<T> s(dim * dim * dim * dim, {0, 0});
    for (const auto &k : kraus) {
        auto kk = kron<T>(k, dim, conjugate<T>(k), dim);
        for (std::size_t i = 0; i < s.size(); i++) {
            s[i] += kk[i];
        }
    }
    return s;
}

/**
 * @brief Superoperator of a unitary: U (x) conj(U).
 */
template <class T> CVec<T> unitaryToSuperop(const CVec<T> &u, std::size_t dim) {
    return kron<T>(u, dim, conjugate<T>(u), dim);
}

} // namespace Pennylane::LightningNoisy
