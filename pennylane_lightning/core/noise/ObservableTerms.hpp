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
 * @file ObservableTerms.hpp
 * Backend-independent observable representation used by the noisy Kokkos
 * backends (lightning.tensornet): an observable is a linear combination of
 * terms, where every term is either a Pauli word or a dense matrix on a few
 * wires.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "GateMatrices.hpp"

namespace Pennylane::LightningNoisy {

/**
 * @brief One term of an observable.
 *
 * If `dense` is false, the term is coeff * P where P is the tensor product of
 * `paulis[i]` acting on `wires[i]` (identity elsewhere; an empty word is the
 * identity). If `dense` is true, the term is coeff * M where M is a
 * 2^k x 2^k row-major matrix acting on `wires` (in the given order).
 */
template <class T> struct ObsTerm {
    std::complex<T> coeff{1, 0};
    std::vector<std::size_t> wires{};
    std::string paulis{};
    bool dense{false};
    CVec<T> matrix{};
};

template <class T> using ObsTerms = std::vector<ObsTerm<T>>;

namespace detail {
/**
 * @brief Single-qubit Pauli product a*b = phase * c.
 */
inline std::pair<std::complex<double>, char> pauliProduct(char a, char b) {
    if (a == 'I') {
        return {{1, 0}, b};
    }
    if (b == 'I') {
        return {{1, 0}, a};
    }
    if (a == b) {
        return {{1, 0}, 'I'};
    }
    const std::string cyc = "XYZ";
    const auto ia = cyc.find(a);
    const auto ib = cyc.find(b);
    const char c = cyc[3 - ia - ib];
    // XY = iZ, YZ = iX, ZX = iY ; reversed order gives -i
    const bool forward = ((ia + 1) % 3) == ib;
    return {forward ? std::complex<double>{0, 1} : std::complex<double>{0, -1},
            c};
}
} // namespace detail

/**
 * @brief Sort the factors of a Pauli term by wire and drop identities.
 */
template <class T> void canonicalize(ObsTerm<T> &t) {
    if (t.dense) {
        return;
    }
    std::map<std::size_t, char> m;
    std::complex<T> phase{1, 0};
    for (std::size_t i = 0; i < t.wires.size(); i++) {
        auto it = m.find(t.wires[i]);
        if (it == m.end()) {
            m[t.wires[i]] = t.paulis[i];
        } else {
            auto [ph, c] = detail::pauliProduct(it->second, t.paulis[i]);
            phase *= std::complex<T>(static_cast<T>(ph.real()),
                                     static_cast<T>(ph.imag()));
            it->second = c;
        }
    }
    t.wires.clear();
    t.paulis.clear();
    for (auto [w, c] : m) {
        if (c != 'I') {
            t.wires.push_back(w);
            t.paulis.push_back(c);
        }
    }
    t.coeff *= phase;
}

/**
 * @brief Dense matrix of a term on its own wires (for Pauli terms: on its
 * non-identity wires). Coefficient is NOT included.
 */
template <class T> CVec<T> termMatrix(const ObsTerm<T> &t) {
    if (t.dense) {
        return t.matrix;
    }
    return pauliWordMatrix<T>(t.paulis);
}

/**
 * @brief Terms of a named single observable.
 */
template <class T>
ObsTerms<T> namedObservable(const std::string &name,
                            const std::vector<std::size_t> &wires,
                            const CVec<T> &matrix = {}) {
    ObsTerm<T> t;
    if (name == "Identity") {
        return {t};
    }
    if (name == "PauliX" || name == "PauliY" || name == "PauliZ") {
        PL_ABORT_IF_NOT(wires.size() == 1, "Pauli observables act on 1 wire");
        t.wires = wires;
        t.paulis = std::string(1, name[5]);
        return {t};
    }
    if (name == "Hadamard") {
        PL_ABORT_IF_NOT(wires.size() == 1, "Hadamard acts on 1 wire");
        const T s = static_cast<T>(M_SQRT1_2);
        ObsTerm<T> tx{{s, 0}, wires, "X", false, {}};
        ObsTerm<T> tz{{s, 0}, wires, "Z", false, {}};
        return {tx, tz};
    }
    if (name == "Hermitian") {
        PL_ABORT_IF_NOT(matrix.size() == pow2(2 * wires.size()),
                        "Hermitian matrix size does not match its wires");
        t.dense = true;
        t.wires = wires;
        t.matrix = matrix;
        return {t};
    }
    PL_ABORT(std::string("Unsupported observable: ") + name);
    return {};
}

/**
 * @brief Product of two terms.
 */
template <class T>
ObsTerm<T> multiplyTerms(const ObsTerm<T> &a, const ObsTerm<T> &b) {
    ObsTerm<T> r;
    r.coeff = a.coeff * b.coeff;
    if (!a.dense && !b.dense) {
        r.wires = a.wires;
        r.wires.insert(r.wires.end(), b.wires.begin(), b.wires.end());
        r.paulis = a.paulis + b.paulis;
        canonicalize(r);
        return r;
    }
    // Dense product on the union of wires (a applied after b: a * b)
    std::vector<std::size_t> uni = a.wires;
    for (auto w : b.wires) {
        if (std::find(uni.begin(), uni.end(), w) == uni.end()) {
            uni.push_back(w);
        }
    }
    PL_ABORT_IF(uni.size() > 14,
                "Observable product acts on too many wires for a dense "
                "representation.");
    auto ma = embedMatrix<T>(termMatrix(a), a.wires, uni);
    auto mb = embedMatrix<T>(termMatrix(b), b.wires, uni);
    r.dense = true;
    r.wires = uni;
    r.matrix = matmul<T>(ma, mb, pow2(uni.size()));
    return r;
}

/**
 * @brief Tensor product (operator product) of several observables.
 */
template <class T>
ObsTerms<T> productObservable(const std::vector<ObsTerms<T>> &factors) {
    ObsTerms<T> acc{ObsTerm<T>{}};
    for (const auto &f : factors) {
        ObsTerms<T> next;
        next.reserve(acc.size() * f.size());
        for (const auto &a : acc) {
            for (const auto &b : f) {
                next.push_back(multiplyTerms(a, b));
            }
        }
        acc = std::move(next);
    }
    return acc;
}

/**
 * @brief Merge Pauli terms with identical words.
 */
template <class T> ObsTerms<T> simplify(const ObsTerms<T> &terms) {
    std::map<std::pair<std::vector<std::size_t>, std::string>, std::size_t>
        index;
    ObsTerms<T> out;
    for (auto t : terms) {
        canonicalize(t);
        if (t.dense) {
            out.push_back(t);
            continue;
        }
        auto key = std::make_pair(t.wires, t.paulis);
        auto it = index.find(key);
        if (it == index.end()) {
            index[key] = out.size();
            out.push_back(t);
        } else {
            out[it->second].coeff += t.coeff;
        }
    }
    ObsTerms<T> res;
    for (auto &t : out) {
        if (t.dense || std::abs(t.coeff) > T{0}) {
            res.push_back(std::move(t));
        }
    }
    return res;
}

/**
 * @brief Linear combination sum_i c_i O_i.
 */
template <class T>
ObsTerms<T> linearCombination(const std::vector<T> &coeffs,
                              const std::vector<ObsTerms<T>> &obs) {
    PL_ABORT_IF_NOT(coeffs.size() == obs.size(),
                    "Coefficient and observable lists differ in length");
    ObsTerms<T> out;
    for (std::size_t i = 0; i < obs.size(); i++) {
        for (auto t : obs[i]) {
            t.coeff *= coeffs[i];
            out.push_back(std::move(t));
        }
    }
    return simplify(out);
}

/**
 * @brief Terms of O^2 (used for variances).
 */
template <class T> ObsTerms<T> squareObservable(const ObsTerms<T> &terms) {
    ObsTerms<T> out;
    out.reserve(terms.size() * terms.size());
    for (const auto &a : terms) {
        for (const auto &b : terms) {
            out.push_back(multiplyTerms(a, b));
        }
    }
    return simplify(out);
}

/**
 * @brief All wires touched by a set of terms (sorted, unique).
 */
template <class T>
std::vector<std::size_t> observableWires(const ObsTerms<T> &terms) {
    std::vector<std::size_t> w;
    for (const auto &t : terms) {
        w.insert(w.end(), t.wires.begin(), t.wires.end());
    }
    std::sort(w.begin(), w.end());
    w.erase(std::unique(w.begin(), w.end()), w.end());
    return w;
}

} // namespace Pennylane::LightningNoisy
