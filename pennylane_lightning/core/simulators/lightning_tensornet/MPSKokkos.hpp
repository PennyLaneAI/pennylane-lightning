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
 * @file MPSKokkos.hpp
 * Kokkos matrix-product-state simulator (lightning.tensornet).
 *
 * Representation. Site i holds a tensor A_i[l, s, r] (row-major, shape
 * D_i x d x D_{i+1}, D_0 = D_N = 1):
 *   - pure mode (d = 2): |psi> = sum A_0[s_0] ... A_{N-1}[s_{N-1}] |s>;
 *   - mixed mode (d = 4): the vectorised density matrix
 *     |rho>> = sum rho[r, c] |r>|c> in Liouville space, with the local
 *     index x = 2 r_i + c_i (a matrix product density operator). A unitary
 *     acts as U (x) conj(U) and a channel as sum_k K_k (x) conj(K_k).
 * The state starts pure and is promoted to mixed mode by the first
 * non-unitary channel (or immediately, if constructed with mixed = true).
 *
 * The MPS is kept in mixed-canonical form around `center_` (left of it
 * left-orthonormal, right of it right-orthonormal; for mixed mode in the
 * Hilbert-Schmidt sense), so two-site SVD truncation is locally optimal.
 * Gates on k sites contract the sites into one tensor, apply the (d^k x
 * d^k) operator and split it again with k - 1 SVDs, keeping at most
 * `max_bond_dim` singular values above `cutoff` (absolute, or relative to
 * the largest one). Non-adjacent sites are brought together with SWAPs.
 *
 * Contractions run on the Kokkos default execution space; SVDs run on the
 * host (LAPACK from scipy-openblas, or a Jacobi fallback).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Error.hpp"
#include "GateMatrices.hpp"
#include "MPSLinAlg.hpp"
#include "NoiseModel.hpp"
#include "ObservableTerms.hpp"

namespace Pennylane::LightningTensornet {

using Pennylane::LightningNoisy::CVec;
using Pennylane::LightningNoisy::KrausChannel;
using Pennylane::LightningNoisy::NoiseModel;
using Pennylane::LightningNoisy::ObsTerm;
using Pennylane::LightningNoisy::ObsTerms;
using Pennylane::LightningNoisy::pow2;
using Pennylane::LightningNoisy::ResolvedNoise;

namespace detail {
KOKKOS_INLINE_FUNCTION std::uint64_t splitmix64(std::uint64_t &x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

inline std::mutex &kokkosInitMutex() {
    static std::mutex m;
    return m;
}

inline void initKokkos(const Kokkos::InitializationSettings &args) {
    const std::lock_guard<std::mutex> lock(kokkosInitMutex());
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize(args);
    }
}

inline void registerKokkosFinalize() {
    const std::lock_guard<std::mutex> lock(kokkosInitMutex());
    static bool registered = false;
    if (!registered) {
        registered = true;
        std::atexit([]() {
            if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
                Kokkos::finalize();
            }
        });
    }
}
} // namespace detail

template <class fp_t = double> class MPSKokkos {
  public:
    using PrecisionT = fp_t;
    using ComplexT = Kokkos::complex<fp_t>;
    using StdComplex = std::complex<fp_t>;
    using KV = KVector<fp_t>;
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using LocalOp = std::array<StdComplex, 4>; ///< row-major 2x2 operator

    /// Largest number of wires of a dense operator (pure / mixed mode)
    static constexpr std::size_t max_dense_wires_pure = 10;
    static constexpr std::size_t max_dense_wires_mixed = 5;
    /// Largest intermediate tensor (complex entries) of a contraction
    static constexpr std::size_t max_contraction_size = std::size_t{1} << 26U;

    /**
     * @param num_qubits Number of qubits.
     * @param max_bond_dim Largest bond dimension kept by SVD truncation.
     * @param cutoff Singular values below the cutoff are discarded.
     * @param cutoff_mode "abs" (absolute cutoff) or "rel" (relative to the
     * largest singular value).
     * @param mixed Start in mixed (density-operator) mode.
     */
    explicit MPSKokkos(std::size_t num_qubits, std::size_t max_bond_dim = 128,
                       fp_t cutoff = 0, const std::string &cutoff_mode = "abs",
                       bool mixed = false,
                       const Kokkos::InitializationSettings &kokkos_args = {})
        : max_bond_{max_bond_dim}, cutoff_{cutoff},
          cutoff_rel_{cutoff_mode == "rel"}, prefer_mixed_{mixed} {
        PL_ABORT_IF(max_bond_dim == 0, "max_bond_dim must be positive.");
        PL_ABORT_IF(cutoff < 0, "cutoff must be non-negative.");
        PL_ABORT_IF_NOT(cutoff_mode == "abs" || cutoff_mode == "rel",
                        "cutoff_mode must be 'abs' or 'rel'.");
        detail::initKokkos(kokkos_args);
        resetTo(num_qubits);
    }

    MPSKokkos(const MPSKokkos &other)
        : n_{other.n_}, d_{other.d_}, max_bond_{other.max_bond_},
          cutoff_{other.cutoff_}, cutoff_rel_{other.cutoff_rel_},
          prefer_mixed_{other.prefer_mixed_}, bond_{other.bond_},
          center_{other.center_}, trunc_err_{other.trunc_err_} {
        sites_.reserve(other.sites_.size());
        for (const auto &s : other.sites_) {
            sites_.push_back(clone<fp_t>(s));
        }
        invalidate();
    }
    MPSKokkos &operator=(const MPSKokkos &) = delete;
    MPSKokkos(MPSKokkos &&) = delete;
    MPSKokkos &operator=(MPSKokkos &&) = delete;

    ~MPSKokkos() { detail::registerKokkosFinalize(); }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    [[nodiscard]] std::size_t getNumQubits() const { return n_; }
    [[nodiscard]] bool isMixed() const { return d_ == 4; }
    [[nodiscard]] std::size_t getMaxBondDim() const { return max_bond_; }
    [[nodiscard]] fp_t getCutoff() const { return cutoff_; }
    [[nodiscard]] std::string getCutoffMode() const {
        return cutoff_rel_ ? "rel" : "abs";
    }
    /// Bond dimensions D_0 .. D_N
    [[nodiscard]] std::vector<std::size_t> bondDims() const { return bond_; }
    /// Sum of the discarded squared singular values (relative weight)
    [[nodiscard]] fp_t truncationError() const { return trunc_err_; }
    [[nodiscard]] std::size_t orthogonalityCenter() const { return center_; }

    // ------------------------------------------------------------------
    // State preparation
    // ------------------------------------------------------------------

    /// |0...0> (in the preferred mode)
    void resetState() { resetTo(n_); }

    /// Basis state on `wires` (other wires in |0>)
    void setBasisState(const std::vector<std::size_t> &state,
                       const std::vector<std::size_t> &wires) {
        PL_ABORT_IF_NOT(state.size() == wires.size(),
                        "state and wires must have equal dimensions.");
        checkWires(wires);
        resetTo(n_);
        for (std::size_t k = 0; k < wires.size(); k++) {
            const std::size_t bit = state[k] != 0U ? 1U : 0U;
            std::vector<StdComplex> t(d_, StdComplex{0, 0});
            t[isMixed() ? 3 * bit : bit] = StdComplex{1, 0};
            sites_[wires[k]] = toDevice<fp_t>(t);
        }
    }

    /**
     * @brief Pure state psi on `wires` (other wires in |0>). The MPS of psi
     * is built with sequential SVDs (subject to the truncation settings).
     */
    void setPureState(const StdComplex *psi, std::size_t length,
                      const std::vector<std::size_t> &wires) {
        const std::size_t k = wires.size();
        PL_ABORT_IF_NOT(length == pow2(k),
                        "Inconsistent state and wires dimensions.");
        checkWires(wires);
        checkUnique(wires);
        const bool want_mixed = prefer_mixed_;
        prefer_mixed_ = false;
        resetTo(n_);
        prefer_mixed_ = want_mixed;
        if (k == 0) {
            return;
        }
        std::vector<std::size_t> sorted = wires;
        std::sort(sorted.begin(), sorted.end());
        // psi in sorted wire order
        std::vector<std::size_t> pos(k); // pos[j]: bit (from MSB) of wires[j]
        for (std::size_t j = 0; j < k; j++) {
            pos[j] = static_cast<std::size_t>(
                std::find(sorted.begin(), sorted.end(), wires[j]) -
                sorted.begin());
        }
        std::vector<StdComplex> rest(length);
        for (std::size_t i = 0; i < length; i++) {
            std::size_t idx = 0;
            for (std::size_t j = 0; j < k; j++) {
                const std::size_t bit = (i >> (k - 1 - j)) & 1U;
                idx |= bit << (k - 1 - pos[j]);
            }
            rest[idx] = psi[i];
        }
        std::size_t D = 1;
        std::size_t next = 0; // index into sorted
        for (std::size_t site = sorted.front(); site <= sorted.back(); site++) {
            if (next < k && site == sorted[next]) {
                const std::size_t cols = pow2(k - 1 - next);
                if (next + 1 == k) {
                    sites_[site] = toDevice<fp_t>(rest);
                    bond_[site] = D;
                    bond_[site + 1] = 1;
                } else {
                    std::vector<StdComplex> U;
                    std::vector<StdComplex> SV;
                    const std::size_t r =
                        svdSplit(rest, D * 2, cols, true, U, SV, true);
                    sites_[site] = toDevice<fp_t>(U);
                    bond_[site] = D;
                    bond_[site + 1] = r;
                    rest = std::move(SV);
                    D = r;
                }
                next++;
            } else {
                // |0> carrying the bond: A[l, s, r] = delta_{l r} delta_{s 0}
                std::vector<StdComplex> t(D * 2 * D, StdComplex{0, 0});
                for (std::size_t l = 0; l < D; l++) {
                    t[(l * 2) * D + l] = StdComplex{1, 0};
                }
                sites_[site] = toDevice<fp_t>(t);
                bond_[site] = D;
                bond_[site + 1] = D;
            }
        }
        center_ = sorted.back();
        invalidate();
        if (prefer_mixed_) {
            toMixed();
        }
    }

    // ------------------------------------------------------------------
    // Gates
    // ------------------------------------------------------------------

    void applyOperation(const std::string &name,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<fp_t> &params = {}) {
        applyOperation(name, {}, {}, wires, inverse, params);
    }

    /**
     * @brief Apply a (possibly controlled) named gate.
     */
    void applyOperation(const std::string &name,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<fp_t> &params = {}) {
        checkWires(wires);
        checkWires(controlled_wires);
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "Controlled wires and values differ in length.");
        PL_ABORT_IF(name == "PauliRot",
                    "Use applyPauliRot for PauliRot operations.");
        if (name == "Identity") {
            return;
        }
        if (controlled_wires.empty()) {
            if (name == "GlobalPhase") {
                if (!isMixed()) {
                    const fp_t ph = inverse ? -params[0] : params[0];
                    scaleCenter(std::exp(StdComplex{0, -ph}));
                }
                return;
            }
            if (name == "MultiRZ" && wires.size() > 3) {
                applyMultiRZ(wires, inverse ? -params[0] : params[0]);
                return;
            }
        }
        const auto m = LightningNoisy::namedGateMatrix<fp_t>(
            name, params, inverse, wires.size());
        applyUnitary(controlledMatrix(m, wires.size(), controlled_values),
                     concat(controlled_wires, wires));
    }

    /**
     * @brief PauliRot(theta, word) = exp(-i theta/2 P).
     */
    void applyPauliRot(const std::vector<std::size_t> &wires, bool inverse,
                       const std::vector<fp_t> &params,
                       const std::string &word) {
        checkWires(wires);
        PL_ABORT_IF_NOT(word.size() == wires.size(),
                        "Pauli word and wires differ in length.");
        PL_ABORT_IF(params.empty(), "PauliRot requires one parameter.");
        const fp_t theta = inverse ? -params[0] : params[0];
        std::vector<std::size_t> act;
        std::string w;
        for (std::size_t i = 0; i < wires.size(); i++) {
            PL_ABORT_IF(word[i] != 'I' && word[i] != 'X' && word[i] != 'Y' &&
                            word[i] != 'Z',
                        "Invalid Pauli word.");
            if (word[i] != 'I') {
                act.push_back(wires[i]);
                w.push_back(word[i]);
            }
        }
        if (act.empty()) {
            if (!isMixed()) {
                scaleCenter(std::exp(StdComplex{0, -theta / 2}));
            }
            return;
        }
        if (act.size() <= 3) {
            applyUnitary(LightningNoisy::pauliRotMatrix<fp_t>(theta, w, false),
                         act);
            return;
        }
        // basis change V P V^dagger = Z...Z, then MultiRZ
        for (std::size_t i = 0; i < act.size(); i++) {
            if (w[i] == 'X') {
                applyOperation("Hadamard", {act[i]});
            } else if (w[i] == 'Y') {
                applyOperation("S", {act[i]}, true);
                applyOperation("Hadamard", {act[i]});
            }
        }
        applyMultiRZ(act, theta);
        for (std::size_t i = 0; i < act.size(); i++) {
            if (w[i] == 'X') {
                applyOperation("Hadamard", {act[i]});
            } else if (w[i] == 'Y') {
                applyOperation("Hadamard", {act[i]});
                applyOperation("S", {act[i]});
            }
        }
    }

    /// Apply a unitary given as a row-major host matrix
    void applyMatrix(const StdComplex *matrix, std::size_t size,
                     const std::vector<std::size_t> &wires,
                     bool inverse = false) {
        applyControlledMatrix(matrix, size, {}, {}, wires, inverse);
    }

    void applyControlledMatrix(const StdComplex *matrix, std::size_t size,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               bool inverse = false) {
        checkWires(wires);
        checkWires(controlled_wires);
        const std::size_t dim = pow2(wires.size());
        PL_ABORT_IF_NOT(size == dim * dim, "Matrix size does not match wires");
        CVec<fp_t> m(matrix, matrix + size);
        if (inverse) {
            m = LightningNoisy::adjoint<fp_t>(m, dim);
        }
        applyUnitary(controlledMatrix(m, wires.size(), controlled_values),
                     concat(controlled_wires, wires));
    }

    // ------------------------------------------------------------------
    // Channels
    // ------------------------------------------------------------------

    /// rho -> sum_k K_k rho K_k^dagger (Kraus operators row-major)
    void applyChannel(const std::vector<CVec<fp_t>> &kraus,
                      const std::vector<std::size_t> &wires) {
        KrausChannel<fp_t> ch;
        ch.num_qubits = wires.size();
        ch.kraus = kraus;
        for (const auto &k : kraus) {
            PL_ABORT_IF_NOT(k.size() == pow2(2 * wires.size()),
                            "Kraus operator size does not match wires");
        }
        ch.finalize();
        applyChannel(ch, wires);
    }

    void applyChannel(const KrausChannel<fp_t> &ch,
                      const std::vector<std::size_t> &wires) {
        checkWires(wires);
        PL_ABORT_IF_NOT(ch.num_qubits == wires.size(),
                        "Channel size does not match wires");
        const std::size_t dim = pow2(wires.size());
        if (ch.kraus.size() == 1 && isUnitary(ch.kraus[0], dim)) {
            applyUnitary(ch.kraus[0], wires);
            return;
        }
        if (!isMixed()) {
            toMixed();
        }
        applyOnWires(wires, ch.superop);
    }

    void applyNoise(const NoiseModel<fp_t> &model,
                    const std::vector<ResolvedNoise> &noise) {
        for (const auto &n : noise) {
            applyChannel(model.channel(n.channel), n.wires);
        }
    }

    /**
     * @brief Switch to mixed (density-operator) mode: A -> A (x) conj(A).
     * The bond dimensions are squared and then truncated to max_bond_dim.
     */
    void toMixed() {
        if (isMixed()) {
            return;
        }
        for (std::size_t i = 0; i < n_; i++) {
            const std::size_t D2 = bond_[i] * bond_[i + 1];
            PL_ABORT_IF(4 * D2 * D2 > max_contraction_size,
                        "The pure MPS is too entangled to switch to mixed mode "
                        "(bond dimension squared); start in mixed mode or "
                        "reduce max_bond_dim.");
        }
        for (std::size_t i = 0; i < n_; i++) {
            const std::size_t Dl = bond_[i];
            const std::size_t Dr = bond_[i + 1];
            const KV a = sites_[i];
            const std::size_t Dl2 = Dl * Dl;
            const std::size_t Dr2 = Dr * Dr;
            KV b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mps_site"),
                 Dl2 * 4 * Dr2);
            Kokkos::parallel_for(
                "mps_to_mixed", Kokkos::RangePolicy<ExecSpace>(0, b.size()),
                KOKKOS_LAMBDA(std::size_t idx) {
                    const std::size_t rr = idx % Dr2;
                    const std::size_t x = (idx / Dr2) % 4;
                    const std::size_t ll = idx / (Dr2 * 4);
                    const std::size_t l = ll / Dl;
                    const std::size_t l2 = ll % Dl;
                    const std::size_t r = rr / Dr;
                    const std::size_t r2 = rr % Dr;
                    const std::size_t s = x >> 1U;
                    const std::size_t s2 = x & 1U;
                    b(idx) = a((l * 2 + s) * Dr + r) *
                             Kokkos::conj(a((l2 * 2 + s2) * Dr + r2));
                });
            sites_[i] = b;
        }
        for (auto &D : bond_) {
            D *= D;
        }
        d_ = 4;
        invalidate();
        if (*std::max_element(bond_.begin(), bond_.end()) > max_bond_) {
            compress();
        }
    }

    // ------------------------------------------------------------------
    // Measurements
    // ------------------------------------------------------------------

    /// <psi|psi> (pure) or Tr(rho) (mixed)
    [[nodiscard]] fp_t normalization() const {
        if (n_ == 0) {
            return fp_t{1};
        }
        const auto v = toHost<fp_t>(leftEnv(n_));
        return v[0].real();
    }

    /// Tr(rho) of the (unnormalised) state; measurements normalise by it
    [[nodiscard]] fp_t trace() const { return normalization(); }

    /// Tr(rho^2) of the normalised state
    [[nodiscard]] fp_t purity() const {
        if (!isMixed() || n_ == 0) {
            return fp_t{1};
        }
        // canonical form: ||rho||_HS^2 = ||A_center||^2
        const auto c = toHost<fp_t>(sites_[center_]);
        fp_t hs = 0;
        for (const auto &x : c) {
            hs += std::norm(x);
        }
        const fp_t tr = normalization();
        return hs / (tr * tr);
    }

    /// Probabilities of the computational basis states of `wires`
    [[nodiscard]] std::vector<fp_t>
    probs(const std::vector<std::size_t> &wires) const {
        checkWires(wires);
        checkUnique(wires);
        const std::size_t k = wires.size();
        if (k == 0) {
            return {fp_t{1}};
        }
        std::vector<fp_t> p_sorted;
        std::vector<std::size_t> sorted = wires;
        std::sort(sorted.begin(), sorted.end());
        if (!isMixed() && (k == n_ || !batchFits(sorted, 2)) && n_ <= 26) {
            const auto psi = stateVector();
            p_sorted.assign(pow2(k), fp_t{0});
            for (std::size_t i = 0; i < psi.size(); i++) {
                std::size_t idx = 0;
                for (std::size_t j = 0; j < k; j++) {
                    idx = (idx << 1U) | ((i >> (n_ - 1 - sorted[j])) & 1U);
                }
                p_sorted[idx] += std::norm(psi[i]);
            }
        } else {
            const LocalOp p0{StdComplex{1, 0}, StdComplex{0, 0},
                             StdComplex{0, 0}, StdComplex{0, 0}};
            const LocalOp p1{StdComplex{0, 0}, StdComplex{0, 0},
                             StdComplex{0, 0}, StdComplex{1, 0}};
            std::vector<std::vector<LocalOp>> ops;
            for (std::size_t s = sorted.front(); s <= sorted.back(); s++) {
                if (std::binary_search(sorted.begin(), sorted.end(), s)) {
                    ops.push_back({p0, p1});
                } else {
                    ops.push_back({identityOp()});
                }
            }
            const auto v = contractRange(sorted.front(), ops);
            p_sorted.resize(v.size());
            for (std::size_t i = 0; i < v.size(); i++) {
                p_sorted[i] = v[i].real();
            }
        }
        fp_t total = 0;
        for (auto &p : p_sorted) {
            p = std::max(p, fp_t{0});
            total += p;
        }
        PL_ABORT_IF(!(total > fp_t{0}), "The state has zero norm.");
        // reorder to the requested wire order
        std::vector<std::size_t> pos(k);
        for (std::size_t j = 0; j < k; j++) {
            pos[j] = static_cast<std::size_t>(
                std::find(sorted.begin(), sorted.end(), wires[j]) -
                sorted.begin());
        }
        std::vector<fp_t> out(pow2(k));
        for (std::size_t i = 0; i < out.size(); i++) {
            std::size_t src = 0;
            for (std::size_t j = 0; j < k; j++) {
                const std::size_t bit = (i >> (k - 1 - j)) & 1U;
                src |= bit << (k - 1 - pos[j]);
            }
            out[i] = p_sorted[src] / total;
        }
        return out;
    }

    [[nodiscard]] std::vector<fp_t> probs() const {
        std::vector<std::size_t> all(n_);
        std::iota(all.begin(), all.end(), 0);
        return probs(all);
    }

    /**
     * @brief Reduced density matrix of `wires` (2^k x 2^k, row-major, the
     * first wire is the most significant bit), normalised.
     */
    [[nodiscard]] CVec<fp_t>
    reducedDensityMatrix(const std::vector<std::size_t> &wires) const {
        checkWires(wires);
        checkUnique(wires);
        const std::size_t k = wires.size();
        const std::size_t dim = pow2(k);
        if (k == 0) {
            return {StdComplex{1, 0}};
        }
        std::vector<std::size_t> sorted = wires;
        std::sort(sorted.begin(), sorted.end());
        CVec<fp_t> rho_sorted(dim * dim);
        if (!isMixed() && (k == n_ || !batchFits(sorted, 4)) && n_ <= 26 &&
            k <= 13) {
            // from the state vector: rho = Tr_rest |psi><psi|
            const auto psi = stateVector();
            const std::size_t nr = n_ - k;
            std::vector<std::size_t> others;
            for (std::size_t w = 0; w < n_; w++) {
                if (!std::binary_search(sorted.begin(), sorted.end(), w)) {
                    others.push_back(w);
                }
            }
            auto index = [&](std::size_t a, std::size_t e) {
                std::size_t i = 0;
                for (std::size_t j = 0; j < k; j++) {
                    i |= ((a >> (k - 1 - j)) & 1U) << (n_ - 1 - sorted[j]);
                }
                for (std::size_t j = 0; j < nr; j++) {
                    i |= ((e >> (nr - 1 - j)) & 1U) << (n_ - 1 - others[j]);
                }
                return i;
            };
            for (std::size_t e = 0; e < pow2(nr); e++) {
                for (std::size_t a = 0; a < dim; a++) {
                    const StdComplex pa = psi[index(a, e)];
                    if (pa == StdComplex{0, 0}) {
                        continue;
                    }
                    for (std::size_t b = 0; b < dim; b++) {
                        rho_sorted[a * dim + b] +=
                            pa * std::conj(psi[index(b, e)]);
                    }
                }
            }
        } else {
            // o = |b><a| gives rho[a, b]; branch j = 2 a + b
            std::vector<LocalOp> branch(4);
            for (std::size_t a = 0; a < 2; a++) {
                for (std::size_t b = 0; b < 2; b++) {
                    LocalOp o{};
                    o.fill(StdComplex{0, 0});
                    o[b * 2 + a] = StdComplex{1, 0};
                    branch[2 * a + b] = o;
                }
            }
            std::vector<std::vector<LocalOp>> ops;
            for (std::size_t s = sorted.front(); s <= sorted.back(); s++) {
                if (std::binary_search(sorted.begin(), sorted.end(), s)) {
                    ops.push_back(branch);
                } else {
                    ops.push_back({identityOp()});
                }
            }
            const auto v = contractRange(sorted.front(), ops);
            const fp_t nrm = normalization();
            // v index: (a_0 b_0 a_1 b_1 ...) MSB first
            for (std::size_t idx = 0; idx < v.size(); idx++) {
                std::size_t a = 0;
                std::size_t b = 0;
                for (std::size_t j = 0; j < k; j++) {
                    const std::size_t pair = (idx >> (2 * (k - 1 - j))) & 3U;
                    a = (a << 1U) | (pair >> 1U);
                    b = (b << 1U) | (pair & 1U);
                }
                rho_sorted[a * dim + b] = v[idx] / nrm;
            }
        }
        // normalise the trace and reorder to the requested wire order
        StdComplex tr{0, 0};
        for (std::size_t a = 0; a < dim; a++) {
            tr += rho_sorted[a * dim + a];
        }
        std::vector<std::size_t> pos(k);
        for (std::size_t j = 0; j < k; j++) {
            pos[j] = static_cast<std::size_t>(
                std::find(sorted.begin(), sorted.end(), wires[j]) -
                sorted.begin());
        }
        auto src = [&](std::size_t i) {
            std::size_t s = 0;
            for (std::size_t j = 0; j < k; j++) {
                s |= ((i >> (k - 1 - j)) & 1U) << (k - 1 - pos[j]);
            }
            return s;
        };
        CVec<fp_t> out(dim * dim);
        for (std::size_t a = 0; a < dim; a++) {
            for (std::size_t b = 0; b < dim; b++) {
                out[a * dim + b] =
                    rho_sorted[src(a) * dim + src(b)] / tr.real();
            }
        }
        return out;
    }

    /// Expectation value of an observable
    [[nodiscard]] fp_t expval(const ObsTerms<fp_t> &terms) const {
        const fp_t nrm = normalization();
        fp_t total = 0;
        for (auto t : terms) {
            LightningNoisy::canonicalize(t);
            if (t.dense) {
                const auto rho = reducedDensityMatrix(t.wires);
                const std::size_t dim = pow2(t.wires.size());
                StdComplex v{0, 0};
                for (std::size_t a = 0; a < dim; a++) {
                    for (std::size_t b = 0; b < dim; b++) {
                        v += t.matrix[a * dim + b] * rho[b * dim + a];
                    }
                }
                total += (t.coeff * v).real();
                continue;
            }
            if (t.wires.empty()) {
                total += t.coeff.real();
                continue;
            }
            std::vector<std::vector<LocalOp>> ops;
            std::size_t j = 0;
            for (std::size_t s = t.wires.front(); s <= t.wires.back(); s++) {
                if (j < t.wires.size() && t.wires[j] == s) {
                    ops.push_back({pauliOp(t.paulis[j])});
                    j++;
                } else {
                    ops.push_back({identityOp()});
                }
            }
            const auto v = contractRange(t.wires.front(), ops);
            total += (t.coeff * v[0]).real() / nrm;
        }
        return total;
    }

    [[nodiscard]] fp_t var(const ObsTerms<fp_t> &terms) const {
        const fp_t e = expval(terms);
        const fp_t e2 = expval(LightningNoisy::squareObservable(terms));
        return e2 - e * e;
    }

    /**
     * @brief Sample the computational-basis outcomes of `wires` (perfect
     * sampling, one sweep per shot, shots in parallel).
     * @return shots x wires.size() bits (row-major).
     */
    template <class RNG>
    [[nodiscard]] std::vector<std::size_t>
    sampleWires(const std::vector<std::size_t> &wires, std::size_t shots,
                RNG &gen) const {
        checkWires(wires);
        const std::size_t k = wires.size();
        std::vector<std::size_t> out(shots * k, 0);
        if (k == 0 || shots == 0) {
            return out;
        }
        const std::size_t last = *std::max_element(wires.begin(), wires.end());
        const std::size_t ns = last + 1; // sampled sites 0..last
        // pack the site tensors and the right environments R_1 .. R_{last+1}
        std::vector<std::size_t> soff(ns + 1, 0);
        std::vector<std::size_t> roff(ns + 1, 0);
        std::size_t dmax = 1;
        for (std::size_t i = 0; i < ns; i++) {
            soff[i + 1] = soff[i] + sites_[i].size();
            roff[i + 1] = roff[i] + rightEnv(i + 1).size();
            dmax = std::max(dmax, bond_[i + 1]);
        }
        KV packed("mps_packed", soff[ns]);
        KV renv("mps_packed_env", roff[ns]);
        for (std::size_t i = 0; i < ns; i++) {
            Kokkos::deep_copy(
                Kokkos::subview(packed, std::make_pair(soff[i], soff[i + 1])),
                sites_[i]);
            Kokkos::deep_copy(
                Kokkos::subview(renv, std::make_pair(roff[i], roff[i + 1])),
                rightEnv(i + 1));
        }
        Kokkos::View<std::size_t *> d_soff("mps_soff", ns + 1);
        Kokkos::View<std::size_t *> d_roff("mps_roff", ns + 1);
        Kokkos::View<std::size_t *> d_bond("mps_bond", ns + 1);
        {
            auto hs = Kokkos::create_mirror_view(d_soff);
            auto hr = Kokkos::create_mirror_view(d_roff);
            auto hb = Kokkos::create_mirror_view(d_bond);
            for (std::size_t i = 0; i <= ns; i++) {
                hs(i) = soff[i];
                hr(i) = roff[i];
                hb(i) = bond_[i];
            }
            Kokkos::deep_copy(d_soff, hs);
            Kokkos::deep_copy(d_roff, hr);
            Kokkos::deep_copy(d_bond, hb);
        }
        const auto seed_hi = static_cast<std::uint64_t>(gen());
        const auto seed_lo = static_cast<std::uint64_t>(gen());
        const std::uint64_t seed = (seed_hi << 32U) ^ seed_lo;
        const bool mixed = isMixed();
        const std::size_t d = d_;
        const std::size_t chunk = std::max<std::size_t>(
            1,
            std::min<std::size_t>({shots, (std::size_t{1} << 22U) / (3 * dmax),
                                   std::size_t{1} << 16U}));
        KV scratch("mps_sample_scratch", chunk * 3 * dmax);
        Kokkos::View<std::uint8_t **> bits("mps_bits", chunk, ns);
        for (std::size_t base = 0; base < shots; base += chunk) {
            const std::size_t cnt = std::min(chunk, shots - base);
            Kokkos::parallel_for(
                "mps_sample", Kokkos::RangePolicy<ExecSpace>(0, cnt),
                KOKKOS_LAMBDA(std::size_t t) {
                    std::uint64_t rs =
                        seed ^ ((base + t + 1) * 0xD1B54A32D192ED03ULL);
                    const std::size_t sbase = t * 3 * dmax;
                    // L, phi0, phi1
                    scratch(sbase) = ComplexT{1, 0};
                    std::size_t Dl = 1;
                    for (std::size_t i = 0; i < ns; i++) {
                        const std::size_t Dr = d_bond(i + 1);
                        const std::size_t so = d_soff(i);
                        const std::size_t ro = d_roff(i);
                        fp_t p[2] = {0, 0};
                        for (std::size_t s = 0; s < 2; s++) {
                            const std::size_t x = mixed ? 3 * s : s;
                            const std::size_t po = sbase + (1 + s) * dmax;
                            for (std::size_t r = 0; r < Dr; r++) {
                                ComplexT acc{0, 0};
                                for (std::size_t l = 0; l < Dl; l++) {
                                    acc += scratch(sbase + l) *
                                           packed(so + (l * d + x) * Dr + r);
                                }
                                scratch(po + r) = acc;
                            }
                            if (mixed) {
                                ComplexT acc{0, 0};
                                for (std::size_t r = 0; r < Dr; r++) {
                                    acc += scratch(po + r) * renv(ro + r);
                                }
                                p[s] = acc.real();
                            } else {
                                // phi^dagger R phi, Rt[ket][bra]
                                ComplexT acc{0, 0};
                                for (std::size_t r = 0; r < Dr; r++) {
                                    ComplexT row{0, 0};
                                    for (std::size_t r2 = 0; r2 < Dr; r2++) {
                                        row += renv(ro + r2 * Dr + r) *
                                               scratch(po + r2);
                                    }
                                    acc += Kokkos::conj(scratch(po + r)) * row;
                                }
                                p[s] = acc.real();
                            }
                            p[s] = p[s] > fp_t{0} ? p[s] : fp_t{0};
                        }
                        const fp_t tot = p[0] + p[1];
                        const double u =
                            static_cast<double>(detail::splitmix64(rs) >> 11U) *
                            (1.0 / 9007199254740992.0);
                        const std::size_t s =
                            (tot > fp_t{0} && u * static_cast<double>(tot) >=
                                                  static_cast<double>(p[0]))
                                ? 1
                                : 0;
                        bits(t, i) = static_cast<std::uint8_t>(s);
                        const fp_t ps = p[s] > fp_t{0} ? p[s] : fp_t{1};
                        const fp_t scale =
                            mixed ? fp_t{1} / ps : fp_t{1} / Kokkos::sqrt(ps);
                        const std::size_t po = sbase + (1 + s) * dmax;
                        for (std::size_t r = 0; r < Dr; r++) {
                            scratch(sbase + r) = scratch(po + r) * scale;
                        }
                        Dl = Dr;
                    }
                });
            auto hbits =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bits);
            for (std::size_t t = 0; t < cnt; t++) {
                for (std::size_t j = 0; j < k; j++) {
                    out[(base + t) * k + j] = hbits(t, wires[j]);
                }
            }
        }
        return out;
    }

    /// Computational-basis samples of all wires
    template <class RNG>
    [[nodiscard]] std::vector<std::size_t> generateSamples(std::size_t shots,
                                                           RNG &gen) const {
        std::vector<std::size_t> all(n_);
        std::iota(all.begin(), all.end(), 0);
        return sampleWires(all, shots, gen);
    }

    // ------------------------------------------------------------------
    // Mid-circuit measurements and qubit management
    // ------------------------------------------------------------------

    /// Project `wire` onto |branch> and renormalise
    void collapse(std::size_t wire, bool branch) {
        checkWires({wire});
        moveCenter(wire);
        const std::size_t keep =
            isMixed() ? (branch ? 3U : 0U) : (branch ? 1U : 0U);
        const std::size_t d = d_;
        const std::size_t Dr = bond_[wire + 1];
        KV a = sites_[wire];
        Kokkos::parallel_for(
            "mps_collapse", Kokkos::RangePolicy<ExecSpace>(0, a.size()),
            KOKKOS_LAMBDA(std::size_t idx) {
                if ((idx / Dr) % d != keep) {
                    a(idx) = ComplexT{0, 0};
                }
            });
        invalidate();
        const fp_t nrm = normalization();
        PL_ABORT_IF(!(nrm > std::numeric_limits<fp_t>::min()),
                    "Collapse onto a zero-probability branch.");
        scaleCenter(StdComplex{
            isMixed() ? fp_t{1} / nrm : fp_t{1} / std::sqrt(nrm), 0});
    }

    /// Mid-circuit measurement (postselect: -1, 0 or 1)
    template <class RNG>
    bool measure(std::size_t wire, int postselect, RNG &gen) {
        const auto p = probs({wire});
        bool outcome = false;
        if (postselect == 0 || postselect == 1) {
            PL_ABORT_IF(p[static_cast<std::size_t>(postselect)] <=
                            std::numeric_limits<fp_t>::epsilon() * 10,
                        "Probability of postselected state is zero.");
            outcome = (postselect == 1);
        } else {
            std::uniform_real_distribution<fp_t> u(0, 1);
            outcome = u(gen) >= p[0];
        }
        collapse(wire, outcome);
        return outcome;
    }

    /// Append a qubit in |0> as the last wire
    void appendQubit() {
        if (n_ == 0) {
            resetTo(1);
            return;
        }
        std::vector<StdComplex> t(d_, StdComplex{0, 0});
        t[0] = StdComplex{1, 0};
        sites_.push_back(toDevice<fp_t>(t, "mps_site"));
        bond_.push_back(1);
        n_++;
        invalidate();
    }

    /**
     * @brief Partial trace over `wires`; the remaining wires keep their
     * relative order and are renumbered. Tracing out a qubit that is
     * entangled with the rest switches to mixed mode.
     */
    void traceOut(const std::vector<std::size_t> &wires) {
        checkWires(wires);
        std::vector<std::size_t> w = wires;
        std::sort(w.begin(), w.end());
        w.erase(std::unique(w.begin(), w.end()), w.end());
        for (auto it = w.rbegin(); it != w.rend(); ++it) {
            removeSite(*it);
        }
    }

    /// Reset `wire` to |0> (rho -> |0><0| (x) Tr_wire(rho))
    void resetQubit(std::size_t wire) {
        checkWires({wire});
        if (!isMixed()) {
            moveCenter(wire);
            LocalOp phi_bra{};
            if (productQubit(wire, phi_bra)) {
                // A'[l, 0, r] = sum_s conj(phi_s) A[l, s, r], A'[l, 1, r] = 0
                const std::size_t Dr = bond_[wire + 1];
                const std::size_t Dl = bond_[wire];
                const KV a = sites_[wire];
                KV b("mps_site", a.size());
                const ComplexT c0{phi_bra[0].real(), phi_bra[0].imag()};
                const ComplexT c1{phi_bra[1].real(), phi_bra[1].imag()};
                Kokkos::parallel_for(
                    "mps_reset", Kokkos::RangePolicy<ExecSpace>(0, Dl * Dr),
                    KOKKOS_LAMBDA(std::size_t idx) {
                        const std::size_t l = idx / Dr;
                        const std::size_t r = idx % Dr;
                        b((l * 2) * Dr + r) = c0 * a((l * 2) * Dr + r) +
                                              c1 * a((l * 2 + 1) * Dr + r);
                    });
                sites_[wire] = b;
                invalidate();
                return;
            }
            toMixed();
        }
        const std::vector<CVec<fp_t>> kraus = {
            {StdComplex{1, 0}, StdComplex{0, 0}, StdComplex{0, 0},
             StdComplex{0, 0}},
            {StdComplex{0, 0}, StdComplex{1, 0}, StdComplex{0, 0},
             StdComplex{0, 0}}};
        applyChannel(kraus, {wire});
    }

    // ------------------------------------------------------------------
    // State vector
    // ------------------------------------------------------------------

    [[nodiscard]] static constexpr bool hasStateVector() { return true; }

    /// Normalised state vector (pure mode only)
    [[nodiscard]] CVec<fp_t> stateVector() const {
        PL_ABORT_IF(isMixed(),
                    "The MPS is in mixed (density-operator) mode: the state "
                    "vector is not defined. Use the density matrix instead.");
        PL_ABORT_IF(n_ > 30, "Too many qubits to build the state vector.");
        if (n_ == 0) {
            return {StdComplex{1, 0}};
        }
        KV t = sites_[0]; // 2 x D_1
        std::size_t rows = 2;
        for (std::size_t i = 1; i < n_; i++) {
            t = matmul<fp_t>(Trans::N, Trans::N, rows, 2 * bond_[i + 1],
                             bond_[i], t, sites_[i]);
            rows *= 2;
        }
        auto psi = toHost<fp_t>(t);
        fp_t nrm = 0;
        for (const auto &x : psi) {
            nrm += std::norm(x);
        }
        const fp_t inv = fp_t{1} / std::sqrt(nrm);
        for (auto &x : psi) {
            x *= inv;
        }
        return psi;
    }

    void getStateVector(StdComplex *out, std::size_t length) const {
        const auto psi = stateVector();
        PL_ABORT_IF_NOT(length == psi.size(), "Invalid state-vector size.");
        std::copy(psi.begin(), psi.end(), out);
    }

    /// Full density matrix (2^N x 2^N), normalised
    [[nodiscard]] CVec<fp_t> getDensityMatrix() const {
        std::vector<std::size_t> all(n_);
        std::iota(all.begin(), all.end(), 0);
        return reducedDensityMatrix(all);
    }

    /// Copy of the site tensors (host, row-major D_i x d x D_{i+1})
    [[nodiscard]] std::vector<CVec<fp_t>> siteTensors() const {
        std::vector<CVec<fp_t>> out;
        out.reserve(n_);
        for (const auto &s : sites_) {
            out.push_back(toHost<fp_t>(s));
        }
        return out;
    }

  private:
    std::size_t n_{0};
    std::size_t d_{2};
    std::size_t max_bond_{128};
    fp_t cutoff_{0};
    bool cutoff_rel_{false};
    bool prefer_mixed_{false};
    std::vector<KV> sites_{};
    std::vector<std::size_t> bond_{1};
    std::size_t center_{0};
    fp_t trunc_err_{0};

    // cached environments: L_i (sites < i) and R_i (sites >= i)
    mutable std::vector<KV> left_env_{};
    mutable std::vector<KV> right_env_{};
    mutable std::vector<char> left_ok_{};
    mutable std::vector<char> right_ok_{};

    // ---------------- helpers ----------------

    void checkWires(const std::vector<std::size_t> &wires) const {
        for (auto w : wires) {
            PL_ABORT_IF_NOT(w < n_, "Wire index out of range for the MPS.");
        }
    }

    static void checkUnique(const std::vector<std::size_t> &wires) {
        std::vector<std::size_t> s = wires;
        std::sort(s.begin(), s.end());
        PL_ABORT_IF(std::adjacent_find(s.begin(), s.end()) != s.end(),
                    "Wires must be unique.");
    }

    static std::vector<std::size_t> concat(const std::vector<std::size_t> &a,
                                           const std::vector<std::size_t> &b) {
        std::vector<std::size_t> out = a;
        out.insert(out.end(), b.begin(), b.end());
        return out;
    }

    static LocalOp identityOp() {
        return {StdComplex{1, 0}, StdComplex{0, 0}, StdComplex{0, 0},
                StdComplex{1, 0}};
    }

    static LocalOp pauliOp(char p) {
        const auto m = LightningNoisy::pauliMatrix<fp_t>(p);
        return {m[0], m[1], m[2], m[3]};
    }

    static bool isUnitary(const CVec<fp_t> &m, std::size_t dim) {
        const fp_t tol = std::sqrt(std::numeric_limits<fp_t>::epsilon());
        for (std::size_t i = 0; i < dim; i++) {
            for (std::size_t j = 0; j < dim; j++) {
                StdComplex s{0, 0};
                for (std::size_t k = 0; k < dim; k++) {
                    s += std::conj(m[k * dim + i]) * m[k * dim + j];
                }
                const StdComplex target{i == j ? fp_t{1} : fp_t{0}, 0};
                if (std::abs(s - target) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    /// Matrix of a controlled gate on (controls + targets)
    static CVec<fp_t> controlledMatrix(const CVec<fp_t> &u, std::size_t kt,
                                       const std::vector<bool> &values) {
        const std::size_t kc = values.size();
        if (kc == 0) {
            return u;
        }
        const std::size_t dt = pow2(kt);
        const std::size_t dim = pow2(kc + kt);
        std::size_t cidx = 0;
        for (std::size_t j = 0; j < kc; j++) {
            cidx = (cidx << 1U) | (values[j] ? 1U : 0U);
        }
        CVec<fp_t> m(dim * dim, StdComplex{0, 0});
        for (std::size_t i = 0; i < dim; i++) {
            if ((i / dt) != cidx) {
                m[i * dim + i] = StdComplex{1, 0};
            }
        }
        const std::size_t o = cidx * dt;
        for (std::size_t a = 0; a < dt; a++) {
            for (std::size_t b = 0; b < dt; b++) {
                m[(o + a) * dim + o + b] = u[a * dt + b];
            }
        }
        return m;
    }

    void invalidate() {
        left_env_.assign(n_ + 1, KV{});
        right_env_.assign(n_ + 1, KV{});
        left_ok_.assign(n_ + 1, 0);
        right_ok_.assign(n_ + 1, 0);
    }

    void resetTo(std::size_t n) {
        n_ = n;
        d_ = prefer_mixed_ ? 4 : 2;
        sites_.clear();
        std::vector<StdComplex> t(d_, StdComplex{0, 0});
        t[0] = StdComplex{1, 0};
        for (std::size_t i = 0; i < n; i++) {
            sites_.push_back(toDevice<fp_t>(t, "mps_site"));
        }
        bond_.assign(n + 1, 1);
        center_ = 0;
        trunc_err_ = 0;
        invalidate();
    }

  public: // nvcc: an extended __host__ __device__ lambda needs a public
          // enclosing function
    void scaleCenter(StdComplex c) {
        if (n_ == 0) {
            return;
        }
        KV a = sites_[center_];
        const ComplexT kc{c.real(), c.imag()};
        Kokkos::parallel_for(
            "mps_scale", Kokkos::RangePolicy<ExecSpace>(0, a.size()),
            KOKKOS_LAMBDA(std::size_t i) { a(i) *= kc; });
        invalidate();
    }

  private:
    /**
     * @brief SVD split of a host row-major rows x cols matrix M.
     * absorb_right: M = U (S Vh), else M = (U S) Vh.
     * @return the kept rank.
     */
    std::size_t svdSplit(const std::vector<StdComplex> &M, std::size_t rows,
                         std::size_t cols, bool truncate,
                         std::vector<StdComplex> &left,
                         std::vector<StdComplex> &right, bool absorb_right) {
        auto res = svd<fp_t>(M, rows, cols);
        const std::size_t r_full = res.S.size();
        const fp_t smax = r_full > 0 ? res.S[0] : fp_t{0};
        const fp_t floor =
            smax * std::numeric_limits<fp_t>::epsilon() * fp_t{64};
        fp_t thr = floor;
        if (truncate) {
            thr = std::max(thr, cutoff_rel_ ? cutoff_ * smax : cutoff_);
        }
        std::size_t r = 0;
        while (r < r_full && res.S[r] > thr) {
            r++;
        }
        if (truncate) {
            r = std::min(r, max_bond_);
        }
        r = std::max<std::size_t>(r, 1);
        fp_t total = 0;
        fp_t kept = 0;
        for (std::size_t i = 0; i < r_full; i++) {
            total += res.S[i] * res.S[i];
            if (i < r) {
                kept += res.S[i] * res.S[i];
            }
        }
        if (total > fp_t{0} && truncate) {
            trunc_err_ += (total - kept) / total;
        }
        // pure mode: keep the norm of the state
        const fp_t renorm = (!isMixed() && kept > fp_t{0} && truncate)
                                ? std::sqrt(total / kept)
                                : fp_t{1};
        left.assign(rows * r, StdComplex{0, 0});
        right.assign(r * cols, StdComplex{0, 0});
        for (std::size_t i = 0; i < rows; i++) {
            for (std::size_t j = 0; j < r; j++) {
                left[i * r + j] = res.U[i * r_full + j] *
                                  (absorb_right ? fp_t{1} : res.S[j] * renorm);
            }
        }
        for (std::size_t j = 0; j < r; j++) {
            const fp_t f = absorb_right ? res.S[j] * renorm : fp_t{1};
            for (std::size_t c = 0; c < cols; c++) {
                right[j * cols + c] = res.Vh[j * cols + c] * f;
            }
        }
        return r;
    }

    /// Move the orthogonality center from site i to i + 1
    void splitRight(std::size_t i, bool truncate) {
        const std::size_t Dl = bond_[i];
        const std::size_t Dr = bond_[i + 1];
        const auto M = toHost<fp_t>(sites_[i]);
        std::vector<StdComplex> U;
        std::vector<StdComplex> SV;
        const std::size_t r = svdSplit(M, Dl * d_, Dr, truncate, U, SV, true);
        sites_[i] = toDevice<fp_t>(U, "mps_site");
        const KV sv = toDevice<fp_t>(SV);
        sites_[i + 1] = matmul<fp_t>(Trans::N, Trans::N, r, d_ * bond_[i + 2],
                                     Dr, sv, sites_[i + 1]);
        bond_[i + 1] = r;
        center_ = i + 1;
    }

    /// Move the orthogonality center from site i to i - 1
    void splitLeft(std::size_t i, bool truncate) {
        const std::size_t Dl = bond_[i];
        const std::size_t Dr = bond_[i + 1];
        const auto M = toHost<fp_t>(sites_[i]);
        std::vector<StdComplex> US;
        std::vector<StdComplex> Vh;
        const std::size_t r = svdSplit(M, Dl, d_ * Dr, truncate, US, Vh, false);
        sites_[i] = toDevice<fp_t>(Vh, "mps_site");
        const KV us = toDevice<fp_t>(US);
        sites_[i - 1] = matmul<fp_t>(Trans::N, Trans::N, bond_[i - 1] * d_, r,
                                     Dl, sites_[i - 1], us);
        bond_[i] = r;
        center_ = i - 1;
    }

    void moveCenter(std::size_t target) {
        if (n_ == 0) {
            return;
        }
        bool moved = false;
        while (center_ < target) {
            splitRight(center_, false);
            moved = true;
        }
        while (center_ > target) {
            splitLeft(center_, false);
            moved = true;
        }
        if (moved) {
            invalidate();
        }
    }

    /// Truncate every bond to the truncation settings (one sweep)
    void compress() {
        if (n_ < 2) {
            return;
        }
        moveCenter(0);
        for (std::size_t i = 0; i + 1 < n_; i++) {
            splitRight(i, true);
        }
        invalidate();
    }

    /**
     * @brief Apply a d^k x d^k operator to the contiguous sites
     * [p, p + k) (operator indices in site order).
     */
    void applyLocal(std::size_t p, std::size_t k,
                    const std::vector<StdComplex> &O) {
        moveCenter(p);
        const std::size_t d = d_;
        const std::size_t Dl = bond_[p];
        KV theta = sites_[p];
        std::size_t rows = Dl * d;
        for (std::size_t j = 1; j < k; j++) {
            theta = matmul<fp_t>(Trans::N, Trans::N, rows, d * bond_[p + j + 1],
                                 bond_[p + j], theta, sites_[p + j]);
            rows *= d;
        }
        const std::size_t Dr = bond_[p + k];
        std::size_t D = 1;
        for (std::size_t j = 0; j < k; j++) {
            D *= d;
        }
        PL_ABORT_IF_NOT(O.size() == D * D, "Local operator size mismatch.");
        PL_ABORT_IF(Dl * D * Dr > max_contraction_size,
                    "Local tensor too large; reduce max_bond_dim.");
        const KV Od = toDevice<fp_t>(O);
        KV th2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mps_theta"),
               Dl * D * Dr);
        batchedGemm<fp_t>(Trans::N, Trans::N, Dl, D, Dr, D, Od, 0, D, 0, theta,
                          0, Dr, D * Dr, th2, 0, Dr, D * Dr);
        if (k == 1) {
            sites_[p] = th2;
            center_ = p;
            invalidate();
            return;
        }
        auto rest = toHost<fp_t>(th2);
        std::size_t left = Dl;
        std::size_t cols = D * Dr;
        for (std::size_t j = 0; j + 1 < k; j++) {
            cols /= d;
            std::vector<StdComplex> U;
            std::vector<StdComplex> SV;
            const std::size_t r =
                svdSplit(rest, left * d, cols, true, U, SV, true);
            sites_[p + j] = toDevice<fp_t>(U, "mps_site");
            bond_[p + j + 1] = r;
            rest = std::move(SV);
            left = r;
        }
        sites_[p + k - 1] = toDevice<fp_t>(rest, "mps_site");
        center_ = p + k - 1;
        invalidate();
    }

    void swapAdjacent(std::size_t p) {
        const std::size_t d = d_;
        std::vector<StdComplex> sw(d * d * d * d, StdComplex{0, 0});
        for (std::size_t a = 0; a < d; a++) {
            for (std::size_t b = 0; b < d; b++) {
                sw[(a * d + b) * d * d + (b * d + a)] = StdComplex{1, 0};
            }
        }
        applyLocal(p, 2, sw);
    }

    /**
     * @brief Apply an operator on arbitrary wires. In pure mode `O` is a
     * 2^k x 2^k unitary; in mixed mode a 4^k x 4^k superoperator in the
     * (rows, columns) ordering of LightningNoisy::krausToSuperop.
     */
    void applyOnWires(const std::vector<std::size_t> &wires,
                      const std::vector<StdComplex> &O) {
        const std::size_t k = wires.size();
        if (k == 0) {
            return;
        }
        checkUnique(wires);
        PL_ABORT_IF(
            k > (isMixed() ? max_dense_wires_mixed : max_dense_wires_pure),
            "Operation acts on too many wires for lightning.tensornet; "
            "decompose it first.");
        std::vector<std::size_t> sorted = wires;
        std::sort(sorted.begin(), sorted.end());
        std::vector<StdComplex> Ob;
        if (!isMixed()) {
            Ob = (sorted == wires)
                     ? O
                     : LightningNoisy::embedMatrix<fp_t>(O, wires, sorted);
        } else {
            constexpr std::size_t off = std::size_t{1} << 40U;
            std::vector<std::size_t> op_w = wires;
            for (auto w : wires) {
                op_w.push_back(w + off);
            }
            std::vector<std::size_t> tgt;
            for (auto w : sorted) {
                tgt.push_back(w);
                tgt.push_back(w + off);
            }
            Ob = LightningNoisy::embedMatrix<fp_t>(O, op_w, tgt);
        }
        const std::size_t p = sorted[0];
        for (std::size_t j = 1; j < k; j++) {
            for (std::size_t x = sorted[j]; x > p + j; x--) {
                swapAdjacent(x - 1);
            }
        }
        applyLocal(p, k, Ob);
        for (std::size_t j = k - 1; j >= 1; j--) {
            for (std::size_t x = p + j; x < sorted[j]; x++) {
                swapAdjacent(x);
            }
        }
    }

    /// Apply a unitary (pure: U, mixed: U (x) conj(U))
    void applyUnitary(const CVec<fp_t> &u,
                      const std::vector<std::size_t> &wires) {
        if (wires.empty()) {
            if (!isMixed() && !u.empty()) {
                scaleCenter(u[0]);
            }
            return;
        }
        if (!isMixed()) {
            applyOnWires(wires, u);
        } else {
            applyOnWires(wires, LightningNoisy::unitaryToSuperop<fp_t>(
                                    u, pow2(wires.size())));
        }
    }

    /// exp(-i theta/2 Z...Z) with a CNOT ladder
    void applyMultiRZ(const std::vector<std::size_t> &wires, fp_t theta) {
        std::vector<std::size_t> w = wires;
        std::sort(w.begin(), w.end());
        for (std::size_t i = 0; i + 1 < w.size(); i++) {
            applyOperation("CNOT", {w[i], w[i + 1]});
        }
        applyOperation("RZ", {w.back()}, false, {theta});
        for (std::size_t i = w.size() - 1; i >= 1; i--) {
            applyOperation("CNOT", {w[i - 1], w[i]});
        }
    }

    /**
     * @brief Pure mode, center at `wire`: whether the qubit is in a product
     * state with the rest; if so, return the bra <phi| of its state.
     */
    bool productQubit(std::size_t wire, LocalOp &phi_bra) const {
        const auto a = toHost<fp_t>(sites_[wire]);
        const std::size_t Dl = bond_[wire];
        const std::size_t Dr = bond_[wire + 1];
        StdComplex rho[2][2] = {{0, 0}, {0, 0}};
        for (std::size_t l = 0; l < Dl; l++) {
            for (std::size_t r = 0; r < Dr; r++) {
                for (std::size_t s = 0; s < 2; s++) {
                    for (std::size_t s2 = 0; s2 < 2; s2++) {
                        rho[s][s2] += a[(l * 2 + s) * Dr + r] *
                                      std::conj(a[(l * 2 + s2) * Dr + r]);
                    }
                }
            }
        }
        const fp_t tr = rho[0][0].real() + rho[1][1].real();
        if (!(tr > fp_t{0})) {
            return false;
        }
        const fp_t pa = rho[0][0].real() / tr;
        const fp_t pd = rho[1][1].real() / tr;
        const StdComplex b = rho[0][1] / tr;
        const fp_t pur = pa * pa + pd * pd + 2 * std::norm(b);
        if (pur < fp_t{1} - std::sqrt(std::numeric_limits<fp_t>::epsilon())) {
            return false;
        }
        const fp_t lam =
            (pa + pd) / 2 + std::sqrt((pa - pd) * (pa - pd) / 4 + std::norm(b));
        StdComplex v0;
        StdComplex v1;
        if (std::abs(b) > std::numeric_limits<fp_t>::epsilon()) {
            v0 = b;
            v1 = StdComplex{lam - pa, 0};
        } else {
            v0 = StdComplex{pa >= pd ? fp_t{1} : fp_t{0}, 0};
            v1 = StdComplex{pa >= pd ? fp_t{0} : fp_t{1}, 0};
        }
        const fp_t nv = std::sqrt(std::norm(v0) + std::norm(v1));
        phi_bra = {std::conj(v0) / nv, std::conj(v1) / nv, StdComplex{0, 0},
                   StdComplex{0, 0}};
        return true;
    }

  public: // nvcc: an extended __host__ __device__ lambda needs a public
          // enclosing function
    /// Remove site i by tracing it out
    void removeSite(std::size_t i) {
        moveCenter(i);
        std::array<StdComplex, 4> w{};
        if (!isMixed()) {
            LocalOp phi_bra{};
            if (productQubit(i, phi_bra)) {
                w = {phi_bra[0], phi_bra[1], StdComplex{0, 0},
                     StdComplex{0, 0}};
            } else {
                toMixed();
                moveCenter(i);
            }
        }
        const std::size_t Dl2 = bond_[i];
        const std::size_t Dr2 = bond_[i + 1];
        if (isMixed()) {
            w = {StdComplex{1, 0}, StdComplex{0, 0}, StdComplex{0, 0},
                 StdComplex{1, 0}};
        }
        // M[l, r] = sum_x w[x] A[l, x, r]
        const std::size_t d = d_;
        const KV a = sites_[i];
        KV M("mps_trace_m", Dl2 * Dr2);
        Kokkos::View<ComplexT *> wd("mps_w", 4);
        {
            auto hw = Kokkos::create_mirror_view(wd);
            for (std::size_t x = 0; x < 4; x++) {
                hw(x) = ComplexT{w[x].real(), w[x].imag()};
            }
            Kokkos::deep_copy(wd, hw);
        }
        Kokkos::parallel_for(
            "mps_trace_site", Kokkos::RangePolicy<ExecSpace>(0, Dl2 * Dr2),
            KOKKOS_LAMBDA(std::size_t idx) {
                const std::size_t l = idx / Dr2;
                const std::size_t r = idx % Dr2;
                ComplexT s{0, 0};
                for (std::size_t x = 0; x < d; x++) {
                    s += wd(x) * a((l * d + x) * Dr2 + r);
                }
                M(idx) = s;
            });
        if (n_ == 1) {
            sites_.clear();
            bond_.assign(1, 1);
            n_ = 0;
            center_ = 0;
            invalidate();
            return;
        }
        if (i > 0) {
            sites_[i - 1] = matmul<fp_t>(Trans::N, Trans::N, bond_[i - 1] * d,
                                         Dr2, Dl2, sites_[i - 1], M);
            sites_.erase(sites_.begin() + static_cast<std::ptrdiff_t>(i));
            bond_.erase(bond_.begin() + static_cast<std::ptrdiff_t>(i));
            center_ = i - 1;
        } else {
            sites_[1] = matmul<fp_t>(Trans::N, Trans::N, 1, d * bond_[2], Dr2,
                                     M, sites_[1]);
            sites_.erase(sites_.begin());
            bond_.erase(bond_.begin() + 1);
            center_ = 0;
        }
        n_--;
        invalidate();
    }

  private:
    // ---------------- environments ----------------

  public: // nvcc: an extended __host__ __device__ lambda needs a public
          // enclosing function
    /// Transfer a batch of nb environments through site i with q local
    /// operators; output entry b * q + j.
    [[nodiscard]] KV transferBatch(const KV &env, std::size_t nb, std::size_t i,
                                   const std::vector<LocalOp> &ops) const {
        const std::size_t Dl = bond_[i];
        const std::size_t Dr = bond_[i + 1];
        const std::size_t q = ops.size();
        const KV &A = sites_[i];
        const bool mixed = isMixed();
        const std::size_t blk = mixed ? Dr : Dr * Dr;
        PL_ABORT_IF(nb * std::max<std::size_t>(q, 4) * blk >
                            max_contraction_size ||
                        (!mixed && nb * Dl * 2 * Dr > max_contraction_size),
                    "Contraction too large for lightning.tensornet (too many "
                    "measured wires for the bond dimension).");
        KV X(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mps_blocks"),
             nb * 4 * blk);
        if (!mixed) {
            // F_b = E_b (Dl x Dl) A (Dl x 2Dr); X_{b,s,s'} = A_s^dag F_{b,s'}
            KV F(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mps_f"),
                 nb * Dl * 2 * Dr);
            batchedGemm<fp_t>(Trans::N, Trans::N, nb, Dl, 2 * Dr, Dl, env, 0,
                              Dl, Dl * Dl, A, 0, 2 * Dr, 0, F, 0, 2 * Dr,
                              Dl * 2 * Dr);
            for (std::size_t s = 0; s < 2; s++) {
                for (std::size_t s2 = 0; s2 < 2; s2++) {
                    batchedGemm<fp_t>(Trans::C, Trans::N, nb, Dr, Dr, Dl, A,
                                      s * Dr, 2 * Dr, 0, F, s2 * Dr, 2 * Dr,
                                      Dl * 2 * Dr, X, (2 * s + s2) * blk, Dr,
                                      4 * blk);
                }
            }
        } else {
            // Y_{b,x} = V_b (1 x Dl) B_x (Dl x Dr)
            batchedGemm<fp_t>(Trans::N, Trans::N, 4, nb, Dr, Dl, env, 0, Dl, 0,
                              A, 0, 4 * Dr, Dr, X, 0, 4 * Dr, Dr);
        }
        // weights: pure w_{2s+s'} = o[s][s'], mixed w_{2r+c} = o[c][r]
        Kokkos::View<ComplexT *> W("mps_weights", q * 4);
        {
            auto hw = Kokkos::create_mirror_view(W);
            for (std::size_t j = 0; j < q; j++) {
                for (std::size_t a = 0; a < 2; a++) {
                    for (std::size_t b = 0; b < 2; b++) {
                        const StdComplex o =
                            mixed ? ops[j][b * 2 + a] : ops[j][a * 2 + b];
                        hw(j * 4 + 2 * a + b) = ComplexT{o.real(), o.imag()};
                    }
                }
            }
            Kokkos::deep_copy(W, hw);
        }
        KV out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mps_env"),
               nb * q * blk);
        Kokkos::parallel_for(
            "mps_combine", Kokkos::RangePolicy<ExecSpace>(0, nb * q * blk),
            KOKKOS_LAMBDA(std::size_t idx) {
                const std::size_t e = idx % blk;
                const std::size_t j = (idx / blk) % q;
                const std::size_t b = idx / (blk * q);
                ComplexT s{0, 0};
                for (std::size_t x = 0; x < 4; x++) {
                    s += W(j * 4 + x) * X((b * 4 + x) * blk + e);
                }
                out(idx) = s;
            });
        return out;
    }

  private:
    [[nodiscard]] KV unitEnv() const {
        const StdComplex one{1, 0};
        return toDevice<fp_t>(&one, 1, "mps_env");
    }

    [[nodiscard]] const KV &leftEnv(std::size_t i) const {
        if (left_ok_[i] != 0) {
            return left_env_[i];
        }
        std::size_t j = i;
        while (j > 0 && left_ok_[j] == 0) {
            j--;
        }
        if (left_ok_[j] == 0) { // j == 0
            left_env_[0] = unitEnv();
            left_ok_[0] = 1;
        }
        for (std::size_t t = j; t < i; t++) {
            left_env_[t + 1] =
                transferBatch(left_env_[t], 1, t, {identityOp()});
            left_ok_[t + 1] = 1;
        }
        return left_env_[i];
    }

  public: // nvcc: an extended __host__ __device__ lambda needs a public
          // enclosing function
    /// Right environment R_i (sites i..N-1): pure Rt[ket][bra], mixed vector
    [[nodiscard]] const KV &rightEnv(std::size_t i) const {
        if (right_ok_[i] != 0) {
            return right_env_[i];
        }
        std::size_t j = i;
        while (j < n_ && right_ok_[j] == 0) {
            j++;
        }
        if (right_ok_[j] == 0) { // j == n_
            right_env_[n_] = unitEnv();
            right_ok_[n_] = 1;
        }
        for (std::size_t t = j; t > i; t--) {
            const std::size_t s = t - 1;
            const std::size_t Dl = bond_[s];
            const std::size_t Dr = bond_[s + 1];
            const KV &A = sites_[s];
            if (!isMixed()) {
                const KV H = matmul<fp_t>(Trans::N, Trans::N, Dl * 2, Dr, Dr, A,
                                          right_env_[t]);
                right_env_[s] =
                    matmul<fp_t>(Trans::N, Trans::C, Dl, Dl, 2 * Dr, H, A);
            } else {
                const KV Z = matmul<fp_t>(Trans::N, Trans::N, Dl * 4, 1, Dr, A,
                                          right_env_[t]);
                KV R("mps_env", Dl);
                Kokkos::parallel_for(
                    "mps_renv", Kokkos::RangePolicy<ExecSpace>(0, Dl),
                    KOKKOS_LAMBDA(std::size_t l) {
                        R(l) = Z(l * 4) + Z(l * 4 + 3);
                    });
                right_env_[s] = R;
            }
            right_ok_[s] = 1;
        }
        return right_env_[i];
    }

  private:
    /// Whether a batched contraction over the sorted wires fits in memory
    [[nodiscard]] bool batchFits(const std::vector<std::size_t> &sorted,
                                 std::size_t q) const {
        std::size_t nb = 1;
        for (std::size_t s = sorted.front(); s <= sorted.back(); s++) {
            if (std::binary_search(sorted.begin(), sorted.end(), s)) {
                if (nb > max_contraction_size / q) {
                    return false;
                }
                nb *= q;
            }
            const std::size_t Dr = bond_[s + 1];
            const std::size_t blk = isMixed() ? Dr : Dr * Dr;
            if (nb * 4 * blk > max_contraction_size) {
                return false;
            }
        }
        return true;
    }

  public: // nvcc: an extended __host__ __device__ lambda needs a public
          // enclosing function
    /**
     * @brief Contract sites [a, a + ops.size()) between the cached
     * environments, applying ops[i - a] (a list of branches) at site i.
     * @return one value per branch combination (last site least
     * significant). Values are not normalised.
     */
    [[nodiscard]] std::vector<StdComplex>
    contractRange(std::size_t a,
                  const std::vector<std::vector<LocalOp>> &ops) const {
        KV env = leftEnv(a);
        std::size_t nb = 1;
        for (std::size_t j = 0; j < ops.size(); j++) {
            env = transferBatch(env, nb, a + j, ops[j]);
            nb *= ops[j].size();
        }
        const std::size_t b = a + ops.size();
        const KV &R = rightEnv(b);
        const std::size_t D = bond_[b];
        const bool mixed = isMixed();
        const std::size_t blk = mixed ? D : D * D;
        KV vals("mps_vals", nb);
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
        using Member = typename TeamPolicy::member_type;
        Kokkos::parallel_for(
            "mps_close", TeamPolicy(static_cast<int>(nb), Kokkos::AUTO),
            KOKKOS_LAMBDA(const Member &team) {
                const std::size_t bb =
                    static_cast<std::size_t>(team.league_rank());
                ComplexT sum{0, 0};
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, blk),
                    [&](std::size_t e, ComplexT &acc) {
                        if (mixed) {
                            acc += env(bb * blk + e) * R(e);
                        } else {
                            const std::size_t r = e / D;
                            const std::size_t r2 = e % D;
                            acc += env(bb * blk + e) * R(r2 * D + r);
                        }
                    },
                    sum);
                Kokkos::single(Kokkos::PerTeam(team),
                               [&]() { vals(bb) = sum; });
            });
        return toHost<fp_t>(vals);
    }
};

} // namespace Pennylane::LightningTensornet
