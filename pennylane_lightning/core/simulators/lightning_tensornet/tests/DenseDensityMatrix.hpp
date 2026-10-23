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
 * @file DenseDensityMatrix.hpp
 * Exact host-side density-matrix simulator used as the reference in the
 * lightning.tensornet C++ tests. Every operation is applied as a dense
 * 2^N x 2^N matrix (rho -> K rho K^dag) in double precision, independently
 * of the MPS contraction and truncation code.
 */
#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include "Error.hpp"
#include "GateMatrices.hpp"
#include "NoiseModel.hpp"
#include "ObservableTerms.hpp"

namespace Pennylane::LightningTensornet::Test {

/**
 * @brief Dense density matrix on N qubits (wire 0 is the most significant
 * bit), with the same interface as the MPS backend for the operations used
 * in the tests.
 *
 * @tparam T Precision of the interface (the state is always stored in double
 * precision).
 */
template <class T> class DenseDensityMatrix {
  public:
    using StdComplex = std::complex<T>;
    using CD = std::complex<double>;
    using DVec = LightningNoisy::CVec<double>;

    explicit DenseDensityMatrix(std::size_t num_qubits)
        : num_qubits_{num_qubits}, rho_(dim() * dim(), CD{0, 0}) {
        rho_[0] = CD{1, 0};
    }

    [[nodiscard]] std::size_t getNumQubits() const { return num_qubits_; }

    void setPureState(const StdComplex *psi, std::size_t length,
                      const std::vector<std::size_t> &wires) {
        PL_ABORT_IF_NOT(length == LightningNoisy::pow2(wires.size()),
                        "Inconsistent state and wires dimensions.");
        DVec full(dim(), CD{0, 0});
        for (std::size_t i = 0; i < length; i++) {
            full[globalIndex(i, wires)] = CD{psi[i].real(), psi[i].imag()};
        }
        for (std::size_t r = 0; r < dim(); r++) {
            for (std::size_t c = 0; c < dim(); c++) {
                rho_[r * dim() + c] = full[r] * std::conj(full[c]);
            }
        }
    }

    [[nodiscard]] LightningNoisy::CVec<T> getDensityMatrix() const {
        return toT(rho_);
    }

    // ------------------------------------------------------------------
    // Operations
    // ------------------------------------------------------------------

    void applyOperation(const std::string &name,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<T> &params = {}) {
        applyOperation(name, {}, {}, wires, inverse, params);
    }

    void applyOperation(const std::string &name,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<T> &params = {}) {
        const auto m = LightningNoisy::namedGateMatrix<double>(
            name, toDouble(params), inverse, wires.size());
        applyControlledUnitary(m, controlled_wires, controlled_values, wires);
    }

    void applyPauliRot(const std::vector<std::size_t> &wires, bool inverse,
                       const std::vector<T> &params, const std::string &word) {
        applyUnitary(LightningNoisy::pauliRotMatrix<double>(
                         static_cast<double>(params.at(0)), word, inverse),
                     wires);
    }

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
        DVec m(matrix, matrix + size);
        if (inverse) {
            m = LightningNoisy::adjoint<double>(
                m, LightningNoisy::pow2(wires.size()));
        }
        applyControlledUnitary(m, controlled_wires, controlled_values, wires);
    }

    /// rho -> sum_k K_k rho K_k^dag
    void applyChannel(const std::vector<LightningNoisy::CVec<T>> &kraus,
                      const std::vector<std::size_t> &wires) {
        std::vector<DVec> ks;
        for (const auto &k : kraus) {
            ks.emplace_back(k.begin(), k.end());
        }
        applyKraus(ks, wires);
    }

    void applyNoise(const LightningNoisy::NoiseModel<T> &model,
                    const std::vector<LightningNoisy::ResolvedNoise> &noise) {
        for (const auto &n : noise) {
            applyChannel(model.channel(n.channel).kraus, n.wires);
        }
    }

    // ------------------------------------------------------------------
    // Measurements
    // ------------------------------------------------------------------

    [[nodiscard]] std::vector<T>
    probs(const std::vector<std::size_t> &wires) const {
        const auto rdm = reduced(wires);
        const std::size_t dk = LightningNoisy::pow2(wires.size());
        std::vector<T> out(dk);
        for (std::size_t i = 0; i < dk; i++) {
            out[i] = static_cast<T>(rdm[i * dk + i].real());
        }
        return out;
    }

    [[nodiscard]] std::vector<T> probs() const { return probs(allWires()); }

    [[nodiscard]] LightningNoisy::CVec<T>
    reducedDensityMatrix(const std::vector<std::size_t> &wires) const {
        return toT(reduced(wires));
    }

    [[nodiscard]] T expval(const LightningNoisy::ObsTerms<T> &terms) const {
        return static_cast<T>(traceWithRho(observableMatrix(terms)));
    }

    [[nodiscard]] T var(const LightningNoisy::ObsTerms<T> &terms) const {
        const auto o = observableMatrix(terms);
        const double e = traceWithRho(o);
        return static_cast<T>(
            traceWithRho(LightningNoisy::matmul<double>(o, o, dim())) - e * e);
    }

    /// Tr(rho^2)
    [[nodiscard]] T purity() const {
        double p = 0;
        for (const auto &x : rho_) {
            p += std::norm(x);
        }
        return static_cast<T>(p);
    }

    // ------------------------------------------------------------------
    // Mid-circuit operations
    // ------------------------------------------------------------------

    /// Project `wire` onto |branch> and renormalise
    void collapse(std::size_t wire, bool branch) {
        const std::size_t sh = num_qubits_ - 1 - wire;
        const std::size_t b = branch ? 1U : 0U;
        double tr = 0;
        for (std::size_t r = 0; r < dim(); r++) {
            for (std::size_t c = 0; c < dim(); c++) {
                if (((r >> sh) & 1U) != b || ((c >> sh) & 1U) != b) {
                    rho_[r * dim() + c] = CD{0, 0};
                }
            }
            tr += rho_[r * dim() + r].real();
        }
        PL_ABORT_IF(tr <= 0,
                    "Cannot collapse onto a zero-probability outcome.");
        for (auto &x : rho_) {
            x /= tr;
        }
    }

    /// Reset `wire` to |0>
    void resetQubit(std::size_t wire) {
        applyKraus({{CD{1, 0}, CD{0, 0}, CD{0, 0}, CD{0, 0}},
                    {CD{0, 0}, CD{1, 0}, CD{0, 0}, CD{0, 0}}},
                   {wire});
    }

    /// Partial trace over `wires`; the remaining wires keep their order
    void traceOut(const std::vector<std::size_t> &wires) {
        std::vector<std::size_t> keep;
        for (std::size_t w = 0; w < num_qubits_; w++) {
            if (std::find(wires.begin(), wires.end(), w) == wires.end()) {
                keep.push_back(w);
            }
        }
        rho_ = reduced(keep);
        num_qubits_ = keep.size();
    }

    /// rho -> rho (x) |0><0| (the new qubit is the last wire)
    void appendQubit() {
        const std::size_t d = dim();
        DVec out(4 * d * d, CD{0, 0});
        for (std::size_t r = 0; r < d; r++) {
            for (std::size_t c = 0; c < d; c++) {
                out[(2 * r) * (2 * d) + 2 * c] = rho_[r * d + c];
            }
        }
        rho_ = std::move(out);
        num_qubits_++;
    }

  private:
    std::size_t num_qubits_;
    DVec rho_;

    [[nodiscard]] std::size_t dim() const {
        return LightningNoisy::pow2(num_qubits_);
    }

    [[nodiscard]] std::vector<std::size_t> allWires() const {
        std::vector<std::size_t> w(num_qubits_);
        std::iota(w.begin(), w.end(), 0);
        return w;
    }

    static std::vector<double> toDouble(const std::vector<T> &v) {
        return {v.begin(), v.end()};
    }

    static LightningNoisy::CVec<T> toT(const DVec &v) {
        LightningNoisy::CVec<T> out(v.size());
        std::transform(v.begin(), v.end(), out.begin(), [](CD x) {
            return StdComplex{static_cast<T>(x.real()),
                              static_cast<T>(x.imag())};
        });
        return out;
    }

    /// Index in the full space of a local index on `wires` (others in |0>)
    [[nodiscard]] std::size_t
    globalIndex(std::size_t local,
                const std::vector<std::size_t> &wires) const {
        const std::size_t k = wires.size();
        std::size_t idx = 0;
        for (std::size_t j = 0; j < k; j++) {
            idx |= ((local >> (k - 1 - j)) & 1U)
                   << (num_qubits_ - 1 - wires[j]);
        }
        return idx;
    }

    /// Controlled-U as a dense matrix on (controlled_wires + wires)
    void
    applyControlledUnitary(const DVec &u,
                           const std::vector<std::size_t> &controlled_wires,
                           const std::vector<bool> &controlled_values,
                           const std::vector<std::size_t> &wires) {
        if (controlled_wires.empty()) {
            applyUnitary(u, wires);
            return;
        }
        const std::size_t nc = controlled_wires.size();
        const std::size_t dt = LightningNoisy::pow2(wires.size());
        const std::size_t d = LightningNoisy::pow2(nc) * dt;
        std::size_t ctrl = 0;
        for (std::size_t i = 0; i < nc; i++) {
            ctrl |= static_cast<std::size_t>(controlled_values[i])
                    << (nc - 1 - i);
        }
        auto m = LightningNoisy::identityMatrix<double>(d);
        for (std::size_t a = 0; a < dt; a++) {
            for (std::size_t b = 0; b < dt; b++) {
                m[(ctrl * dt + a) * d + ctrl * dt + b] = u[a * dt + b];
            }
        }
        std::vector<std::size_t> all = controlled_wires;
        all.insert(all.end(), wires.begin(), wires.end());
        applyUnitary(m, all);
    }

    void applyUnitary(const DVec &u, const std::vector<std::size_t> &wires) {
        applyKraus({u}, wires);
    }

    void applyKraus(const std::vector<DVec> &kraus,
                    const std::vector<std::size_t> &wires) {
        for (auto w : wires) {
            PL_ABORT_IF_NOT(w < num_qubits_, "Wire index out of range");
        }
        const std::size_t d = dim();
        const auto all = allWires();
        DVec out(d * d, CD{0, 0});
        for (const auto &k : kraus) {
            const auto kf = LightningNoisy::embedMatrix<double>(k, wires, all);
            // K rho K^dag = (K (K rho)^dag)^dag, using left products only
            const auto a = LightningNoisy::matmul<double>(kf, rho_, d);
            const auto b = LightningNoisy::matmul<double>(
                kf, LightningNoisy::adjoint<double>(a, d), d);
            const auto c = LightningNoisy::adjoint<double>(b, d);
            for (std::size_t i = 0; i < out.size(); i++) {
                out[i] += c[i];
            }
        }
        rho_ = std::move(out);
    }

    /// Reduced density matrix on `wires` (first wire = MSB)
    [[nodiscard]] DVec reduced(const std::vector<std::size_t> &wires) const {
        const std::size_t k = wires.size();
        const std::size_t dk = LightningNoisy::pow2(k);
        std::size_t mask = 0;
        for (auto w : wires) {
            mask |= std::size_t{1} << (num_qubits_ - 1 - w);
        }
        auto local = [&](std::size_t idx) {
            std::size_t l = 0;
            for (std::size_t j = 0; j < k; j++) {
                l |= ((idx >> (num_qubits_ - 1 - wires[j])) & 1U)
                     << (k - 1 - j);
            }
            return l;
        };
        DVec out(dk * dk, CD{0, 0});
        for (std::size_t r = 0; r < dim(); r++) {
            for (std::size_t c = 0; c < dim(); c++) {
                if ((r & ~mask) == (c & ~mask)) {
                    out[local(r) * dk + local(c)] += rho_[r * dim() + c];
                }
            }
        }
        return out;
    }

    /// sum_t coeff_t O_t as a dense matrix on all wires
    [[nodiscard]] DVec
    observableMatrix(const LightningNoisy::ObsTerms<T> &terms) const {
        const std::size_t d = dim();
        const auto all = allWires();
        DVec o(d * d, CD{0, 0});
        for (const auto &t : terms) {
            const CD coeff{t.coeff.real(), t.coeff.imag()};
            if (t.wires.empty()) {
                for (std::size_t i = 0; i < d; i++) {
                    o[i * d + i] += coeff;
                }
                continue;
            }
            const auto tm = LightningNoisy::termMatrix<T>(t);
            const auto full = LightningNoisy::embedMatrix<double>(
                DVec(tm.begin(), tm.end()), t.wires, all);
            for (std::size_t i = 0; i < o.size(); i++) {
                o[i] += coeff * full[i];
            }
        }
        return o;
    }

    /// Re Tr(M rho)
    [[nodiscard]] double traceWithRho(const DVec &m) const {
        const std::size_t d = dim();
        CD s{0, 0};
        for (std::size_t a = 0; a < d; a++) {
            for (std::size_t b = 0; b < d; b++) {
                s += m[a * d + b] * rho_[b * d + a];
            }
        }
        return s.real();
    }
};

} // namespace Pennylane::LightningTensornet::Test
