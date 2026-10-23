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
 * Tests of the Kokkos MPS backend against an exact dense density matrix
 * (DenseDensityMatrix), which is the reference for every gate, channel and
 * measurement.
 */
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DenseDensityMatrix.hpp"
#include "GateMatrices.hpp"
#include "MPSKokkos.hpp"
#include "NoiseModel.hpp"
#include "ObservableTerms.hpp"

using namespace Pennylane::LightningTensornet;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using Pennylane::LightningNoisy::adjoint;
using Pennylane::LightningNoisy::CVec;
using Pennylane::LightningNoisy::ObsTerm;
using Pennylane::LightningNoisy::ObsTerms;
using Pennylane::LightningNoisy::pow2;
using Pennylane::LightningTensornet::Test::DenseDensityMatrix;

namespace {

template <class T> T tol() {
    return std::is_same_v<T, float> ? T{2e-4} : T{1e-9};
}

struct GateSpec {
    std::string name;
    std::size_t num_params;
    std::size_t num_wires;
};

const std::vector<GateSpec> &gateList() {
    static const std::vector<GateSpec> gates = {
        {"PauliX", 0, 1},
        {"PauliY", 0, 1},
        {"Hadamard", 0, 1},
        {"S", 0, 1},
        {"T", 0, 1},
        {"SX", 0, 1},
        {"RX", 1, 1},
        {"RY", 1, 1},
        {"RZ", 1, 1},
        {"PhaseShift", 1, 1},
        {"Rot", 3, 1},
        {"CNOT", 0, 2},
        {"CY", 0, 2},
        {"CZ", 0, 2},
        {"SWAP", 0, 2},
        {"IsingXX", 1, 2},
        {"IsingYY", 1, 2},
        {"IsingZZ", 1, 2},
        {"IsingXY", 1, 2},
        {"CRX", 1, 2},
        {"CRY", 1, 2},
        {"CRZ", 1, 2},
        {"CRot", 3, 2},
        {"ControlledPhaseShift", 1, 2},
        {"SingleExcitation", 1, 2},
        {"SingleExcitationPlus", 1, 2},
        {"DoubleExcitation", 1, 4},
        {"Toffoli", 0, 3},
        {"CSWAP", 0, 3},
        {"MultiRZ", 1, 2},
        {"MultiRZ", 1, 5},
    };
    return gates;
}

std::vector<std::size_t> randomWires(std::size_t n, std::size_t k,
                                     std::mt19937 &g) {
    std::vector<std::size_t> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), g);
    all.resize(k);
    return all;
}

template <class T> CVec<T> randomUnitary(std::size_t k, std::mt19937 &g) {
    // exp(i H) via a product of PauliRot matrices (exactly unitary)
    std::uniform_real_distribution<double> u(-3, 3);
    const std::size_t dim = pow2(k);
    CVec<T> m = Pennylane::LightningNoisy::identityMatrix<T>(dim);
    const std::string letters = "IXYZ";
    for (int rep = 0; rep < 6; rep++) {
        std::string w;
        for (std::size_t i = 0; i < k; i++) {
            w.push_back(letters[static_cast<std::size_t>(g() % 4)]);
        }
        m = Pennylane::LightningNoisy::matmul<T>(
            Pennylane::LightningNoisy::pauliRotMatrix<T>(static_cast<T>(u(g)),
                                                         w, false),
            m, dim);
    }
    return m;
}

template <class T> std::vector<CVec<T>> depolarizing(T p) {
    using C = std::complex<T>;
    const T a = std::sqrt(1 - p);
    const T b = std::sqrt(p / 3);
    return {{C{a, 0}, C{0, 0}, C{0, 0}, C{a, 0}},
            {C{0, 0}, C{b, 0}, C{b, 0}, C{0, 0}},
            {C{0, 0}, C{0, -b}, C{0, b}, C{0, 0}},
            {C{b, 0}, C{0, 0}, C{0, 0}, C{-b, 0}}};
}

template <class T> std::vector<CVec<T>> amplitudeDamping(T gamma) {
    using C = std::complex<T>;
    return {{C{1, 0}, C{0, 0}, C{0, 0}, C{std::sqrt(1 - gamma), 0}},
            {C{0, 0}, C{std::sqrt(gamma), 0}, C{0, 0}, C{0, 0}}};
}

/// Two-qubit correlated dephasing (not a product channel)
template <class T> std::vector<CVec<T>> correlatedDephasing(T p) {
    using C = std::complex<T>;
    CVec<T> k0(16, C{0, 0});
    CVec<T> k1(16, C{0, 0});
    const T a = std::sqrt(1 - p);
    const T b = std::sqrt(p);
    const T zz[4] = {1, -1, -1, 1};
    for (std::size_t i = 0; i < 4; i++) {
        k0[i * 4 + i] = C{a, 0};
        k1[i * 4 + i] = C{b * zz[i], 0};
    }
    return {k0, k1};
}

/// Apply the same random operation to both states
template <class T>
void applyRandomOp(MPSKokkos<T> &mps, DenseDensityMatrix<T> &dm,
                   std::mt19937 &g, bool with_channels) {
    const std::size_t n = mps.getNumQubits();
    std::uniform_real_distribution<double> ang(-3.1, 3.1);
    const auto kind = g() % (with_channels ? 8U : 6U);
    if (kind <= 2) {
        const auto &gates = gateList();
        const auto &gs = gates[g() % gates.size()];
        if (gs.num_wires > n) {
            return;
        }
        const auto wires = randomWires(n, gs.num_wires, g);
        std::vector<T> params(gs.num_params);
        for (auto &p : params) {
            p = static_cast<T>(ang(g));
        }
        const bool inv = (g() % 2) == 1;
        mps.applyOperation(gs.name, wires, inv, params);
        dm.applyOperation(gs.name, wires, inv, params);
    } else if (kind == 3) {
        // controlled named gate with mixed control values
        if (n < 3) {
            return;
        }
        const auto w = randomWires(n, 3, g);
        const std::vector<std::size_t> ctrls = {w[0], w[1]};
        const std::vector<bool> vals = {true, (g() % 2) == 1};
        const std::vector<T> params = {static_cast<T>(ang(g))};
        mps.applyOperation("RY", ctrls, vals, {w[2]}, false, params);
        dm.applyOperation("RY", ctrls, vals, {w[2]}, false, params);
    } else if (kind == 4) {
        const std::size_t k = 1 + g() % std::min<std::size_t>(3, n);
        const auto wires = randomWires(n, k, g);
        const auto u = randomUnitary<T>(k, g);
        mps.applyMatrix(u.data(), u.size(), wires);
        dm.applyMatrix(u.data(), u.size(), wires);
    } else if (kind == 5) {
        const std::size_t k = 1 + g() % std::min<std::size_t>(5, n);
        const auto wires = randomWires(n, k, g);
        const std::string letters = "IXYZ";
        std::string w;
        for (std::size_t i = 0; i < k; i++) {
            w.push_back(letters[g() % 4]);
        }
        const std::vector<T> params = {static_cast<T>(ang(g))};
        mps.applyPauliRot(wires, false, params, w);
        dm.applyPauliRot(wires, false, params, w);
    } else if (kind == 6) {
        const auto wires = randomWires(n, 1, g);
        const auto k = (g() % 2) == 0 ? depolarizing<T>(T{0.1})
                                      : amplitudeDamping<T>(T{0.3});
        mps.applyChannel(k, wires);
        dm.applyChannel(k, wires);
    } else {
        if (n < 2) {
            return;
        }
        const auto wires = randomWires(n, 2, g);
        const auto k = correlatedDephasing<T>(T{0.2});
        mps.applyChannel(k, wires);
        dm.applyChannel(k, wires);
    }
}

template <class T>
void requireClose(const CVec<T> &a, const CVec<T> &b, T eps) {
    REQUIRE(a.size() == b.size());
    T err = 0;
    for (std::size_t i = 0; i < a.size(); i++) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    CHECK(err <= eps);
}

template <class T>
void requireSameState(const MPSKokkos<T> &mps, const DenseDensityMatrix<T> &dm,
                      T eps = tol<T>()) {
    requireClose<T>(mps.getDensityMatrix(), dm.getDensityMatrix(), eps);
}

template <class T>
ObsTerms<T> randomObservable(std::size_t n, std::mt19937 &g) {
    ObsTerms<T> obs;
    const std::string letters = "XYZ";
    std::uniform_real_distribution<double> c(-1, 1);
    for (int t = 0; t < 3; t++) {
        ObsTerm<T> term;
        term.coeff = {static_cast<T>(c(g)), 0};
        const std::size_t k = 1 + g() % std::min<std::size_t>(3, n);
        term.wires = randomWires(n, k, g);
        for (std::size_t i = 0; i < k; i++) {
            term.paulis.push_back(letters[g() % 3]);
        }
        obs.push_back(term);
    }
    // a dense (Hermitian) term
    ObsTerm<T> h;
    h.dense = true;
    h.wires = randomWires(n, std::min<std::size_t>(2, n), g);
    const std::size_t dim = pow2(h.wires.size());
    CVec<T> a(dim * dim);
    for (auto &x : a) {
        x = {static_cast<T>(c(g)), static_cast<T>(c(g))};
    }
    h.matrix.assign(dim * dim, {0, 0});
    for (std::size_t i = 0; i < dim; i++) {
        for (std::size_t j = 0; j < dim; j++) {
            h.matrix[i * dim + j] = a[i * dim + j] + std::conj(a[j * dim + i]);
        }
    }
    obs.push_back(h);
    return obs;
}

template <class T>
void checkMeasurements(const MPSKokkos<T> &mps, const DenseDensityMatrix<T> &dm,
                       std::mt19937 &g) {
    const std::size_t n = mps.getNumQubits();
    const T eps = tol<T>() * 10;
    // probabilities: all wires, a subset in arbitrary order
    const auto pa = mps.probs();
    const auto pb = dm.probs();
    for (std::size_t i = 0; i < pa.size(); i++) {
        CHECK(pa[i] == Approx(pb[i]).margin(eps));
    }
    const auto sub = randomWires(n, std::min<std::size_t>(3, n), g);
    const auto sa = mps.probs(sub);
    const auto sb = dm.probs(sub);
    for (std::size_t i = 0; i < sa.size(); i++) {
        CHECK(sa[i] == Approx(sb[i]).margin(eps));
    }
    requireClose<T>(mps.reducedDensityMatrix(sub), dm.reducedDensityMatrix(sub),
                    eps);
    const auto obs = randomObservable<T>(n, g);
    CHECK(mps.expval(obs) == Approx(dm.expval(obs)).margin(eps * 10));
    CHECK(mps.var(obs) == Approx(dm.var(obs)).margin(eps * 50));
    CHECK(mps.purity() == Approx(dm.purity()).margin(eps * 10));
}

} // namespace

TEMPLATE_TEST_CASE("MPSKokkos initial state and state preparation",
                   "[LightningTensornet]", float, double) {
    using T = TestType;
    using C = std::complex<T>;
    SECTION("|0...0>") {
        const MPSKokkos<T> mps(4);
        CHECK(mps.getNumQubits() == 4);
        CHECK_FALSE(mps.isMixed());
        const auto p = mps.probs();
        CHECK(p[0] == Approx(1));
        const auto bd = mps.bondDims();
        CHECK(bd == std::vector<std::size_t>(5, 1));
        CHECK(mps.normalization() == Approx(1));
    }
    SECTION("basis state") {
        for (bool mixed : {false, true}) {
            MPSKokkos<T> mps(4, 16, 0, "abs", mixed);
            CHECK(mps.isMixed() == mixed);
            mps.setBasisState({1, 0, 1}, {3, 0, 1});
            const auto p = mps.probs();
            // wires 0..3 = 0, 1, 0, 1 -> index 0b0101
            CHECK(p[5] == Approx(1));
        }
    }
    SECTION("pure state on a subset of wires") {
        std::mt19937 g(3);
        for (bool mixed : {false, true}) {
            const std::size_t n = 6;
            const std::vector<std::size_t> wires = {4, 1, 2};
            std::normal_distribution<double> nd(0, 1);
            std::vector<C> psi(8);
            double nrm = 0;
            for (auto &x : psi) {
                x = {static_cast<T>(nd(g)), static_cast<T>(nd(g))};
                nrm += std::norm(x);
            }
            for (auto &x : psi) {
                x /= static_cast<T>(std::sqrt(nrm));
            }
            MPSKokkos<T> mps(n, 64, 0, "abs", mixed);
            mps.setPureState(psi.data(), psi.size(), wires);
            DenseDensityMatrix<T> dm(n);
            dm.setPureState(psi.data(), psi.size(), wires);
            CHECK(mps.isMixed() == mixed);
            requireSameState<T>(mps, dm);
            if (!mixed) {
                std::vector<C> sv(pow2(n));
                mps.getStateVector(sv.data(), sv.size());
                // compare |sv><sv| with the density matrix
                const auto rho = dm.getDensityMatrix();
                for (std::size_t i = 0; i < sv.size(); i++) {
                    CHECK(std::abs(sv[i] * std::conj(sv[i]) -
                                   rho[i * sv.size() + i]) <= tol<T>());
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("MPSKokkos random unitary circuits match the density matrix",
                   "[LightningTensornet]", float, double) {
    using T = TestType;
    std::mt19937 g(1234);
    for (std::size_t n : {1U, 2U, 5U, 7U}) {
        MPSKokkos<T> mps(n, 256);
        DenseDensityMatrix<T> dm(n);
        for (int step = 0; step < 60; step++) {
            applyRandomOp<T>(mps, dm, g, false);
        }
        CHECK_FALSE(mps.isMixed());
        CHECK(mps.truncationError() == Approx(0).margin(tol<T>()));
        requireSameState<T>(mps, dm, tol<T>() * 20);
        checkMeasurements<T>(mps, dm, g);
    }
}

TEMPLATE_TEST_CASE("MPSKokkos noisy circuits match the density matrix",
                   "[LightningTensornet]", float, double) {
    using T = TestType;
    std::mt19937 g(99);
    for (std::size_t n : {1U, 3U, 6U}) {
        for (bool start_mixed : {false, true}) {
            MPSKokkos<T> mps(n, 1024, 0, "abs", start_mixed);
            DenseDensityMatrix<T> dm(n);
            for (int step = 0; step < 50; step++) {
                applyRandomOp<T>(mps, dm, g, true);
            }
            requireSameState<T>(mps, dm, tol<T>() * 50);
            checkMeasurements<T>(mps, dm, g);
        }
    }
}

TEST_CASE("MPSKokkos unitary channels keep the state pure",
          "[LightningTensornet]") {
    MPSKokkos<double> mps(3);
    std::mt19937 g(5);
    const auto u = randomUnitary<double>(2, g);
    mps.applyChannel({u}, {0, 2});
    CHECK_FALSE(mps.isMixed());
    mps.applyChannel(depolarizing<double>(0.0), {1}); // 4 Kraus ops
    CHECK(mps.isMixed());
    mps.toMixed(); // no-op
    CHECK(mps.isMixed());
}

TEST_CASE("MPSKokkos GlobalPhase and identity PauliRot",
          "[LightningTensornet]") {
    using C = std::complex<double>;
    MPSKokkos<double> mps(2);
    mps.applyOperation("Hadamard", {0});
    mps.applyOperation("GlobalPhase", {}, false, {0.7});
    mps.applyPauliRot({0, 1}, false, {0.4}, "II");
    std::vector<C> sv(4);
    mps.getStateVector(sv.data(), sv.size());
    const C ph = std::exp(C{0, -0.7}) * std::exp(C{0, -0.2});
    CHECK(std::abs(sv[0] - ph / std::sqrt(2.0)) < 1e-12);
    CHECK(std::abs(sv[2] - ph / std::sqrt(2.0)) < 1e-12);
    // controlled global phase = phase shift on the control
    MPSKokkos<double> a(2);
    DenseDensityMatrix<double> b(2);
    for (auto *op : {"Hadamard"}) {
        a.applyOperation(op, {0});
        b.applyOperation(op, {0});
    }
    a.applyOperation("GlobalPhase", {0}, {true}, {1}, false, {0.3});
    b.applyOperation("GlobalPhase", {0}, {true}, {1}, false, {0.3});
    requireSameState<double>(a, b);
}

TEST_CASE("MPSKokkos bond dimensions and truncation", "[LightningTensornet]") {
    const std::size_t n = 12;
    SECTION("GHZ state has bond dimension 2") {
        MPSKokkos<double> mps(n);
        mps.applyOperation("Hadamard", {0});
        for (std::size_t i = 0; i + 1 < n; i++) {
            mps.applyOperation("CNOT", {i, i + 1});
        }
        const auto bd = mps.bondDims();
        CHECK(*std::max_element(bd.begin(), bd.end()) == 2);
        const auto p = mps.probs();
        CHECK(p.front() == Approx(0.5));
        CHECK(p.back() == Approx(0.5));
        ObsTerms<double> zz(1);
        zz[0].wires = {0, n - 1};
        zz[0].paulis = "ZZ";
        CHECK(mps.expval(zz) == Approx(1));
        // long-range gate between the first and last qubits
        mps.applyOperation("CNOT", {0, n - 1});
        CHECK(mps.probs({n - 1})[0] == Approx(1));
    }
    SECTION("max_bond_dim limits the bonds and records the error") {
        std::mt19937 g(17);
        MPSKokkos<double> exact(n, 4096);
        MPSKokkos<double> trunc(n, 4);
        std::uniform_real_distribution<double> ang(-3, 3);
        for (int layer = 0; layer < 6; layer++) {
            for (std::size_t i = 0; i < n; i++) {
                const std::vector<double> p = {ang(g), ang(g), ang(g)};
                exact.applyOperation("Rot", {i}, false, p);
                trunc.applyOperation("Rot", {i}, false, p);
            }
            for (std::size_t i = layer % 2; i + 1 < n; i += 2) {
                exact.applyOperation("CNOT", {i, i + 1});
                trunc.applyOperation("CNOT", {i, i + 1});
            }
        }
        const auto bd = trunc.bondDims();
        CHECK(*std::max_element(bd.begin(), bd.end()) <= 4);
        CHECK(trunc.truncationError() > 0);
        CHECK(exact.truncationError() == Approx(0).margin(1e-12));
        const auto be = exact.bondDims();
        CHECK(*std::max_element(be.begin(), be.end()) > 4);
        // truncated state stays normalised
        const auto p = trunc.probs();
        CHECK(std::accumulate(p.begin(), p.end(), 0.0) == Approx(1));
        CHECK(trunc.normalization() == Approx(1).epsilon(1e-10));
    }
    SECTION("relative cutoff") {
        MPSKokkos<double> mps(4, 64, 0.5, "rel");
        CHECK(mps.getCutoffMode() == "rel");
        mps.applyOperation("RY", {0}, false, {0.2});
        mps.applyOperation("CNOT", {0, 1});
        // Schmidt coefficients cos(0.1), sin(0.1): the second is dropped
        const auto bd = mps.bondDims();
        CHECK(bd[1] == 1);
        CHECK(mps.truncationError() == Approx(std::pow(std::sin(0.1), 2)));
    }
}

TEMPLATE_TEST_CASE("MPSKokkos sampling", "[LightningTensornet]", float,
                   double) {
    using T = TestType;
    std::mt19937 g(5);
    for (bool mixed : {false, true}) {
        const std::size_t n = 5;
        MPSKokkos<T> mps(n, 64, 0, "abs", mixed);
        DenseDensityMatrix<T> dm(n);
        for (int s = 0; s < 30; s++) {
            applyRandomOp<T>(mps, dm, g, mixed);
        }
        const std::vector<std::size_t> wires = {3, 0, 4};
        const auto p = dm.probs(wires);
        std::mt19937_64 rng(42);
        const std::size_t shots = 10000;
        const auto samples = mps.sampleWires(wires, shots, rng);
        REQUIRE(samples.size() == shots * wires.size());
        std::vector<double> freq(8, 0);
        for (std::size_t s = 0; s < shots; s++) {
            std::size_t idx = 0;
            for (std::size_t j = 0; j < wires.size(); j++) {
                const auto b = samples[s * wires.size() + j];
                REQUIRE(b <= 1);
                idx = (idx << 1U) | b;
            }
            freq[idx] += 1.0 / shots;
        }
        for (std::size_t i = 0; i < 8; i++) {
            // 5 sigma binomial bound
            const double sigma = std::sqrt(p[i] * (1 - p[i]) / shots);
            CHECK(std::abs(freq[i] - p[i]) <= 5 * sigma + 1e-3);
        }
        // deterministic for a given seed
        std::mt19937_64 r1(7);
        std::mt19937_64 r2(7);
        CHECK(mps.sampleWires(wires, 100, r1) ==
              mps.sampleWires(wires, 100, r2));
    }
}

TEMPLATE_TEST_CASE("MPSKokkos mid-circuit operations", "[LightningTensornet]",
                   float, double) {
    using T = TestType;
    std::mt19937 g(21);
    for (bool noisy : {false, true}) {
        const std::size_t n = 5;
        MPSKokkos<T> mps(n, 256);
        DenseDensityMatrix<T> dm(n);
        for (int s = 0; s < 25; s++) {
            applyRandomOp<T>(mps, dm, g, noisy);
        }
        SECTION("collapse") {
            const auto p = dm.probs({2});
            const bool branch = p[1] > p[0];
            mps.collapse(2, branch);
            dm.collapse(2, branch);
            requireSameState<T>(mps, dm, tol<T>() * 50);
        }
        SECTION("reset") {
            mps.resetQubit(1);
            dm.resetQubit(1);
            requireSameState<T>(mps, dm, tol<T>() * 50);
            CHECK(mps.probs({1})[0] == Approx(1).margin(tol<T>() * 10));
        }
        SECTION("trace out and append") {
            mps.traceOut({3, 0});
            dm.traceOut({3, 0});
            REQUIRE(mps.getNumQubits() == 3);
            requireSameState<T>(mps, dm, tol<T>() * 50);
            mps.appendQubit();
            dm.appendQubit();
            REQUIRE(mps.getNumQubits() == 4);
            requireSameState<T>(mps, dm, tol<T>() * 50);
            mps.applyOperation("CNOT", {0, 3});
            dm.applyOperation("CNOT", {0, 3});
            requireSameState<T>(mps, dm, tol<T>() * 50);
            mps.traceOut({0, 1, 2, 3});
            CHECK(mps.getNumQubits() == 0);
            mps.appendQubit();
            CHECK(mps.probs()[0] == Approx(1));
        }
    }
}

TEST_CASE("MPSKokkos tracing out a product qubit keeps the state pure",
          "[LightningTensornet]") {
    MPSKokkos<double> mps(3);
    mps.applyOperation("Hadamard", {0});
    mps.applyOperation("CNOT", {0, 2});
    mps.applyOperation("RY", {1}, false, {0.9});
    mps.traceOut({1});
    CHECK_FALSE(mps.isMixed());
    CHECK(mps.getNumQubits() == 2);
    const auto p = mps.probs();
    CHECK(p[0] == Approx(0.5));
    CHECK(p[3] == Approx(0.5));
    mps.resetQubit(0); // entangled: switches to mixed
    CHECK(mps.isMixed());
    CHECK(mps.purity() == Approx(0.5));
    const auto q = mps.probs();
    CHECK(q[0] == Approx(0.5));
    CHECK(q[1] == Approx(0.5));
}

TEST_CASE("MPSKokkos copy is independent", "[LightningTensornet]") {
    MPSKokkos<double> a(3);
    a.applyOperation("Hadamard", {0});
    MPSKokkos<double> b(a);
    b.applyOperation("PauliX", {1});
    CHECK(a.probs({1})[0] == Approx(1));
    CHECK(b.probs({1})[1] == Approx(1));
    CHECK(b.getMaxBondDim() == a.getMaxBondDim());
}

TEST_CASE("MPSKokkos measure with postselection", "[LightningTensornet]") {
    MPSKokkos<double> mps(2);
    mps.applyOperation("Hadamard", {0});
    mps.applyOperation("CNOT", {0, 1});
    std::mt19937 g(1);
    CHECK(mps.measure(0, 1, g));
    CHECK(mps.probs({1})[1] == Approx(1));
    CHECK_THROWS_WITH(mps.measure(1, 0, g), ContainsSubstring("zero"));
}

TEST_CASE("MPSKokkos error handling", "[LightningTensornet]") {
    MPSKokkos<double> mps(3);
    CHECK_THROWS_WITH(mps.applyOperation("PauliX", {3}),
                      ContainsSubstring("out of range"));
    CHECK_THROWS_WITH(MPSKokkos<double>(2, 0), ContainsSubstring("positive"));
    CHECK_THROWS_WITH(MPSKokkos<double>(2, 4, 0, "bad"),
                      ContainsSubstring("cutoff_mode"));
    mps.applyChannel(depolarizing<double>(0.1), {0});
    std::vector<std::complex<double>> sv(8);
    CHECK_THROWS_WITH(mps.getStateVector(sv.data(), sv.size()),
                      ContainsSubstring("mixed"));
    MPSKokkos<double> big(12);
    const auto u = Pennylane::LightningNoisy::identityMatrix<double>(pow2(11));
    CHECK_THROWS_WITH(
        big.applyMatrix(u.data(), u.size(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
        ContainsSubstring("too many wires"));
}

TEMPLATE_TEST_CASE("MPSKokkos with a native noise model",
                   "[LightningTensornet]", float, double) {
    using T = TestType;
    using Pennylane::LightningNoisy::NoiseModel;
    using Pennylane::LightningNoisy::ResolvedNoise;
    SECTION("unitary noise keeps the MPS pure") {
        // bit flip p = 1 after every PauliX on any wire -> X then X = I
        const std::string spec =
            "LNM1;K=1/1/0/0/1/0/1/0/0/0;G=PauliX/*1/a0@r0;M=probs/*1/b0@r0";
        NoiseModel<T> nm(spec);
        MPSKokkos<T> mps(2);
        std::vector<ResolvedNoise> before;
        std::vector<ResolvedNoise> after;
        nm.lookupGate("PauliX", {1}, before, after);
        mps.applyNoise(nm, before);
        mps.applyOperation("PauliX", {1});
        mps.applyNoise(nm, after);
        CHECK_FALSE(mps.isMixed());
        CHECK(mps.probs()[0] == Approx(1.0));
        std::vector<ResolvedNoise> ro;
        nm.lookupReadout("probs", {0}, ro);
        MPSKokkos<T> copy(mps);
        copy.applyNoise(nm, ro);
        CHECK(copy.probs()[2] == Approx(1.0)); // |10>
        CHECK(mps.probs()[0] == Approx(1.0));  // original untouched
    }
    SECTION("bit flip after every Hadamard matches the density matrix") {
        const double p = 0.2;
        auto f = [](double x) { return std::to_string(x); };
        const std::string a = f(std::sqrt(1 - p));
        const std::string b = f(std::sqrt(p));
        const std::string spec = "LNM1;K=1/2/" + a + "/0/0/0/0/0/" + a +
                                 "/0/0/0/" + b + "/0/" + b +
                                 "/0/0/0;G=Hadamard/*1/a0@r0";
        NoiseModel<T> nm(spec);
        MPSKokkos<T> mps(3, 64, 0, "abs", !nm.empty());
        DenseDensityMatrix<T> dm(3);
        CHECK(mps.isMixed());
        for (std::size_t w = 0; w < 3; w++) {
            std::vector<ResolvedNoise> before;
            std::vector<ResolvedNoise> after;
            nm.lookupGate("Hadamard", {w}, before, after);
            mps.applyNoise(nm, before);
            mps.applyOperation("Hadamard", {w});
            mps.applyNoise(nm, after);
            dm.applyNoise(nm, before);
            dm.applyOperation("Hadamard", {w});
            dm.applyNoise(nm, after);
            if (w > 0) {
                mps.applyOperation("CNOT", {w - 1, w});
                dm.applyOperation("CNOT", {w - 1, w});
            }
        }
        requireSameState<T>(mps, dm, tol<T>() * 20);
    }
}
