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
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DenseDensityMatrix.hpp"
#include "LightningTensornetSimulator.hpp"
#include "QuantumDevice.hpp"

using namespace Catalyst::Runtime::Simulator;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using CD = std::complex<double>;
using DM = Pennylane::LightningTensornet::Test::DenseDensityMatrix<double>;
using MPS = Pennylane::LightningTensornet::MPSKokkos<double>;

extern "C" Catalyst::Runtime::QuantumDevice *
LightningTensornetSimulatorFactory(const char *kwargs);

namespace {

/// Bit flip with probability p as an "LNM1" channel entry
std::string bitflipChannel(double p) {
    const double a = std::sqrt(1 - p);
    const double b = std::sqrt(p);
    auto f = [](double x) {
        std::ostringstream os;
        os.precision(17);
        os << x;
        return os.str();
    };
    return "K=1/2/" + f(a) + "/0/0/0/0/0/" + f(a) + "/0/0/0/" + f(b) + "/0/" +
           f(b) + "/0/0/0";
}

std::string kwargsFor(const std::string &spec) {
    return "{'noise_model': '" + spec + "'}";
}

double probsZ(LightningTensornetSimulator &sim, QubitIdType q) {
    std::vector<double> p(2);
    DataView<double, 1> view(p);
    sim.PartialProbs(view, {q});
    return p[0] - p[1];
}

template <class A, class B>
void requireSameDM(const A &a, const B &b, double eps = 1e-10) {
    const auto x = a.getDensityMatrix();
    const auto y = b.getDensityMatrix();
    REQUIRE(x.size() == y.size());
    for (std::size_t i = 0; i < x.size(); i++) {
        REQUIRE(std::abs(x[i] - y[i]) < eps);
    }
}

} // namespace

TEST_CASE("LightningTensornetSimulator factory and kwargs",
          "[LTensornetCatalyst]") {
    std::unique_ptr<Catalyst::Runtime::QuantumDevice> dev(
        LightningTensornetSimulatorFactory("{}"));
    REQUIRE(dev->GetNumQubits() == 0);
    dev->AllocateQubits(3);
    REQUIRE(dev->GetNumQubits() == 3);

    LightningTensornetSimulator sim(
        kwargsFor("LNM1;" + bitflipChannel(0.1) + ";G=RX/*1/a0@r0"));
    REQUIRE(!sim.getNoiseModel().empty());
    REQUIRE(sim.getNoiseModel().numChannels() == 1);

    REQUIRE_THROWS_WITH(LightningTensornetSimulator(kwargsFor("LNM1;Q=1")),
                        ContainsSubstring("Invalid noise model"));
}

TEST_CASE("LightningTensornetSimulator gates match the dense density matrix",
          "[LTensornetCatalyst]") {
    LightningTensornetSimulator sim;
    auto q = sim.AllocateQubits(3);
    DM ref(3);

    sim.NamedOperation("Hadamard", {}, {q[0]}, false, {}, {}, {});
    ref.applyOperation("Hadamard", {0});
    sim.NamedOperation("RX", {0.3}, {q[1]}, true, {}, {}, {});
    ref.applyOperation("RX", {1}, true, {0.3});
    sim.NamedOperation("CRY", {0.7}, {q[0], q[2]}, false, {}, {}, {});
    ref.applyOperation("CRY", {0, 2}, false, {0.7});
    sim.NamedOperation("RZ", {1.1}, {q[1]}, false, {q[0], q[2]}, {true, false},
                       {});
    ref.applyOperation("RZ", {0, 2}, {true, false}, {1}, false, {1.1});
    sim.NamedOperation("PauliRot", {0.4}, {q[0], q[1], q[2]}, false, {}, {},
                       {"XIY"});
    ref.applyPauliRot({0, 2}, false, {0.4}, "XY");
    const std::vector<CD> u{
        CD{0, 1} / std::sqrt(2.0), CD{1, 0} / std::sqrt(2.0),
        CD{1, 0} / std::sqrt(2.0), CD{0, 1} / std::sqrt(2.0)};
    sim.MatrixOperation(u, {q[2]}, true, {q[1]}, {true});
    ref.applyControlledMatrix(u.data(), u.size(), {1}, {true}, {2}, true);
    sim.NamedOperation("GlobalPhase", {0.2}, {}, false, {q[1]}, {true}, {});
    ref.applyOperation("GlobalPhase", {1}, {true}, {}, false, {0.2});
    requireSameDM(sim.getState(), ref);

    SECTION("channels given as operations") {
        sim.NamedOperation("AmplitudeDamping", {0.3}, {q[0]}, false, {}, {},
                           {});
        using Pennylane::LightningNoisy::CVec;
        std::vector<CVec<double>> k{
            {CD{1, 0}, CD{0, 0}, CD{0, 0}, CD{std::sqrt(0.7), 0}},
            {CD{0, 0}, CD{std::sqrt(0.3), 0}, CD{0, 0}, CD{0, 0}}};
        ref.applyChannel(k, {0});
        requireSameDM(sim.getState(), ref);
        REQUIRE_THROWS_WITH(
            sim.NamedOperation("BitFlip", {0.1, 0.2}, {q[0]}, false, {}, {},
                               {}),
            ContainsSubstring("Wrong number of channel parameters"));
    }
    SECTION("pure states are available as a vector") {
        std::vector<CD> buf(8);
        DataView<CD, 1> view(buf);
        sim.State(view);
        const auto rho = ref.getDensityMatrix();
        for (std::size_t i = 0; i < 8; i++) {
            for (std::size_t j = 0; j < 8; j++) {
                REQUIRE(std::abs(buf[i] * std::conj(buf[j]) - rho[i * 8 + j]) <
                        1e-10);
            }
        }
    }
    SECTION("mixed states are not available as a vector") {
        sim.NamedOperation("AmplitudeDamping", {0.3}, {q[0]}, false, {}, {},
                           {});
        std::vector<CD> buf(8);
        DataView<CD, 1> view(buf);
        REQUIRE_THROWS_WITH(sim.State(view), ContainsSubstring("mixed"));
    }
    SECTION("invalid wires") {
        REQUIRE_THROWS_WITH(
            sim.NamedOperation("PauliX", {}, {17}, false, {}, {}, {}),
            ContainsSubstring("do not refer to qubits"));
    }
}

TEST_CASE("LightningTensornetSimulator applies gate noise",
          "[LTensornetCatalyst]") {
    const double p = 0.1;
    const std::string ch = bitflipChannel(p);

    SECTION("wildcard rule after a gate") {
        LightningTensornetSimulator sim(
            kwargsFor("LNM1;" + ch + ";G=RX/*1/a0@r0"));
        auto q = sim.AllocateQubits(2);
        sim.NamedOperation("RX", {0.0}, {q[1]}, false, {}, {}, {});
        REQUIRE(probsZ(sim, q[1]) == Approx(1 - 2 * p));
        REQUIRE(probsZ(sim, q[0]) == Approx(1.0));
        // other gates are noiseless
        sim.NamedOperation("RY", {0.0}, {q[0]}, false, {}, {}, {});
        REQUIRE(probsZ(sim, q[0]) == Approx(1.0));
    }

    SECTION("before-noise, absolute wires and exact wire tuples") {
        // CNOT on (0, 1): flip wire 0 before and wire 1 after; CNOT on
        // (1, 0) is explicitly noiseless.
        LightningTensornetSimulator sim(
            kwargsFor("LNM1;" + ch + ";G=CNOT/0_1/b0@0+a0@r1;G=CNOT/1_0/-"));
        auto q = sim.AllocateQubits(2);
        sim.NamedOperation("CNOT", {}, {q[0], q[1]}, false, {}, {}, {});
        // before: wire0 flipped w.p. p; CNOT copies it to wire1; after: wire1
        // flipped again w.p. p
        REQUIRE(probsZ(sim, q[0]) == Approx(1 - 2 * p));
        const double p1 = p * (1 - p) + (1 - p) * p;
        REQUIRE(probsZ(sim, q[1]) == Approx(1 - 2 * p1));
        LightningTensornetSimulator sim2(
            kwargsFor("LNM1;" + ch + ";G=CNOT/0_1/b0@0+a0@r1;G=CNOT/1_0/-"));
        auto q2 = sim2.AllocateQubits(2);
        sim2.NamedOperation("CNOT", {}, {q2[1], q2[0]}, false, {}, {}, {});
        REQUIRE(probsZ(sim2, q2[0]) == Approx(1.0));
        REQUIRE(probsZ(sim2, q2[1]) == Approx(1.0));
    }

    SECTION("keys for adjoint, controlled, matrix and PauliRot gates") {
        const std::string spec = "LNM1;" + ch +
                                 ";G=Adjoint(S)/*1/a0@r0"
                                 ";G=C(RX)/*3/a0@r2"
                                 ";G=QubitUnitary/*1/a0@r0"
                                 ";G=PauliRot/*2/a0@r1";
        LightningTensornetSimulator sim(kwargsFor(spec));
        auto q = sim.AllocateQubits(3);
        sim.NamedOperation("S", {}, {q[0]}, false, {}, {}, {});
        REQUIRE(probsZ(sim, q[0]) == Approx(1.0));
        sim.NamedOperation("S", {}, {q[0]}, true, {}, {}, {});
        REQUIRE(probsZ(sim, q[0]) == Approx(1 - 2 * p));

        LightningTensornetSimulator sim2(kwargsFor(spec));
        q = sim2.AllocateQubits(3);
        sim2.NamedOperation("RX", {0.0}, {q[2]}, false, {q[0], q[1]},
                            {true, true}, {});
        REQUIRE(probsZ(sim2, q[2]) == Approx(1 - 2 * p));
        REQUIRE(probsZ(sim2, q[0]) == Approx(1.0));

        LightningTensornetSimulator sim3(kwargsFor(spec));
        q = sim3.AllocateQubits(3);
        const std::vector<CD> id{CD{1, 0}, CD{0, 0}, CD{0, 0}, CD{1, 0}};
        sim3.MatrixOperation(id, {q[1]}, false, {}, {});
        REQUIRE(probsZ(sim3, q[1]) == Approx(1 - 2 * p));
        sim3.NamedOperation("PauliRot", {0.0}, {q[0], q[2]}, false, {}, {},
                            {"ZZ"});
        REQUIRE(probsZ(sim3, q[2]) == Approx(1 - 2 * p));
        REQUIRE(probsZ(sim3, q[0]) == Approx(1.0));
    }

    SECTION("wire labels are stable across executions of a pooled device") {
        LightningTensornetSimulator sim(
            kwargsFor("LNM1;" + ch + ";G=RX/0/a0@0"));
        for (int rep = 0; rep < 3; rep++) {
            auto q = sim.AllocateQubits(2);
            sim.NamedOperation("RX", {0.0}, {q[0]}, false, {}, {}, {});
            sim.NamedOperation("RX", {0.0}, {q[1]}, false, {}, {}, {});
            REQUIRE(probsZ(sim, q[0]) == Approx(1 - 2 * p));
            REQUIRE(probsZ(sim, q[1]) == Approx(1.0));
            sim.ReleaseQubits(q);
            REQUIRE(sim.GetNumQubits() == 0);
        }
    }
}

TEST_CASE("LightningTensornetSimulator readout noise", "[LTensornetCatalyst]") {
    const double p = 0.2;
    LightningTensornetSimulator sim(kwargsFor(
        "LNM1;" + bitflipChannel(p) + ";M=expval/*1/a0@r0;M=counts/*1/a0@r0"));
    auto q = sim.AllocateQubits(2);
    const auto z0 = sim.Observable(ObsId::PauliZ, {}, {q[0]});
    // expval sees the readout error, probs does not
    REQUIRE(sim.Expval(z0) == Approx(1 - 2 * p));
    REQUIRE(probsZ(sim, q[0]) == Approx(1.0));
    // the device state is untouched
    REQUIRE(sim.getState().probs({0})[0] == Approx(1.0));
    REQUIRE(sim.Var(z0) == Approx(0.0).margin(1e-12));

    const std::size_t shots = 20000;
    sim.SetDeviceShots(shots);
    std::mt19937 gen(42);
    sim.SetDevicePRNG(&gen);
    std::vector<double> eig(4);
    std::vector<int64_t> counts(4);
    DataView<double, 1> ev(eig);
    DataView<int64_t, 1> cv(counts);
    sim.PartialCounts(ev, cv, {q[0], q[1]});
    // the single-wire rule applies to every measured wire
    const std::vector<double> expected{(1 - p) * (1 - p), (1 - p) * p,
                                       p * (1 - p), p * p};
    for (std::size_t k = 0; k < 4; k++) {
        REQUIRE(static_cast<double>(counts[k]) / shots ==
                Approx(expected[k]).margin(0.02));
    }
}

TEST_CASE("LightningTensornetSimulator measurements", "[LTensornetCatalyst]") {
    LightningTensornetSimulator sim;
    auto q = sim.AllocateQubits(3);
    sim.NamedOperation("RY", {0.7}, {q[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {q[0], q[1]}, false, {}, {}, {});
    sim.NamedOperation("RX", {1.3}, {q[2]}, false, {}, {}, {});
    sim.NamedOperation("CRZ", {0.4}, {q[1], q[2]}, false, {}, {}, {});
    sim.NamedOperation("AmplitudeDamping", {0.25}, {q[1]}, false, {}, {}, {});

    const auto x0 = sim.Observable(ObsId::PauliX, {}, {q[0]});
    const auto y2 = sim.Observable(ObsId::PauliY, {}, {q[2]});
    const auto z1 = sim.Observable(ObsId::PauliZ, {}, {q[1]});
    const auto h2 = sim.Observable(ObsId::Hadamard, {}, {q[2]});
    const std::vector<CD> herm{CD{1, 0}, CD{0.5, -0.2}, CD{0.5, 0.2},
                               CD{-0.3, 0}};
    const auto m0 = sim.Observable(ObsId::Hermitian, herm, {q[0]});
    const auto t = sim.TensorObservable({x0, z1, y2});
    // Hermitian on (0, 1) that commutes with Z1, so Z1 * H01 is Hermitian
    std::vector<CD> herm01(16, CD{0, 0});
    for (std::size_t a = 0; a < 2; a++) {
        for (std::size_t b = 0; b < 2; b++) {
            for (std::size_t c = 0; c < 2; c++) {
                herm01[(2 * a + c) * 4 + 2 * b + c] =
                    herm[2 * a + b] * (c == 0 ? 1.0 : 2.0);
            }
        }
    }
    const auto m01 = sim.Observable(ObsId::Hermitian, herm01, {q[0], q[1]});
    const auto t_overlap = sim.TensorObservable({z1, m01});
    const auto ham = sim.HamiltonianObservable({0.5, -1.2, 0.7}, {t, h2, m0});

    std::vector<ObsIdType> all{x0, y2, z1, h2, m0, t, t_overlap, ham};
    std::vector<double> e_exact;
    std::vector<double> v_exact;
    for (auto k : all) {
        e_exact.push_back(sim.Expval(k));
        v_exact.push_back(sim.Var(k));
    }
    // cross-checks of the analytic values
    const auto &dm = sim.getState();
    REQUIRE(e_exact[2] == Approx(dm.probs({1})[0] - dm.probs({1})[1]));
    REQUIRE(v_exact[2] == Approx(1 - e_exact[2] * e_exact[2]));
    REQUIRE(e_exact[7] ==
            Approx(0.5 * e_exact[5] - 1.2 * e_exact[3] + 0.7 * e_exact[4]));

    SECTION("finite-shot estimates") {
        const std::size_t shots = 40000;
        sim.SetDeviceShots(shots);
        std::mt19937 gen(7);
        sim.SetDevicePRNG(&gen);
        for (std::size_t i = 0; i < all.size(); i++) {
            INFO("observable " << i);
            const double e = sim.Expval(all[i]);
            const double sd = std::sqrt(std::max(v_exact[i], 1e-3) / shots);
            REQUIRE(e == Approx(e_exact[i]).margin(6 * sd + 1e-3));
            if (i != 7) {
                REQUIRE(sim.Var(all[i]) == Approx(v_exact[i]).margin(0.03));
            }
        }
        // Hamiltonian variance with shots follows lightning.kokkos: sum of
        // c_i^2 var(O_i)
        const double v_terms =
            0.25 * v_exact[5] + 1.44 * v_exact[3] + 0.49 * v_exact[4];
        REQUIRE(sim.Var(ham) == Approx(v_terms).margin(0.05));

        // probs from shots
        std::vector<double> pr(4);
        DataView<double, 1> pv(pr);
        sim.PartialProbs(pv, {q[2], q[0]});
        const auto pe = dm.probs({2, 0});
        for (std::size_t k = 0; k < 4; k++) {
            REQUIRE(pr[k] == Approx(pe[k]).margin(0.02));
        }
        // samples: shots x wires, consistent with probabilities
        std::vector<double> buf(shots * 2);
        std::size_t sizes[2] = {shots, 2};
        std::size_t strides[2] = {2, 1};
        DataView<double, 2> sv(buf.data(), 0, sizes, strides);
        sim.PartialSample(sv, {q[2], q[0]});
        std::vector<double> hist(4, 0);
        for (std::size_t s = 0; s < shots; s++) {
            hist[static_cast<std::size_t>(buf[2 * s] * 2 + buf[2 * s + 1])] +=
                1.0 / shots;
        }
        for (std::size_t k = 0; k < 4; k++) {
            REQUIRE(hist[k] == Approx(pe[k]).margin(0.02));
        }
        std::vector<double> full(shots * 3);
        std::size_t sizes3[2] = {shots, 3};
        std::size_t strides3[2] = {3, 1};
        DataView<double, 2> fv(full.data(), 0, sizes3, strides3);
        sim.Sample(fv);
        std::vector<double> eig(8);
        std::vector<int64_t> cnt(8);
        DataView<double, 1> ev(eig);
        DataView<int64_t, 1> cv(cnt);
        sim.Counts(ev, cv);
        int64_t total = 0;
        for (auto c : cnt) {
            total += c;
        }
        REQUIRE(total == static_cast<int64_t>(shots));
        REQUIRE(eig[5] == 5.0);
    }

    SECTION("analytic probs") {
        std::vector<double> pr(8);
        DataView<double, 1> pv(pr);
        sim.Probs(pv);
        const auto pe = dm.probs();
        for (std::size_t k = 0; k < 8; k++) {
            REQUIRE(pr[k] == Approx(pe[k]));
        }
    }
}

TEST_CASE("LightningTensornetSimulator mid-circuit and Pauli measurements",
          "[LTensornetCatalyst]") {
    SECTION("postselection") {
        LightningTensornetSimulator sim;
        auto q = sim.AllocateQubits(2);
        sim.NamedOperation("Hadamard", {}, {q[0]}, false, {}, {}, {});
        sim.NamedOperation("CNOT", {}, {q[0], q[1]}, false, {}, {}, {});
        auto r = sim.Measure(q[0], 1);
        REQUIRE(*r);
        REQUIRE(probsZ(sim, q[1]) == Approx(-1.0));
        REQUIRE_THROWS(sim.Measure(q[0], 0));
    }
    SECTION("measurement statistics with a seeded generator") {
        std::mt19937 gen(3);
        std::size_t ones = 0;
        const std::size_t reps = 4000;
        LightningTensornetSimulator sim;
        sim.SetDevicePRNG(&gen);
        for (std::size_t i = 0; i < reps; i++) {
            auto q = sim.AllocateQubits(1);
            sim.NamedOperation("RY", {2.0}, {q[0]}, false, {}, {}, {});
            ones += *sim.Measure(q[0], std::nullopt) ? 1U : 0U;
            sim.ReleaseQubits(q);
        }
        const double p1 = std::sin(1.0) * std::sin(1.0);
        REQUIRE(static_cast<double>(ones) / reps == Approx(p1).margin(0.03));
    }
    SECTION("Pauli product measurement keeps the eigenstate intact") {
        LightningTensornetSimulator sim;
        auto q = sim.AllocateQubits(2);
        sim.NamedOperation("Hadamard", {}, {q[0]}, false, {}, {}, {});
        sim.NamedOperation("CNOT", {}, {q[0], q[1]}, false, {}, {}, {});
        MPS before(sim.getState());
        for (const char *w : {"XX", "ZZ"}) {
            INFO(w);
            auto r = sim.PauliMeasure(w, {q[0], q[1]});
            REQUIRE(!*r); // Bell state is a +1 eigenstate of XX and ZZ
            requireSameDM(sim.getState(), before, 1e-10);
        }
        auto r = sim.PauliMeasure("YY", {q[0], q[1]});
        REQUIRE(*r); // YY |Phi+> = -|Phi+>
        requireSameDM(sim.getState(), before, 1e-10);
    }
    SECTION("Pauli measurement projects onto an eigenspace") {
        std::mt19937 gen(11);
        LightningTensornetSimulator sim;
        sim.SetDevicePRNG(&gen);
        auto q = sim.AllocateQubits(2);
        sim.NamedOperation("RY", {0.9}, {q[0]}, false, {}, {}, {});
        sim.NamedOperation("RX", {0.4}, {q[1]}, false, {}, {}, {});
        auto r = sim.PauliMeasure("XZ", {q[0], q[1]});
        const auto xz =
            sim.TensorObservable({sim.Observable(ObsId::PauliX, {}, {q[0]}),
                                  sim.Observable(ObsId::PauliZ, {}, {q[1]})});
        REQUIRE(sim.Expval(xz) == Approx(*r ? -1.0 : 1.0));
    }
}

TEST_CASE("LightningTensornetSimulator dynamic qubits",
          "[LTensornetCatalyst]") {
    LightningTensornetSimulator sim;
    auto q = sim.AllocateQubits(2);
    sim.NamedOperation("Hadamard", {}, {q[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {q[0], q[1]}, false, {}, {}, {});

    SECTION("releasing an entangled qubit traces it out") {
        sim.ReleaseQubit(q[0]);
        REQUIRE(sim.GetNumQubits() == 1);
        REQUIRE(sim.getState().getNumQubits() == 1);
        REQUIRE(sim.getState().purity() == Approx(0.5));
        REQUIRE(probsZ(sim, q[1]) == Approx(0.0).margin(1e-12));
    }
    SECTION("allocating after a release resets the reused qubit") {
        sim.ReleaseQubit(q[1]);
        auto extra = sim.AllocateQubit();
        REQUIRE(sim.GetNumQubits() == 2);
        REQUIRE(probsZ(sim, extra) == Approx(1.0));
        REQUIRE(probsZ(sim, q[0]) == Approx(0.0).margin(1e-12));
        REQUIRE(sim.getState().getNumQubits() == 2);
    }
    SECTION("allocating a fresh qubit appends |0>") {
        auto extra = sim.AllocateQubit();
        REQUIRE(sim.getState().getNumQubits() == 3);
        sim.NamedOperation("CNOT", {}, {q[1], extra}, false, {}, {}, {});
        const auto zz =
            sim.TensorObservable({sim.Observable(ObsId::PauliZ, {}, {q[0]}),
                                  sim.Observable(ObsId::PauliZ, {}, {extra})});
        REQUIRE(sim.Expval(zz) == Approx(1.0));
        sim.ReleaseQubit(q[0]);
        sim.ReleaseQubit(q[1]);
        REQUIRE(sim.getState().getNumQubits() == 1);
        REQUIRE(probsZ(sim, extra) == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("LightningTensornetSimulator state preparation",
          "[LTensornetCatalyst]") {
    LightningTensornetSimulator sim;
    auto q = sim.AllocateQubits(3);
    std::vector<int8_t> bits{1, 0};
    std::vector<QubitIdType> w{q[2], q[0]};
    DataView<int8_t, 1> bv(bits);
    sim.SetBasisState(bv, w);
    REQUIRE(probsZ(sim, q[2]) == Approx(-1.0));
    REQUIRE(probsZ(sim, q[0]) == Approx(1.0));

    std::vector<CD> psi{CD{0.6, 0}, CD{0, 0.8}};
    std::vector<QubitIdType> w1{q[1]};
    DataView<CD, 1> pv(psi);
    sim.SetState(pv, w1);
    REQUIRE(probsZ(sim, q[1]) == Approx(0.36 - 0.64));
    const auto y1 = sim.Observable(ObsId::PauliY, {}, {q[1]});
    REQUIRE(sim.Expval(y1) == Approx(2 * 0.6 * 0.8));
}

TEST_CASE("LightningTensornetSimulator MPS options", "[LTensornetCatalyst]") {
    SECTION("defaults") {
        LightningTensornetSimulator sim;
        REQUIRE(sim.getMaxBondDim() == 128);
        REQUIRE(sim.getCutoff() == 0.0);
        REQUIRE(sim.getCutoffMode() == "abs");
        sim.AllocateQubits(2);
        REQUIRE_FALSE(sim.getState().isMixed());
    }
    SECTION("kwargs are forwarded to the MPS") {
        LightningTensornetSimulator sim(
            "{'max_bond_dim': 2, 'cutoff': 1e-3, 'cutoff_mode': 'rel'}");
        REQUIRE(sim.getMaxBondDim() == 2);
        REQUIRE(sim.getCutoff() == Approx(1e-3));
        REQUIRE(sim.getCutoffMode() == "rel");
        auto q = sim.AllocateQubits(6);
        for (int layer = 0; layer < 4; layer++) {
            for (std::size_t i = 0; i < 6; i++) {
                sim.NamedOperation("RY", {0.3 + 0.1 * static_cast<double>(i)},
                                   {q[i]}, false, {}, {}, {});
            }
            for (std::size_t i = layer % 2; i + 1 < 6; i += 2) {
                sim.NamedOperation("CNOT", {}, {q[i], q[i + 1]}, false, {}, {},
                                   {});
            }
        }
        const auto bd = sim.getState().bondDims();
        REQUIRE(*std::max_element(bd.begin(), bd.end()) <= 2);
        REQUIRE(sim.getState().getMaxBondDim() == 2);
        REQUIRE(sim.getState().truncationError() > 0);
    }
    SECTION("invalid options") {
        REQUIRE_THROWS_WITH(LightningTensornetSimulator("{'max_bond_dim': 0}"),
                            ContainsSubstring("max_bond_dim"));
        REQUIRE_THROWS_WITH(
            LightningTensornetSimulator("{'cutoff_mode': 'foo'}"),
            ContainsSubstring("cutoff_mode"));
    }
    SECTION("a noise model selects the density-operator representation") {
        LightningTensornetSimulator sim(
            kwargsFor("LNM1;" + bitflipChannel(0.1) + ";G=RX/*1/a0@r0"));
        auto q = sim.AllocateQubits(2);
        REQUIRE(sim.getState().isMixed());
        sim.NamedOperation("RX", {0.0}, {q[0]}, false, {}, {}, {});
        REQUIRE(probsZ(sim, q[0]) == Approx(0.8));
    }
}

TEST_CASE("LightningTensornetSimulator matches the density matrix on a noisy "
          "circuit",
          "[LTensornetCatalyst]") {
    const double p = 0.05;
    const std::string spec = "LNM1;" + bitflipChannel(p) +
                             ";G=CNOT/*2/a0@r0+a0@r1;G=Hadamard/*1/b0@r0";
    LightningTensornetSimulator sim(kwargsFor(spec));
    auto q = sim.AllocateQubits(5);
    const Pennylane::LightningNoisy::NoiseModel<double> nm(spec);
    DM ref(5);
    auto apply = [&](const std::string &name, std::vector<std::size_t> w,
                     std::vector<double> params) {
        std::vector<QubitIdType> ids;
        for (auto x : w) {
            ids.push_back(q[x]);
        }
        sim.NamedOperation(name, params, ids, false, {}, {}, {});
        std::vector<Pennylane::LightningNoisy::ResolvedNoise> before;
        std::vector<Pennylane::LightningNoisy::ResolvedNoise> after;
        nm.lookupGate(name, w, before, after);
        ref.applyNoise(nm, before);
        ref.applyOperation(name, w, false, params);
        ref.applyNoise(nm, after);
    };
    for (std::size_t i = 0; i < 5; i++) {
        apply("Hadamard", {i}, {});
    }
    apply("CNOT", {0, 3}, {});
    apply("CNOT", {4, 1}, {});
    apply("RZ", {2}, {0.4});
    apply("CNOT", {2, 0}, {});
    apply("IsingXY", {1, 3}, {0.8});
    requireSameDM(sim.getState(), ref, 1e-9);
}
