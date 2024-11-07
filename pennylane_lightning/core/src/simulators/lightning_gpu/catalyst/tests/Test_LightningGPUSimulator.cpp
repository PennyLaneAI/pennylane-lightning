// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch.hpp>

#include "LightningGPUSimulator.hpp"
#include "QuantumDevice.hpp"
#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Catalyst::Runtime::Simulator;
using namespace Pennylane::Util;
using LGPUSimulator = LightningGPUSimulator;
using QDevice = Catalyst::Runtime::QuantumDevice;

GENERATE_DEVICE_FACTORY(LightningGPUSimulator,
                        Catalyst::Runtime::Simulator::LightningGPUSimulator);
} // namespace
/// @endcond

/**
 * @brief Tests the LightningGPUSimulator class.
 *
 */
TEST_CASE("LightningGPUSimulator::constructor", "[constructibility]") {
    SECTION("LightningGPUSimulator") {
        REQUIRE(std::is_constructible<LGPUSimulator>::value);
    }
    SECTION("LightningGPUSimulator(string))") {
        REQUIRE(std::is_constructible<LGPUSimulator, std::string>::value);
    }
}

TEST_CASE("Test the device factory method", "[constructibility]") {
    std::unique_ptr<QDevice> LGPUsim(LightningGPUSimulatorFactory(""));
    REQUIRE(LGPUsim->GetNumQubits() == 0);
}

TEST_CASE("LightningGPUSimulator::unit_tests", "[unit tests]") {
    SECTION("Managing Qubits") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(0);
        REQUIRE(LGPUsim->GetNumQubits() == 0);
        LGPUsim->AllocateQubits(2);
        REQUIRE(LGPUsim->GetNumQubits() == 2);
        LGPUsim->AllocateQubits(2);
        REQUIRE(LGPUsim->GetNumQubits() == 4);
        LGPUsim->ReleaseQubit(0);
        REQUIRE(
            LGPUsim->GetNumQubits() ==
            4); // releasing only one qubit does not change the total number.
        LGPUsim->ReleaseAllQubits();
        REQUIRE(LGPUsim->GetNumQubits() ==
                0); // releasing all qubits resets the simulator.
    }
    SECTION("Tape recording") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(1);
        REQUIRE_NOTHROW(LGPUsim->StartTapeRecording());
        REQUIRE_THROWS_WITH(
            LGPUsim->StartTapeRecording(),
            Catch::Matchers::Contains("Cannot re-activate the cache manager"));
        REQUIRE_NOTHROW(LGPUsim->StopTapeRecording());
        REQUIRE_THROWS_WITH(
            LGPUsim->StopTapeRecording(),
            Catch::Matchers::Contains(
                "Cannot stop an already stopped cache manager"));
    }
}

TEST_CASE("LightningGPUSimulator::GateSet", "[GateSet]") {
    SECTION("Identity gate") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 10;
        std::vector<intptr_t> Qs;
        Qs.reserve(n_qubits);
        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            Qs[ind] = LGPUsim->AllocateQubit();
        }

        for (std::size_t ind = 0; ind < n_qubits; ind += 2) {
            LGPUsim->NamedOperation("Identity", {}, {Qs[ind]}, false);
        }

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state.at(0) == std::complex<double>{1, 0});

        std::complex<double> sum{0, 0};
        for (std::size_t ind = 1; ind < state.size(); ind++) {
            sum += state[ind];
        }

        CHECK(sum == std::complex<double>{0, 0});
    }

    SECTION("PauliX gate") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 3;
        std::vector<intptr_t> Qs;
        Qs.reserve(n_qubits);
        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            Qs[ind] = LGPUsim->AllocateQubit();
        }

        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            LGPUsim->NamedOperation("PauliX", {}, {Qs[ind]}, false);
        }
        for (std::size_t ind = n_qubits; ind > 0; ind--) {
            LGPUsim->NamedOperation("PauliX", {}, {Qs[ind - 1]}, false);
        }

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state.at(0) == std::complex<double>{1, 0});

        std::complex<double> sum{0, 0};
        for (std::size_t ind = 1; ind < state.size(); ind++) {
            sum += state[ind];
        }

        CHECK(sum == std::complex<double>{0, 0});
    }

    SECTION("PauliY gate") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs;
        Qs.reserve(n_qubits);
        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            Qs[ind] = LGPUsim->AllocateQubit();
        }

        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            LGPUsim->NamedOperation("PauliY", {}, {Qs[ind]}, false);
        }

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state.at(0) == std::complex<double>{0, 0});
        CHECK(state.at(1) == std::complex<double>{0, 0});
        CHECK(state.at(2) == std::complex<double>{0, 0});
        CHECK(state.at(3) == std::complex<double>{-1, 0});
    }

    SECTION("PauliY and PauliZ gates") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs;
        Qs.reserve(n_qubits);
        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            Qs[ind] = LGPUsim->AllocateQubit();
        }

        LGPUsim->NamedOperation("PauliY", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("PauliZ", {}, {Qs[1]}, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state.at(0) == std::complex<double>{0, 0});
        CHECK(state.at(1) == std::complex<double>{0, 0});
        CHECK(state.at(2) == std::complex<double>{0, 1});
        CHECK(state.at(3) == std::complex<double>{0, 0});
    }

    SECTION("Hadamard gate") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs;
        Qs.reserve(n_qubits);
        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            Qs[ind] = LGPUsim->AllocateQubit();
        }

        for (std::size_t ind = 0; ind < n_qubits; ind++) {
            LGPUsim->NamedOperation("Hadamard", {}, {Qs[ind]}, false);
        }

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state[0] ==
              PLApproxComplex(std::complex<double>{0.5, 0}).epsilon(1e-5));
        CHECK(state.at(1) == state.at(0));
        CHECK(state.at(2) == state.at(0));
        CHECK(state.at(3) == state.at(0));
    }

    SECTION("R(X, Y, Z) and PauliX gates") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 4;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("PauliX", {}, {Qs[0]}, false);

        LGPUsim->NamedOperation("RX", {0.123}, {Qs[1]}, false);
        LGPUsim->NamedOperation("RY", {0.456}, {Qs[2]}, false);
        LGPUsim->NamedOperation("RZ", {0.789}, {Qs[3]}, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        // calculated by pennylane.
        CHECK(state.at(0) == std::complex<double>{0, 0});
        CHECK(state.at(1) == std::complex<double>{0, 0});
        CHECK(state.at(2) == std::complex<double>{0, 0});
        CHECK(state.at(3) == std::complex<double>{0, 0});
        CHECK(state.at(4) == std::complex<double>{0, 0});
        CHECK(state.at(5) == std::complex<double>{0, 0});
        CHECK(state.at(6) == std::complex<double>{0, 0});
        CHECK(state.at(7) == std::complex<double>{0, 0});
        CHECK(state[8] ==
              PLApproxComplex(
                  std::complex<double>{0.8975969498074641, -0.3736920921192206})
                  .epsilon(1e-5));
        CHECK(state.at(9) == std::complex<double>{0, 0});
        CHECK(state[10] ==
              PLApproxComplex(std::complex<double>{0.20827363966052723,
                                                   -0.08670953277495183})
                  .epsilon(1e-5));
        CHECK(state.at(11) == std::complex<double>{0, 0});
        CHECK(state[12] ==
              PLApproxComplex(std::complex<double>{-0.023011082205037697,
                                                   -0.055271914055973925})
                  .epsilon(1e-5));
        CHECK(state.at(13) == std::complex<double>{0, 0});
        CHECK(state[14] ==
              PLApproxComplex(std::complex<double>{-0.005339369573836912,
                                                   -0.012825002038956146})
                  .epsilon(1e-5));
        CHECK(state.at(15) == std::complex<double>{0, 0});
    }

    SECTION("Hadamard, RX, PhaseShift with cache manager") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs;
        Qs.reserve(n_qubits);

        Qs[0] = LGPUsim->AllocateQubit();
        Qs[1] = LGPUsim->AllocateQubit();

        LGPUsim->StartTapeRecording();
        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("RX", {0.123}, {Qs[1]}, false);
        LGPUsim->NamedOperation("PhaseShift", {0.456}, {Qs[0]}, false);
        LGPUsim->StopTapeRecording();

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        // calculated by pennylane.
        CHECK(state[0] == PLApproxComplex(std::complex<double>{0.7057699753, 0})
                              .epsilon(1e-5));
        CHECK(state[1] == PLApproxComplex(std::complex<double>{0, -0.04345966})
                              .epsilon(1e-5));
        CHECK(state[2] ==
              PLApproxComplex(std::complex<double>{0.63365519, 0.31079312})
                  .epsilon(1e-5));
        CHECK(state[3] ==
              PLApproxComplex(std::complex<double>{0.01913791, -0.039019})
                  .epsilon(1e-5));

        std::tuple<std::size_t, std::size_t, std::size_t,
                   std::vector<std::string>, std::vector<intptr_t>>
            expected{3, 0, 2, {"Hadamard", "RX", "PhaseShift"}, {}};
        REQUIRE(LGPUsim->CacheManagerInfo() == expected);
    }

    // ============= 2-qubit operations =============

    SECTION("PauliX and CNOT") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs;
        Qs.reserve(n_qubits);

        for (std::size_t i = 0; i < n_qubits; i++) {
            Qs[i] = LGPUsim->AllocateQubit();
        }

        LGPUsim->NamedOperation("PauliX", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state.at(0) == std::complex<double>{0, 0});
        CHECK(state.at(1) == std::complex<double>{0, 0});
        CHECK(state.at(2) == std::complex<double>{0, 0});
        CHECK(state.at(3) == std::complex<double>{1, 0});
    }

    SECTION("Hadamard and CR(X, Y, Z)") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 4;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("CRX", {0.123}, {Qs[0], Qs[1]}, false);
        LGPUsim->NamedOperation("CRY", {0.456}, {Qs[0], Qs[2]}, false);
        LGPUsim->NamedOperation("CRZ", {0.789}, {Qs[0], Qs[3]}, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        // calculated by pennylane.
        CHECK(
            state[0] ==
            PLApproxComplex(std::complex<double>{M_SQRT1_2, 0}).epsilon(1e-5));
        CHECK(state.at(1) == std::complex<double>{0, 0});
        CHECK(state.at(2) == std::complex<double>{0, 0});
        CHECK(state.at(3) == std::complex<double>{0, 0});
        CHECK(state.at(4) == std::complex<double>{0, 0});
        CHECK(state.at(5) == std::complex<double>{0, 0});
        CHECK(state.at(6) == std::complex<double>{0, 0});
        CHECK(state.at(7) == std::complex<double>{0, 0});
        CHECK(state[8] ==
              PLApproxComplex(
                  std::complex<double>{0.6346968899812189, -0.2642402124132889})
                  .epsilon(1e-5));
        CHECK(state.at(9) == std::complex<double>{0, 0});
        CHECK(state[10] ==
              PLApproxComplex(std::complex<double>{0.14727170294636227,
                                                   -0.061312898618685635})
                  .epsilon(1e-5));
        CHECK(state.at(11) == std::complex<double>{0, 0});
        CHECK(state[12] ==
              PLApproxComplex(std::complex<double>{-0.016271292269623247,
                                                   -0.03908314523813921})
                  .epsilon(1e-5));
        CHECK(state.at(13) == std::complex<double>{0, 0});
        CHECK(state[14] ==
              PLApproxComplex(std::complex<double>{-0.0037755044329212074,
                                                   -0.009068645910477189})
                  .epsilon(1e-5));
        CHECK(state.at(15) == std::complex<double>{0, 0});
    }

    SECTION("Hadamard and CRot") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("CRot", {M_PI, M_PI_2, 0.5}, {Qs[0], Qs[1]},
                                false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(
            state[0] ==
            PLApproxComplex(std::complex<double>{M_SQRT1_2, 0}).epsilon(1e-5));

        CHECK(state[1] ==
              PLApproxComplex(std::complex<double>{0, 0}).epsilon(1e-5));

        CHECK(state[2] == PLApproxComplex(std::complex<double>{-0.1237019796,
                                                               -0.4844562109})
                              .epsilon(1e-5));
        CHECK(state[3] ==
              PLApproxComplex(std::complex<double>{0.1237019796, -0.4844562109})
                  .epsilon(1e-5));
    }

    SECTION("Hadamard, PauliZ, IsingXY, SWAP") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("PauliZ", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("IsingXY", {0.2}, {Qs[1], Qs[0]}, false);
        LGPUsim->NamedOperation("SWAP", {}, {Qs[0], Qs[1]}, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(
            state[0] ==
            PLApproxComplex(std::complex<double>{M_SQRT1_2, 0}).epsilon(1e-5));
        CHECK(state[1] == PLApproxComplex(std::complex<double>{-0.70357419, 0})
                              .epsilon(1e-5));
        CHECK(state[2] == PLApproxComplex(std::complex<double>{0, -0.07059289})
                              .epsilon(1e-5));
        CHECK(state[3] ==
              PLApproxComplex(std::complex<double>{0, 0}).epsilon(1e-5));
    }

    SECTION("Hadamard, PauliX and Toffoli") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 3;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("PauliX", {}, {Qs[1]}, false);
        LGPUsim->NamedOperation("Toffoli", {}, {Qs[0], Qs[1], Qs[2]}, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state.at(0) == std::complex<double>{0, 0});
        CHECK(state.at(1) == std::complex<double>{0, 0});
        CHECK(
            state[2] ==
            PLApproxComplex(std::complex<double>{M_SQRT1_2, 0}).epsilon(1e-5));
        CHECK(state.at(3) == std::complex<double>{0, 0});
        CHECK(state.at(4) == std::complex<double>{0, 0});
        CHECK(state.at(5) == std::complex<double>{0, 0});
        CHECK(state.at(6) == std::complex<double>{0, 0});
        CHECK(
            state[7] ==
            PLApproxComplex(std::complex<double>{M_SQRT1_2, 0}).epsilon(1e-5));
    }

    SECTION("RX, Hadamard and MultiRZ") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("RX", {M_PI}, {Qs[1]}, false);
        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
        LGPUsim->NamedOperation("MultiRZ", {M_PI}, {Qs[0], Qs[1]}, false);
        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("Hadamard", {}, {Qs[1]}, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state[2] ==
              PLApproxComplex(std::complex<double>{-1, 0}).margin(1e-5));
    }

    SECTION("Hadamard, CNOT and Matrix") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

        const std::vector<intptr_t> wires = {Qs[0]};
        std::vector<std::complex<double>> matrix{
            {-0.6709485262524046, -0.6304426335363695},
            {-0.14885403153998722, 0.3608498832392019},
            {-0.2376311670004963, 0.3096798175687841},
            {-0.8818365947322423, -0.26456390390903695},
        };
        LGPUsim->MatrixOperation(matrix, wires, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state[0] ==
              PLApproxComplex(std::complex<double>{-0.474432, -0.44579})
                  .epsilon(1e-5));
        CHECK(state[1] ==
              PLApproxComplex(std::complex<double>{-0.105256, 0.255159})
                  .epsilon(1e-5));
        CHECK(state[2] ==
              PLApproxComplex(std::complex<double>{-0.168031, 0.218977})
                  .epsilon(1e-5));
        CHECK(state[3] ==
              PLApproxComplex(std::complex<double>{-0.623553, -0.187075})
                  .epsilon(1e-5));
    }

    SECTION("Hadamard, CR(X, Y, Z) and Matrix") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 4;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("CRX", {0.123}, {Qs[0], Qs[1]}, false);
        LGPUsim->NamedOperation("CRY", {0.456}, {Qs[0], Qs[2]}, false);
        LGPUsim->NamedOperation("CRZ", {0.789}, {Qs[0], Qs[3]}, false);

        const std::vector<intptr_t> wires = {Qs[0], Qs[1], Qs[2]};
        std::vector<std::complex<double>> matrix{
            {-0.14601911598243822, -0.18655250647340088},
            {-0.03917826201290317, -0.031161687050443518},
            {0.11497626236175404, 0.38310733543366354},
            {-0.0929691815340695, 0.1219804125497268},
            {0.07306514883467692, 0.017445444816725875},
            {-0.27330866098918355, -0.6007032759764033},
            {0.4530754397715841, -0.08267189625512258},
            {0.32125201986075, -0.036845158875036116},
            {0.032317572838307884, 0.02292755555300329},
            {-0.18775945295623664, -0.060215004737844156},
            {-0.3093351335745536, -0.2061961962889725},
            {0.4216087567144761, 0.010534488410902099},
            {0.2769943541718527, -0.26016137877135465},
            {0.18727884147867532, 0.02830415812286322},
            {0.3367562196770689, -0.5250999173939218},
            {0.05770014289220745, 0.26595514845958573},
            {0.37885720163317027, 0.3110931426403546},
            {0.13436510737129648, -0.4083415934958021},
            {-0.5443665467635203, 0.2458343977310266},
            {-0.050346912365833024, 0.08709833123617361},
            {0.11505259829552131, 0.010155858056939438},
            {-0.2930849061531229, 0.019339259194141145},
            {0.011825409829453282, 0.011597907736881019},
            {-0.10565527258356637, -0.3113689446440079},
            {0.0273191284561944, -0.2479498526173881},
            {-0.5528072425836249, -0.06114469689935285},
            {-0.20560364740746587, -0.3800208994544297},
            {-0.008236143958221483, 0.3017421511504845},
            {0.04817188123334976, 0.08550951191632741},
            {-0.24081054643565586, -0.3412671345149831},
            {-0.38913538197001885, 0.09288402897806938},
            {-0.07937578245883717, 0.013979426755633685},
            {0.22246583652015395, -0.18276674810033927},
            {0.22376666162382491, 0.2995723155125488},
            {-0.1727191441070097, -0.03880522034607489},
            {0.075780203819001, 0.2818783673816625},
            {-0.6161322400651016, 0.26067347179217193},
            {-0.021161519614267765, -0.08430919051054794},
            {0.1676500381348944, -0.30645601624407504},
            {-0.28858251997285883, 0.018089595494883842},
            {-0.19590767481842053, -0.12844366632033652},
            {0.18707834504831794, -0.1363932722670649},
            {-0.07224221779769334, -0.11267803536286894},
            {-0.23897684826459387, -0.39609971967853685},
            {-0.0032110880452929555, -0.29294331305690136},
            {-0.3188741682462722, -0.17338979346647143},
            {0.08194395032821632, -0.002944814673179825},
            {-0.5695791830944521, 0.33299548924055095},
            {-0.4983660307441444, -0.4222358493977972},
            {0.05533914327048402, -0.42575842134560576},
            {-0.2187623521182678, -0.03087596187054778},
            {0.11278255885846857, 0.07075886163492914},
            {-0.3054684775292515, -0.1739796870866232},
            {0.14151567663565712, 0.20399935744127418},
            {0.06720165377364941, 0.07543463072363207},
            {0.08019665306716581, -0.3473013434358584},
            {-0.2600167605995786, -0.08795704036197827},
            {0.125680477777759, 0.266342700305046},
            {-0.1586772594600269, 0.187360909108502},
            {-0.4653314704208982, 0.4048609954619629},
            {0.39992560380733094, -0.10029244177901954},
            {0.2533527906886461, 0.05222114898540775},
            {-0.15840033949128557, -0.2727320427534386},
            {-0.21590866323269536, -0.1191163626522938},
        };
        LGPUsim->MatrixOperation(matrix, wires, false);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        CHECK(state[0] ==
              PLApproxComplex(std::complex<double>{-0.141499, -0.230993})
                  .epsilon(1e-5));
        CHECK(state[2] ==
              PLApproxComplex(std::complex<double>{0.135423, -0.235563})
                  .epsilon(1e-5));
        CHECK(state[4] ==
              PLApproxComplex(std::complex<double>{0.299458, 0.218321})
                  .epsilon(1e-5));
        CHECK(state[6] ==
              PLApproxComplex(std::complex<double>{0.0264869, -0.154913})
                  .epsilon(1e-5));
        CHECK(state[8] ==
              PLApproxComplex(std::complex<double>{-0.186607, 0.188884})
                  .epsilon(1e-5));
        CHECK(state[10] ==
              PLApproxComplex(std::complex<double>{-0.271843, -0.281136})
                  .epsilon(1e-5));
        CHECK(state[12] ==
              PLApproxComplex(std::complex<double>{-0.560499, -0.310176})
                  .epsilon(1e-5));
        CHECK(state[14] ==
              PLApproxComplex(std::complex<double>{0.0756372, -0.226334})
                  .epsilon(1e-5));
    }

    SECTION("Hadamard and IsingZZ and cache manager") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();

        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        LGPUsim->StartTapeRecording();
        LGPUsim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
        LGPUsim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
        LGPUsim->NamedOperation("IsingZZ", {M_PI_4}, {Qs[0], Qs[1]}, false);
        LGPUsim->StopTapeRecording();

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        std::complex<double> c1{0.4619397663, -0.1913417162};
        std::complex<double> c2{0.4619397663, 0.1913417162};

        CHECK(state[0] == PLApproxComplex(c1).epsilon(1e-5));
        CHECK(state[1] == PLApproxComplex(c2).epsilon(1e-5));
        CHECK(state[2] == PLApproxComplex(c2).epsilon(1e-5));
        CHECK(state[3] == PLApproxComplex(c1).epsilon(1e-5));

        std::tuple<std::size_t, std::size_t, std::size_t,
                   std::vector<std::string>, std::vector<intptr_t>>
            expected{3, 0, 1, {"Hadamard", "Hadamard", "IsingZZ"}, {}};
        REQUIRE(LGPUsim->CacheManagerInfo() == expected);
    }

    SECTION("Test setStateVector") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();
        constexpr std::size_t n_qubits = 2;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        std::vector<std::complex<double>> data = {{0.5, 0.5}, {0.0, 0.0}};
        DataView<std::complex<double>, 1> data_view(data);
        std::vector<QubitIdType> wires = {1};
        LGPUsim->SetState(data_view, wires);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        std::complex<double> c1{0.5, 0.5};
        std::complex<double> c2{0.0, 0.0};
        CHECK(state[0] == PLApproxComplex(c1).epsilon(1e-5));
        CHECK(state[1] == PLApproxComplex(c2).epsilon(1e-5));
        CHECK(state[2] == PLApproxComplex(c2).epsilon(1e-5));
        CHECK(state[3] == PLApproxComplex(c2).epsilon(1e-5));
    }

    SECTION("Test setBasisState") {
        std::unique_ptr<LGPUSimulator> LGPUsim =
            std::make_unique<LGPUSimulator>();
        constexpr std::size_t n_qubits = 1;
        std::vector<intptr_t> Qs = LGPUsim->AllocateQubits(n_qubits);

        std::vector<int8_t> data = {0};
        DataView<int8_t, 1> data_view(data);
        std::vector<QubitIdType> wires = {0};
        LGPUsim->SetBasisState(data_view, wires);

        std::vector<std::complex<double>> state(1U << LGPUsim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        LGPUsim->State(view);

        std::complex<double> c1{1.0, 0.0};
        std::complex<double> c2{0.0, 0.0};
        CHECK(state[0] == PLApproxComplex(c1).epsilon(1e-5));
        CHECK(state[1] == PLApproxComplex(c2).epsilon(1e-5));
    }
}
