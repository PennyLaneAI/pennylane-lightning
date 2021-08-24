#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointDiff.hpp"
#include "StateVector.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::Algorithms;

/**
 * @brief Tests the constructability of the StateVector class.
 *
 */
TEMPLATE_TEST_CASE("AdjointJacobian::AdjointJacobian", "[AdjointJacobian]",
                   float, double) {
    SECTION("AdjointJacobian") {
        REQUIRE(std::is_constructible<AdjointJacobian<>>::value);
    }
    SECTION("AdjointJacobian<TestType> {}") {
        REQUIRE(std::is_constructible<AdjointJacobian<TestType>>::value);
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian", "[AdjointJacobian]") {

    AdjointJacobian<double> adj;
    std::vector<double> param{1, -2, 1.623, -0.051, 0};

    SECTION("RX gradient") {
        const size_t num_qubits = 1;
        const size_t num_params = 1;
        const size_t num_obs = 1;
        for (const auto &p : param) {
            std::vector<double> jacobian(num_obs * num_params, 0.0);

            std::vector<std::complex<double>> cdata(0b1 << num_qubits);
            StateVector<double> psi(cdata.data(), cdata.size());
            cdata[0] = std::complex<double>{1, 0};

            adj.adjointJacobian(psi, jacobian, {"PauliZ"}, {{}}, {{0}}, {"RX"},
                                {{p}}, {{0}}, {0}, 1);
            CAPTURE(jacobian);
            CHECK(-sin(p) == Approx(jacobian.front()));
        }
    }

    SECTION("RY gradient") {
        const size_t num_qubits = 1;
        const size_t num_params = 1;
        const size_t num_obs = 1;
        for (const auto &p : param) {
            std::vector<double> jacobian(num_obs * num_params, 0.0);

            std::vector<std::complex<double>> cdata(0b1 << num_qubits);
            StateVector<double> psi(cdata.data(), cdata.size());
            cdata[0] = std::complex<double>{1, 0};

            adj.adjointJacobian(psi, jacobian, {"PauliX"}, {{}}, {{0}}, {"RY"},
                                {{p}}, {{0}}, {0}, 1);
            CAPTURE(jacobian);
            CHECK(cos(p) == Approx(jacobian.front()).epsilon(0.01));
        }
    }
    SECTION("Single RX gradient, multiple wires") {
        const size_t num_qubits = 3;
        const size_t num_params = 1;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        std::vector<std::complex<double>> cdata(0b1 << num_qubits);
        StateVector<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        adj.adjointJacobian(psi, jacobian, {"PauliZ", "PauliZ", "PauliZ"}, {{}},
                            {{0}, {1}, {2}}, {"RX"}, {{param[0]}}, {{0}}, {0},
                            num_params);
        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]));
    }
    SECTION("Multiple RX gradient") {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        std::vector<std::complex<double>> cdata(0b1 << num_qubits);
        StateVector<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        adj.adjointJacobian(psi, jacobian, {"PauliZ", "PauliZ", "PauliZ"}, {{}},
                            {{0}, {1}, {2}}, {"RX", "RX", "RX"},
                            {{param[0]}, {param[1]}, {param[2]}},
                            {{0}, {1}, {2}}, {0, 1, 2}, num_params);
        CAPTURE(jacobian);
        for (size_t i = 0; i < num_params; i++) {
            CHECK(-cos(param[i]) == Approx(jacobian[i * num_params + i]));
        }
    }
}
