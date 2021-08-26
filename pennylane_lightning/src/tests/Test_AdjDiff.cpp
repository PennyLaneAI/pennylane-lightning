#define _USE_MATH_DEFINES

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
    // std::vector<double> param{1, -2, 1.623, -0.051, 0};
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};

    SECTION("RX gradient") {
        const size_t num_qubits = 1;
        const size_t num_params = 1;
        const size_t num_obs = 1;

        auto obs = adj.createObs({"PauliZ"}, {{}}, {{0}});

        for (const auto &p : param) {
            auto ops = adj.createOpsData({"RX"}, {{p}}, {{0}}, {false});

            std::vector<double> jacobian(num_obs * num_params, 0.0);

            std::vector<std::complex<double>> cdata(0b1 << num_qubits);
            StateVector<double> psi(cdata.data(), cdata.size());
            cdata[0] = std::complex<double>{1, 0};

            adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                                ops, {0}, 1);
            CAPTURE(jacobian);
            CHECK(-sin(p) == Approx(jacobian.front()));
        }
    }

    SECTION("RY gradient") {
        const size_t num_qubits = 1;
        const size_t num_params = 1;
        const size_t num_obs = 1;

        auto obs = adj.createObs({"PauliX"}, {{}}, {{0}});

        for (const auto &p : param) {
            auto ops = adj.createOpsData({"RY"}, {{p}}, {{0}}, {false});

            std::vector<double> jacobian(num_obs * num_params, 0.0);

            std::vector<std::complex<double>> cdata(0b1 << num_qubits);
            StateVector<double> psi(cdata.data(), cdata.size());
            cdata[0] = std::complex<double>{1, 0};

            adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                                ops, {0}, 1);

            CAPTURE(jacobian);
            CHECK(cos(p) == Approx(jacobian.front()).margin(1e-7));
        }
    }
    SECTION("Single RX gradient, 2 expval") {
        const size_t num_qubits = 2;
        const size_t num_params = 1;
        const size_t num_obs = 2;
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        std::vector<std::complex<double>> cdata(0b1 << num_qubits);
        StateVector<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        auto obs1 = adj.createObs({"PauliZ"}, {{}}, {{0}});
        auto obs2 = adj.createObs({"PauliZ"}, {{}}, {{1}});

        auto ops = adj.createOpsData({"RX"}, {{param[0]}}, {{0}}, {false});

        adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian,
                            {obs1, obs2}, ops, {0}, num_params);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
    }
    SECTION("Multiple RX gradient, single expval per wire") {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        std::vector<std::complex<double>> cdata(0b1 << num_qubits);
        StateVector<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        auto obs1 = adj.createObs({"PauliZ"}, {{}}, {{0}});
        auto obs2 = adj.createObs({"PauliZ"}, {{}}, {{1}});
        auto obs3 = adj.createObs({"PauliZ"}, {{}}, {{2}});

        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian,
                            {obs1, obs2, obs3}, ops, {0, 1, 2}, num_params);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
    }
    SECTION("Multiple RX gradient, tensor expval") {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        std::vector<std::complex<double>> cdata(0b1 << num_qubits);
        StateVector<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        auto obs = adj.createObs({"PauliZ", "PauliZ", "PauliZ"}, {{}, {}, {}},
                                 {{0}, {1}, {2}});
        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                            ops, {0, 1, 2}, num_params);
        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-0.1755096592645253 == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.26478810666384334 == Approx(jacobian[1]).margin(1e-7));
        CHECK(-0.6312451595102775 == Approx(jacobian[2]).margin(1e-7));
    }

    SECTION("Mixed gradient, tensor expval") {
        const size_t num_qubits = 3;
        const size_t num_params = 6;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        std::vector<std::complex<double>> cdata(0b1 << num_qubits);
        StateVector<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        auto obs = adj.createObs({"PauliZ", "PauliZ", "PauliZ"}, {{}, {}, {}},
                                 {{0}, {1}, {2}});
        auto ops =
            adj.createOpsData({"RX", "RX", "RX", "RY", "RY", "RY"},
                              {{param[0]},
                               {param[1]},
                               {param[2]},
                               {param[0]},
                               {param[1]},
                               {param[2]}},
                              {{0}, {1}, {2}, {0}, {1}, {2}},
                              {false, false, false, false, false, false});

        adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                            ops, {0, 1, 2, 3, 4, 5}, num_params);
        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(0.06396442 == Approx(jacobian[0]).margin(1e-7));
        CHECK(-0.09650191 == Approx(jacobian[1]).margin(1e-7));
        CHECK(0.23005702 == Approx(jacobian[2]).margin(1e-7));
        CHECK(0.06396442 == Approx(jacobian[3]).margin(1e-7));
        CHECK(-0.09650191 == Approx(jacobian[4]).margin(1e-7));
        CHECK(0.23005702 == Approx(jacobian[5]).margin(1e-7));
    }
}
