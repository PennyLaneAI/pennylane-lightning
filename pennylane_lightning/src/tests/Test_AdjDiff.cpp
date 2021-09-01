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
 * @brief Tests the constructability of the AdjointDiff.hpp classes.
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

TEMPLATE_TEST_CASE("ObsDatum::ObsDatum", "[AdjointJacobian]", float, double) {
    SECTION("ObsDatum") {
        REQUIRE_FALSE(std::is_constructible<ObsDatum<>>::value);
    }
    SECTION("ObsDatum<TestType> {}") {
        REQUIRE_FALSE(std::is_constructible<ObsDatum<TestType>>::value);
    }
    SECTION("ObsDatum<std::complex<TestType>> {}") {
        REQUIRE_FALSE(
            std::is_constructible<ObsDatum<std::complex<TestType>>>::value);
    }
    SECTION("ObsDatum<TestType> {const std::vector<std::string> &, const "
            "std::vector<std::vector<TestType>> &, const "
            "std::vector<std::vector<size_t>> &}") {
        REQUIRE(std::is_constructible<
                ObsDatum<TestType>, const std::vector<std::string> &,
                const std::vector<std::vector<TestType>> &,
                const std::vector<std::vector<size_t>> &>::value);
    }
    SECTION("ObsDatum<std::complex<TestType>> {const std::vector<std::string> "
            "&, const std::vector<std::vector<std::complex<TestType>>> &, "
            "const std::vector<std::vector<size_t>> &}") {
        REQUIRE(std::is_constructible<
                ObsDatum<std::complex<TestType>>,
                const std::vector<std::string> &,
                const std::vector<std::vector<std::complex<TestType>>> &,
                const std::vector<std::vector<size_t>> &>::value);
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian", "[AdjointJacobian]") {
    AdjointJacobian<double> adj;
    // std::vector<double> param{1, -2, 1.623, -0.051, 0};
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};

    SECTION("RX gradient") {
        const size_t num_qubits = 1;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        auto obs = adj.createObs({"PauliZ"}, {{}}, {{0}});
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        for (const auto &p : param) {
            auto ops = adj.createOpsData({"RX"}, {{p}}, {{0}}, {false});

            std::vector<std::complex<double>> cdata(0b1 << num_qubits);
            cdata[0] = std::complex<double>{1, 0};

            StateVector<double> psi(cdata.data(), cdata.size());
            adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                                ops, {0}, 1, true);
            CAPTURE(jacobian);
            CHECK(-sin(p) == Approx(jacobian.front()));
        }
    }

    SECTION("RY gradient") {
        const size_t num_qubits = 1;
        const size_t num_params = 3;
        const size_t num_obs = 1;

        auto obs = adj.createObs({"PauliX"}, {{}}, {{0}});
        std::vector<double> jacobian(num_obs * num_params, 0.0);

        for (const auto &p : param) {
            auto ops = adj.createOpsData({"RY"}, {{p}}, {{0}}, {false});

            std::vector<std::complex<double>> cdata(0b1 << num_qubits);
            cdata[0] = std::complex<double>{1, 0};

            StateVector<double> psi(cdata.data(), cdata.size());

            adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                                ops, {0}, 1, true);

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
                            {obs1, obs2}, ops, {0}, num_params, true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.0 == Approx(jacobian[1]).margin(1e-7));
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
                            {obs1, obs2, obs3}, ops, {0, 1, 2}, num_params,
                            true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) ==
              Approx(jacobian[0 * num_params + 0]).margin(1e-7));
        CHECK(-sin(param[1]) ==
              Approx(jacobian[1 * num_params + 1]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[2 * num_params + 2]).margin(1e-7));
    }
    SECTION("Multiple RX gradient, single expval per wire, subset of params") {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_params * num_obs, 0.0);
        std::set<size_t> t_params{0, 2};

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
                            {obs1, obs2, obs3}, ops, t_params, num_params,
                            true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) ==
              Approx(jacobian[0 * num_params + 0]).margin(1e-7));
        CHECK(0 == Approx(jacobian[1 * num_params + 1]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[1 * num_params + 2]).margin(1e-7));
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
                            ops, {0, 1, 2}, num_params, true);
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

        auto obs = adj.createObs({"PauliX", "PauliX", "PauliX"}, {{}, {}, {}},
                                 {{0}, {1}, {2}});
        auto ops = adj.createOpsData(
            {"RZ", "RY", "RZ", "CNOT", "CNOT", "RZ", "RY", "RZ"},
            {{param[0]},
             {param[1]},
             {param[2]},
             {},
             {},
             {param[0]},
             {param[1]},
             {param[2]}},
            {{0}, {0}, {0}, {0, 1}, {1, 2}, {1}, {1}, {1}},
            {false, false, false, false, false, false, false, false});

        adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                            ops, {0, 1, 2, 3, 4, 5}, num_params, true);
        CAPTURE(jacobian);

        // Computed with PennyLane using default.qubit.adjoint_jacobian
        CHECK(0.0 == Approx(jacobian[0]).margin(1e-7));
        CHECK(-0.674214427 == Approx(jacobian[1]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[2]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[3]).margin(1e-7));
        CHECK(-0.0129093062 == Approx(jacobian[4]).margin(1e-7));
        CHECK(0.323846156 == Approx(jacobian[5]).margin(1e-7));
    }

    SECTION("Decomposed Rot gate, non computational basis state") {
        const size_t num_qubits = 1;
        const size_t num_params = 3;
        const size_t num_obs = 1;

        const auto thetas = Util::linspace(-2 * M_PI, 2 * M_PI, 7);
        std::unordered_map<double, std::vector<double>> expec_results{
            {thetas[0], {0, -9.90819496e-01, 0}},
            {thetas[1], {-8.18996553e-01, 1.62526544e-01, 0}},
            {thetas[2], {-0.203949, 0.48593716, 0}},
            {thetas[3], {0, 1, 0}},
            {thetas[4], {-2.03948985e-01, 4.85937177e-01, 0}},
            {thetas[5], {-8.18996598e-01, 1.62526487e-01, 0}},
            {thetas[6], {0, -9.90819511e-01, 0}}};

        for (const auto &theta : thetas) {
            std::vector<double> local_params{theta, std::pow(theta, 3),
                                             SQRT2<double>() * theta};
            std::vector<double> jacobian(num_obs * num_params, 0);

            std::vector<std::complex<double>> cdata{INVSQRT2<double>(),
                                                    -INVSQRT2<double>()};
            StateVector<double> psi(cdata.data(), cdata.size());

            auto obs = adj.createObs({"PauliZ"}, {{}}, {{0}});
            auto ops = adj.createOpsData(
                {"RZ", "RY", "RZ"},
                {{local_params[0]}, {local_params[1]}, {local_params[2]}},
                {{0}, {0}, {0}}, {false, false, false});

            adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                                ops, {0, 1, 2}, num_params, true);
            CAPTURE(theta);
            CAPTURE(jacobian);

            // Computed with PennyLane using default.qubit
            CHECK(expec_results[theta][0] == Approx(jacobian[0]).margin(1e-7));
            CHECK(expec_results[theta][1] == Approx(jacobian[1]).margin(1e-7));
            CHECK(expec_results[theta][2] == Approx(jacobian[2]).margin(1e-7));
        }
    }
}
