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
#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

using namespace Pennylane;
using namespace Pennylane::Algorithms;

TEST_CASE("Algorithms::adjointJacobian Op=RX, Obs=Z", "[Algorithms]") {
    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 1;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        const auto obs = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{0});
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        for (const auto &p : param) {
            auto ops = OpsData<double>({"RX"}, {{p}}, {{0}}, {false});

            std::vector<std::complex<double>> cdata(1U << num_qubits);
            cdata[0] = std::complex<double>{1, 0};

            StateVectorRaw<double> psi(cdata.data(), cdata.size());

            JacobianData<double> tape{
                num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};

            adjointJacobian(jacobian, tape, true);

            CAPTURE(jacobian);
            CHECK(-sin(p) == Approx(jacobian[0]));
        }
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=RY, Obs=X", "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 1;
        const size_t num_params = 3;
        const size_t num_obs = 1;

        const auto obs = std::make_shared<NamedObs<double>>(
            "PauliX", std::vector<size_t>{0});
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        for (const auto &p : param) {
            auto ops = OpsData<double>({"RY"}, {{p}}, {{0}}, {false});

            std::vector<std::complex<double>> cdata(1U << num_qubits);
            cdata[0] = std::complex<double>{1, 0};

            StateVectorRaw<double> psi(cdata.data(), cdata.size());

            JacobianData<double> tape{
                num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};

            adjointJacobian(jacobian, tape, true);

            CAPTURE(jacobian);
            CHECK(cos(p) == Approx(jacobian[0]).margin(1e-7));
        }
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=RX, Obs=[Z,Z]", "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_params = 1;
        const size_t num_obs = 2;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        const auto obs1 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{1});

        auto ops = OpsData<double>({"RX"}, {{param[0]}}, {{0}}, {false});

        JacobianData<double> tape{
            num_params, psi.getLength(), psi.getData(), {obs1, obs2}, ops, tp};

        adjointJacobian(jacobian, tape, true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.0 == Approx(jacobian[1 * num_obs - 1]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=[RX,RX,RX], Obs=[Z,Z,Z]",
          "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        const auto obs1 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{2});

        auto ops = OpsData<double>({"RX", "RX", "RX"},
                                   {{param[0]}, {param[1]}, {param[2]}},
                                   {{0}, {1}, {2}}, {false, false, false});

        JacobianData<double> tape{num_params,    psi.getLength(),
                                  psi.getData(), {obs1, obs2, obs3},
                                  ops,           tp};

        adjointJacobian(jacobian, tape, true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(-sin(param[1]) ==
              Approx(jacobian[1 * num_params + 1]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[2 * num_params + 2]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=[RX,RX,RX], Obs=[Z,Z,Z], "
          "TParams=[0,2]",
          "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_obs * t_params.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        const auto obs1 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{2});

        auto ops = OpsData<double>({"RX", "RX", "RX"},
                                   {{param[0]}, {param[1]}, {param[2]}},
                                   {{0}, {1}, {2}}, {false, false, false});

        JacobianData<double> tape{num_params,    psi.getLength(),
                                  psi.getData(), {obs1, obs2, obs3},
                                  ops,           t_params};

        adjointJacobian(jacobian, tape, true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(0 == Approx(jacobian[1 * t_params.size() + 1]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[2 * t_params.size() + 1]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=[RX,RX,RX], Obs=[ZZZ]",
          "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        const auto obs = std::make_shared<TensorProdObs<double>>(
            std::make_shared<NamedObs<double>>("PauliZ",
                                               std::vector<size_t>{0}),
            std::make_shared<NamedObs<double>>("PauliZ",
                                               std::vector<size_t>{1}),
            std::make_shared<NamedObs<double>>("PauliZ",
                                               std::vector<size_t>{2}));
        auto ops = OpsData<double>({"RX", "RX", "RX"},
                                   {{param[0]}, {param[1]}, {param[2]}},
                                   {{0}, {1}, {2}}, {false, false, false});

        JacobianData<double> tape{
            num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};

        adjointJacobian(jacobian, tape, true);

        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-0.1755096592645253 == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.26478810666384334 == Approx(jacobian[1]).margin(1e-7));
        CHECK(-0.6312451595102775 == Approx(jacobian[2]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=Mixed, Obs=[XXX]", "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2, 3, 4, 5};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 6;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        const auto obs = std::make_shared<TensorProdObs<double>>(
            std::make_shared<NamedObs<double>>("PauliX",
                                               std::vector<size_t>{0}),
            std::make_shared<NamedObs<double>>("PauliX",
                                               std::vector<size_t>{1}),
            std::make_shared<NamedObs<double>>("PauliX",
                                               std::vector<size_t>{2}));
        auto ops = OpsData<double>(
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

        JacobianData<double> tape{
            num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};

        adjointJacobian(jacobian, tape, true);

        CAPTURE(jacobian);

        // Computed with PennyLane using default.qubit.adjoint_jacobian
        CHECK(0.0 == Approx(jacobian[0]).margin(1e-7));
        CHECK(-0.674214427 == Approx(jacobian[1]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[2]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[3]).margin(1e-7));
        CHECK(-0.0129093062 == Approx(jacobian[4]).margin(1e-7));
        CHECK(0.323846156 == Approx(jacobian[5]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Decomposed Rot gate, non "
          "computational basis state",
          "[Algorithms]") {
    using namespace Pennylane::Util;
    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_params = 3;
        const size_t num_obs = 1;

        const auto thetas = Util::linspace(-2 * M_PI, 2 * M_PI, 7);
        std::vector<std::vector<double>> expec_results{
            {0, -9.90819496e-01, 0},
            {-8.18996553e-01, 1.62526544e-01, 0},
            {-0.203949, 0.48593716, 0},
            {0, 1, 0},
            {-2.03948985e-01, 4.85937177e-01, 0},
            {-8.18996598e-01, 1.62526487e-01, 0},
            {0, -9.90819511e-01, 0}};

        const auto obs = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{0});

        for (size_t i = 0; i < thetas.size(); i++) {
            const auto theta = thetas[i];
            std::vector<double> local_params{theta, std::pow(theta, 3),
                                             SQRT2<double>() * theta};
            std::vector<double> jacobian(num_obs * tp.size(), 0);

            std::vector<std::complex<double>> cdata{INVSQRT2<double>(),
                                                    -INVSQRT2<double>()};
            StateVectorRaw<double> psi(cdata.data(), cdata.size());

            auto ops = OpsData<double>(
                {"RZ", "RY", "RZ"},
                {{local_params[0]}, {local_params[1]}, {local_params[2]}},
                {{0}, {0}, {0}}, {false, false, false});

            JacobianData<double> tape{
                num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};

            adjointJacobian(jacobian, tape, true);

            CAPTURE(theta);
            CAPTURE(jacobian);

            // Computed with PennyLane using default.qubit
            CHECK(expec_results[i][0] == Approx(jacobian[0]).margin(1e-7));
            CHECK(expec_results[i][1] == Approx(jacobian[1]).margin(1e-7));
            CHECK(expec_results[i][2] == Approx(jacobian[2]).margin(1e-7));
        }
    }
}

TEST_CASE("Algorithms::adjointJacobian Mixed Ops, Obs and TParams",
          "[Algorithms]") {
    using namespace Pennylane::Util;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> t_params{1, 2, 3};
    {
        const size_t num_obs = 1;

        const auto thetas = Util::linspace(-2 * M_PI, 2 * M_PI, 8);

        std::vector<double> local_params{0.543, 0.54, 0.1,  0.5, 1.3,
                                         -2.3,  0.5,  -0.5, 0.5};
        std::vector<double> jacobian(num_obs * t_params.size(), 0);

        std::vector<std::complex<double>> cdata{ONE<double>(), ZERO<double>(),
                                                ZERO<double>(), ZERO<double>()};
        StateVectorRaw<double> psi(cdata.data(), cdata.size());

        const auto obs = std::make_shared<TensorProdObs<double>>(
            std::make_shared<NamedObs<double>>("PauliX",
                                               std::vector<size_t>{0}),
            std::make_shared<NamedObs<double>>("PauliZ",
                                               std::vector<size_t>{1}));

        auto ops = OpsData<double>(
            {"Hadamard", "RX", "CNOT", "RZ", "RY", "RZ", "RZ", "RY", "RZ", "RZ",
             "RY", "CNOT"},
            {{},
             {local_params[0]},
             {},
             {local_params[1]},
             {local_params[2]},
             {local_params[3]},
             {local_params[4]},
             {local_params[5]},
             {local_params[6]},
             {local_params[7]},
             {local_params[8]},
             {}},
            {{0}, {0}, {0, 1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {0, 1}},
            {false, false, false, false, false, false, false, false, false,
             false, false, false});

        JacobianData<double> tape{
            t_params.size(), psi.getLength(), psi.getData(), {obs}, ops,
            t_params};

        adjointJacobian(jacobian, tape, true);

        std::vector<double> expected{-0.71429188, 0.04998561, -0.71904837};
        // Computed with PennyLane using default.qubit
        CHECK(expected[0] == Approx(jacobian[0]));
        CHECK(expected[1] == Approx(jacobian[1]));
        CHECK(expected[2] == Approx(jacobian[2]));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=RX, Obs=Ham[Z0+Z1]", "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_params = 1;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        const auto obs1 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<double>>(
            "PauliZ", std::vector<size_t>{1});

        auto ham = Hamiltonian<double>::create({0.3, 0.7}, {obs1, obs2});

        auto ops = OpsData<double>({"RX"}, {{param[0]}}, {{0}}, {false});

        JacobianData<double> tape{
            num_params, psi.getLength(), psi.getData(), {ham}, ops, tp};

        adjointJacobian(jacobian, tape, true);

        CAPTURE(jacobian);
        CHECK(-0.3 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=[RX,RX,RX], Obs=Ham[Z0+Z1+Z2], "
          "TParams=[0,2]",
          "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * t_params.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        auto obs1 = std::make_shared<NamedObs<double>>("PauliZ",
                                                       std::vector<size_t>{0});
        auto obs2 = std::make_shared<NamedObs<double>>("PauliZ",
                                                       std::vector<size_t>{1});
        auto obs3 = std::make_shared<NamedObs<double>>("PauliZ",
                                                       std::vector<size_t>{2});

        auto ham =
            Hamiltonian<double>::create({0.47, 0.32, 0.96}, {obs1, obs2, obs3});

        auto ops = OpsData<double>({"RX", "RX", "RX"},
                                   {{param[0]}, {param[1]}, {param[2]}},
                                   {{0}, {1}, {2}}, {false, false, false});

        JacobianData<double> tape{
            num_params, psi.getLength(), psi.getData(), {ham}, ops, t_params};

        adjointJacobian(jacobian, tape, true);

        CAPTURE(jacobian);
        CHECK((-0.47 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7)));
        CHECK((-0.96 * sin(param[2]) == Approx(jacobian[1]).margin(1e-7)));
    }
}

TEST_CASE("Algorithms::adjointJacobian Test HermitianObs", "[Algorithms]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        std::vector<double> jacobian1(num_obs * t_params.size(), 0);
        std::vector<double> jacobian2(num_obs * t_params.size(), 0);

        std::vector<std::complex<double>> cdata(1U << num_qubits);
        StateVectorRaw<double> psi(cdata.data(), cdata.size());
        cdata[0] = std::complex<double>{1, 0};

        auto obs1 = std::make_shared<TensorProdObs<double>>(
            std::make_shared<NamedObs<double>>("PauliZ",
                                               std::vector<size_t>{0}),
            std::make_shared<NamedObs<double>>("PauliZ",
                                               std::vector<size_t>{1}));
        auto obs2 = std::make_shared<HermitianObs<double>>(
            std::vector<std::complex<double>>{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1,
                                              0, 0, 0, 0, 1},
            std::vector<size_t>{0, 1});

        auto ops = OpsData<double>({"RX", "RX", "RX"},
                                   {{param[0]}, {param[1]}, {param[2]}},
                                   {{0}, {1}, {2}}, {false, false, false});

        JacobianData<double> tape1{
            num_params, psi.getLength(), psi.getData(), {obs1}, ops, t_params};

        JacobianData<double> tape2{
            num_params, psi.getLength(), psi.getData(), {obs2}, ops, t_params};

        adjointJacobian(jacobian1, tape1, true);
        adjointJacobian(jacobian2, tape2, true);

        CHECK((jacobian1 == PLApprox(jacobian2).margin(1e-7)));
    }
}

/*
TEST_CASE("Algorithms::applyObservable visitor checks",
          "[Algorithms]") {
    SECTION("Obs with params 0") {
        std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
        std::vector<double> expected_results{0.90096887, 0.80901699, -0.5};

        auto obs_term = std::make_shared<ObsTerm<double>>({"PauliZ"}, {{}},
{{0}}); auto ops = OpsData<double>({"RX"}, {{expec_results[0]}}, {{0}},
{false}); std::vector<double> out_data(1);

        for (std::size_t i = 0; i < param.size(); i++) {
            StateVectorManaged<double> psi(2);
            JacobianData<double> jd(1, psi.getLength(), psi.getData(),
                                    {obs_default}, ops, {1});
            adjointJacobian(out_data, jd, true);
            REQUIRE(out_data == expected_results[i]);
        }
    }

    SECTION("Obs with params std::vector<std::complex<double>>") {
        std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
        std::vector<double> expec_results{0.90096887, 0.80901699, -0.5};
        using v_type = std::vector<std::complex<double>>;

        v_type z_par{ONE<double>(), ZERO<double>(), ZERO<double>(),
                     ZERO<double>()};

        auto obs_term = std::make_shared<ObsDatum<double>>
            ({"MyPauliZ"}, {z_par}, {{0}});

        auto ops =
            OpsData<double>({"RX"}, {{expec_results[0]}}, {{0}}, {false});
        std::vector<double> out_data(1);

        for (std::size_t i = 0; i < param.size(); i++) {
            StateVectorManaged<double> psi(2);
            JacobianData<double> jd(1, psi.getLength(), psi.getData(),
                                    {obs_default}, ops, {1});
            adjointJacobian(out_data, jd, true);
            REQUIRE(out_data == expec_results[i]);
        }
    }
    SECTION("Obs no params") {
        std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
        std::vector<double> expec_results{0.90096887, 0.80901699, -0.5};
        using v_type = std::vector<std::complex<double>>;

        v_type z_par{ONE<double>(), ZERO<double>(), ZERO<double>(),
                     ZERO<double>()};

        auto obs_default = ObsDatum<double>({"PauliZ"}, {}, {{0}});

        auto ops =
            OpsData<double>({"RX"}, {{expec_results[0]}}, {{0}}, {false});
        std::vector<double> out_data(1);

        for (std::size_t i = 0; i < param.size(); i++) {
            StateVectorManaged<double> psi(2);
            JacobianData<double> jd(1, psi.getLength(), psi.getData(),
                                    {obs_default}, ops, {1});
            adj.adjointJacobian(out_data, jd, true);
        }
    }
}
*/
