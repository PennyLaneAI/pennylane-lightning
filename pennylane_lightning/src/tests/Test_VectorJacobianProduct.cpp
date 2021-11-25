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
TEMPLATE_TEST_CASE("VectorJacobianProduct::VectorJacobianProduct",
                   "[VectorJacobianProduct]", float, double) {
    SECTION("VectorJacobianProduct") {
        REQUIRE(std::is_constructible<VectorJacobianProduct<>>::value);
    }
    SECTION("VectorJacobianProduct<TestType> {}") {
        REQUIRE(std::is_constructible<VectorJacobianProduct<TestType>>::value);
    }
}

TEST_CASE("VectorJacobianProduct::vectorJacobianProduct Op=RX, Obs=Z",
          "[VectorJacobianProduct]") {
    AdjointJacobian<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};

    {
        const size_t num_qubits = 1;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        auto obs = ObsDatum<double>({"PauliZ"}, {{}}, {{0}});
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(num_params, 0));

        for (const auto &p : param) {
            auto ops = adj.createOpsData({"RX"}, {{p}}, {{0}}, {false});

            std::vector<std::complex<double>> cdata(0b1 << num_qubits);
            cdata[0] = std::complex<double>{1, 0};

            StateVector<double> psi(cdata.data(), cdata.size());
            adj.adjointJacobian(psi.getData(), psi.getLength(), jacobian, {obs},
                                ops, {0}, true);
            CAPTURE(jacobian);
            CHECK(-sin(p) == Approx(jacobian[0].front()));
        }
    }
}