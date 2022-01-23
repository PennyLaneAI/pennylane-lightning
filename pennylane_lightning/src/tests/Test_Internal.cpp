#include "GateImplementationsPI.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

#include <random>

/**
 * We test internal functions for test suite.
 */

using namespace Pennylane;

TEMPLATE_TEST_CASE("createProductState", "[Test_Internal]", float, double) {
    using PrecisionT = TestType;
    using Pennylane::GateImplementationsPI;
    using TestHelper::Approx;

    SECTION("createProductState(\"+-0\") == |+-0> ") {
        const auto st = createProductState<PrecisionT>("+-0");

        auto expected = createZeroState<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);

        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
    SECTION("createProductState(\"+-0\") == |+-1> ") {
        const auto st = createProductState<PrecisionT>("+-0");

        auto expected = createZeroState<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {2}, false);

        REQUIRE_THAT(st, !Approx(expected).margin(1e-7));
    }
}

/**
 * @brief Test randomUnitary is correct
 */
TEMPLATE_TEST_CASE("randomUnitary", "[Test_Internal]", float, double) {
    using PrecisionT = TestType;
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using TestHelper::Approx;

    std::mt19937 re{1337};

    for (size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        const size_t dim = (1U << num_qubits);
        const auto unitary = randomUnitary<PrecisionT>(re, num_qubits);

        std::vector<std::complex<PrecisionT>> unitary_dagger =
            Transpose(unitary, dim, dim);
        std::transform(
            unitary_dagger.begin(), unitary_dagger.end(),
            unitary_dagger.begin(),
            [](const std::complex<PrecisionT> &v) { return std::conj(v); });

        std::vector<std::complex<PrecisionT>> mat(dim * dim);
        matrixMatProd(unitary.data(), unitary_dagger.data(), mat.data(), dim,
                      dim, dim);

        std::vector<std::complex<PrecisionT>> identity(
            dim * dim, std::complex<PrecisionT>{});
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = std::complex<PrecisionT>{1.0, 0.0};
        }

        REQUIRE_THAT(mat, Approx(identity).margin(1e-5));
        /*
        for(size_t row = 0; row < dim; row++) {
            const ComplexPrecisionT* row_p = unitary.data() + row*dim;
            std::cout << row << "\t" << squaredNorm(row_p, dim) << std::endl;
        }
        */
    }
}
