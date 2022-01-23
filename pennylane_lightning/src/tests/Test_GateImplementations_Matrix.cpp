#include "SelectGateOps.hpp"
#include "TestHelpers.hpp"
#include "TestMacros.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

/**
 * @brief Generate random unitary matrix
 *
 * @return Generated unitary matrix in row-major format
 */
template <typename PrecisionT, class RandomEngine>
auto randomUnitary(RandomEngine &re, size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t dim = (1U << num_qubits);
    std::vector<ComplexPrecisionT> res(dim * dim, ComplexPrecisionT{});

    std::normal_distribution<PrecisionT> dist;

    auto generator = [&dist, &re]() { return dist(re); };

    std::generate(res.begin(), res.end(), generator);

    // Simple algorithm to make rows orthogonal with Gram-Schmidt
    // This algorithm is unstable but works for a small matrix.
    // Use QR decomposition when we have LAPACK support.

    for (size_t row2 = 1; row2 < num_qubits; row2++) {
        ComplexPrecisionT *row2_p = res.data() + row2 * dim;
        for (size_t row1 = 0; row1 < row2; row1++) {
            const ComplexPrecisionT *row1_p = res.data() + row1 * dim;
            ComplexPrecisionT dot = innerProd(row1_p, row2_p, dim);

            // orthogonalize row2
            std::transform(row2_p, row2_p + dim, row1_p, row2_p,
                           [dot](auto &elt2, const auto &elt1) {
                               return elt2 - dot * elt1;
                           });
        }

        PrecisionT norm2 = std::sqrt(squareNorm(row2_p, row2_p + dim));
        // noramlize row2
        std::transform(row2_p, row2_p + dim,
                       [norm2](auto c) { return 1.0 / norm2 * c; });
    }
    return res;
}

template <typename PrecisionT, class GateImplementation>
void testApplyMatrix() {}

TEMPLATE_TEST_CASE("GateImplementation::applyMatrix",
                   "[GateImplementations_Matrix]", float, double) {
    using PrecisionT = TestType;
}
