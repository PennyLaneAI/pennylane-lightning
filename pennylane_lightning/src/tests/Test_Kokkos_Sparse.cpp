#include <complex>
#include <cstdio>
#include <vector>

#include "Kokkos_Sparse.hpp"

#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

using namespace Pennylane;
using namespace Pennylane::Util;

namespace {
using std::complex;
using std::size_t;
using std::string;
using std::vector;
}; // namespace

TEMPLATE_TEST_CASE("apply_Sparse_Matrix_Kokkos", "[Kokkos Sparse]", float,
                   double) {
    long num_qubits = 3;
    long data_size = Util::exp2(num_qubits);

    std::vector<std::vector<complex<TestType>>> vectors = {
        {0.33160916, 0.90944626, 0.81097291, 0.46112135, 0.42801563, 0.38077181,
         0.23550137, 0.57416324},
        {{0.26752544, 0.00484225},
         {0.49189265, 0.21231633},
         {0.28691029, 0.87552205},
         {0.13499786, 0.63862517},
         {0.31748372, 0.25701515},
         {0.96968437, 0.69821151},
         {0.53674213, 0.58564544},
         {0.02213429, 0.3050882}}};

    const std::vector<std::vector<complex<TestType>>> result_refs = {
        {-1.15200034, -0.23313581, -0.5595947, -0.7778672, -0.41387753,
         -0.28274519, -0.71943368, 0.00705271},
        {{-0.24650151, -0.51256229},
         {-0.06254307, -0.66804797},
         {-0.33998022, 0.02458055},
         {-0.46939616, -0.49391203},
         {-0.7871985, -1.07982153},
         {0.11545852, -0.14444908},
         {-0.45507653, -0.41765428},
         {-0.78213328, -0.28539948}}};

    std::vector<long> row_map;
    std::vector<long> entries;
    std::vector<complex<TestType>> values;
    write_CSR_vectors(row_map, entries, values, data_size);

    if constexpr (USE_KOKKOS) {
        SECTION("Testing sparse matrix dense vector product:") {
            for (size_t vec = 0; vec < vectors.size(); vec++) {
                std::vector<complex<TestType>> result;
                apply_Sparse_Matrix_Kokkos(
                    vectors[vec].data(), static_cast<long>(vectors[vec].size()),
                    row_map.data(), static_cast<long>(row_map.size()),
                    entries.data(), values.data(),
                    static_cast<long>(values.size()), result);
                REQUIRE(result_refs[vec] == approx(result).margin(1e-6));
            };
        }
    } else {
        SECTION(
            "Testing if apply_Sparse_Matrix_Kokkos is throwing an exception:") {
            size_t vec = 0;
            std::vector<complex<TestType>> result;
            PL_CHECK_THROWS_MATCHES(
                apply_Sparse_Matrix_Kokkos(
                    vectors[vec].data(), static_cast<long>(vectors[vec].size()),
                    row_map.data(), static_cast<long>(row_map.size()),
                    entries.data(), values.data(),
                    static_cast<long>(values.size()), result),
                LightningException,
                "Executing the product of a Sparse matrix and a vector "
                "needs Kokkos and Kokkos Kernels installation.");
        }
    }
}

TEMPLATE_TEST_CASE("apply_Sparse_Matrix", "[Kokkos Sparse]", float, double) {
    long num_qubits = 3;
    long data_size = Util::exp2(num_qubits);

    std::vector<std::vector<complex<TestType>>> vectors = {
        {0.33160916, 0.90944626, 0.81097291, 0.46112135, 0.42801563, 0.38077181,
         0.23550137, 0.57416324},
        {{0.26752544, 0.00484225},
         {0.49189265, 0.21231633},
         {0.28691029, 0.87552205},
         {0.13499786, 0.63862517},
         {0.31748372, 0.25701515},
         {0.96968437, 0.69821151},
         {0.53674213, 0.58564544},
         {0.02213429, 0.3050882}}};

    const std::vector<std::vector<complex<TestType>>> result_refs = {
        {-1.15200034, -0.23313581, -0.5595947, -0.7778672, -0.41387753,
         -0.28274519, -0.71943368, 0.00705271},
        {{-0.24650151, -0.51256229},
         {-0.06254307, -0.66804797},
         {-0.33998022, 0.02458055},
         {-0.46939616, -0.49391203},
         {-0.7871985, -1.07982153},
         {0.11545852, -0.14444908},
         {-0.45507653, -0.41765428},
         {-0.78213328, -0.28539948}}};

    std::vector<long> row_map;
    std::vector<long> entries;
    std::vector<complex<TestType>> values;
    write_CSR_vectors(row_map, entries, values, data_size);

    if constexpr (USE_KOKKOS) {
        SECTION("Testing sparse matrix dense vector product:") {
            for (size_t vec = 0; vec < vectors.size(); vec++) {
                std::vector<complex<TestType>> result = apply_Sparse_Matrix(
                    vectors[vec].data(), static_cast<long>(vectors[vec].size()),
                    row_map.data(), static_cast<long>(row_map.size()),
                    entries.data(), values.data(),
                    static_cast<long>(values.size()));
                REQUIRE(result_refs[vec] == approx(result).margin(1e-6));
            };
        }
    } else {
        SECTION("Testing if apply_Sparse_Matrix is throwing an exception:") {
            size_t vec = 0;
            PL_CHECK_THROWS_MATCHES(
                apply_Sparse_Matrix(
                    vectors[vec].data(), static_cast<long>(vectors[vec].size()),
                    row_map.data(), static_cast<long>(row_map.size()),
                    entries.data(), values.data(),
                    static_cast<long>(values.size())),
                LightningException,
                "Executing the product of a Sparse matrix and a vector "
                "needs Kokkos and Kokkos Kernels installation.");
        }
    }
}