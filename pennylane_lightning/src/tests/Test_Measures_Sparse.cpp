#include <complex>
#include <cstdio>
#include <vector>

#include "Kokkos_Sparse.hpp"
#include "Measures.hpp"
#include "StateVectorManagedCPU.hpp"
#include "StateVectorRawCPU.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

using namespace Pennylane;
using namespace Pennylane::Util;
using namespace Pennylane::Simulators;

namespace {
using std::complex;
using std::size_t;
using std::string;
using std::vector;
}; // namespace

TEMPLATE_TEST_CASE("Expected Values - Sparse Hamiltonian [Kokkos]",
                   "[Measures]", float, double) {
    // Defining the State Vector that will be measured.
    auto Measured_StateVector = Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // This object attaches to the statevector allowing several measures.
    Measures<TestType, StateVectorManagedCPU<TestType>> Measurer(
        Measured_StateVector);

    if constexpr (USE_KOKKOS) {
        SECTION("Testing Sparse Hamiltonian:") {
            long num_qubits = 3;
            long data_size = Util::exp2(num_qubits);

            std::vector<long> row_map;
            std::vector<long> entries;
            std::vector<complex<TestType>> values;
            write_CSR_vectors(row_map, entries, values, data_size);

            TestType exp_values = Measurer.expval(
                row_map.data(), static_cast<long>(row_map.size()),
                entries.data(), values.data(),
                static_cast<long>(values.size()));
            TestType exp_values_ref = 0.5930885;
            REQUIRE(exp_values == Approx(exp_values_ref).margin(1e-6));
        }

        SECTION("Testing Sparse Hamiltonian (incompatible sizes):") {
            long num_qubits = 4;
            long data_size = Util::exp2(num_qubits);

            std::vector<long> row_map;
            std::vector<long> entries;
            std::vector<complex<TestType>> values;
            write_CSR_vectors(row_map, entries, values, data_size);

            PL_CHECK_THROWS_MATCHES(
                Measurer.expval(row_map.data(),
                                static_cast<long>(row_map.size()),
                                entries.data(), values.data(),
                                static_cast<long>(values.size())),
                LightningException,
                "Statevector and Hamiltonian have incompatible sizes.");
        }
    } else {
        SECTION("Testing Sparse Hamiltonian:") {
            long num_qubits = 3;
            long data_size = Util::exp2(num_qubits);

            std::vector<long> row_map;
            std::vector<long> entries;
            std::vector<complex<TestType>> values;
            write_CSR_vectors(row_map, entries, values, data_size);

            PL_CHECK_THROWS_MATCHES(
                Measurer.expval(row_map.data(),
                                static_cast<long>(row_map.size()),
                                entries.data(), values.data(),
                                static_cast<long>(values.size())),
                LightningException,
                "Executing the product of a Sparse matrix and a vector needs "
                "Kokkos and Kokkos Kernels installation.");
        }
    }
}
