#include <complex>
#include <cstdio>
#include <vector>

#include "Kokkos_Sparse.hpp"
#include "Measures.hpp"
#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"
#include "Util.hpp"

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

template <typename T = double>
StateVectorManaged<T> Initializing_StateVector() {
    // Defining a StateVector in a non-trivial configuration:
    size_t num_qubits = 3;
    size_t data_size = Util::exp2(num_qubits);

    std::vector<std::complex<T>> arr(data_size, 0);
    arr[0] = 1;
    StateVectorManaged<T> Measured_StateVector(arr.data(), data_size);

    std::vector<size_t> wires;

    T alpha = 0.7;
    T beta = 0.5;
    T gamma = 0.2;
    Measured_StateVector.applyOperations(
        {"RX", "RY", "RX", "RY", "RX", "RY"}, {{0}, {0}, {1}, {1}, {2}, {2}},
        {false, false, false, false, false, false},
        {{alpha}, {alpha}, {beta}, {beta}, {gamma}, {gamma}});

    return Measured_StateVector;
}

#ifdef _ENABLE_KOKKOS
/**
 * @brief Fills the empty vectors with the CRS (Compressed Row Storage) sparse
 * matrix representation, for a tridiagonal + periodic boundary conditions
 * Hamiltonian.
 *
 * @param row_map the j element encodes the total number of non-zeros above
 * row j.
 * @param entries column indices.
 * @param values  matrix non-zero elements.
 * @param numRows matrix number of rows.
 */
template <class fp_precision>
void write_CRS_vectors(std::vector<index_type> &row_map,
                       std::vector<index_type> &entries,
                       std::vector<complex<fp_precision>> &values,
                       index_type numRows) {
    const data_type<fp_precision> SC_ONE =
        Kokkos::ArithTraits<data_type<fp_precision>>::one();

    row_map.resize(numRows + 1);
    for (index_type rowIdx = 1; rowIdx < (index_type)row_map.size(); ++rowIdx) {
        row_map[rowIdx] = row_map[rowIdx - 1] + 3;
    };
    const index_type numNNZ = row_map[numRows];

    entries.resize(numNNZ);
    values.resize(numNNZ);
    for (index_type rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        if (rowIdx == 0) {
            entries[0] = rowIdx;
            entries[1] = rowIdx + 1;
            entries[2] = numRows - 1;

            values[0] = SC_ONE;
            values[1] = -SC_ONE;
            values[2] = -SC_ONE;
        } else if (rowIdx == numRows - 1) {
            entries[row_map[rowIdx]] = 0;
            entries[row_map[rowIdx] + 1] = rowIdx - 1;
            entries[row_map[rowIdx] + 2] = rowIdx;

            values[row_map[rowIdx]] = -SC_ONE;
            values[row_map[rowIdx] + 1] = -SC_ONE;
            values[row_map[rowIdx] + 2] = SC_ONE;
        } else {
            entries[row_map[rowIdx]] = rowIdx - 1;
            entries[row_map[rowIdx] + 1] = rowIdx;
            entries[row_map[rowIdx] + 2] = rowIdx + 1;

            values[row_map[rowIdx]] = -SC_ONE;
            values[row_map[rowIdx] + 1] = SC_ONE;
            values[row_map[rowIdx] + 2] = -SC_ONE;
        }
    }
};
#endif

TEST_CASE("Probabilities", "[Measures]") {
    // Probabilities calculated with Pennylane default_qbit:
    std::vector<std::vector<double>> Expected_Probabilities = {
        {0.687573, 0.013842, 0.089279, 0.001797, 0.180036, 0.003624, 0.023377,
         0.000471}, // probs(0,1,2)
        {0.687573, 0.180036, 0.089279, 0.023377, 0.013842, 0.003624, 0.001797,
         0.000471}, // probs(2,1,0)
        {0.687573, 0.180036, 0.013842, 0.003624, 0.089279, 0.023377, 0.001797,
         0.000471}, // probs(2,0,1)
        {0.687573, 0.089279, 0.180036, 0.023377, 0.013842, 0.001797, 0.003624,
         0.000471},                               // probs(1,2,0)
        {0.701415, 0.091077, 0.183660, 0.023848}, // probs(0,1)
        {0.776852, 0.015640, 0.203413, 0.004095}, // probs(0,2)
        {0.867609, 0.017467, 0.112656, 0.002268}, // probs(1,2)
        {0.867609, 0.112656, 0.017467, 0.002268}, // probs(2,1)
        {0.792492, 0.207508},                     // probs(0)
        {0.885076, 0.114924},                     // probs(1)
        {0.980265, 0.019735}                      // probs(2)
    };

    std::vector<std::vector<size_t>> Wires_Configuration = {
        {0, 1, 2}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {0, 1}, {0, 2},
        {1, 2},    {2, 1},    {0},       {1},       {2}};

    // Defining the State Vector that will be measured.
    StateVectorManaged<double> Measured_StateVector =
        Initializing_StateVector();

    // Initializing the measures class.
    // It will attach to the StateVector, allowing measures to keep been taken.
    Measures<double, StateVectorManaged<double>> Measurer(Measured_StateVector);

    vector<double> probabilities;

    SECTION("No wires provided:") {
        probabilities = Measurer.probs();
        REQUIRE_THAT(probabilities,
                     Catch::Approx(Expected_Probabilities[0]).margin(1e-6));
    }

    SECTION("Looping over different wire configuration:") {
        for (size_t conf = 0; conf < Expected_Probabilities.size(); conf++) {
            probabilities = Measurer.probs(Wires_Configuration[conf]);
            REQUIRE_THAT(
                probabilities,
                Catch::Approx(Expected_Probabilities[conf]).margin(1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("Expected Values", "[Measures]", float, double) {
    // Defining the State Vector that will be measured.
    StateVectorManaged<TestType> Measured_StateVector =
        Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // It will attach to the StateVector, allowing measures to keep been taken.
    Measures<TestType, StateVectorManaged<TestType>> Measurer(
        Measured_StateVector);

    SECTION("Testing single operation defined by a matrix:") {
        vector<std::complex<TestType>> PauliX = {0, 1, 1, 0};
        vector<size_t> wires_single = {0};
        TestType exp_value = Measurer.expval(PauliX, wires_single);
        TestType exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        vector<size_t> wires_single = {0};
        TestType exp_value = Measurer.expval("PauliX", wires_single);
        TestType exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        vector<std::complex<TestType>> PauliX = {0, 1, 1, 0};
        vector<std::complex<TestType>> PauliY = {0, {0, -1}, {0, 1}, 0};
        vector<std::complex<TestType>> PauliZ = {1, 0, 0, -1};

        vector<TestType> exp_values;
        vector<TestType> exp_values_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<vector<std::complex<TestType>>> operations_list;

        operations_list = {PauliX, PauliX, PauliX};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.492725, 0.420735, 0.194709};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.644218, -0.479426, -0.198669};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.584984, 0.770151, 0.960530};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        vector<TestType> exp_values;
        vector<TestType> exp_values_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.492725, 0.420735, 0.194709};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.644218, -0.479426, -0.198669};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.584984, 0.770151, 0.960530};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }
}

#ifdef _ENABLE_KOKKOS
TEMPLATE_TEST_CASE("Expected Values - Sparse Operator [Kokkos]", "[Measures]",
                   float, double) {
    // Defining the State Vector that will be measured.
    StateVectorManaged<TestType> Measured_StateVector =
        Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // It will attach to the StateVector, allowing measures to keep been taken.
    Measures<TestType, StateVectorManaged<TestType>> Measurer(
        Measured_StateVector);

    SECTION("Testing Sparse Hamiltonian:") {
        index_type num_qubits = 3;
        index_type data_size = Util::exp2(num_qubits);

        std::vector<index_type> row_map;
        std::vector<index_type> entries;
        std::vector<complex<TestType>> values;
        write_CRS_vectors(row_map, entries, values, data_size);

        TestType exp_values =
            Measurer.expval(row_map.data(), row_map.size(), entries.data(),
                            values.data(), values.size());
        TestType exp_values_ref = 0.71999;
        REQUIRE(exp_values == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing Sparse Hamiltonian (incompatible sizes):") {
        index_type num_qubits = 4;
        index_type data_size = Util::exp2(num_qubits);

        std::vector<index_type> row_map;
        std::vector<index_type> entries;
        std::vector<complex<TestType>> values;
        write_CRS_vectors(row_map, entries, values, data_size);

        PL_CHECK_THROWS_MATCHES(
            Measurer.expval(row_map.data(), row_map.size(), entries.data(),
                            values.data(), values.size()),
            LightningException,
            "Statevector and Hamiltonian have incompatible sizes.");
    }
}
#endif

TEMPLATE_TEST_CASE("Sample", "[Measures]", float, double) {
    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    // Defining the State Vector that will be measured.
    StateVectorManaged<TestType> Measured_StateVector =
        Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // It will attach to the StateVector, allowing measures to keep been taken.
    Measures<TestType, StateVectorManaged<TestType>> Measurer(
        Measured_StateVector);
    vector<TestType> expected_probabilities = {0.687573, 0.013842, 0.089279,
                                               0.001797, 0.180036, 0.003624,
                                               0.023377, 0.000471};

    size_t num_qubits = 3;
    size_t N = std::pow(2, num_qubits);
    size_t num_samples = 100000;
    auto &&samples = Measurer.generate_samples(num_samples);

    std::vector<size_t> counts(N, 0);
    std::vector<size_t> samples_decimal(num_samples, 0);

    // convert samples to decimal and then bin them in counts
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < num_qubits; j++) {
            if (samples[i * num_qubits + j] != 0) {
                samples_decimal[i] += twos[(num_qubits - 1 - j)];
            }
        }
        counts[samples_decimal[i]] += 1;
    }

    // compute estimated probabilities from histogram
    std::vector<TestType> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (TestType)num_samples;
    }

    // compare estimated probabilities to real probabilities
    SECTION("No wires provided:") {
        REQUIRE_THAT(probabilities,
                     Catch::Approx(expected_probabilities).margin(.05));
    }
}

TEMPLATE_TEST_CASE("Variances", "[Measures]", float, double) {
    // Defining the State Vector that will be measured.
    StateVectorManaged<TestType> Measured_StateVector =
        Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // It will attach to the StateVector, allowing measures to keep been taken.
    Measures<TestType, StateVectorManaged<TestType>> Measurer(
        Measured_StateVector);

    SECTION("Testing single operation defined by a matrix:") {
        vector<std::complex<TestType>> PauliX = {0, 1, 1, 0};
        vector<size_t> wires_single = {0};
        TestType variance = Measurer.var(PauliX, wires_single);
        TestType variances_ref = 0.757222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        vector<size_t> wires_single = {0};
        TestType variance = Measurer.var("PauliX", wires_single);
        TestType variances_ref = 0.757222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        vector<std::complex<TestType>> PauliX = {
            {0, 0}, {1, 0}, {1, 0}, {0, 0}};
        vector<std::complex<TestType>> PauliY = {
            {0, 0}, {0, -1}, {0, 1}, {0, 0}};
        vector<std::complex<TestType>> PauliZ = {
            {1, 0}, {0, 0}, {0, 0}, {-1, 0}};

        vector<TestType> variances;
        vector<TestType> variances_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<vector<std::complex<TestType>>> operations_list;

        operations_list = {PauliX, PauliX, PauliX};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.757222, 0.822982, 0.962088};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.584984, 0.770151, 0.960530};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.657794, 0.406867, 0.077381};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        vector<TestType> variances;
        vector<TestType> variances_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.757222, 0.822982, 0.962088};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.584984, 0.770151, 0.960530};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.657794, 0.406867, 0.077381};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }
}
