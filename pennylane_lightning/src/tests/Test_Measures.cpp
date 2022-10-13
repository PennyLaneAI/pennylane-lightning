#include <complex>
#include <cstdio>
#include <vector>

#include "Kokkos_Sparse.hpp"
#include "Measures.hpp"
#include "StateVectorManagedCPU.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

using namespace Pennylane;
using namespace Pennylane::Util;
using namespace Pennylane::Simulators;

using std::complex;
using std::size_t;
using std::string;
using std::vector;

TEST_CASE("Probabilities", "[Measures]") {
    // Probabilities calculated with Pennylane default.qubit:
    std::vector<std::pair<std::vector<size_t>, std::vector<double>>> input = {
        {{0, 1, 2},
         {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072, 0.00801973,
          0.02280642, 0.00104134}},
        {{0, 2, 1},
         {0.67078706, 0.0870997, 0.03062806, 0.00397696, 0.17564072, 0.02280642,
          0.00801973, 0.00104134}},
        {{1, 0, 2},
         {0.67078706, 0.03062806, 0.17564072, 0.00801973, 0.0870997, 0.00397696,
          0.02280642, 0.00104134}},
        {{1, 2, 0},
         {0.67078706, 0.0870997, 0.17564072, 0.02280642, 0.03062806, 0.00397696,
          0.00801973, 0.00104134}},
        {{2, 0, 1},
         {0.67078706, 0.17564072, 0.03062806, 0.00801973, 0.0870997, 0.02280642,
          0.00397696, 0.00104134}},
        {{2, 1, 0},
         {0.67078706, 0.17564072, 0.0870997, 0.02280642, 0.03062806, 0.00801973,
          0.00397696, 0.00104134}},
        {{0, 1}, {0.70141512, 0.09107666, 0.18366045, 0.02384776}},
        {{0, 2}, {0.75788676, 0.03460502, 0.19844714, 0.00906107}},
        {{1, 2}, {0.84642778, 0.0386478, 0.10990612, 0.0050183}},
        {{2, 1}, {0.84642778, 0.10990612, 0.0386478, 0.0050183}},
        {{0}, {0.79249179, 0.20750821}},
        {{1}, {0.88507558, 0.11492442}},
        {{2}, {0.9563339, 0.0436661}}};

    // Defining the State Vector that will be measured.
    auto Measured_StateVector = Initializing_StateVector();

    // Initializing the measures class.
    // This object attaches to the statevector allowing several measures.
    Measures<double, StateVectorManagedCPU<double>> Measurer(
        Measured_StateVector);

    vector<double> probabilities;

    SECTION("Looping over different wire configurations:") {
        for (const auto &term : input) {
            probabilities = Measurer.probs(term.first);
            REQUIRE_THAT(term.second,
                         Catch::Approx(probabilities).margin(1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("Expected Values", "[Measures]", float, double) {
    // Defining the State Vector that will be measured.
    auto Measured_StateVector = Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // This object attaches to the statevector allowing several measures.
    Measures<TestType, StateVectorManagedCPU<TestType>> Measurer(
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
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        vector<TestType> exp_values;
        vector<TestType> exp_values_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }
}

TEMPLATE_TEST_CASE("Sample", "[Measures]", float, double) {
    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    // Defining the State Vector that will be measured.
    StateVectorManagedCPU<TestType> Measured_StateVector =
        Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // This object attaches to the statevector allowing several measures.
    Measures<TestType, StateVectorManagedCPU<TestType>> Measurer(
        Measured_StateVector);
    vector<TestType> expected_probabilities = {
        0.67078706, 0.03062806, 0.0870997,  0.00397696,
        0.17564072, 0.00801973, 0.02280642, 0.00104134};

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
    StateVectorManagedCPU<TestType> Measured_StateVector =
        Initializing_StateVector<TestType>();

    // Initializing the measures class.
    // This object attaches to the statevector allowing several measures.
    Measures<TestType, StateVectorManagedCPU<TestType>> Measurer(
        Measured_StateVector);

    SECTION("Testing single operation defined by a matrix:") {
        vector<std::complex<TestType>> PauliX = {0, 1, 1, 0};
        vector<size_t> wires_single = {0};
        TestType variance = Measurer.var(PauliX, wires_single);
        TestType variances_ref = 0.7572222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        vector<size_t> wires_single = {0};
        TestType variance = Measurer.var("PauliX", wires_single);
        TestType variances_ref = 0.7572222;
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
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        vector<TestType> variances;
        vector<TestType> variances_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }
}
