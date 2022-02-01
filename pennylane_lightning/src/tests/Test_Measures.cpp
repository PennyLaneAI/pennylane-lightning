#include <complex>
#include <cstdio>
#include <vector>

#include "Measures.hpp"
#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

namespace {
using std::complex;
using std::size_t;
using std::string;
using std::vector;
}; // namespace

StateVectorManaged<double> Initializing_StateVector() {
    // Defining a StateVector in a non-trivial configuration:
    size_t num_qubits = 3;
    size_t data_size = std::pow(2, num_qubits);

    std::vector<std::complex<double>> arr(data_size, 0);
    arr[0] = 1;
    StateVectorManaged<double> Measured_StateVector(arr.data(), data_size);

    std::vector<size_t> wires;

    double alpha = 0.7;
    double beta = 0.5;
    double gamma = 0.2;
    Measured_StateVector.applyOperations(
        {"RX", "RY", "RX", "RY", "RX", "RY"}, {{0}, {0}, {1}, {1}, {2}, {2}},
        {false, false, false, false, false, false},
        {{alpha}, {alpha}, {beta}, {beta}, {gamma}, {gamma}});

    return Measured_StateVector;
}

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

TEST_CASE("Expected Values", "[Measures]") {
    // Defining the State Vector that will be measured.
    StateVectorManaged<double> Measured_StateVector =
        Initializing_StateVector();

    // Initializing the measures class.
    // It will attach to the StateVector, allowing measures to keep been taken.
    Measures<double, StateVectorManaged<double>> Measurer(Measured_StateVector);

    SECTION("Testing single operation defined by a matrix:") {
        vector<std::complex<double>> PauliX = {0, 1, 1, 0};
        vector<size_t> wires_single = {0};
        double exp_value = Measurer.expval(PauliX, wires_single);
        double exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        vector<size_t> wires_single = {0};
        double exp_value = Measurer.expval("PauliX", wires_single);
        double exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        vector<std::complex<double>> PauliX = {0, 1, 1, 0};
        vector<std::complex<double>> PauliY = {0, {0, -1}, {0, 1}, 0};
        vector<std::complex<double>> PauliZ = {1, 0, 0, -1};

        vector<double> exp_values;
        vector<double> exp_values_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<vector<std::complex<double>>> operations_list;

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
        vector<double> exp_values;
        vector<double> exp_values_ref;
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

TEST_CASE("Variances", "[Measures]") {
    // Defining the State Vector that will be measured.
    StateVectorManaged<double> Measured_StateVector =
        Initializing_StateVector();

    // Initializing the measures class.
    // It will attach to the StateVector, allowing measures to keep been taken.
    Measures<double, StateVectorManaged<double>> Measurer(Measured_StateVector);

    SECTION("Testing single operation defined by a matrix:") {
        vector<std::complex<double>> PauliX = {0, 1, 1, 0};
        vector<size_t> wires_single = {0};
        double variance = Measurer.var(PauliX, wires_single);
        double variances_ref = 0.757222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        vector<size_t> wires_single = {0};
        double variance = Measurer.var("PauliX", wires_single);
        double variances_ref = 0.757222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        vector<std::complex<double>> PauliX = {0, 1, 1, 0};
        vector<std::complex<double>> PauliY = {0, {0, -1}, {0, 1}, 0};
        vector<std::complex<double>> PauliZ = {1, 0, 0, -1};

        vector<double> variances;
        vector<double> variances_ref;
        vector<vector<size_t>> wires_list = {{0}, {1}, {2}};
        vector<vector<std::complex<double>>> operations_list;

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
        vector<double> variances;
        vector<double> variances_ref;
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
