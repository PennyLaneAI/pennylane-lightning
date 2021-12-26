#include "Gates.hpp"
#include "GateOperationsLM.hpp"
#include "GateOperationsPI.hpp"
#include "Util.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Pennylane;

namespace {
    using std::vector;
    /**
     * @brief create |0>^N
     */
    template<typename fp_t>
    auto create_zero_state(size_t num_qubits) -> std::vector<std::complex<fp_t>> {
        std::vector<std::complex<fp_t>> res(1U << num_qubits, 0.0);
        res[0] = std::complex<fp_t>{1.0, 0.0};
        return res;
    }

    /**
     * @brief create |+>^N
     */
    template<typename fp_t>
    auto create_plus_state(size_t num_qubits) -> std::vector<std::complex<fp_t>> {
        std::vector<std::complex<fp_t>> res(1U << num_qubits, 1.0);
        for(auto& elt: res) {
            elt /= std::sqrt(1U << num_qubits);
        }
        return res;
    }
}


TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRX", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 1;
    SVData<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{{0.1}, {0.6}};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, -0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, -0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, -0.867423225594017}}};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, 0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, 0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, 0.867423225594017}}};

    const auto init_state = svdat.cdata;
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};

                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRX(int_idx, ext_idx, false, angles[index]);

                CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index],
                                    1e-7));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};
                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRX(int_idx, ext_idx, true,
                                        {angles[index]});
                CHECK(isApproxEqual(svdat_direct.cdata,
                                    expected_results_adj[index], 1e-7));
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRX", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
TEMPLATE_TEST_CASE("StateVector::applyRY", "[StateVector_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 1;
    SVData<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.8731983044562817, 0.04786268954660339},
                          {0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, 0.16439396602553008},
                          {0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, 0.47593196040758534},
                          {0.8711876098966215, -0.0577721051072477}}};
    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.8731983044562817, -0.04786268954660339},
                          {-0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, -0.16439396602553008},
                          {-0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, -0.47593196040758534},
                          {-0.8711876098966215, -0.0577721051072477}}};

    const std::vector<cp_t> init_state{{0.8775825618903728, 0.0},
                                       {0.0, -0.47942553860420306}};
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};
                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRY(int_idx, ext_idx, false,
                                        {angles[index]});
                CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index],
                                    1e-5));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};
                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRY(int_idx, ext_idx, true,
                                        {angles[index]});
                CHECK(isApproxEqual(svdat_direct.cdata,
                                    expected_results_adj[index], 1e-5));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.sv.applyOperation("RY", {0}, true,
                                                 {angles[index]});
                CHECK(isApproxEqual(svdat_dispatch.cdata,
                                    expected_results_adj[index], 1e-5));
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRX", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
TEMPLATE_TEST_CASE("StateVector::applyRZ", "[StateVector_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> rz_data;
    rz_data.reserve(angles.size());
    for (auto &a : angles) {
        rz_data.push_back(Gates::getRZ<TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {rz_data[0][0], rz_data[0][0], rz_data[0][0], rz_data[0][0],
         rz_data[0][3], rz_data[0][3], rz_data[0][3], rz_data[0][3]},
        {
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
        },
        {rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3],
         rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_direct{num_qubits, init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.sv.applyRZ(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits, init_state};
            svdat_dispatch.sv.applyOperation("RZ", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRX", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
TEMPLATE_TEST_CASE("StateVector::applyPhaseShift", "[StateVector_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
        {
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
        },
        {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
         ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.cdata;
    for (size_t index = 0; index < num_qubits; index++) {
        SVData<TestType> svdat_direct{num_qubits, init_state};
        auto int_idx = svdat_direct.getInternalIndices({index});
        auto ext_idx = svdat_direct.getExternalIndices({index});

        svdat_direct.sv.applyPhaseShift(int_idx, ext_idx, false,
                                        {angles[index]});

        CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
    }
    
}


