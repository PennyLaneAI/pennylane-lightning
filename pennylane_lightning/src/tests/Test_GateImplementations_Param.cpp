#include "AvailableKernels.hpp"
#include "Gates.hpp"
#include "TestHelpers.hpp"
#include "TestMacros.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file This file contains tests for parameterized gates. List of such gates is
 * [RX, RY, RZ, PhaseShift, Rot, ControlledPhaseShift, CRX, CRY, CRZ, CRot]
 */

using namespace Pennylane;
/**
 * @brief Run test suit only when the gate is defined
 */
#define PENNYLANE_RUN_TEST(GATE_NAME)                                          \
    template <typename PrecisionT, typename ParamT, class GateImplementation,  \
              typename U = void>                                               \
    struct TestApply##GATE_NAME##IfDefined {                                   \
        static void run() {}                                                   \
    };                                                                         \
    template <typename PrecisionT, typename ParamT, class GateImplementation>  \
    struct TestApply##GATE_NAME##IfDefined<                                    \
        PrecisionT, ParamT, GateImplementation,                                \
        std::enable_if_t<std::is_pointer_v<decltype(                           \
            &GateImplementation::template apply##GATE_NAME<PrecisionT,         \
                                                           ParamT>)>>> {       \
        static void run() {                                                    \
            testApply##GATE_NAME<PrecisionT, ParamT, GateImplementation>();    \
        }                                                                      \
    };                                                                         \
    template <typename PrecisionT, typename ParamT, typename TypeList>         \
    struct TestApply##GATE_NAME##ForKernels {                                  \
        static void run() {                                                    \
            TestApply##GATE_NAME##IfDefined<PrecisionT, ParamT,                \
                                            typename TypeList::Type>::run();   \
            TestApply##GATE_NAME##ForKernels<PrecisionT, ParamT,               \
                                             typename TypeList::Next>::run();  \
        }                                                                      \
    };                                                                         \
    template <typename PrecisionT, typename ParamT>                            \
    struct TestApply##GATE_NAME##ForKernels<PrecisionT, ParamT, void> {        \
        static void run() {}                                                   \
    };                                                                         \
    TEMPLATE_TEST_CASE("GateImplementation::apply" #GATE_NAME,                 \
                       "[GateImplementations_Param]", float, double) {         \
        using PrecisionT = TestType;                                           \
        using ParamT = TestType;                                               \
        TestApply##GATE_NAME##ForKernels<PrecisionT, ParamT,                   \
                                         AvailableKernels>::run();             \
    }

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRX() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 1;

    const std::vector<PrecisionT> angles{{0.1}, {0.6}};
    std::vector<std::vector<ComplexPrecisionT>> expected_results{
        std::vector<ComplexPrecisionT>{{0.9987502603949663, 0.0},
                                       {0.0, -0.04997916927067834}},
        std::vector<ComplexPrecisionT>{{0.9553364891256061, 0.0},
                                       {0, -0.2955202066613395}},
        std::vector<ComplexPrecisionT>{{0.49757104789172696, 0.0},
                                       {0, -0.867423225594017}}};

    for (size_t index = 0; index < angles.size(); index++) {
        auto st = create_zero_state<PrecisionT>(num_qubits);

        GateImplementation::applyRX(st.data(), num_qubits, {0}, false,
                                    {angles[index]});

        CHECK(isApproxEqual(st, expected_results[index], 1e-7));
    }
}
PENNYLANE_RUN_TEST(RX)

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRY() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 1;

    const std::vector<PrecisionT> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<ComplexPrecisionT>> expected_results{
        std::vector<ComplexPrecisionT>{
            {0.8731983044562817, 0.04786268954660339},
            {0.0876120655431924, -0.47703040785184303}},
        std::vector<ComplexPrecisionT>{
            {0.8243771119105122, 0.16439396602553008},
            {0.3009211363333468, -0.45035926880694604}},
        std::vector<ComplexPrecisionT>{
            {0.10575112905629831, 0.47593196040758534},
            {0.8711876098966215, -0.0577721051072477}}};
    std::vector<std::vector<ComplexPrecisionT>> expected_results_adj{
        std::vector<ComplexPrecisionT>{
            {0.8731983044562817, -0.04786268954660339},
            {-0.0876120655431924, -0.47703040785184303}},
        std::vector<ComplexPrecisionT>{
            {0.8243771119105122, -0.16439396602553008},
            {-0.3009211363333468, -0.45035926880694604}},
        std::vector<ComplexPrecisionT>{
            {0.10575112905629831, -0.47593196040758534},
            {-0.8711876098966215, -0.0577721051072477}}};

    const std::vector<ComplexPrecisionT> init_state{
        {0.8775825618903728, 0.0}, {0.0, -0.47942553860420306}};
    SECTION("adj = false") {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = init_state;
            GateImplementation::applyRY(st.data(), num_qubits, {0}, false,
                                        {angles[index]});
            CHECK(isApproxEqual(st, expected_results[index], 1e-5));
        }
    }
    SECTION("adj = true") {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = init_state;

            GateImplementation::applyRY(st.data(), num_qubits, {0}, true,
                                        {angles[index]});

            CHECK(isApproxEqual(st, expected_results_adj[index], 1e-5));
        }
    }
}
PENNYLANE_RUN_TEST(RY)

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    // Test using |+++> state

    const std::vector<PrecisionT> angles{0.2, 0.7, 2.9};
    const ComplexPrecisionT coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<ComplexPrecisionT>> rz_data;
    rz_data.reserve(angles.size());
    for (auto &a : angles) {
        rz_data.push_back(Gates::getRZ<PrecisionT>(a));
    }

    std::vector<std::vector<ComplexPrecisionT>> expected_results = {
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

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<PrecisionT>(num_qubits);

        GateImplementation::applyRZ(st.data(), num_qubits, {index}, false,
                                    {angles[index]});

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(RZ)

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyPhaseShift() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    // Test using |+++> state

    const std::vector<PrecisionT> angles{0.3, 0.8, 2.4};
    const ComplexPrecisionT coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<ComplexPrecisionT>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<PrecisionT>(a));
    }

    std::vector<std::vector<ComplexPrecisionT>> expected_results = {
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

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<PrecisionT>(num_qubits);

        GateImplementation::applyPhaseShift(st.data(), num_qubits, {index},
                                            false, {angles[index]});

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(PhaseShift)

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRot() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st = create_zero_state<PrecisionT>(num_qubits);

    const std::vector<std::vector<PrecisionT>> angles{
        std::vector<PrecisionT>{0.3, 0.8, 2.4},
        std::vector<PrecisionT>{0.5, 1.1, 3.0},
        std::vector<PrecisionT>{2.3, 0.1, 0.4}};

    std::vector<std::vector<ComplexPrecisionT>> expected_results{
        std::vector<ComplexPrecisionT>(0b1 << num_qubits),
        std::vector<ComplexPrecisionT>(0b1 << num_qubits),
        std::vector<ComplexPrecisionT>(0b1 << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat =
            Gates::getRot<PrecisionT>(angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_zero_state<PrecisionT>(num_qubits);
        GateImplementation::applyRot(st.data(), num_qubits, {index}, false,
                                     angles[index][0], angles[index][1],
                                     angles[index][2]);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(Rot)

/*******************************************************************************
 * Two-qubit gates
 ******************************************************************************/

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyControlledPhaseShift() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    const size_t num_qubits = 3;

    // Test using |+++> state
    auto ini_st = create_plus_state<PrecisionT>(num_qubits);

    const std::vector<PrecisionT> angles{0.3, 2.4};
    const ComplexPrecisionT coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<ComplexPrecisionT>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<PrecisionT>(a));
    }

    std::vector<std::vector<ComplexPrecisionT>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    auto st = ini_st;

    GateImplementation::applyControlledPhaseShift(st.data(), num_qubits, {0, 1},
                                                  false, {angles[0]});
    CAPTURE(st);
    CHECK(isApproxEqual(st, expected_results[0]));
}
PENNYLANE_RUN_TEST(ControlledPhaseShift)


template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingXX() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using TestHelper::Approx;
    using std::cos;
    using std::sin;
    
    SECTION("IsingXX0,1 |000> -> a|000> + b|110>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_zero_state<PrecisionT>(num_qubits);
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingXX0,1 |100> -> a|100> + b|010>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("100");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingXX0,1 |010> -> a|010> + b|100>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("010");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingXX0,1 |110> -> a|110> + b|000>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("110");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingXX0,2") {
        const size_t num_qubits = 3;
        std::vector<ComplexPrecisionT> ini_st {
            ComplexPrecisionT{0.125681356503, 0.252712197380},
            ComplexPrecisionT{0.262591068130, 0.370189000494},
            ComplexPrecisionT{0.129300299863, 0.371057794075},
            ComplexPrecisionT{0.392248682814, 0.195795523118},
            ComplexPrecisionT{0.303908059240, 0.082981563244},
            ComplexPrecisionT{0.189140284321, 0.179512645957},
            ComplexPrecisionT{0.173146612336, 0.092249594834},
            ComplexPrecisionT{0.298857179897, 0.269627836165},
        };
        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexPrecisionT> expected {
            ComplexPrecisionT{0.148459317603, 0.225284945157},
            ComplexPrecisionT{0.271300438716, 0.326438461763},
            ComplexPrecisionT{0.164042082006, 0.327971890339},
            ComplexPrecisionT{0.401037861022, 0.171003883572},
            ComplexPrecisionT{0.350482432141, 0.047287216587},
            ComplexPrecisionT{0.221097705423, 0.161184442326},
            ComplexPrecisionT{0.197669694288, 0.039212892562},
            ComplexPrecisionT{0.345592157995, 0.250015865318},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, wires, false, angle);
        REQUIRE_THAT(st, Approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingXX)


template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingYY() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using TestHelper::Approx;
    using std::cos;
    using std::sin;
    
    SECTION("IsingYY0,1 |000> -> a|000> + b|110>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_zero_state<PrecisionT>(num_qubits);
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingYY0,1 |100> -> a|100> + b|010>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("100");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingYY0,1 |010> -> a|010> + b|100>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("010");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingYY0,1 |110> -> a|110> + b|000>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("110");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingYY0,1") {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st {
            ComplexPrecisionT{0.276522701942, 0.192601873155},
            ComplexPrecisionT{0.035951282872, 0.224882549474},
            ComplexPrecisionT{0.142578003191, 0.016769549184},
            ComplexPrecisionT{0.207510965432, 0.068085008177},
            ComplexPrecisionT{0.231177902264, 0.039974505646},
            ComplexPrecisionT{0.038587049391, 0.058503643276},
            ComplexPrecisionT{0.023121176451, 0.294843178966},
            ComplexPrecisionT{0.297936734810, 0.061981734524},
            ComplexPrecisionT{0.140961289031, 0.061129422308},
            ComplexPrecisionT{0.204531438234, 0.159178277448},
            ComplexPrecisionT{0.143828437747, 0.031972463787},
            ComplexPrecisionT{0.291528706380, 0.138875986482},
            ComplexPrecisionT{0.297088897520, 0.179914971203},
            ComplexPrecisionT{0.032991360504, 0.024025500927},
            ComplexPrecisionT{0.121553926676, 0.263606060346},
            ComplexPrecisionT{0.177173454285, 0.267447421480},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected {
            ComplexPrecisionT{0.245211756573, 0.236421160261},
            ComplexPrecisionT{0.031781919269, 0.227277526275},
            ComplexPrecisionT{0.099890674345, 0.035451505339},
            ComplexPrecisionT{0.163438308608, 0.094785319724},
            ComplexPrecisionT{0.237868187763, 0.017588203228},
            ComplexPrecisionT{0.062849689541, 0.026015566111},
            ComplexPrecisionT{0.027807906892, 0.268916455494},
            ComplexPrecisionT{0.315895675672, 0.015934827233},
            ComplexPrecisionT{0.145460308037, 0.024469450691},
            ComplexPrecisionT{0.211137338769, 0.151250126997},
            ComplexPrecisionT{0.187891084547, 0.027991919467},
            ComplexPrecisionT{0.297618553419, 0.090899723116},
            ComplexPrecisionT{0.263557070771, 0.220692990352},
            ComplexPrecisionT{-0.002348824386, 0.029319431141},
            ComplexPrecisionT{0.117472403735, 0.282557065430},
            ComplexPrecisionT{0.164443742376, 0.296440286247},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, wires, false, angle);
        REQUIRE_THAT(st, Approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingYY)

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingZZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using TestHelper::Approx;
    using std::cos;
    using std::sin;
    
    SECTION("IsingZZ0,1 |000> -> |000>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_zero_state<PrecisionT>(num_qubits);
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{cos(angle/2), -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingZZ0,1 |100> -> |100>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("100");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }

    SECTION("IsingZZ0,1 |010> -> |010>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("010");
        ParamT angle = 0.312;
        
        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }

    SECTION("IsingZZ0,1 |110> -> |110>") {
        const size_t num_qubits = 3;
        const auto ini_st = create_product_state<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle/2), -sin(angle/2)},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                angle);
        REQUIRE_THAT(st, Approx(expected_results).margin(1e-7));
    }
    SECTION("IsingZZ0,1") {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st {
            ComplexPrecisionT{0.267462841882, 0.010768564798},
            ComplexPrecisionT{0.228575129706, 0.010564590956},
            ComplexPrecisionT{0.099492749900, 0.260849823392},
            ComplexPrecisionT{0.093690204310, 0.189847108173},
            ComplexPrecisionT{0.033390732374, 0.203836830144},
            ComplexPrecisionT{0.226979395737, 0.081852150975},
            ComplexPrecisionT{0.031235505729, 0.176933497281},
            ComplexPrecisionT{0.294287602843, 0.145156781198},
            ComplexPrecisionT{0.152742706049, 0.111628061129},
            ComplexPrecisionT{0.012553863703, 0.120027860480},
            ComplexPrecisionT{0.237156555364, 0.154658769755},
            ComplexPrecisionT{0.117001120872, 0.228059505033},
            ComplexPrecisionT{0.041495873225, 0.065934827444},
            ComplexPrecisionT{0.089653239407, 0.221581340372},
            ComplexPrecisionT{0.217892322429, 0.291261296999},
            ComplexPrecisionT{0.292993251871, 0.186570798697},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected {
            ComplexPrecisionT{0.265888039508, -0.030917377350},
            ComplexPrecisionT{0.227440863156, -0.025076966901},
            ComplexPrecisionT{0.138812299373, 0.242224241539},
            ComplexPrecisionT{0.122048663851, 0.172985266764},
            ComplexPrecisionT{0.001315529800, 0.206549421962},
            ComplexPrecisionT{0.211505899280, 0.116123534558},
            ComplexPrecisionT{0.003366392733, 0.179637932181},
            ComplexPrecisionT{0.268161243812, 0.189116978698},
            ComplexPrecisionT{0.133544466595, 0.134003857126},
            ComplexPrecisionT{-0.006247074818, 0.120520790080},
            ComplexPrecisionT{0.210247652980, 0.189627242850},
            ComplexPrecisionT{0.080147179284, 0.243468334233},
            ComplexPrecisionT{0.051236139067, 0.058687025978},
            ComplexPrecisionT{0.122991206449, 0.204961354585},
            ComplexPrecisionT{0.260499076094, 0.253870909435},
            ComplexPrecisionT{0.318422472324, 0.138783420076},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, wires, false, angle);
        REQUIRE_THAT(st, Approx(expected).margin(1e-5));
    }

}
PENNYLANE_RUN_TEST(IsingZZ)

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRot() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    const auto ini_st = create_zero_state<PrecisionT>(num_qubits);

    const std::vector<PrecisionT> angles{0.3, 0.8, 2.4};

    std::vector<ComplexPrecisionT> expected_results(8);
    const auto rot_mat =
        Gates::getRot<PrecisionT>(angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    SECTION("CRot0,1 |000> -> |000>") {
        auto st = create_zero_state<PrecisionT>(num_qubits);
        GateImplementation::applyCRot(st.data(), num_qubits, {0, 1}, false,
                                      angles[0], angles[1], angles[2]);

        CHECK(isApproxEqual(st, ini_st));
    }
    SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
        auto st = create_zero_state<PrecisionT>(num_qubits);
        GateImplementation::applyPauliX(st.data(), num_qubits, {0}, false);

        GateImplementation::applyCRot(st.data(), num_qubits, {0, 1}, false,
                                      angles[0], angles[1], angles[2]);

        CHECK(isApproxEqual(st, expected_results));
    }
}
PENNYLANE_RUN_TEST(CRot)

/*******************************************************************************
 * Multi-qubit gates
 ******************************************************************************/
template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyMultiRZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using TestHelper::Approx;

    SECTION("MultiRZ0 |++++>") {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = create_plus_state<PrecisionT>(num_qubits);

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0}, false,
                                         angle);

        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
    SECTION("MultiRZ0 |++++>") {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = create_plus_state<PrecisionT>(num_qubits);

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0}, false,
                                         angle);

        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
    SECTION("MultiRZ01 |++++>") {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = create_plus_state<PrecisionT>(num_qubits);

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);

        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
    SECTION("MultiRZ012 |++++>") {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = create_plus_state<PrecisionT>(num_qubits);

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, +0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0, 1, 2},
                                         false, angle);

        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
    SECTION("MultiRZ0123 |++++>") {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = create_plus_state<PrecisionT>(num_qubits);

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, -0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, -0.25}, ComplexPrecisionT{0, +0.25},
            ComplexPrecisionT{0, +0.25}, ComplexPrecisionT{0, -0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0, 1, 2, 3},
                                         false, angle);

        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }

    SECTION("MultiRZ013") {
        const size_t num_qubits = 4;
        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.029963367200, 0.181037777550},
            ComplexPrecisionT{0.070992796807, 0.263183826811},
            ComplexPrecisionT{0.086883003918, 0.090811332201},
            ComplexPrecisionT{0.156989157753, 0.153911449950},
            ComplexPrecisionT{0.193120178047, 0.257383787598},
            ComplexPrecisionT{0.262262890778, 0.163282579388},
            ComplexPrecisionT{0.110853627976, 0.247870990381},
            ComplexPrecisionT{0.202098107411, 0.160525183734},
            ComplexPrecisionT{0.025750679341, 0.172601520950},
            ComplexPrecisionT{0.235737282225, 0.008347360496},
            ComplexPrecisionT{0.085757778150, 0.248516366527},
            ComplexPrecisionT{0.047549845173, 0.223003660220},
            ComplexPrecisionT{0.086414423346, 0.250866254986},
            ComplexPrecisionT{0.112429154107, 0.111787742027},
            ComplexPrecisionT{0.240562329064, 0.010449374903},
            ComplexPrecisionT{0.267984502939, 0.236708607552},
        };
        const std::vector<size_t> wires = {0, 1, 3};
        const ParamT angle = 0.6746272767672288;
        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.088189897518, 0.160919303534},
            ComplexPrecisionT{-0.020109410195, 0.271847963971},
            ComplexPrecisionT{0.112041208417, 0.056939635075},
            ComplexPrecisionT{0.097204863997, 0.197194179664},
            ComplexPrecisionT{0.097055284752, 0.306793234914},
            ComplexPrecisionT{0.301522534529, 0.067284365065},
            ComplexPrecisionT{0.022572982655, 0.270590123918},
            ComplexPrecisionT{0.243835640173, 0.084594090888},
            ComplexPrecisionT{-0.032823490356, 0.171397202432},
            ComplexPrecisionT{0.225215396328, -0.070141071525},
            ComplexPrecisionT{-0.001322233373, 0.262893576650},
            ComplexPrecisionT{0.118674074836, 0.194699985129},
            ComplexPrecisionT{0.164569740491, 0.208130081842},
            ComplexPrecisionT{0.069096925107, 0.142696982805},
            ComplexPrecisionT{0.230464206558, -0.069754376895},
            ComplexPrecisionT{0.174543309361, 0.312059756876},
        };

        auto st = ini_st;

        GateImplementation::applyMultiRZ(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
}
PENNYLANE_RUN_TEST(MultiRZ)
