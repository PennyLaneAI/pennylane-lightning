#include "CPUMemoryModel.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"
#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

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
    struct Apply##GATE_NAME##IsDefined {                                       \
        constexpr static bool value = false;                                   \
    };                                                                         \
    template <typename PrecisionT, typename ParamT, class GateImplementation>  \
    struct Apply##GATE_NAME##IsDefined<                                        \
        PrecisionT, ParamT, GateImplementation,                                \
        std::enable_if_t<std::is_pointer_v<                                    \
            decltype(&GateImplementation::template apply##GATE_NAME<           \
                     PrecisionT, ParamT>)>>> {                                 \
        constexpr static bool value = true;                                    \
    };                                                                         \
    template <typename PrecisionT, typename ParamT, typename TypeList>         \
    void testApply##GATE_NAME##ForKernels() {                                  \
        if constexpr (!std::is_same_v<TypeList, void>) {                       \
            using GateImplementation = typename TypeList::Type;                \
            if constexpr (Apply##GATE_NAME##IsDefined<                         \
                              PrecisionT, ParamT,                              \
                              GateImplementation>::value) {                    \
                testApply##GATE_NAME<PrecisionT, ParamT,                       \
                                     GateImplementation>();                    \
            } else {                                                           \
                SUCCEED("Member function apply" #GATE_NAME                     \
                        " is not defined for kernel "                          \
                        << GateImplementation::name);                          \
            }                                                                  \
            testApply##GATE_NAME##ForKernels<PrecisionT, ParamT,               \
                                             typename TypeList::Next>();       \
        }                                                                      \
    }                                                                          \
    TEMPLATE_TEST_CASE("GateImplementation::apply" #GATE_NAME,                 \
                       "[GateImplementations_Param]", float, double) {         \
        using PrecisionT = TestType;                                           \
        using ParamT = TestType;                                               \
        testApply##GATE_NAME##ForKernels<PrecisionT, ParamT, TestKernels>();   \
    }                                                                          \
    static_assert(true, "Require semicolon")

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyPhaseShift() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    // Test using |+++> state
    const auto isqrt2 = PrecisionT{Util::INVSQRT2<PrecisionT>()};
    const std::vector<PrecisionT> angles{0.3, 0.8, 2.4};
    const ComplexPrecisionT coef{isqrt2 / PrecisionT{2.0}, PrecisionT{0.0}};

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
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyPhaseShift(st.data(), num_qubits, {index},
                                            false, {angles[index]});

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(PhaseShift);

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
        auto st = createZeroState<PrecisionT>(num_qubits);

        GateImplementation::applyRX(st.data(), num_qubits, {0}, false,
                                    {angles[index]});

        CHECK(st == approx(expected_results[index]).epsilon(1e-7));
    }
}
PENNYLANE_RUN_TEST(RX);

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

    const TestVector<ComplexPrecisionT> init_state{
        {{0.8775825618903728, 0.0}, {0.0, -0.47942553860420306}},
        getBestAllocator<ComplexPrecisionT>()};
    DYNAMIC_SECTION(GateImplementation::name
                    << ", RY - " << PrecisionToName<PrecisionT>::value) {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = init_state;
            GateImplementation::applyRY(st.data(), num_qubits, {0}, false,
                                        {angles[index]});
            CHECK(st == approx(expected_results[index]).epsilon(1e-5));
        }
    }
}
PENNYLANE_RUN_TEST(RY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    // Test using |+++> state
    const auto isqrt2 = PrecisionT{Util::INVSQRT2<PrecisionT>()};

    const std::vector<PrecisionT> angles{0.2, 0.7, 2.9};
    const ComplexPrecisionT coef{isqrt2 / PrecisionT{2.0}, PrecisionT{0.0}};

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
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyRZ(st.data(), num_qubits, {index}, false,
                                    {angles[index]});

        CHECK(st == approx(expected_results[index]));
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyRZ(st.data(), num_qubits, {index}, true,
                                    {-angles[index]});
        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(RZ);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRot() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st = createZeroState<PrecisionT>(num_qubits);

    const std::vector<std::vector<PrecisionT>> angles{
        std::vector<PrecisionT>{0.3, 0.8, 2.4},
        std::vector<PrecisionT>{0.5, 1.1, 3.0},
        std::vector<PrecisionT>{2.3, 0.1, 0.4}};

    std::vector<std::vector<ComplexPrecisionT>> expected_results{
        std::vector<ComplexPrecisionT>(1U << num_qubits),
        std::vector<ComplexPrecisionT>(1U << num_qubits),
        std::vector<ComplexPrecisionT>(1U << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat =
            Gates::getRot<PrecisionT>(angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][size_t{1U} << (num_qubits - i - 1)] = rot_mat[2];
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createZeroState<PrecisionT>(num_qubits);
        GateImplementation::applyRot(st.data(), num_qubits, {index}, false,
                                     angles[index][0], angles[index][1],
                                     angles[index][2]);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(Rot);

/*******************************************************************************
 * Two-qubit gates
 ******************************************************************************/
template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingXX() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |000> -> a|000> + b|110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |100> -> a|100> + b|010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |010> -> a|010> + b|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |110> -> a|110> + b|000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, -sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,2 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = TestVector<ComplexPrecisionT>{
            {
                ComplexPrecisionT{0.125681356503, 0.252712197380},
                ComplexPrecisionT{0.262591068130, 0.370189000494},
                ComplexPrecisionT{0.129300299863, 0.371057794075},
                ComplexPrecisionT{0.392248682814, 0.195795523118},
                ComplexPrecisionT{0.303908059240, 0.082981563244},
                ComplexPrecisionT{0.189140284321, 0.179512645957},
                ComplexPrecisionT{0.173146612336, 0.092249594834},
                ComplexPrecisionT{0.298857179897, 0.269627836165},
            },
            getBestAllocator<ComplexPrecisionT>()};
        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexPrecisionT> expected{
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
        GateImplementation::applyIsingXX(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingXX);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingXY() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |000> -> a|000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{1.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |100> -> a|100> + b|010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |010> -> a|010> + b|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |110> -> a|110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{1.0, 0.0}, ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
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

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.267462849617, 0.010768564418},
            ComplexPrecisionT{0.228575125337, 0.010564590804},
            ComplexPrecisionT{0.099492751062, 0.260849833488},
            ComplexPrecisionT{0.093690201640, 0.189847111702},
            ComplexPrecisionT{0.015641822883, 0.225092900621},
            ComplexPrecisionT{0.205574608177, 0.082808663337},
            ComplexPrecisionT{0.006827173322, 0.211631480575},
            ComplexPrecisionT{0.255280800811, 0.161572331669},
            ComplexPrecisionT{0.119218164572, 0.115460377284},
            ComplexPrecisionT{-0.000315789761, 0.153835664378},
            ComplexPrecisionT{0.206786872079, 0.157633689097},
            ComplexPrecisionT{0.093027614553, 0.271012980118},
            ComplexPrecisionT{0.041495874524, 0.065934829414},
            ComplexPrecisionT{0.089653238654, 0.221581339836},
            ComplexPrecisionT{0.217892318964, 0.291261285543},
            ComplexPrecisionT{0.292993247509, 0.186570793390},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingXY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingYY() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |000> -> a|000> + b|110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |100> -> a|100> + b|010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |010> -> a|010> + b|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, -sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |110> -> a|110> + b|000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        const auto ini_st = TestVector<ComplexPrecisionT>{
            {ComplexPrecisionT{0.276522701942, 0.192601873155},
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
             ComplexPrecisionT{0.177173454285, 0.267447421480}},
            getBestAllocator<ComplexPrecisionT>()};

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
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
        GateImplementation::applyIsingYY(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingYY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingZZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |000> -> |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{cos(angle / 2), -sin(angle / 2)},
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
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |100> -> |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |010> -> |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |110> -> |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{cos(angle / 2), -sin(angle / 2)},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        TestVector<ComplexPrecisionT> ini_st{
            {ComplexPrecisionT{0.267462841882, 0.010768564798},
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
             ComplexPrecisionT{0.292993251871, 0.186570798697}},
            getBestAllocator<ComplexPrecisionT>()};

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
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
        GateImplementation::applyIsingZZ(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingZZ);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyControlledPhaseShift() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    const size_t num_qubits = 3;

    // Test using |+++> state
    auto ini_st = createPlusState<PrecisionT>(num_qubits);

    const auto isqrt2 = Util::INVSQRT2<PrecisionT>();

    const std::vector<PrecisionT> angles{0.3, 2.4};
    const ComplexPrecisionT coef{isqrt2 / PrecisionT{2.0}, PrecisionT{0.0}};

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
                                                  false, angles[0]);
    CAPTURE(st);
    CHECK(st == approx(expected_results[0]));
}
PENNYLANE_RUN_TEST(ControlledPhaseShift);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRX() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRX0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.188018120185, 0.267344585187},
            ComplexPrecisionT{0.172684792903, 0.187465336044},
            ComplexPrecisionT{0.218892658302, 0.241508557821},
            ComplexPrecisionT{0.107094509452, 0.233123916768},
            ComplexPrecisionT{0.144398681914, 0.102112687699},
            ComplexPrecisionT{0.266641428689, 0.096286886834},
            ComplexPrecisionT{0.037126289559, 0.047222166486},
            ComplexPrecisionT{0.136865047634, 0.203178369592},
            ComplexPrecisionT{0.001562711889, 0.224933454573},
            ComplexPrecisionT{0.009933412610, 0.080866505038},
            ComplexPrecisionT{0.000948295069, 0.280652963863},
            ComplexPrecisionT{0.109817299553, 0.150776413412},
            ComplexPrecisionT{0.297480913626, 0.232588348025},
            ComplexPrecisionT{0.247386444054, 0.077608200535},
            ComplexPrecisionT{0.192650977126, 0.054764192471},
            ComplexPrecisionT{0.033093927690, 0.243038790593},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.188018120185, 0.267344585187},
            ComplexPrecisionT{0.172684792903, 0.187465336044},
            ComplexPrecisionT{0.218892658302, 0.241508557821},
            ComplexPrecisionT{0.107094509452, 0.233123916768},
            ComplexPrecisionT{0.144398681914, 0.102112687699},
            ComplexPrecisionT{0.266641428689, 0.096286886834},
            ComplexPrecisionT{0.037126289559, 0.047222166486},
            ComplexPrecisionT{0.136865047634, 0.203178369592},
            ComplexPrecisionT{0.037680529583, 0.175982985869},
            ComplexPrecisionT{0.021870621269, 0.041448569986},
            ComplexPrecisionT{0.009445384485, 0.247313095111},
            ComplexPrecisionT{0.146244209335, 0.143803745197},
            ComplexPrecisionT{0.328815969263, 0.229521152393},
            ComplexPrecisionT{0.256946415396, 0.075122442730},
            ComplexPrecisionT{0.233916049255, 0.053951837341},
            ComplexPrecisionT{0.056117891609, 0.223025389250},
        };

        auto st = ini_st;
        GateImplementation::applyCRX(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRX0,2 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.052996853820, 0.268704529517},
            ComplexPrecisionT{0.082642978242, 0.195193762273},
            ComplexPrecisionT{0.275869474800, 0.221416497403},
            ComplexPrecisionT{0.198695648566, 0.006071386515},
            ComplexPrecisionT{0.067983147697, 0.276232498024},
            ComplexPrecisionT{0.136067312263, 0.055703741794},
            ComplexPrecisionT{0.157173013237, 0.279061453647},
            ComplexPrecisionT{0.104219108364, 0.247711145514},
            ComplexPrecisionT{0.176998514444, 0.152305581694},
            ComplexPrecisionT{0.055177054767, 0.009344289143},
            ComplexPrecisionT{0.047003532929, 0.014270464770},
            ComplexPrecisionT{0.067602001658, 0.237978418468},
            ComplexPrecisionT{0.191357285454, 0.247486891611},
            ComplexPrecisionT{0.059014417923, 0.240820754268},
            ComplexPrecisionT{0.017675906958, 0.280795663824},
            ComplexPrecisionT{0.149294381068, 0.236647612943},
        };

        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.052996853820, 0.268704529517},
            ComplexPrecisionT{0.082642978242, 0.195193762273},
            ComplexPrecisionT{0.275869474800, 0.221416497403},
            ComplexPrecisionT{0.198695648566, 0.006071386515},
            ComplexPrecisionT{0.067983147697, 0.276232498024},
            ComplexPrecisionT{0.136067312263, 0.055703741794},
            ComplexPrecisionT{0.157173013237, 0.279061453647},
            ComplexPrecisionT{0.104219108364, 0.247711145514},
            ComplexPrecisionT{0.177066334766, 0.143153236251},
            ComplexPrecisionT{0.091481259734, -0.001272371824},
            ComplexPrecisionT{0.070096171606, -0.013402737499},
            ComplexPrecisionT{0.068232891172, 0.226515814342},
            ComplexPrecisionT{0.232660238337, 0.241735302419},
            ComplexPrecisionT{0.095065259834, 0.214700810780},
            ComplexPrecisionT{0.055912814010, 0.247655060549},
            ComplexPrecisionT{0.184897295154, 0.224604965678},
        };

        auto st = ini_st;
        GateImplementation::applyCRX(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRX1,3 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.192438300910, 0.082027221475},
            ComplexPrecisionT{0.217147770013, 0.101186506864},
            ComplexPrecisionT{0.172631211937, 0.036301903892},
            ComplexPrecisionT{0.006532319481, 0.086171029910},
            ComplexPrecisionT{0.042291498813, 0.282934641945},
            ComplexPrecisionT{0.231739267944, 0.188873888944},
            ComplexPrecisionT{0.278594048803, 0.306941867941},
            ComplexPrecisionT{0.126901023080, 0.220266540060},
            ComplexPrecisionT{0.229998291616, 0.200076737619},
            ComplexPrecisionT{0.016698938983, 0.160673755090},
            ComplexPrecisionT{0.123754272868, 0.123889666882},
            ComplexPrecisionT{0.128913058161, 0.104905508280},
            ComplexPrecisionT{0.004957334386, 0.000151477546},
            ComplexPrecisionT{0.286109480550, 0.287939421742},
            ComplexPrecisionT{0.180882613126, 0.180408714716},
            ComplexPrecisionT{0.169404192357, 0.128550443286},
        };

        const std::vector<size_t> wires = {1, 3};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.192438300910, 0.082027221475},
            ComplexPrecisionT{0.217147770013, 0.101186506864},
            ComplexPrecisionT{0.172631211937, 0.036301903892},
            ComplexPrecisionT{0.006532319481, 0.086171029910},
            ComplexPrecisionT{0.071122903322, 0.243493995118},
            ComplexPrecisionT{0.272884177375, 0.180009581467},
            ComplexPrecisionT{0.309433364794, 0.283498205063},
            ComplexPrecisionT{0.173048974802, 0.174307158347},
            ComplexPrecisionT{0.229998291616, 0.200076737619},
            ComplexPrecisionT{0.016698938983, 0.160673755090},
            ComplexPrecisionT{0.123754272868, 0.123889666882},
            ComplexPrecisionT{0.128913058161, 0.104905508280},
            ComplexPrecisionT{0.049633717487, -0.044302629247},
            ComplexPrecisionT{0.282658689673, 0.283672663198},
            ComplexPrecisionT{0.198658723032, 0.151897953530},
            ComplexPrecisionT{0.195376806318, 0.098886035231},
        };

        auto st = ini_st;
        GateImplementation::applyCRX(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(CRX);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRY() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRY0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.024509081663, 0.005606762650},
            ComplexPrecisionT{0.261792037054, 0.259257414596},
            ComplexPrecisionT{0.168380715455, 0.096012484887},
            ComplexPrecisionT{0.169761107379, 0.042890935442},
            ComplexPrecisionT{0.012169527484, 0.082631086139},
            ComplexPrecisionT{0.155790166500, 0.292998574950},
            ComplexPrecisionT{0.150529463310, 0.282021216715},
            ComplexPrecisionT{0.097100202708, 0.134938013786},
            ComplexPrecisionT{0.062640753523, 0.251735121160},
            ComplexPrecisionT{0.121654204141, 0.116964600258},
            ComplexPrecisionT{0.152865184550, 0.084800955456},
            ComplexPrecisionT{0.300145205424, 0.101098965771},
            ComplexPrecisionT{0.288274703880, 0.038180155037},
            ComplexPrecisionT{0.041378441702, 0.206525491532},
            ComplexPrecisionT{0.033201995261, 0.096777018650},
            ComplexPrecisionT{0.303210250465, 0.300817738868},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.024509081663, 0.005606762650},
            ComplexPrecisionT{0.261792037054, 0.259257414596},
            ComplexPrecisionT{0.168380715455, 0.096012484887},
            ComplexPrecisionT{0.169761107379, 0.042890935442},
            ComplexPrecisionT{0.012169527484, 0.082631086139},
            ComplexPrecisionT{0.155790166500, 0.292998574950},
            ComplexPrecisionT{0.150529463310, 0.282021216715},
            ComplexPrecisionT{0.097100202708, 0.134938013786},
            ComplexPrecisionT{0.017091411508, 0.242746239557},
            ComplexPrecisionT{0.113748028260, 0.083456799483},
            ComplexPrecisionT{0.145850361424, 0.068735133269},
            ComplexPrecisionT{0.249391258812, 0.053133825802},
            ComplexPrecisionT{0.294506455875, 0.076828111036},
            ComplexPrecisionT{0.059777143539, 0.222190141515},
            ComplexPrecisionT{0.056549175144, 0.108777179774},
            ComplexPrecisionT{0.346161234622, 0.312872353290},
        };

        auto st = ini_st;
        GateImplementation::applyCRY(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRY0,2 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.102619838050, 0.054477528511},
            ComplexPrecisionT{0.202715827962, 0.019268690848},
            ComplexPrecisionT{0.009985085718, 0.046864154650},
            ComplexPrecisionT{0.095353410397, 0.178365407785},
            ComplexPrecisionT{0.265491448756, 0.075474015573},
            ComplexPrecisionT{0.155542525434, 0.336145304405},
            ComplexPrecisionT{0.264473386058, 0.073102790542},
            ComplexPrecisionT{0.275654487087, 0.027356694914},
            ComplexPrecisionT{0.040156237615, 0.323407814320},
            ComplexPrecisionT{0.111584643322, 0.148005654537},
            ComplexPrecisionT{0.143440399478, 0.139829784016},
            ComplexPrecisionT{0.104105862006, 0.036845342185},
            ComplexPrecisionT{0.254859090295, 0.077839069459},
            ComplexPrecisionT{0.166580751989, 0.081673415646},
            ComplexPrecisionT{0.322693919290, 0.244062536913},
            ComplexPrecisionT{0.203101217204, 0.182142660415},
        };

        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.102619838050, 0.054477528511},
            ComplexPrecisionT{0.202715827962, 0.019268690848},
            ComplexPrecisionT{0.009985085718, 0.046864154650},
            ComplexPrecisionT{0.095353410397, 0.178365407785},
            ComplexPrecisionT{0.265491448756, 0.075474015573},
            ComplexPrecisionT{0.155542525434, 0.336145304405},
            ComplexPrecisionT{0.264473386058, 0.073102790542},
            ComplexPrecisionT{0.275654487087, 0.027356694914},
            ComplexPrecisionT{0.017382553849, 0.297755483640},
            ComplexPrecisionT{0.094054909639, 0.140483782705},
            ComplexPrecisionT{0.147937549133, 0.188379019063},
            ComplexPrecisionT{0.120178355382, 0.059393264033},
            ComplexPrecisionT{0.201627929216, 0.038974326513},
            ComplexPrecisionT{0.133002468018, 0.052382480362},
            ComplexPrecisionT{0.358372291916, 0.253192504889},
            ComplexPrecisionT{0.226516213248, 0.192620277535},
        };

        auto st = ini_st;
        GateImplementation::applyCRY(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRY1,3 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.058899496683, 0.031397556785},
            ComplexPrecisionT{0.069961513798, 0.130434904124},
            ComplexPrecisionT{0.217689437802, 0.274984586300},
            ComplexPrecisionT{0.306390652950, 0.298990481245},
            ComplexPrecisionT{0.209944539032, 0.220900665872},
            ComplexPrecisionT{0.003587823096, 0.069341448987},
            ComplexPrecisionT{0.114578641694, 0.136714993752},
            ComplexPrecisionT{0.131460200149, 0.288466810023},
            ComplexPrecisionT{0.153891247725, 0.128222510215},
            ComplexPrecisionT{0.161391493466, 0.264248676428},
            ComplexPrecisionT{0.102366240850, 0.123871730768},
            ComplexPrecisionT{0.094155009506, 0.178235083697},
            ComplexPrecisionT{0.137480035766, 0.038860712805},
            ComplexPrecisionT{0.181542539134, 0.186931324992},
            ComplexPrecisionT{0.130801257167, 0.165524479895},
            ComplexPrecisionT{0.303475658073, 0.099907724058},
        };

        const std::vector<size_t> wires = {1, 3};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.058899496683, 0.031397556785},
            ComplexPrecisionT{0.069961513798, 0.130434904124},
            ComplexPrecisionT{0.217689437802, 0.274984586300},
            ComplexPrecisionT{0.306390652950, 0.298990481245},
            ComplexPrecisionT{0.206837677400, 0.207444748683},
            ComplexPrecisionT{0.036162925095, 0.102820314015},
            ComplexPrecisionT{0.092762561137, 0.090236295654},
            ComplexPrecisionT{0.147665692045, 0.306204998241},
            ComplexPrecisionT{0.153891247725, 0.128222510215},
            ComplexPrecisionT{0.161391493466, 0.264248676428},
            ComplexPrecisionT{0.102366240850, 0.123871730768},
            ComplexPrecisionT{0.094155009506, 0.178235083697},
            ComplexPrecisionT{0.107604661198, 0.009345661471},
            ComplexPrecisionT{0.200698008554, 0.190699066265},
            ComplexPrecisionT{0.082062476397, 0.147991992696},
            ComplexPrecisionT{0.320112783074, 0.124411723198},
        };

        auto st = ini_st;
        GateImplementation::applyCRY(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}

PENNYLANE_RUN_TEST(CRY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRZ0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.264968228755, 0.059389110312},
            ComplexPrecisionT{0.004927738580, 0.117198819444},
            ComplexPrecisionT{0.192517901751, 0.061524928233},
            ComplexPrecisionT{0.285160768924, 0.013212111581},
            ComplexPrecisionT{0.278645646186, 0.212116779981},
            ComplexPrecisionT{0.171786665640, 0.141260537212},
            ComplexPrecisionT{0.199480649113, 0.218261452113},
            ComplexPrecisionT{0.071007710848, 0.294720535623},
            ComplexPrecisionT{0.169589173252, 0.010528306669},
            ComplexPrecisionT{0.061973371011, 0.033143783035},
            ComplexPrecisionT{0.177570977662, 0.116785656786},
            ComplexPrecisionT{0.070266502325, 0.084338553411},
            ComplexPrecisionT{0.053744021753, 0.146932844792},
            ComplexPrecisionT{0.254428637803, 0.138916780809},
            ComplexPrecisionT{0.260354050166, 0.267004004472},
            ComplexPrecisionT{0.008910554792, 0.316282675508},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.264968228755, 0.059389110312},
            ComplexPrecisionT{0.004927738580, 0.117198819444},
            ComplexPrecisionT{0.192517901751, 0.061524928233},
            ComplexPrecisionT{0.285160768924, 0.013212111581},
            ComplexPrecisionT{0.278645646186, 0.212116779981},
            ComplexPrecisionT{0.171786665640, 0.141260537212},
            ComplexPrecisionT{0.199480649113, 0.218261452113},
            ComplexPrecisionT{0.071007710848, 0.294720535623},
            ComplexPrecisionT{0.169165556003, -0.015948278519},
            ComplexPrecisionT{0.066370291483, 0.023112625918},
            ComplexPrecisionT{0.193559430151, 0.087778634862},
            ComplexPrecisionT{0.082516747253, 0.072397233118},
            ComplexPrecisionT{0.030262722499, 0.153498691785},
            ComplexPrecisionT{0.229755796458, 0.176759943762},
            ComplexPrecisionT{0.215708594452, 0.304212379961},
            ComplexPrecisionT{-0.040337866447, 0.313826361773},
        };

        auto st = ini_st;
        GateImplementation::applyCRZ(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRZ0,2 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.148770394604, 0.083378238599},
            ComplexPrecisionT{0.274356796683, 0.083823071640},
            ComplexPrecisionT{0.028016616540, 0.165919229565},
            ComplexPrecisionT{0.123329104424, 0.295826835858},
            ComplexPrecisionT{0.222343815006, 0.093160444663},
            ComplexPrecisionT{0.288857659956, 0.138646598905},
            ComplexPrecisionT{0.199272938656, 0.123099916175},
            ComplexPrecisionT{0.182062963782, 0.098622669183},
            ComplexPrecisionT{0.270467177482, 0.282942493365},
            ComplexPrecisionT{0.147717133688, 0.038580110182},
            ComplexPrecisionT{0.279040367487, 0.114344708857},
            ComplexPrecisionT{0.229917326705, 0.222777886314},
            ComplexPrecisionT{0.047595071834, 0.026542458656},
            ComplexPrecisionT{0.133654136834, 0.275281854777},
            ComplexPrecisionT{0.126723771272, 0.071649311030},
            ComplexPrecisionT{0.040467231551, 0.098358909396},
        };

        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.148770394604, 0.083378238599},
            ComplexPrecisionT{0.274356796683, 0.083823071640},
            ComplexPrecisionT{0.028016616540, 0.165919229565},
            ComplexPrecisionT{0.123329104424, 0.295826835858},
            ComplexPrecisionT{0.222343815006, 0.093160444663},
            ComplexPrecisionT{0.288857659956, 0.138646598905},
            ComplexPrecisionT{0.199272938656, 0.123099916175},
            ComplexPrecisionT{0.182062963782, 0.098622669183},
            ComplexPrecisionT{0.311143020471, 0.237484672050},
            ComplexPrecisionT{0.151917469671, 0.015161098089},
            ComplexPrecisionT{0.257886371956, 0.156310134957},
            ComplexPrecisionT{0.192512799579, 0.255794420869},
            ComplexPrecisionT{0.051140958142, 0.018825391755},
            ComplexPrecisionT{0.174801129192, 0.251173432304},
            ComplexPrecisionT{0.114052908458, 0.090468071985},
            ComplexPrecisionT{0.024693993739, 0.103451817578},
        };

        auto st = ini_st;
        GateImplementation::applyCRZ(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRZ1,3 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.190769680625, 0.287992363388},
            ComplexPrecisionT{0.098068639739, 0.098569855389},
            ComplexPrecisionT{0.037728060139, 0.188330976218},
            ComplexPrecisionT{0.091809561053, 0.200107659880},
            ComplexPrecisionT{0.299856248683, 0.162326250675},
            ComplexPrecisionT{0.064700651300, 0.038667789709},
            ComplexPrecisionT{0.119630787356, 0.257575730461},
            ComplexPrecisionT{0.061392768321, 0.055938727834},
            ComplexPrecisionT{0.052661991695, 0.274401532393},
            ComplexPrecisionT{0.238974614805, 0.213527036406},
            ComplexPrecisionT{0.163750665141, 0.107235582319},
            ComplexPrecisionT{0.260992375359, 0.008326988206},
            ComplexPrecisionT{0.240406501616, 0.032737802983},
            ComplexPrecisionT{0.152754313527, 0.107245249982},
            ComplexPrecisionT{0.162638949527, 0.306372397719},
            ComplexPrecisionT{0.231663044710, 0.107293515032},
        };

        const std::vector<size_t> wires = {1, 3};
        const ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.190769680625, 0.287992363388},
            ComplexPrecisionT{0.098068639739, 0.098569855389},
            ComplexPrecisionT{0.037728060139, 0.188330976218},
            ComplexPrecisionT{0.091809561053, 0.200107659880},
            ComplexPrecisionT{0.321435301661, 0.113766991605},
            ComplexPrecisionT{0.057907230634, 0.048250646420},
            ComplexPrecisionT{0.158197104346, 0.235861099766},
            ComplexPrecisionT{0.051956164721, 0.064797918341},
            ComplexPrecisionT{0.052661991695, 0.274401532393},
            ComplexPrecisionT{0.238974614805, 0.213527036406},
            ComplexPrecisionT{0.163750665141, 0.107235582319},
            ComplexPrecisionT{0.260992375359, 0.008326988206},
            ComplexPrecisionT{0.242573571004, -0.005011228787},
            ComplexPrecisionT{0.134236881868, 0.129676071390},
            ComplexPrecisionT{0.208264445871, 0.277383118761},
            ComplexPrecisionT{0.212179898392, 0.141983644728},
        };

        auto st = ini_st;
        GateImplementation::applyCRZ(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(CRZ);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRot() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    const std::vector<PrecisionT> angles{0.3, 0.8, 2.4};

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRot0,1 |000> -> |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);

        auto st = createZeroState<PrecisionT>(num_qubits);
        GateImplementation::applyCRot(st.data(), num_qubits, {0, 1}, false,
                                      angles[0], angles[1], angles[2]);

        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRot0,1 |100> -> |1>(a|0>+b|1>)|0> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;

        auto st = createZeroState<PrecisionT>(num_qubits);

        std::vector<ComplexPrecisionT> expected_results(8);
        const auto rot_mat =
            Gates::getRot<PrecisionT>(angles[0], angles[1], angles[2]);
        expected_results[size_t{1U} << (num_qubits - 1)] = rot_mat[0];
        expected_results[(size_t{1U} << num_qubits) - 2] = rot_mat[2];

        GateImplementation::applyPauliX(st.data(), num_qubits, {0}, false);

        GateImplementation::applyCRot(st.data(), num_qubits, {0, 1}, false,
                                      angles[0], angles[1], angles[2]);

        CHECK(st == approx(expected_results));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRot0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.234734234199, 0.088957328814},
            ComplexPrecisionT{0.065109443398, 0.284054307559},
            ComplexPrecisionT{0.272603451516, 0.101758170511},
            ComplexPrecisionT{0.049922391489, 0.280849666080},
            ComplexPrecisionT{0.012676439023, 0.283581988298},
            ComplexPrecisionT{0.074837215146, 0.119865583718},
            ComplexPrecisionT{0.220666349215, 0.083019197512},
            ComplexPrecisionT{0.228645004012, 0.109144153614},
            ComplexPrecisionT{0.186515011731, 0.009044330588},
            ComplexPrecisionT{0.268705684298, 0.278878779206},
            ComplexPrecisionT{0.007225255939, 0.104466710409},
            ComplexPrecisionT{0.092186772555, 0.167323294042},
            ComplexPrecisionT{0.198642540305, 0.317101356672},
            ComplexPrecisionT{0.061416756317, 0.014463767792},
            ComplexPrecisionT{0.109767506116, 0.244842265274},
            ComplexPrecisionT{0.044108879936, 0.124327196075},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT phi = 0.128;
        const ParamT theta = -0.563;
        const ParamT omega = 1.414;

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.234734234199, 0.088957328814},
            ComplexPrecisionT{0.065109443398, 0.284054307559},
            ComplexPrecisionT{0.272603451516, 0.101758170511},
            ComplexPrecisionT{0.049922391489, 0.280849666080},
            ComplexPrecisionT{0.012676439023, 0.283581988298},
            ComplexPrecisionT{0.074837215146, 0.119865583718},
            ComplexPrecisionT{0.220666349215, 0.083019197512},
            ComplexPrecisionT{0.228645004012, 0.109144153614},
            ComplexPrecisionT{0.231541411002, -0.081215269214},
            ComplexPrecisionT{0.387885772871, 0.005250582985},
            ComplexPrecisionT{0.140096879751, 0.103289147066},
            ComplexPrecisionT{0.206040689190, 0.073864544104},
            ComplexPrecisionT{-0.115373527531, 0.318376165756},
            ComplexPrecisionT{0.019345803102, -0.055678858513},
            ComplexPrecisionT{-0.072480957773, 0.217744954736},
            ComplexPrecisionT{-0.045461901445, 0.062632338099},
        };

        auto st = ini_st;
        GateImplementation::applyCRot(st.data(), num_qubits, wires, false, phi,
                                      theta, omega);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(CRot);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplySingleExcitation() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;
        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},           ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{-0.1553680335, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, 0.0},  ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},           ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},          ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.1553680335, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},          ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        std::vector<ComplexPrecisionT> ini_st{
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
        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.125681, 0.252712},
            ComplexPrecisionT{0.219798, 0.355848},
            ComplexPrecisionT{0.1293, 0.371058},
            ComplexPrecisionT{0.365709, 0.181773},
            ComplexPrecisionT{0.336159, 0.131522},
            ComplexPrecisionT{0.18914, 0.179513},
            ComplexPrecisionT{0.223821, 0.117493},
            ComplexPrecisionT{0.298857, 0.269628},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, wires,
                                                  false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(SingleExcitation);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplySingleExcitationMinus() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.9878566567, -0.1553680335},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},           ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{-0.1553680335, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, 0.0},  ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},           ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},          ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.1553680335, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},          ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, -0.1553680335},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        std::vector<ComplexPrecisionT> ini_st{
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
        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.158204, 0.233733},
            ComplexPrecisionT{0.219798, 0.355848},
            ComplexPrecisionT{0.177544, 0.350543},
            ComplexPrecisionT{0.365709, 0.181773},
            ComplexPrecisionT{0.336159, 0.131522},
            ComplexPrecisionT{0.211353, 0.152737},
            ComplexPrecisionT{0.223821, 0.117493},
            ComplexPrecisionT{0.33209, 0.227445}};

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(SingleExcitationMinus);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplySingleExcitationPlus() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.9878566567, 0.1553680335},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},           ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{-0.1553680335, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, 0.0},  ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},           ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},          ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.1553680335, 0.0}, ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},          ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexPrecisionT> expected_results{
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.0, 0.0},
            ComplexPrecisionT{0.9878566567, 0.1553680335},
            ComplexPrecisionT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        std::vector<ComplexPrecisionT> ini_st{
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
        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.090922, 0.267194},
            ComplexPrecisionT{0.219798, 0.355848},
            ComplexPrecisionT{0.0787548, 0.384968},
            ComplexPrecisionT{0.365709, 0.181773},
            ComplexPrecisionT{0.336159, 0.131522},
            ComplexPrecisionT{0.16356, 0.203093},
            ComplexPrecisionT{0.223821, 0.117493},
            ComplexPrecisionT{0.260305, 0.307012}};

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(SingleExcitationPlus);

/*******************************************************************************
 * Four-qubit gates
 ******************************************************************************/
template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyDoubleExcitation() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 |0000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;
        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits,
                                                  {0, 1, 2, 3}, false, angle);
        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 |1100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("1100");
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[3] = ComplexPrecisionT{-0.1553680335, 0};
        expected_results[12] = ComplexPrecisionT{0.9878566566949545, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits,
                                                  {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 |0011> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("0011");
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[3] = ComplexPrecisionT{0.9878566566949545, 0};
        expected_results[12] = ComplexPrecisionT{0.15536803346720587, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits,
                                                  {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.125681356503, 0.252712197380},
            ComplexPrecisionT{0.262591068130, 0.370189000494},
            ComplexPrecisionT{0.129300299863, 0.371057794075},
            ComplexPrecisionT{0.392248682814, 0.195795523118},
            ComplexPrecisionT{0.303908059240, 0.082981563244},
            ComplexPrecisionT{0.189140284321, 0.179512645957},
            ComplexPrecisionT{0.173146612336, 0.092249594834},
            ComplexPrecisionT{0.298857179897, 0.269627836165},
            ComplexPrecisionT{0.125681356503, 0.252712197380},
            ComplexPrecisionT{0.262591068130, 0.370189000494},
            ComplexPrecisionT{0.129300299863, 0.371057794075},
            ComplexPrecisionT{0.392248682814, 0.195795523118},
            ComplexPrecisionT{0.303908059240, 0.082981563244},
            ComplexPrecisionT{0.189140284321, 0.179512645957},
            ComplexPrecisionT{0.173146612336, 0.092249594834},
            ComplexPrecisionT{0.298857179897, 0.269627836165},
        };
        const std::vector<size_t> wires = {0, 1, 2, 3};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.125681, 0.252712},
            ComplexPrecisionT{0.262591, 0.370189},
            ComplexPrecisionT{0.1293, 0.371058},
            ComplexPrecisionT{0.348302, 0.183007},
            ComplexPrecisionT{0.303908, 0.0829816},
            ComplexPrecisionT{0.18914, 0.179513},
            ComplexPrecisionT{0.173147, 0.0922496},
            ComplexPrecisionT{0.298857, 0.269628},
            ComplexPrecisionT{0.125681, 0.252712},
            ComplexPrecisionT{0.262591, 0.370189},
            ComplexPrecisionT{0.1293, 0.371058},
            ComplexPrecisionT{0.392249, 0.195796},
            ComplexPrecisionT{0.353419, 0.108307},
            ComplexPrecisionT{0.18914, 0.179513},
            ComplexPrecisionT{0.173147, 0.0922496},
            ComplexPrecisionT{0.298857, 0.269628},
        };

        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits, wires,
                                                  false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(DoubleExcitation);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyDoubleExcitationMinus() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 |0000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[0] =
            ComplexPrecisionT{0.9878566566949545, -0.15536803346720587};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 |1100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("1100");
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[3] = ComplexPrecisionT{-0.1553680335, 0};
        expected_results[12] = ComplexPrecisionT{0.9878566566949545, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 |0011> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("0011");
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[3] = ComplexPrecisionT{0.9878566566949545, 0};
        expected_results[12] = ComplexPrecisionT{0.15536803346720587, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.125681356503, 0.252712197380},
            ComplexPrecisionT{0.262591068130, 0.370189000494},
            ComplexPrecisionT{0.129300299863, 0.371057794075},
            ComplexPrecisionT{0.392248682814, 0.195795523118},
            ComplexPrecisionT{0.303908059240, 0.082981563244},
            ComplexPrecisionT{0.189140284321, 0.179512645957},
            ComplexPrecisionT{0.173146612336, 0.092249594834},
            ComplexPrecisionT{0.298857179897, 0.269627836165},
            ComplexPrecisionT{0.125681356503, 0.252712197380},
            ComplexPrecisionT{0.262591068130, 0.370189000494},
            ComplexPrecisionT{0.129300299863, 0.371057794075},
            ComplexPrecisionT{0.392248682814, 0.195795523118},
            ComplexPrecisionT{0.303908059240, 0.082981563244},
            ComplexPrecisionT{0.189140284321, 0.179512645957},
            ComplexPrecisionT{0.173146612336, 0.092249594834},
            ComplexPrecisionT{0.298857179897, 0.269627836165},
        };
        const std::vector<size_t> wires = {0, 1, 2, 3};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.158204, 0.233733},
            ComplexPrecisionT{0.309533, 0.331939},
            ComplexPrecisionT{0.177544, 0.350543},
            ComplexPrecisionT{0.348302, 0.183007},
            ComplexPrecisionT{0.31225, 0.0417871},
            ComplexPrecisionT{0.211353, 0.152737},
            ComplexPrecisionT{0.183886, 0.0683795},
            ComplexPrecisionT{0.33209, 0.227445},
            ComplexPrecisionT{0.158204, 0.233733},
            ComplexPrecisionT{0.309533, 0.331939},
            ComplexPrecisionT{0.177544, 0.350543},
            ComplexPrecisionT{0.414822, 0.141837},
            ComplexPrecisionT{0.353419, 0.108307},
            ComplexPrecisionT{0.211353, 0.152737},
            ComplexPrecisionT{0.183886, 0.0683795},
            ComplexPrecisionT{0.33209, 0.227445},
        };

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(st.data(), num_qubits,
                                                       wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(DoubleExcitationMinus);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyDoubleExcitationPlus() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 |0000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createZeroState<PrecisionT>(num_qubits);
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[0] =
            ComplexPrecisionT{0.9878566566949545, 0.15536803346720587};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 |1100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("1100");
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[3] = ComplexPrecisionT{-0.1553680335, 0};
        expected_results[12] = ComplexPrecisionT{0.9878566566949545, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 |0011> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("0011");
        ParamT angle = 0.312;

        std::vector<ComplexPrecisionT> expected_results(16,
                                                        ComplexPrecisionT{});
        expected_results[3] = ComplexPrecisionT{0.9878566566949545, 0};
        expected_results[12] = ComplexPrecisionT{0.15536803346720587, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.125681356503, 0.252712197380},
            ComplexPrecisionT{0.262591068130, 0.370189000494},
            ComplexPrecisionT{0.129300299863, 0.371057794075},
            ComplexPrecisionT{0.392248682814, 0.195795523118},
            ComplexPrecisionT{0.303908059240, 0.082981563244},
            ComplexPrecisionT{0.189140284321, 0.179512645957},
            ComplexPrecisionT{0.173146612336, 0.092249594834},
            ComplexPrecisionT{0.298857179897, 0.269627836165},
            ComplexPrecisionT{0.125681356503, 0.252712197380},
            ComplexPrecisionT{0.262591068130, 0.370189000494},
            ComplexPrecisionT{0.129300299863, 0.371057794075},
            ComplexPrecisionT{0.392248682814, 0.195795523118},
            ComplexPrecisionT{0.303908059240, 0.082981563244},
            ComplexPrecisionT{0.189140284321, 0.179512645957},
            ComplexPrecisionT{0.173146612336, 0.092249594834},
            ComplexPrecisionT{0.298857179897, 0.269627836165},
        };
        const std::vector<size_t> wires = {0, 1, 2, 3};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.090922, 0.267194},
            ComplexPrecisionT{0.210975, 0.40185},
            ComplexPrecisionT{0.0787548, 0.384968},
            ComplexPrecisionT{0.348302, 0.183007},
            ComplexPrecisionT{0.290157, 0.122699},
            ComplexPrecisionT{0.16356, 0.203093},
            ComplexPrecisionT{0.159325, 0.114478},
            ComplexPrecisionT{0.260305, 0.307012},
            ComplexPrecisionT{0.090922, 0.267194},
            ComplexPrecisionT{0.210975, 0.40185},
            ComplexPrecisionT{0.0787548, 0.384968},
            ComplexPrecisionT{0.362694, 0.246269},
            ComplexPrecisionT{0.353419, 0.108307},
            ComplexPrecisionT{0.16356, 0.203093},
            ComplexPrecisionT{0.159325, 0.114478},
            ComplexPrecisionT{0.260305, 0.307012},
        };

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(st.data(), num_qubits,
                                                      wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(DoubleExcitationPlus);

/*******************************************************************************
 * Multi-qubit gates
 ******************************************************************************/
template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyMultiRZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ0 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

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

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ0 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

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

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ01 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

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

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ012 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

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

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ0123 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

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

        REQUIRE(st == approx(expected).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ013 - "
                    << PrecisionToName<PrecisionT>::value) {
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
        REQUIRE(st == approx(expected).margin(1e-7));
    }
}
PENNYLANE_RUN_TEST(MultiRZ);
