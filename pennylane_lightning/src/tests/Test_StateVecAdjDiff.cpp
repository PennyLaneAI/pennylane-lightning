#include "AdjointDiff.hpp"
#include "AlgUtil.hpp"
#include "Constant.hpp"
#include "GateOperation.hpp"
#include "StateVecAdjDiff.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

#include <cmath>
#include <numbers>

using namespace Pennylane;
using namespace Pennylane::Util;
using namespace Pennylane::Algorithms;
using namespace Pennylane::Simulators;

/**
 * @brief
 *
 * @param length Size of the gate sequence
 * @param
 */
template <class PrecisionT, class RandomEngine>
auto createRandomOps(RandomEngine &re, size_t length, size_t wires)
    -> OpsData<PrecisionT> {
    using namespace Pennylane::Gates;

    std::array gates_to_use = {GateOperation::RX, GateOperation::RY,
                               GateOperation::RZ};

    std::vector<std::string> ops_names;
    std::vector<std::vector<PrecisionT>> ops_params;
    std::vector<std::vector<size_t>> ops_wires;
    std::vector<bool> ops_inverses;

    std::uniform_int_distribution<size_t> gate_dist(0, gates_to_use.size() - 1);
    std::uniform_real_distribution<PrecisionT> param_dist(0.0, 2 * M_PI);
    std::uniform_int_distribution<int> inverse_dist(0, 1);

    for (size_t i = 0; i < length; i++) {
        const auto gate_op = gates_to_use[gate_dist(re)];
        const auto gate_name =
            Util::lookup(Gates::Constant::gate_names, gate_op);
        ops_names.emplace_back(gate_name);
        ops_params.emplace_back(std::vector<PrecisionT>{param_dist(re)});
        ops_inverses.emplace_back(inverse_dist(re));
        ops_wires.emplace_back(createWires(gate_op, wires));
    }

    return {ops_names, ops_params, ops_wires, ops_inverses, {{}}};
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEMPLATE_TEST_CASE("StateVector VJP", "[Test_StateVecAdjDiff]", float, double) {
    using std::cos;
    using std::sin;
    using std::sqrt;

    using PrecisionT = TestType;
    using ComplexPrecisionT = std::complex<PrecisionT>;

    constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();

    SECTION("Do nothing if the tape does not have trainable parameters") {
        std::vector<ComplexPrecisionT> vjp(1);
        OpsData<TestType> ops_data{
            {"CNOT", "RX"},   // names
            {{}, {M_PI / 7}}, // params
            {{0, 1}, {1}},    // wires
            {false, false},   // inverses
            {}                // matrices
        };

        auto dy = std::vector<ComplexPrecisionT>(4);
        std::vector<ComplexPrecisionT> ini_st{
            {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};
        JacobianData<TestType> jd{1, 4, ini_st.data(), {}, ops_data, {}};
        REQUIRE_NOTHROW(statevectorVJP(
            std::span{vjp}, jd, std::span<const ComplexPrecisionT>{dy}, true));
    }

    SECTION("CNOT RX1") {
        const PrecisionT theta = std::numbers::pi_v<PrecisionT> / 7;
        OpsData<TestType> ops_data{
            {"CNOT", "RX"}, // names
            {{}, {theta}},  // params
            {{0, 1}, {1}},  // wires
            {false, false}, // inverses
            {}              // matrices
        };

        auto dy = std::vector<ComplexPrecisionT>(4);

        std::vector<std::vector<ComplexPrecisionT>> expected = {
            {{-isqrt2 / PrecisionT{2.0} * sin(theta / 2), 0.0}},
            {{0.0, -isqrt2 / PrecisionT{2.0} * cos(theta / 2)}},
            {{0.0, -isqrt2 / PrecisionT{2.0} * cos(theta / 2)}},
            {{-isqrt2 / PrecisionT{2.0} * sin(theta / 2), 0.0}},
        };

        SECTION("with apply_operations = true") {
            std::vector<ComplexPrecisionT> ini_st{
                {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};
            JacobianData<TestType> jd{1, 4, ini_st.data(), {}, ops_data, {0}};

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(), ComplexPrecisionT{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexPrecisionT> vjp(1);
                statevectorVJP(std::span{vjp}, jd,
                               std::span<const ComplexPrecisionT>{dy}, true);

                REQUIRE(vjp == approx(expected[i]).margin(1e-5));
            }
        }

        SECTION("with apply_operations = false") {
            std::vector<std::complex<TestType>> final_st{
                {cos(theta / 2) * isqrt2, 0.0},
                {0.0, -isqrt2 * sin(theta / 2)},
                {0.0, -isqrt2 * sin(theta / 2)},
                {cos(theta / 2) * isqrt2, 0.0}};
            JacobianData<TestType> jd{1, 4, final_st.data(), {}, ops_data, {0}};

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(),
                          std::complex<TestType>{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexPrecisionT> vjp(1);
                statevectorVJP(std::span{vjp}, jd,
                               std::span<const ComplexPrecisionT>{dy}, false);

                REQUIRE(vjp == approx(expected[i]).margin(1e-5));
            }
        }
    }

    SECTION("CNOT0,1 RX1 CNOT1,0 RX0 CNOT0,1 RX1 CNOT1,0 RX0") {
        std::vector<std::complex<TestType>> ini_st{
            {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

        OpsData<TestType> ops_data{
            {"CNOT", "RX", "CNOT", "RX", "CNOT", "RX", "CNOT", "RX"}, // names
            {{}, {M_PI}, {}, {M_PI}, {}, {M_PI}, {}, {M_PI}},         // params
            {{0, 1}, {1}, {1, 0}, {0}, {0, 1}, {1}, {1, 0}, {0}},     // wires
            {false, false, false, false, false, false, false,
             false}, // inverses
            {}       // matrices
        };

        std::vector<std::complex<TestType>> expected_der0 = {
            {0.0, -isqrt2 / 2.0},
            {0.0, 0.0},
            {0.0, 0.0},
            {0.0, -isqrt2 / 2.0},
        }; // For trainable_param == 0
        std::vector<std::complex<TestType>> expected_der1 = {
            {0.0, 0.0},
            {0.0, -isqrt2 / 2.0},
            {0.0, -isqrt2 / 2.0},
            {0.0, 0.0},
        }; // For trainable_param == 1

        SECTION("with apply_operations = true") {
            std::vector<std::complex<TestType>> ini_st{
                {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

            JacobianData<TestType> jd{
                1, 4, ini_st.data(), {}, ops_data, {1, 2} // trainable params
            };

            auto dy = std::vector<std::complex<TestType>>(4);

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(),
                          std::complex<TestType>{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexPrecisionT> vjp(2);
                statevectorVJP(std::span{vjp}, jd,
                               std::span<const ComplexPrecisionT>{dy}, true);

                REQUIRE(vjp[0] == approx(expected_der0[i]).margin(1e-5));
                REQUIRE(vjp[1] == approx(expected_der1[i]).margin(1e-5));
            }
        }

        SECTION("with apply_operations = false") {
            std::vector<std::complex<TestType>> final_st{
                {0.0, 0.0}, {isqrt2, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

            JacobianData<TestType> jd{
                4, 4, final_st.data(), {}, ops_data, {1, 2} // trainable params
            };

            auto dy = std::vector<std::complex<TestType>>(4);

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(),
                          std::complex<TestType>{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexPrecisionT> vjp(2);
                statevectorVJP(std::span{vjp}, jd,
                               std::span<const ComplexPrecisionT>{dy}, false);

                REQUIRE(vjp[0] == approx(expected_der0[i]).margin(1e-5));
                REQUIRE(vjp[1] == approx(expected_der1[i]).margin(1e-5));
            }
        }
    }

    SECTION("Test complex dy") {
        OpsData<TestType> ops_data1{
            {"CNOT", "RX"},   // names
            {{}, {M_PI / 7}}, // params
            {{0, 1}, {1}},    // wires
            {false, false},   // inverses
            {}                // matrices
        };

        auto dy1 = std::vector<ComplexPrecisionT>{
            {0.4, 0.4}, {0.4, 0.4}, {0.4, 0.4}, {0.4, 0.4}};

        OpsData<TestType> ops_data2{
            {"CNOT", "RX"},    // names
            {{}, {-M_PI / 7}}, // params
            {{0, 1}, {1}},     // wires
            {false, false},    // inverses
            {}                 // matrices
        };

        auto dy2 = std::vector<ComplexPrecisionT>{
            {0.4, -0.4}, {0.4, -0.4}, {0.4, -0.4}, {0.4, -0.4}};
        std::vector<ComplexPrecisionT> ini_st{
            {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

        JacobianData<TestType> jd1{1, 4, ini_st.data(), {}, ops_data1, {0}};
        JacobianData<TestType> jd2{1, 4, ini_st.data(), {}, ops_data2, {0}};

        std::vector<ComplexPrecisionT> vjp1(1);
        std::vector<ComplexPrecisionT> vjp2(1);

        statevectorVJP(std::span{vjp1}, jd1,
                       std::span<const ComplexPrecisionT>{dy1}, true);

        statevectorVJP(std::span{vjp2}, jd2,
                       std::span<const ComplexPrecisionT>{dy2}, true);

        REQUIRE(vjp1[0] == approx(-std::conj(vjp2[0])));
    }

    SECTION(
        "Check the result is consistent with adjoint diff with observables") {
        std::mt19937 re{1337};
        auto ops_data = createRandomOps<TestType>(re, 10, 3);
        auto obs = std::make_shared<NamedObs<TestType>>("PauliZ",
                                                        std::vector<size_t>{0});

        const size_t num_params = [&]() {
            size_t r = 0;
            for (const auto &ops_params : ops_data.getOpsParams()) {
                if (!ops_params.empty()) {
                    ++r;
                }
            }
            return r;
        }();

        std::vector<size_t> trainable_params(num_params);
        std::iota(trainable_params.begin(), trainable_params.end(), 0);

        const auto ini_st = createProductState<TestType>("+++");

        StateVectorManagedCPU<TestType> sv(ini_st.data(), ini_st.size());
        applyOperations(sv, ops_data);
        JacobianData<TestType> jd{
            num_params, 8,        sv.getDataVector().data(),
            {obs},      ops_data, trainable_params};

        auto o_sv = sv;
        applyObservable(o_sv, *obs);

        std::vector<TestType> grad_vjp = [&]() {
            std::vector<ComplexPrecisionT> vjp(num_params);
            statevectorVJP(
                std::span{vjp}, jd,
                std::span<const ComplexPrecisionT>{o_sv.getDataVector()},
                false);
            std::vector<TestType> res(vjp.size());
            std::transform(vjp.begin(), vjp.end(), res.begin(),
                           [](const auto &x) { return 2 * std::real(x); });
            return res;
        }();

        std::vector<TestType> jac(num_params);
        adjointJacobian<TestType>(std::span{jac}, jd);

        REQUIRE(grad_vjp == approx(jac).margin(1e-5));
    }
}
