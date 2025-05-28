// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch.hpp>

#include "Gates.hpp" // getHadamard
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "TestHelpersWires.hpp"
#include "Util.hpp"

/**
 * @file
 *  Tests for non-parametric gates functionality defined in the class
 * StateVectorKokkos.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Gates;
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkos::CopyConstructor",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        const std::size_t num_qubits = 3;
        StateVectorKokkos<TestType> kokkos_sv_1{num_qubits};
        kokkos_sv_1.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                    {{0}, {1}, {2}},
                                    {{false}, {false}, {false}});
        StateVectorKokkos<TestType> kokkos_sv_2{kokkos_sv_1};

        CHECK(kokkos_sv_1.getLength() == kokkos_sv_2.getLength());
        CHECK(kokkos_sv_1.getNumQubits() == kokkos_sv_2.getNumQubits());

        std::vector<Kokkos::complex<TestType>> kokkos_sv_1_host(
            kokkos_sv_1.getLength());
        std::vector<Kokkos::complex<TestType>> kokkos_sv_2_host(
            kokkos_sv_2.getLength());
        kokkos_sv_1.DeviceToHost(kokkos_sv_1_host.data(),
                                 kokkos_sv_1.getLength());
        kokkos_sv_2.DeviceToHost(kokkos_sv_2_host.data(),
                                 kokkos_sv_2.getLength());

        for (std::size_t i = 0; i < kokkos_sv_1_host.size(); i++) {
            CHECK(kokkos_sv_1_host[i] == kokkos_sv_2_host[i]);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyOperation",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        const std::size_t num_qubits = 3;
        StateVectorKokkos<TestType> state_vector{num_qubits};
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("XXX", {0}), LightningException,
            "Operation does not exist for XXX and no matrix provided.");
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("XXX", {0}, {true}, {1}),
            LightningException,
            "Operation does not exist for XXX and no matrix provided.");
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyMatrix/Param-Operation",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using StateVectorT = StateVectorKokkos<TestType>;
    using PrecisionT = StateVectorT::PrecisionT;

    const std::size_t num_qubits = 4;
    const TestType EP = 1e-4;
    auto ini_st = createNonTrivialState<StateVectorT>(num_qubits);

    std::unordered_map<std::string, GateOperation> str_to_gates_{};
    for (const auto &[gate_op, gate_name] : Constant::gate_names) {
        str_to_gates_.emplace(gate_name, gate_op);
    }

    const bool inverse = GENERATE(false, true);
    const std::string gate_name =
        GENERATE("Identity", "PauliX", "PauliY", "PauliZ", "Hadamard", "S",
                 "SX", "T", "CNOT", "SWAP", "CY", "CZ", "CSWAP", "Toffoli");
    DYNAMIC_SECTION("Matrix - Gate = " << gate_name
                                       << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, inverse);

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires = createWires(str_to_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", wires, false, {}, gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorKokkos::applyCY",
                           "[StateVectorKokkos_Nonparam]", (StateVectorKokkos),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    const bool inverse = GENERATE(true, false);

    SECTION("Apply::CY") {
        // Defining the statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT sv1(statevector_data.data(), statevector_data.size());
        StateVectorT sv2(statevector_data.data(), statevector_data.size());

        const std::vector<std::size_t> wires{0, 1};
        std::vector<ComplexT> matrix = getCY<Kokkos::complex, PrecisionT>();
        sv1.applyOperation("CY", wires, inverse);
        sv2.applyMatrix(matrix, wires, inverse);

        auto result1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           sv1.getView());
        auto result2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           sv2.getView());

        for (std::size_t j = 0; j < sv1.getView().size(); j++) {
            CHECK(imag(result1[j]) == Approx(imag(result2[j])));
            CHECK(real(result1[j]) == Approx(real(result2[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyHadamard",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        const std::size_t num_qubits = 3;
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv(num_qubits);
                kokkos_sv.applyOperation("Hadamard", {index}, inverse);
                Kokkos::complex<TestType> expected(1.0 / std::sqrt(2), 0);
                auto result_subview = Kokkos::subview(kokkos_sv.getView(), 0);
                Kokkos::complex<TestType> result;
                Kokkos::deep_copy(result, result_subview);
                CHECK(expected.real() == Approx(result.real()));
            }
        }
        SECTION("Apply using matrix") {
            using ComplexT = StateVectorKokkos<TestType>::ComplexT;
            const auto isqrt2 = ComplexT{INVSQRT2<TestType>()};
            const std::vector<ComplexT> matrix = {isqrt2, isqrt2, isqrt2,
                                                  -isqrt2};
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv(num_qubits);
                kokkos_sv.applyOperation("Hadamard", {index}, inverse, {},
                                         matrix);
                Kokkos::complex<TestType> expected(1.0 / std::sqrt(2), 0);
                auto result_subview = Kokkos::subview(kokkos_sv.getView(), 0);
                Kokkos::complex<TestType> result;
                Kokkos::deep_copy(result, result_subview);
                CHECK(expected.real() == Approx(result.real()));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliX",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("PauliX", {index}, false);
                auto result_subview_0 = Kokkos::subview(kokkos_sv.getView(), 0);
                auto result_subview_1 = Kokkos::subview(
                    kokkos_sv.getView(),
                    0b1 << (kokkos_sv.getNumQubits() - index - 1));
                Kokkos::complex<TestType> result_0, result_1;
                Kokkos::deep_copy(result_0, result_subview_0);
                Kokkos::deep_copy(result_1, result_subview_1);

                CHECK(result_0 == ComplexT{ZERO<TestType>()});
                CHECK(result_1 == ComplexT{ONE<TestType>()});
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliY",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const auto p = ComplexT{HALF<TestType>()} *
                       ComplexT{INVSQRT2<TestType>()} *
                       ComplexT{IMAG<TestType>()};
        const auto m = ComplexT{NEGONE<TestType>()} * p;

        const std::vector<std::vector<ComplexT>> expected_results = {
            {m, m, m, m, p, p, p, p},
            {m, m, p, p, m, m, p, p},
            {m, p, m, p, m, p, m, p}};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyOperation("PauliY", {index}, false);
                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliZ",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const auto p =
            ComplexT{HALF<TestType>()} * ComplexT{INVSQRT2<TestType>()};
        const auto m = ComplexT{NEGONE<TestType>()} * p;

        const std::vector<std::vector<ComplexT>> expected_results = {
            {p, p, p, p, m, m, m, m},
            {p, p, m, m, p, p, m, m},
            {p, m, p, m, p, m, p, m}};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyOperation("PauliZ", {index}, false);
                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyS", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto r = ComplexT{HALF<TestType>()} * ComplexT{INVSQRT2<TestType>()};
        auto i = r * ComplexT{IMAG<TestType>()};

        const std::vector<std::vector<ComplexT>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyOperation("S", {index}, false);
                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}
TEMPLATE_TEST_CASE("StateVectorKokkos::applySX", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        const ComplexT z(0.0, 0.0);
        ComplexT p(0.5, 0.5);
        ComplexT m(0.5, -0.5);

        if (inverse) {
            p = conj(p);
            m = conj(m);
        }

        const std::vector<std::vector<ComplexT>> expected_results = {
            {p, z, z, z, m, z, z, z},
            {p, z, m, z, z, z, z, z},
            {p, m, z, z, z, z, z, z}};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("SX", {index}, inverse);
                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyT", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}},
                                  {{inverse}, {inverse}, {inverse}});

        auto r = ComplexT{HALF<TestType>()} * ComplexT{INVSQRT2<TestType>()};
        auto i = ComplexT{HALF<TestType>()} * ComplexT{HALF<TestType>()} *
                 (ComplexT{IMAG<TestType>()} + ComplexT{ONE<TestType>()});
        if (inverse) {
            i = conj(i);
        }

        const std::vector<std::vector<ComplexT>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{inverse}, {inverse}, {inverse}});
                kokkos_sv.applyOperation("T", {index}, inverse);

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCNOT",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation("Hadamard", {0}, false);

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            Kokkos::deep_copy(kokkos_sv.getView(), ini_sv);
            auto result = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, kokkos_sv.getView());
            for (std::size_t index = 1; index < num_qubits; index++) {
                kokkos_sv.applyOperation("CNOT", {index - 1, index}, false);
            }
            Kokkos::deep_copy(result, kokkos_sv.getView());
            CHECK(imag(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(imag(result[0])));
            CHECK(real(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(real(result[0])));
            CHECK(imag(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(imag(result[7])));
            CHECK(real(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(real(result[7])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applySWAP",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("SWAP0,1 |+10> -> 1+0>") {
                const std::vector<ComplexT> expected_results = {z, z, z, z,
                                                                i, z, i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};
                Kokkos::deep_copy(svdat01.getView(), ini_sv);
                Kokkos::deep_copy(svdat10.getView(), ini_sv);

                svdat01.applyOperation("SWAP", {0, 1}, false);
                svdat10.applyOperation("SWAP", {1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getView());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("SWAP0,2 |+10> -> |01+>") {
                const std::vector<ComplexT> expected_results = {z, z, i, i,
                                                                z, z, z, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getView(), ini_sv);
                Kokkos::deep_copy(svdat20.getView(), ini_sv);

                svdat02.applyOperation("SWAP", {0, 2}, false);
                svdat20.applyOperation("SWAP", {2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getView());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("SWAP1,2 |+10> -> |+01>") {
                const std::vector<ComplexT> expected_results = {z, i, z, z,
                                                                z, i, z, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getView(), ini_sv);
                Kokkos::deep_copy(svdat21.getView(), ini_sv);

                svdat12.applyOperation("SWAP", {1, 2}, false);
                svdat21.applyOperation("SWAP", {2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getView());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCZ", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("CZ0,1 |+10> -> 1+0>") {
                const std::vector<ComplexT> expected_results = {z, z, i,  z,
                                                                z, z, -i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};
                Kokkos::deep_copy(svdat01.getView(), ini_sv);
                Kokkos::deep_copy(svdat10.getView(), ini_sv);

                svdat01.applyOperation("CZ", {0, 1}, false);
                svdat10.applyOperation("CZ", {1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getView());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("CZ0,2 |+10> -> |01+>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, z, i, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getView(), ini_sv);
                Kokkos::deep_copy(svdat20.getView(), ini_sv);

                svdat02.applyOperation("CZ", {0, 2}, false);
                svdat20.applyOperation("CZ", {2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getView());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("CZ1,2 |+10> -> |+01>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, z, i, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getView(), ini_sv);
                Kokkos::deep_copy(svdat21.getView(), ini_sv);

                svdat12.applyOperation("CZ", {1, 2}, false);
                svdat21.applyOperation("CZ", {2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getView());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyToffoli",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("Toffoli [0,1,2],[1,0,2] |+10> -> +1+>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, z, z, i};

                StateVectorKokkos<TestType> svdat012{num_qubits};
                StateVectorKokkos<TestType> svdat102{num_qubits};
                Kokkos::deep_copy(svdat012.getView(), ini_sv);
                Kokkos::deep_copy(svdat102.getView(), ini_sv);

                svdat012.applyOperation("Toffoli", {0, 1, 2}, false);
                svdat102.applyOperation("Toffoli", {1, 0, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getView());
                auto sv102 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat102.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv102[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv102[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCSWAP",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("CSWAP [0,1,2]|+10> -> |010> + |101>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, i, z, z};

                StateVectorKokkos<TestType> svdat012{num_qubits};
                Kokkos::deep_copy(svdat012.getView(), ini_sv);

                svdat012.applyOperation("CSWAP", {0, 1, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getView());

                for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyMatrix/Controlled-Operation",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using StateVectorT = StateVectorKokkos<TestType>;
    using PrecisionT = StateVectorT::PrecisionT;

    const std::size_t num_qubits = 7;
    const TestType EP = 1e-4;
    auto ini_st = createNonTrivialState<StateVectorT>(num_qubits);

    std::unordered_map<std::string, GateOperation> str_to_gates_{};
    for (const auto &[gate_op, gate_name] : Constant::gate_names) {
        str_to_gates_.emplace(gate_name, gate_op);
    }

    std::unordered_map<std::string, ControlledGateOperation>
        str_to_controlled_gates_{};
    for (const auto &[gate_op, controlled_gate_name] :
         Constant::controlled_gate_names) {
        str_to_controlled_gates_.emplace(controlled_gate_name, gate_op);
    }

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("PauliX", "PauliY", "PauliZ",
                                           "Hadamard", "S", "SX", "T", "SWAP");
    DYNAMIC_SECTION("1-controlled Matrix - Gate = "
                    << gate_name << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, false);

        std::vector<std::size_t> controlled_wires = {4};
        std::vector<bool> controlled_values = {true};

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires =
            createWires(str_to_controlled_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, controlled_wires,
                                     controlled_values, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", controlled_wires,
                                     controlled_values, wires, inverse, {},
                                     gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }

    DYNAMIC_SECTION("2-controlled Matrix (c {4, 5})- Gate = "
                    << gate_name << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, false);

        std::vector<std::size_t> controlled_wires = {4, 5};
        std::vector<bool> controlled_values = {true, false};

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires =
            createWires(str_to_controlled_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, controlled_wires,
                                     controlled_values, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", controlled_wires,
                                     controlled_values, wires, inverse, {},
                                     gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }

    DYNAMIC_SECTION("2-controlled Matrix (c {4, 6})- Gate = "
                    << gate_name << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, false);

        std::vector<std::size_t> controlled_wires = {4, 6};
        std::vector<bool> controlled_values = {true, false};

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires =
            createWires(str_to_controlled_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, controlled_wires,
                                     controlled_values, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", controlled_wires,
                                     controlled_values, wires, inverse, {},
                                     gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }

    DYNAMIC_SECTION("3-controlled Matrix (c {4, 5, 6})- Gate = "
                    << gate_name << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, false);

        std::vector<std::size_t> controlled_wires = {4, 5, 6};
        std::vector<bool> controlled_values = {true, true, false};

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires =
            createWires(str_to_controlled_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, controlled_wires,
                                     controlled_values, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", controlled_wires,
                                     controlled_values, wires, inverse, {},
                                     gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }

    DYNAMIC_SECTION("3-controlled Matrix (c {2, 4, 6})- Gate = "
                    << gate_name << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, false);

        std::vector<std::size_t> controlled_wires = {2, 4, 6};
        std::vector<bool> controlled_values = {true, true, false};

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires =
            createWires(str_to_controlled_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, controlled_wires,
                                     controlled_values, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", controlled_wires,
                                     controlled_values, wires, inverse, {},
                                     gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }

    DYNAMIC_SECTION("4-controlled Matrix (c {2, 3, 4, 5})- Gate = "
                    << gate_name << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, false);

        std::vector<std::size_t> controlled_wires = {2, 3, 4, 5};
        std::vector<bool> controlled_values = {true, true, false, false};

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires =
            createWires(str_to_controlled_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, controlled_wires,
                                     controlled_values, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", controlled_wires,
                                     controlled_values, wires, inverse, {},
                                     gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }

    DYNAMIC_SECTION("4-controlled Matrix (c {2, 3, 5, 6})- Gate = "
                    << gate_name << " Inverse = " << inverse) {
        auto gate_matrix = getMatrix<Kokkos::complex, PrecisionT>(
            str_to_gates_.at(gate_name), {}, false);

        std::vector<std::size_t> controlled_wires = {2, 3, 5, 6};
        std::vector<bool> controlled_values = {true, true, false, false};

        StateVectorT kokkos_sv_ops{ini_st.data(), ini_st.size()};
        StateVectorT kokkos_sv_mat{ini_st.data(), ini_st.size()};

        const auto wires =
            createWires(str_to_controlled_gates_.at(gate_name), num_qubits);
        kokkos_sv_ops.applyOperation(gate_name, controlled_wires,
                                     controlled_values, wires, inverse, {});
        kokkos_sv_mat.applyOperation("Matrix", controlled_wires,
                                     controlled_values, wires, inverse, {},
                                     gate_matrix);

        auto result_ops = kokkos_sv_ops.getDataVector();
        auto result_mat = kokkos_sv_mat.getDataVector();

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(real(result_ops[j]) ==
                  Approx(real(result_mat[j])).margin(EP));
            CHECK(imag(result_ops[j]) ==
                  Approx(imag(result_mat[j])).margin(EP));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyMultiQubitOp",
                   "[StateVectorKokkos_Nonparam][Inverse]", float, double) {
    const bool inverse = GENERATE(true, false);
    std::size_t num_qubits = 3;
    StateVectorKokkos<TestType> sv_normal{num_qubits};
    StateVectorKokkos<TestType> sv_mq{num_qubits};
    using UnmanagedComplexHostView =
        Kokkos::View<Kokkos::complex<TestType> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    SECTION("Single Qubit via applyOperation") {
        auto matrix = getHadamard<Kokkos::complex, TestType>();
        std::vector<std::size_t> wires = {0};
        sv_normal.applyOperation("Hadamard", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        sv_mq.applyOperation("MatrixHadamard", wires, inverse, {}, matrix);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Single Qubit") {
        auto matrix = getHadamard<Kokkos::complex, TestType>();
        std::vector<std::size_t> wires = {0};
        sv_normal.applyOperation("Hadamard", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, inverse);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Two Qubit") {
        auto matrix = getCNOT<Kokkos::complex, TestType>();
        std::vector<std::size_t> wires = {0, 1};
        sv_normal.applyOperation("CNOT", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, inverse);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Three Qubit") {
        auto matrix = getToffoli<Kokkos::complex, TestType>();
        std::vector<std::size_t> wires =
            GENERATE(std::vector<std::size_t>{0, 1, 2},
                     std::vector<std::size_t>{2, 0, 1},
                     std::vector<std::size_t>{1, 2, 0},
                     std::vector<std::size_t>{0, 2, 1},
                     std::vector<std::size_t>{1, 0, 2},
                     std::vector<std::size_t>{2, 1, 0});
        sv_normal.applyOperation("Toffoli", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, inverse);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyNCMultiQubitOp",
                   "[StateVectorKokkos_Nonparam][Inverse]", float, double) {
    const bool inverse = GENERATE(true, false);
    std::size_t num_qubits = 3;
    StateVectorKokkos<TestType> sv_normal{num_qubits};
    StateVectorKokkos<TestType> sv_mq{num_qubits};
    using UnmanagedComplexHostView =
        Kokkos::View<Kokkos::complex<TestType> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    SECTION("0 Controlled Single Qubit via applyOperation") {
        std::vector<std::size_t> wires = {0};
        sv_normal.applyOperation("PauliX", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        std::vector<std::size_t> controlled_wire = {};
        std::vector<bool> controlled_value = {};
        std::vector<std::size_t> wire = {0};
        auto matrix = getPauliX<Kokkos::complex, TestType>();
        sv_mq.applyOperation("MatrixPauliX", controlled_wire, controlled_value,
                             wire, inverse, {}, matrix);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Single Qubit via applyNCMultiQubitOp") {
        std::vector<std::size_t> wires = {0, 1};
        sv_normal.applyOperation("CNOT", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        std::vector<std::size_t> controlled_wire = {0};
        std::vector<bool> controlled_value = {true};
        std::vector<std::size_t> wire = {1};
        auto matrix = getPauliX<Kokkos::complex, TestType>();
        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyNCMultiQubitOp(device_matrix, controlled_wire,
                                  controlled_value, wire, inverse);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Controlled Single Qubit via applyOperation") {
        std::vector<std::size_t> wires = {0, 1};
        sv_normal.applyOperation("CNOT", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        std::vector<std::size_t> controlled_wire = {0};
        std::vector<bool> controlled_value = {true};
        std::vector<std::size_t> wire = {1};
        auto matrix = getPauliX<Kokkos::complex, TestType>();
        sv_mq.applyOperation("MatrixCNOT", controlled_wire, controlled_value,
                             wire, inverse, {}, matrix);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyOperation non-param "
                   "one-qubit with controls",
                   "[StateVectorKokkos_NonParam]", float, double) {
    const bool inverse = GENERATE(true, false);
    const std::size_t num_qubits = 4;
    const std::size_t control = GENERATE(0, 1, 2, 3);
    const std::size_t wire = GENERATE(0, 1, 2, 3);
    StateVectorKokkos<TestType> kokkos_sv{num_qubits};

    kokkos_sv.applyOperations(
        {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
        {{0}, {1}, {2}, {3}}, {{false}, {false}, {false}, {false}});

    auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      kokkos_sv.getView());
    StateVectorKokkos<TestType> sv_gate{num_qubits};
    StateVectorKokkos<TestType> sv_control{num_qubits};

    SECTION("N-controlled PauliX ") {
        if (control == wire) {
            Kokkos::deep_copy(sv_control.getView(), ini_sv);

            REQUIRE_THROWS_AS(sv_control.applyOperation(
                                  "PauliX", std::vector<std::size_t>{control},
                                  std::vector<bool>{true},
                                  std::vector<std::size_t>{wire}),
                              LightningException);
        }

        if (control != wire) {
            Kokkos::deep_copy(sv_gate.getView(), ini_sv);
            Kokkos::deep_copy(sv_control.getView(), ini_sv);

            sv_gate.applyOperation("CNOT", {control, wire}, inverse);
            sv_control.applyOperation(
                "PauliX", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire});
            auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_gate.getView());
            auto sv_control_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_control.getView());

            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(sv_gate_host[j]) ==
                      Approx(imag(sv_control_host[j])));
                CHECK(real(sv_gate_host[j]) ==
                      Approx(real(sv_control_host[j])));
            }
        }

        if (control != 0 && wire != 0 && control != wire) {
            Kokkos::deep_copy(sv_gate.getView(), ini_sv);
            Kokkos::deep_copy(sv_control.getView(), ini_sv);

            sv_gate.applyOperation("Toffoli", {0, control, wire}, inverse);
            sv_control.applyOperation(
                "PauliX", std::vector<std::size_t>{0, control},
                std::vector<bool>{true, true}, std::vector<std::size_t>{wire});
            auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_gate.getView());
            auto sv_control_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_control.getView());
            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(sv_gate_host[j]) ==
                      Approx(imag(sv_control_host[j])));
                CHECK(real(sv_gate_host[j]) ==
                      Approx(real(sv_control_host[j])));
            }

            sv_gate.applyOperation("Toffoli", {control, 0, wire});
            sv_control.applyOperation(
                "PauliX", std::vector<std::size_t>{control, 0},
                std::vector<bool>{true, true}, std::vector<std::size_t>{wire});
            Kokkos::deep_copy(sv_gate_host, sv_gate.getView());
            Kokkos::deep_copy(sv_control_host, sv_control.getView());
            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(sv_gate_host[j]) ==
                      Approx(imag(sv_control_host[j])));
                CHECK(real(sv_gate_host[j]) ==
                      Approx(real(sv_control_host[j])));
            }
        }
    }

    SECTION("N-controlled PauliY ") {
        if (control != wire) {
            Kokkos::deep_copy(sv_gate.getView(), ini_sv);
            Kokkos::deep_copy(sv_control.getView(), ini_sv);

            sv_gate.applyOperation("CY", {control, wire}, inverse);
            sv_control.applyOperation(
                "PauliY", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire});
            auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_gate.getView());
            auto sv_control_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_control.getView());

            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(sv_gate_host[j]) ==
                      Approx(imag(sv_control_host[j])));
                CHECK(real(sv_gate_host[j]) ==
                      Approx(real(sv_control_host[j])));
            }
        }
    }

    SECTION("N-controlled PauliZ ") {
        if (control != wire) {
            Kokkos::deep_copy(sv_gate.getView(), ini_sv);
            Kokkos::deep_copy(sv_control.getView(), ini_sv);

            sv_gate.applyOperation("CZ", {control, wire}, inverse);
            sv_control.applyOperation(
                "PauliZ", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire});
            auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_gate.getView());
            auto sv_control_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, sv_control.getView());

            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(sv_gate_host[j]) ==
                      Approx(imag(sv_control_host[j])));
                CHECK(real(sv_gate_host[j]) ==
                      Approx(real(sv_control_host[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyOperation non-param "
                   "one-qubit with multiple-controls",
                   "[StateVectorKokkos_NonParam]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const bool inverse = false;
    const std::size_t num_qubits = 3;
    StateVectorKokkos<TestType> kokkos_sv{num_qubits};

    kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                              {{0}, {1}, {2}}, {{false}, {false}, {false}});

    auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      kokkos_sv.getView());
    StateVectorKokkos<TestType> sv_gate{num_qubits};
    StateVectorKokkos<TestType> expected_result{num_qubits};

    SECTION("2-controlled PauliY") {
        Kokkos::deep_copy(sv_gate.getView(), ini_sv);

        const std::vector<std::size_t> control_wires = {0, 2};
        const std::vector<bool> control_values = {true, false};
        const std::vector<std::size_t> target_wire = {1};
        sv_gate.applyOperation("PauliY", control_wires, control_values,
                               target_wire, inverse);
        auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_gate.getView());

        std::vector<ComplexT> expected_result{
            // Generated using Pennylane
            ComplexT{0.35355339, 0.0},  ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0},  ComplexT{0.35355339, 0.0},
            ComplexT{0.0, -0.35355339}, ComplexT{0.35355339, 0.0},
            ComplexT{0.0, 0.35355339},  ComplexT{0.35355339, 0.0},
        };
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_gate_host[j]) == Approx(imag(expected_result[j])));
            CHECK(real(sv_gate_host[j]) == Approx(real(expected_result[j])));
        }
    }

    SECTION("2-controlled PauliZ") {
        Kokkos::deep_copy(sv_gate.getView(), ini_sv);

        const std::vector<std::size_t> control_wires = {0, 2};
        const std::vector<bool> control_values = {true, false};
        const std::vector<std::size_t> target_wire = {1};
        sv_gate.applyOperation("PauliZ", control_wires, control_values,
                               target_wire, inverse);
        auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_gate.getView());

        std::vector<ComplexT> expected_result{
            // Generated using Pennylane
            ComplexT{0.35355339, 0.0},  ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0},  ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0},  ComplexT{0.35355339, 0.0},
            ComplexT{-0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
        };
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_gate_host[j]) == Approx(imag(expected_result[j])));
            CHECK(real(sv_gate_host[j]) == Approx(real(expected_result[j])));
        }
    }

    SECTION("2-controlled Hadamard") {
        Kokkos::deep_copy(sv_gate.getView(), ini_sv);

        const std::vector<std::size_t> control_wires = {0, 2};
        const std::vector<bool> control_values = {true, false};
        const std::vector<std::size_t> target_wire = {1};
        sv_gate.applyOperation("Hadamard", control_wires, control_values,
                               target_wire, inverse);
        auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_gate.getView());

        std::vector<ComplexT> expected_result{
            // Generated using Pennylane
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.5, 0.0},        ComplexT{0.35355339, 0.0},
            ComplexT{0.0, 0.0},        ComplexT{0.35355339, 0.0},
        };
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_gate_host[j]) == Approx(imag(expected_result[j])));
            CHECK(real(sv_gate_host[j]) == Approx(real(expected_result[j])));
        }
    }

    SECTION("2-controlled S") {
        Kokkos::deep_copy(sv_gate.getView(), ini_sv);

        const std::vector<std::size_t> control_wires = {0, 2};
        const std::vector<bool> control_values = {true, false};
        const std::vector<std::size_t> target_wire = {1};
        sv_gate.applyOperation("S", control_wires, control_values, target_wire,
                               inverse);
        auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_gate.getView());

        std::vector<ComplexT> expected_result{
            // Generated using Pennylane
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.0, 0.35355339}, ComplexT{0.35355339, 0.0},
        };
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_gate_host[j]) == Approx(imag(expected_result[j])));
            CHECK(real(sv_gate_host[j]) == Approx(real(expected_result[j])));
        }
    }

    SECTION("2-controlled SX") {
        Kokkos::deep_copy(sv_gate.getView(), ini_sv);

        const std::vector<std::size_t> control_wires = {0, 2};
        const std::vector<bool> control_values = {true, false};
        const std::vector<std::size_t> target_wire = {1};
        sv_gate.applyOperation("SX", control_wires, control_values, target_wire,
                               inverse);
        auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_gate.getView());

        std::vector<ComplexT> expected_result{
            // Generated using Pennylane
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
        };
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_gate_host[j]) == Approx(imag(expected_result[j])));
            CHECK(real(sv_gate_host[j]) == Approx(real(expected_result[j])));
        }
    }

    SECTION("2-controlled T") {
        Kokkos::deep_copy(sv_gate.getView(), ini_sv);

        const std::vector<std::size_t> control_wires = {0, 2};
        const std::vector<bool> control_values = {true, false};
        const std::vector<std::size_t> target_wire = {1};
        sv_gate.applyOperation("T", control_wires, control_values, target_wire,
                               inverse);
        auto sv_gate_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_gate.getView());

        std::vector<ComplexT> expected_result{
            // Generated using Pennylane
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.35355339, 0.0}, ComplexT{0.35355339, 0.0},
            ComplexT{0.25, 0.25},      ComplexT{0.35355339, 0.0},
        };
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_gate_host[j]) == Approx(imag(expected_result[j])));
            CHECK(real(sv_gate_host[j]) == Approx(real(expected_result[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyOperation non-param "
                   "two-qubit with controls",
                   "[StateVectorKokkos_NonParam]", float, double) {
    using StateVectorT = StateVectorKokkos<TestType>;

    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    const bool inverse = GENERATE(true, false);
    const std::size_t control = GENERATE(0, 1, 2, 3);
    const std::size_t wire0 = GENERATE(0, 1, 2, 3);
    const std::size_t wire1 = GENERATE(0, 1, 2, 3);

    auto ini_st = createNonTrivialState<StateVectorT>(num_qubits);

    SECTION("N-controlled SWAP") {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            StateVectorT kokkos_sv0{ini_st.data(), ini_st.size()};
            StateVectorT kokkos_sv1{ini_st.data(), ini_st.size()};
            kokkos_sv0.applyOperation("CSWAP", {control, wire0, wire1},
                                      inverse);
            auto matrix = getSWAP<Kokkos::complex, TestType>();
            kokkos_sv1.applyOperation("SWAP", std::vector<std::size_t>{control},
                                      std::vector<bool>{true},
                                      std::vector<std::size_t>{wire0, wire1},
                                      inverse);
            auto result_sv0 = kokkos_sv0.getDataVector();
            auto result_sv1 = kokkos_sv1.getDataVector();
            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(real(result_sv0[j]) ==
                      Approx(real(result_sv1[j])).margin(EP));
                CHECK(imag(result_sv0[j]) ==
                      Approx(imag(result_sv1[j])).margin(EP));
            }
        }
    }

    SECTION("N-controlled SWAP with matrix") {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            StateVectorT kokkos_sv0{ini_st.data(), ini_st.size()};
            StateVectorT kokkos_sv1{ini_st.data(), ini_st.size()};
            kokkos_sv0.applyOperation("CSWAP", {control, wire0, wire1},
                                      inverse);
            auto matrix = getSWAP<Kokkos::complex, TestType>();
            kokkos_sv1.applyOperation(
                "MatrixCSWAP", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire0, wire1},
                inverse, {}, matrix);
            auto result_sv0 = kokkos_sv0.getDataVector();
            auto result_sv1 = kokkos_sv1.getDataVector();
            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(real(result_sv0[j]) ==
                      Approx(real(result_sv1[j])).margin(EP));
                CHECK(imag(result_sv0[j]) ==
                      Approx(imag(result_sv1[j])).margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorKokkos::applyOperation controlled matrix (PauliX/Toffoli)",
    "[StateVectorKokkos_NonParam]", float, double) {
    using StateVectorT = StateVectorKokkos<TestType>;

    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;
    const bool inverse = GENERATE(true, false);
    const std::size_t control = GENERATE(0, 1, 2);

    auto ini_st = createNonTrivialState<StateVectorT>(num_qubits);
    StateVectorT kokkos_sv0{ini_st.data(), ini_st.size()};
    StateVectorT kokkos_sv1{ini_st.data(), ini_st.size()};
    auto matrix = getToffoli<Kokkos::complex, TestType>();
    kokkos_sv0.applyOperation(
        "Matrix", std::vector<std::size_t>{control}, std::vector<bool>{true},
        std::vector<std::size_t>{3, 4, 5}, inverse, {}, matrix);
    kokkos_sv1.applyOperation("PauliX", std::vector<std::size_t>{control, 3, 4},
                              std::vector<bool>{true, true, true},
                              std::vector<std::size_t>{5}, inverse);
    auto result_sv0 = kokkos_sv0.getDataVector();
    auto result_sv1 = kokkos_sv1.getDataVector();
    for (std::size_t j = 0; j < exp2(num_qubits); j++) {
        CHECK(real(result_sv0[j]) == Approx(real(result_sv1[j])).margin(EP));
        CHECK(imag(result_sv0[j]) == Approx(imag(result_sv1[j])).margin(EP));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyOperation non-param "
                   "two-qubit with multiple-controls",
                   "[StateVectorKokkos_NonParam]", float, double) {
    SECTION("2-controlled SWAP") {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{ComplexT{0.33377149, 0.2511291},
                                     ComplexT{0.33013572, -0.14778699},
                                     ComplexT{0.17288832, -0.041744},
                                     ComplexT{0.11004883, -0.15105962},
                                     ComplexT{0.17120967, 0.04376507},
                                     ComplexT{0.33010645, 0.20088901},
                                     ComplexT{-0.12770624, -0.17329647},
                                     ComplexT{-0.34301996, 0.11944278},
                                     ComplexT{0.0195779, -0.03687076},
                                     ComplexT{0.34155068, 0.07464519},
                                     ComplexT{0.00730597, 0.03670807},
                                     ComplexT{0.08876188, -0.18019018},
                                     ComplexT{-0.04946055, -0.10383813},
                                     ComplexT{0.0715367, 0.04895361},
                                     ComplexT{-0.12377521, -0.04781011},
                                     ComplexT{-0.14509767, 0.2102171}};

        std::vector<ComplexT> expected{ComplexT{0.33377149, 0.2511291},
                                       ComplexT{0.33013572, -0.14778699},
                                       ComplexT{0.17288832, -0.041744},
                                       ComplexT{0.11004883, -0.15105962},
                                       ComplexT{0.17120967, 0.04376507},
                                       ComplexT{0.33010645, 0.20088901},
                                       ComplexT{-0.12770624, -0.17329647},
                                       ComplexT{-0.34301996, 0.11944278},
                                       ComplexT{0.0195779, -0.03687076},
                                       ComplexT{-0.04946055, -0.10383813},
                                       ComplexT{0.00730597, 0.03670807},
                                       ComplexT{0.08876188, -0.18019018},
                                       ComplexT{0.34155068, 0.07464519},
                                       ComplexT{0.0715367, 0.04895361},
                                       ComplexT{-0.12377521, -0.04781011},
                                       ComplexT{-0.14509767, 0.2102171}};

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("SWAP", {0, 2}, {true, false}, {1, 3},
                                     false);
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::SetStateVector",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const std::size_t num_qubits = 3;

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        std::vector<ComplexT> init_state{
            ComplexT{0.267462849617, 0.010768564418},
            ComplexT{0.228575125337, 0.010564590804},
            ComplexT{0.099492751062, 0.260849833488},
            ComplexT{0.093690201640, 0.189847111702},
            ComplexT{0.015641822883, 0.225092900621},
            ComplexT{0.205574608177, 0.082808663337},
            ComplexT{0.006827173322, 0.211631480575},
            ComplexT{0.255280800811, 0.161572331669},
        };
        auto expected_state = init_state;

        for (std::size_t i = 0; i < exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
        }

        StateVectorKokkos<PrecisionT> kokkos_sv{num_qubits};
        std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.HostToDevice(init_state.data(), init_state.size());

        // The setStates will shuffle the state vector values on the device with
        // the following indices and values setting on host. For example, the
        // values[i] is used to set the indices[i] th element of state vector on
        // the device. For example, values[2] (init_state[5]) will be copied to
        // indices[2]th or (4th) element of the state vector.
        std::vector<std::size_t> indices = {0, 2, 4, 6, 1, 3, 5, 7};

        std::vector<Kokkos::complex<PrecisionT>> values = {
            init_state[1], init_state[3], init_state[5], init_state[7],
            init_state[0], init_state[2], init_state[4], init_state[6]};
        kokkos_sv.setStateVector(indices, values);
        kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(expected_state[j]) == Approx(imag(result_sv[j])));
            CHECK(real(expected_state[j]) == Approx(real(result_sv[j])));
        }

        kokkos_sv.setStateVector(init_state, {0, 1, 2});
        kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());
        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(init_state[j]) == Approx(imag(result_sv[j])));
            CHECK(real(init_state[j]) == Approx(real(result_sv[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::SetIthStates",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const std::size_t num_qubits = 3;

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        std::vector<ComplexT> init_state{
            ComplexT{0.267462849617, 0.010768564418},
            ComplexT{0.228575125337, 0.010564590804},
            ComplexT{0.099492751062, 0.260849833488},
            ComplexT{0.093690201640, 0.189847111702},
            ComplexT{0.015641822883, 0.225092900621},
            ComplexT{0.205574608177, 0.082808663337},
            ComplexT{0.006827173322, 0.211631480575},
            ComplexT{0.255280800811, 0.161572331669},
        };

        std::vector<ComplexT> expected_state{
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{1.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
        };

        StateVectorKokkos<PrecisionT> kokkos_sv{num_qubits};
        std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.HostToDevice(init_state.data(), init_state.size());

        std::size_t index = 3;

        kokkos_sv.setBasisState(index);

        kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(expected_state[j]) == Approx(imag(result_sv[j])));
            CHECK(real(expected_state[j]) == Approx(real(result_sv[j])));
        }
    }
}
