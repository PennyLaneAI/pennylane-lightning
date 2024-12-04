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
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointJacobianKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Observables;
using Pennylane::Algorithms::OpsData;
using std::size_t;
} // namespace
/// @endcond

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif
TEMPLATE_PRODUCT_TEST_CASE(
    "Algorithms::adjointJacobian with exceedingly complicated Hamiltonian",
    "[Algorithms]", (StateVectorKokkos), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using namespace std::literals;
    using Pennylane::LightningKokkos::Observables::detail::
        HamiltonianApplyInPlace;

    std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<std::size_t> t_params{0, 2};

    std::mt19937 re{1337};
    const std::size_t num_qubits = 8;
    const std::size_t n_terms = 1024;

    std::array<std::string_view, 4> pauli_strs = {""sv, "PauliX"sv, "PauliY"sv,
                                                  "PauliZ"sv};

    std::vector<PrecisionT> coeffs;
    std::vector<std::shared_ptr<Observable<StateVectorT>>> terms;

    std::uniform_real_distribution<PrecisionT> dist(-1.0, 1.0);

    for (std::size_t k = 0; k < n_terms; k++) {
        auto term_pauli = randomIntVector(re, num_qubits, 0, 3);

        std::vector<std::shared_ptr<Observable<StateVectorT>>> term_comp;
        for (std::size_t i = 0; i < num_qubits; i++) {
            if (term_pauli[i] == 0) {
                continue;
            }
            auto wires = std::vector<std::size_t>();
            wires.emplace_back(i);
            auto ob = std::make_shared<NamedObs<StateVectorT>>(
                std::string{pauli_strs[term_pauli[i]]}, wires);
            term_comp.push_back(std::move(ob));
        }

        coeffs.emplace_back(dist(re));
        terms.emplace_back(TensorProdObs<StateVectorT>::create(term_comp));
    }
    std::vector<ComplexT> psi(std::size_t{1} << num_qubits);
    std::normal_distribution<PrecisionT> ndist;
    for (auto &e : psi) {
        e = ndist(re);
    }
    std::vector<ComplexT> phi = psi;

    StateVectorT sv1(psi.data(), psi.size());
    StateVectorT sv2(phi.data(), phi.size());

    HamiltonianApplyInPlace<StateVectorT, false>::run(coeffs, terms, sv1);
    HamiltonianApplyInPlace<StateVectorT, true>::run(coeffs, terms, sv2);

    PrecisionT eps = std::numeric_limits<PrecisionT>::epsilon() * 1e4;
    REQUIRE(isApproxEqual(sv1.getDataVector(), sv2.getDataVector(), eps));
}

TEMPLATE_PRODUCT_TEST_CASE(
    "AdjointJacobianKokkos::adjointJacobian Op=Controlled-Mixed, Obs=[XXX]",
    "[AdjointJacobianKokkos]", (StateVectorKokkos), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    AdjointJacobian<StateVectorT> adj;
    std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<std::size_t> tp{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const PrecisionT ep = 1e-6;
    {
        const std::size_t num_qubits = 3;
        const std::size_t num_obs = 1;
        std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);

        const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<std::size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<std::size_t>{1}),
            std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<std::size_t>{2}));
        auto ops = OpsData<StateVectorT>(
            {"RZ", "RY", "RZ", "CNOT", "CNOT", "RX", "RY", "RZ",
             "SingleExcitation", "RZ", "RY", "RZ"},
            {{param[0]},
             {param[1]},
             {param[2]},
             {},
             {},
             {param[0]},
             {param[1]},
             {param[2]},
             {param[0]},
             {param[0]},
             {param[1]},
             {param[2]}},
            {{0},
             {0},
             {0},
             {0, 1},
             {1, 2},
             {1},
             {2},
             {1},
             {0, 2},
             {1},
             {1},
             {1}},
            {false, false, false, false, false, false, false, false, false,
             false, false, false},
            {{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}},
            {{}, {}, {}, {}, {}, {0}, {0}, {0}, {1}, {}, {}, {}},
            {{}, {}, {}, {}, {}, {true}, {true}, {true}, {true}, {}, {}, {}});

        JacobianData<StateVectorT> tape{
            param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        CAPTURE(jacobian);

        // Computed with PennyLane using default.qubit.adjoint_jacobian
        std::vector<PrecisionT> expected_jacobian{
            0.0,        0.03722967,  0.53917582, -0.06895157, -0.0020095,
            0.25057513, -0.00139217, 0.52016303, -0.09895398, 0.51843232};
        CHECK(expected_jacobian == PLApprox(jacobian).margin(ep));
    }
}

/**
 * @brief Tests the constructability of the AdjointDiff.hpp classes.
 *
 */
TEMPLATE_TEST_CASE("AdjointJacobian::AdjointJacobian", "[AdjointJacobian]",
                   float, double) {
    SECTION("AdjointJacobian<TestType> {}") {
        REQUIRE(std::is_constructible<
                AdjointJacobian<StateVectorKokkos<TestType>>>::value);
    }
}
