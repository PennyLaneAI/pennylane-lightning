// Copyright 2025 Xanadu Quantum Technologies Inc.

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
#include <limits> // numeric_limits
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointJacobianKokkosMPI.hpp"
#include "MPIManagerKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData

/**
 * @file
 *  Tests for functionality for the class StateVectorKokkosMPI.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::Util;

using Pennylane::Util::isApproxEqual;
using Pennylane::Util::randomUnitary;

std::mt19937_64 re{1337};
} // namespace
/// @endcond

// expval
TEMPLATE_PRODUCT_TEST_CASE("Adjoint", "[LKMPI_Adjoint]", (StateVectorKokkosMPI),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<std::size_t> tp{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const PrecisionT ep = 1e-6;
    {
        const std::size_t num_qubits = 5;
        const std::size_t num_obs = 1;
        std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

        MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 4);

        StateVectorT psi(mpi_manager, num_qubits);

        const auto obs = std::make_shared<TensorProdObsMPI<StateVectorT>>(
            std::make_shared<NamedObsMPI<StateVectorT>>(
                "PauliX", std::vector<std::size_t>{0}),
            std::make_shared<NamedObsMPI<StateVectorT>>(
                "PauliX", std::vector<std::size_t>{1}),
            std::make_shared<NamedObsMPI<StateVectorT>>(
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
