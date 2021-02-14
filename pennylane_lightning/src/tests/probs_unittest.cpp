// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//  
#include <math.h>       /* sqrt */
#include <unsupported/Eigen/CXX11/Tensor>
#include "gtest/gtest.h"

#include "../operations.hpp"
#include "../lightning_qubit.hpp"
#include "../statistics.hpp"

const double tol = 1.0e-6f;

using Matrix_2q = Eigen::Matrix<std::complex<double>, 4, 4>;
using Vector_3q = Eigen::Matrix<std::complex<double>, 8, 1>;

template<class State>
Eigen::VectorXcd vectorize(State state) {
    Eigen::Map<Eigen::VectorXcd> out(state.data(), state.size());
    return out;
}

namespace probs_unit {

TEST(Marginal, Basic) {

    const int qubits = 3;
    int len = int(std::pow(2, qubits));
    VectorXcd v(len);
    v(0) = 0;
    v(1) = 0;
    v(2) = 0.5;
    v(3) = 0;
    v(4) = 0;
    v(5) = 0;
    v(6) = 0.5;
    v(7) = 0;

    const vector<int> wires = {2,1};
    const int M = 4;

    // M > qubits so we expect an error
    EXPECT_THROW(marginal_probs(v, qubits, wires, M), const char*); 
}

}  // namespace probs_unit
