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

#define _USE_MATH_DEFINES

#include "Gates.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <vector>

#include "Util.hpp"

using std::conj;
using std::string;
using std::swap;
using std::unique_ptr;
using std::vector;

template <class Precision>
Pennylane::AbstractGate<Precision>::AbstractGate(size_t numQubits)
    : numQubits(numQubits), length(exp2(numQubits)) {}

template class Pennylane::XGate<float>;
template class Pennylane::XGate<double>;

template class Pennylane::YGate<float>;
template class Pennylane::YGate<double>;

template class Pennylane::ZGate<float>;
template class Pennylane::ZGate<double>;

template class Pennylane::HadamardGate<float>;
template class Pennylane::HadamardGate<double>;

template class Pennylane::SGate<float>;
template class Pennylane::SGate<double>;

template class Pennylane::TGate<float>;
template class Pennylane::TGate<double>;

template class Pennylane::RotationXGate<float>;
template class Pennylane::RotationXGate<double>;

template class Pennylane::RotationYGate<float>;
template class Pennylane::RotationYGate<double>;

template class Pennylane::RotationZGate<float>;
template class Pennylane::RotationZGate<double>;

template class Pennylane::PhaseShiftGate<float>;
template class Pennylane::PhaseShiftGate<double>;

template class Pennylane::CNOTGate<float>;
template class Pennylane::CNOTGate<double>;

template class Pennylane::SWAPGate<float>;
template class Pennylane::SWAPGate<double>;

template class Pennylane::CZGate<float>;
template class Pennylane::CZGate<double>;

template class Pennylane::CRotationXGate<float>;
template class Pennylane::CRotationXGate<double>;

template class Pennylane::CRotationYGate<float>;
template class Pennylane::CRotationYGate<double>;

template class Pennylane::CRotationZGate<float>;
template class Pennylane::CRotationZGate<double>;

template class Pennylane::CGeneralRotationGate<float>;
template class Pennylane::CGeneralRotationGate<double>;

template class Pennylane::ToffoliGate<float>;
template class Pennylane::ToffoliGate<double>;

template class Pennylane::CSWAPGate<float>;
template class Pennylane::CSWAPGate<double>;
