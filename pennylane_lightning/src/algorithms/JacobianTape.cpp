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

#include "JacobianTape.hpp"

template class Pennylane::Algorithms::ObsDatum<float>;
template class Pennylane::Algorithms::ObsDatum<double>;

template class Pennylane::Algorithms::ObsDatum<std::complex<float>>;
template class Pennylane::Algorithms::ObsDatum<std::complex<double>>;

template class Pennylane::Algorithms::JacobianData<float>;
template class Pennylane::Algorithms::JacobianData<double>;