// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Observables.hpp"

template class Pennylane::Simulators::NamedObs<float>;
template class Pennylane::Simulators::NamedObs<double>;

template class Pennylane::Simulators::HermitianObs<float>;
template class Pennylane::Simulators::HermitianObs<double>;

template class Pennylane::Simulators::TensorProdObs<float>;
template class Pennylane::Simulators::TensorProdObs<double>;

template class Pennylane::Simulators::Hamiltonian<float>;
template class Pennylane::Simulators::Hamiltonian<double>;
