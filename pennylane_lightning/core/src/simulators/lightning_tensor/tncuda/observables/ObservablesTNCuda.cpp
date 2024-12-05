// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ObservablesTNCuda.hpp"
#include "ExactTNCuda.hpp"
#include "MPSTNCuda.hpp"

using namespace Pennylane::LightningTensor::TNCuda::Observables;

template class ObservableTNCuda<MPSTNCuda<float>>;
template class ObservableTNCuda<MPSTNCuda<double>>;

template class NamedObsTNCuda<MPSTNCuda<float>>;
template class NamedObsTNCuda<MPSTNCuda<double>>;

template class HermitianObsTNCuda<MPSTNCuda<float>>;
template class HermitianObsTNCuda<MPSTNCuda<double>>;

template class TensorProdObsTNCuda<MPSTNCuda<float>>;
template class TensorProdObsTNCuda<MPSTNCuda<double>>;

template class HamiltonianTNCuda<MPSTNCuda<float>>;
template class HamiltonianTNCuda<MPSTNCuda<double>>;

template class ObservableTNCuda<ExactTNCuda<float>>;
template class ObservableTNCuda<ExactTNCuda<double>>;

template class NamedObsTNCuda<ExactTNCuda<float>>;
template class NamedObsTNCuda<ExactTNCuda<double>>;

template class HermitianObsTNCuda<ExactTNCuda<float>>;
template class HermitianObsTNCuda<ExactTNCuda<double>>;

template class TensorProdObsTNCuda<ExactTNCuda<float>>;
template class TensorProdObsTNCuda<ExactTNCuda<double>>;

template class HamiltonianTNCuda<ExactTNCuda<float>>;
template class HamiltonianTNCuda<ExactTNCuda<double>>;
