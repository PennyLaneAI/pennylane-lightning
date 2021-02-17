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
/**
 * @file
 * Contains methods that produce gates from the requisite parameters.
 */
#pragma once

#include <memory>
#include <string>

#include "Gates.hpp"

namespace Pennylane {

    /**
     * Produces the requested gate, defined by a label and the list of parameters
     *
     * @param label unique string corresponding to a gate type
     * @param parameters defines the gate parameterisation (may be zero-length for some gates)
     * @return the gate wrapped in std::unique_ptr
     * @throws std::invalid_argument thrown if the gate type is not defined, or if the number of parameters to the gate is incorrect
     */
    std::unique_ptr<AbstractGate> constructGate(const std::string& label, const std::vector<double>& parameters);

}
