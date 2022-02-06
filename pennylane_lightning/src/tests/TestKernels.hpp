#pragma once
/**
 * @brief We define test kernels. Note that kernels not registered to
 * AvailableKernels can be also tested by adding it to here.
 */
#include "GateImplementationsLM.hpp"
#include "GateImplementationsPI.hpp"

#include "TypeList.hpp"

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI>;
