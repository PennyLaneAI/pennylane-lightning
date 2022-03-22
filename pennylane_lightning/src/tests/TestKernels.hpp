#pragma once
/**
 * @brief We define test kernels. Note that kernels not registered to
 * AvailableKernels can be also tested by adding it to here.
 */
#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include "TypeList.hpp"

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI, void>;
