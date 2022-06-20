#pragma once
/**
 * @file
 * We define test kernels. Note that kernels not registered to
 * AvailableKernels can also be tested by adding it here.
 */
#include "AvailableKernels.hpp"
#include "Macros.hpp"
#include "TypeList.hpp"

#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI, void>;
