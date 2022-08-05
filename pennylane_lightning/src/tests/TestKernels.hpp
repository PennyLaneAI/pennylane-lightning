#pragma once
/**
 * @file
 * We define default test kernels.
 */
#include "Macros.hpp"
#include "TypeList.hpp"

#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI, void>;
