#include "TestHelpers.hpp"
#include "TestKernels.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file Test_GateImplementations_Generator.cpp
 *
 * This file contains tests for generators. We test generators satisfy
 * -I*G |\psi> = \parital{U}/\partial{\theta} |\psi>
 */
using namespace Pennylane;


