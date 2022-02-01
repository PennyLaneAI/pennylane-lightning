#include <complex>

#include "GateOperations.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

TEST_CASE("Constant::gate_names is well defined", "GateOperations") {
    static_assert(count_unique(Util::first_elts_of(Constant::gate_names)),
                  "Gate operations in gate_names are not distinct.");

    static_assert(count_unique(Util::second_elts_of(Constant::gate_names)),
                  "Gate names in gate_names are not distinct.");
}
