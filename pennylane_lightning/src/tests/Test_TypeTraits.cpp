#include "TypeTraits.hpp"

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

using namespace Pennylane::Util;

TEST_CASE("Test remove_complex") {
    SECTION("remove_complex returns the floating point if the given type is "
            "std::complex") {
        STATIC_REQUIRE(
            std::is_same_v<remove_complex_t<std::complex<float>>, float>);
        STATIC_REQUIRE(
            std::is_same_v<remove_complex_t<std::complex<double>>, double>);
    }
    SECTION("remove_complex returns the same type if not") {
        STATIC_REQUIRE(std::is_same_v<remove_complex_t<float>, float>);
        STATIC_REQUIRE(std::is_same_v<remove_complex_t<double>, double>);
    }
}

TEST_CASE("Test is_complex") {
    SECTION("is_complex returns true if the given type is std::complex") {
        STATIC_REQUIRE(is_complex_v<std::complex<double>>);
        STATIC_REQUIRE(is_complex_v<std::complex<float>>);
    }
    SECTION("remove_complex returns false if not") {
        STATIC_REQUIRE(!is_complex_v<int>);
        STATIC_REQUIRE(!is_complex_v<long>);
        STATIC_REQUIRE(!is_complex_v<float>);
        STATIC_REQUIRE(!is_complex_v<double>);
    }
}

std::pair<int, int> g(std::tuple<int, int, int>);

TEST_CASE("Test FuncReturn") {
    SECTION("FuncReturn gives correct return types") {
        STATIC_REQUIRE(
            std::is_same_v<FuncReturn<decltype(g)>::Type,
                           std::pair<int, int>>); // return type of g is
                                                  // std::pair<int, int>

        using FuncPtr = std::pair<int, int> (*)(std::tuple<int, int, int>);
        STATIC_REQUIRE(
            std::is_same_v<FuncReturn<FuncPtr>::Type,
                           std::pair<int, int>>); // return type of g is
                                                  // std::pair<int, int>
    }
}
