#include <cstring>
#include <exception>

#include <catch2/catch.hpp>

#include "Error.hpp"

/**
 * @brief Test LightningException class behaves correctly
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Error.hpp", "[Error]") {
    SECTION("Raw exception") {
        const auto e = Pennylane::Util::LightningException("Test exception e");
        auto e_mut =
            Pennylane::Util::LightningException("Test exception e_mut");

        REQUIRE_THROWS_WITH(throw e,
                            Catch::Matchers::Contains("Test exception e"));
        REQUIRE_THROWS_AS(throw e, Pennylane::Util::LightningException);

        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const Pennylane::Util::LightningException e_copy(e);
        REQUIRE_THROWS_WITH(throw e_copy,
                            Catch::Matchers::Contains("Test exception e"));
        REQUIRE_THROWS_AS(throw e_copy, Pennylane::Util::LightningException);

        Pennylane::Util::LightningException e_move(std::move(e_mut));
        REQUIRE_THROWS_WITH(throw e_move,
                            Catch::Matchers::Contains("Test exception e_mut"));
        REQUIRE_THROWS_AS(throw e_move, Pennylane::Util::LightningException);

        REQUIRE(std::strcmp(e.what(), "Test exception e") == 0);
        REQUIRE(std::strcmp(e_copy.what(), "Test exception e") == 0);
        REQUIRE(std::strcmp(e_move.what(), "Test exception e_mut") == 0);
    }
    SECTION("Abort") {
        REQUIRE_THROWS_WITH(
            Pennylane::Util::Abort("Test abort", __FILE__, __LINE__, __func__),
            Catch::Matchers::Contains("Test abort"));
        REQUIRE_THROWS_AS(
            Pennylane::Util::Abort("Test abort", __FILE__, __LINE__, __func__),
            Pennylane::Util::LightningException);

        REQUIRE_THROWS_WITH(PL_ABORT("Test abort"),
                            Catch::Matchers::Contains("Test abort"));
        REQUIRE_THROWS_AS(PL_ABORT("Test abort"),
                          Pennylane::Util::LightningException);
    }
}
