#include <catch2/catch.hpp>

#include <exception>

class TestExceptionInConstexprConstructor {
  public:
    explicit constexpr TestExceptionInConstexprConstructor(int t) {
        if (t == 0) {
            throw std::invalid_argument("Throw");
        }
    }
};

TEST_CASE("Test exception in a constexpr constructor") {
    [[maybe_unused]] constexpr static auto t =
        TestExceptionInConstexprConstructor(1);
    REQUIRE_THROWS(TestExceptionInConstexprConstructor(0));
}
