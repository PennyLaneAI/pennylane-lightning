#include <catch2/catch.hpp>

#include <cassert>
#include <exception>

class TestExceptionInConstexprConstructor {
  public:
    explicit constexpr TestExceptionInConstexprConstructor(int t) {
        if (t == 0) {
            throw std::invalid_argument("Throw");
        }
    }
};

constexpr int testAssertInConstexprFunction(int t) {
    assert(t != 0);
    return t;
}

TEST_CASE("Test exception in a constexpr constructor") {
    [[maybe_unused]] constexpr static auto t =
        TestExceptionInConstexprConstructor(1);

    // The following line must not be compiled
    //[[maybe_unused]] constexpr static auto t1 =
    // TestExceptionInConstexprConstructor(0);
}

TEST_CASE("Test assert in a constexpr function") {
    [[maybe_unused]] constexpr static auto t = testAssertInConstexprFunction(1);

    // The following line must not be compiled
    //[[maybe_unused]] constexpr static auto t1 =
    // testAssertInConstexprFunction(0);
}
