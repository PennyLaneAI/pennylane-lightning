#include <catch2/catch.hpp>

#include "Cache.hpp"

using namespace Pennylane::Util;

namespace {

TEST_CASE("LRU_Cache", "[Caching_Indices]") {

    LRU_cache<std::pair<const std::vector<size_t>, size_t>, std::vector<size_t>>
        cache_container{10};

    REQUIRE(cache_container.size() == 0);
    REQUIRE(cache_container.capacity() == 10);

    std::vector<std::pair<const std::vector<size_t>, size_t>> keys = {
        {{1, 1, 1, 1}, 1},  {{1, 1, 1, 2}, 2},  {{1, 1, 2, 1}, 3},
        {{1, 2, 1, 1}, 4},  {{1, 2, 1, 3}, 5},  {{1, 2, 3, 1}, 6},
        {{1, 2, 3, 4}, 7},  {{1, 1, 1, 1}, 8},  {{1, 1, 1, 2}, 9},
        {{1, 1, 2, 1}, 10}, {{1, 2, 1, 1}, 11}, {{1, 2, 1, 3}, 12},
        {{1, 2, 3, 1}, 13}, {{1, 2, 3, 4}, 14}};

    size_t number_of_keys = keys.size();

    std::vector<std::vector<size_t>> indices = {
        {1, 1, 1, 1},     {3, 3, 3, 3},     {5, 5, 5, 5},     {7, 7, 7, 7},
        {9, 9, 9, 9},     {11, 11, 11, 11}, {13, 13, 13, 13}, {15, 15, 15, 15},
        {17, 17, 17, 17}, {19, 19, 19, 19}, {21, 21, 21, 21}, {23, 23, 23, 23},
        {25, 25, 25, 25}, {27, 27, 27, 27}};

    size_t nn = 0;
    for (auto &term : keys) {
        auto element_it = cache_container.check_cache(term);
        if (element_it == cache_container.end()) {
            cache_container.insert(term, indices[nn++]);
        }
    }

    SECTION("Introducing elements in the cache. [beyond capacity] ") {
        REQUIRE(cache_container.size() == 10);
        REQUIRE(cache_container.capacity() == 10);
    }

    SECTION("Testing a single missing value.") {
        auto element_it = cache_container.check_cache(keys[0]);
        REQUIRE(element_it == cache_container.end());
    }

    SECTION("Getting a single value.") {
        auto element_it = cache_container.check_cache(keys.back());
        REQUIRE(cache_container.get_key(element_it) == keys.back());
        REQUIRE(cache_container.get(element_it) == indices.back());
    }

    SECTION("Checking if elements in the cache are the ones most recently "
            "introduced.") {
        for (int nn = 0; nn < (number_of_keys - cache_container.size()); nn++) {
            auto element_it = cache_container.check_cache(keys[nn]);
            REQUIRE(element_it == cache_container.end());
        }

        for (int nn = (number_of_keys - cache_container.size());
             nn < number_of_keys; nn++) {
            auto element_it = cache_container.check_cache(keys[nn]);
            REQUIRE(cache_container.get_key(element_it) == keys[nn]);
        }
    }

    cache_container.set_cache_size(5);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 5);
        REQUIRE(cache_container.capacity() == 5);
    }

    SECTION("After re-scaling, check if elements in the cache are the ones "
            "most recently introduced.") {
        for (int nn = 0; nn < (number_of_keys - cache_container.size()); nn++) {
            auto element_it = cache_container.check_cache(keys[nn]);
            REQUIRE(element_it == cache_container.end());
        }

        for (int nn = (number_of_keys - cache_container.size());
             nn < number_of_keys; nn++) {
            auto element_it = cache_container.check_cache(keys[nn]);
            REQUIRE(cache_container.get_key(element_it) == keys[nn]);
        }
    }

    cache_container.set_cache_size(7);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 5);
        REQUIRE(cache_container.capacity() == 7);
    }

    for (int nn = 0; nn < number_of_keys; nn++) {
        // keeping the first element alive, if it is there.
        auto element_it_0 = cache_container.check_cache(keys[0]);
        if (element_it_0 != cache_container.end()) {
            auto renewed_element = cache_container.get(element_it_0);
        }

        auto element_it = cache_container.check_cache(keys[nn]);
        if (element_it == cache_container.end()) {
            cache_container.insert(keys[nn], indices[nn]);
        }
    }

    cache_container.set_cache_size(7);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 7);
        REQUIRE(cache_container.capacity() == 7);
    }

    SECTION("Checking if we only kept the most recently updated elements.") {
        {
            int nn = 0;
            auto element_it = cache_container.check_cache(keys[nn]);
            REQUIRE(cache_container.get_key(element_it) == keys[nn]);
        }

        for (int nn = 1; nn < (number_of_keys - cache_container.size() + 1);
             nn++) {
            auto element_it = cache_container.check_cache(keys[nn]);
            REQUIRE(element_it == cache_container.end());
        }

        for (int nn = (number_of_keys - cache_container.size() + 1);
             nn < number_of_keys; nn++) {
            auto element_it = cache_container.check_cache(keys[nn]);
            REQUIRE(cache_container.get_key(element_it) == keys[nn]);
        }
    }
}

} // namespace
