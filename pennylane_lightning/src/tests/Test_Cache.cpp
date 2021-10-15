#include <catch2/catch.hpp>

#include "Cache.hpp"

using namespace Pennylane::Util;

// Boost implementation of a hash combine:
// https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
struct hash_function {
    std::size_t
    operator()(const std::pair<std::vector<size_t>, size_t> &key) const {
        std::size_t combined_hash_value = 0;

        for (auto &term : key.first) {
            combined_hash_value ^= std::hash<size_t>()(term) + 0x9e3779b9 +
                                   (combined_hash_value << 6) +
                                   (combined_hash_value >> 2);
        };
        combined_hash_value ^= std::hash<size_t>()(key.second) + 0x9e3779b9 +
                               (combined_hash_value << 6) +
                               (combined_hash_value >> 2);
        return combined_hash_value;
    }
};

TEST_CASE("LRU_Cache for doubles, with sise_t keys", "[Caching_Doubles]") {

    LRU_cache<size_t, double> cache_container{10};

    REQUIRE(cache_container.size() == 0);
    REQUIRE(cache_container.capacity() == 10);

    std::vector<size_t> keys = {1, 2,  3,  4,  5,  6,  7, 8,
                                9, 10, 11, 12, 13, 14, 15};

    std::vector<double> values_to_store = {1.2,  2.3,  3.4,  4.5,  5.6,
                                           6.7,  7.8,  8.9,  9.1,  10.2,
                                           11.3, 12.4, 13.5, 14.6, 15.7};

    size_t number_of_values = values_to_store.size();

    size_t value_index = 0;
    for (auto &term : keys) {
        auto element_it = cache_container.check_cache(term);
        if (element_it == cache_container.end()) {
            cache_container.insert(term, values_to_store[value_index++]);
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
        REQUIRE(cache_container.get(element_it) == values_to_store.back());
    }

    SECTION("Checking if elements in the cache are the ones most recently "
            "introduced.") {
        for (size_t value_index = 0;
             value_index < (number_of_values - cache_container.size());
             value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(element_it == cache_container.end());
        }

        for (size_t value_index = (number_of_values - cache_container.size());
             value_index < number_of_values; value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }
    }

    cache_container.set_cache_size(5);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 5);
        REQUIRE(cache_container.capacity() == 5);
    }

    SECTION("After re-scaling, check if elements in the cache are the ones "
            "most recently introduced.") {
        for (size_t value_index = 0;
             value_index < (number_of_values - cache_container.size());
             value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(element_it == cache_container.end());
        }

        for (size_t value_index = (number_of_values - cache_container.size());
             value_index < number_of_values; value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }
    }

    cache_container.set_cache_size(7);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 5);
        REQUIRE(cache_container.capacity() == 7);
    }

    for (size_t value_index = 0; value_index < number_of_values;
         value_index++) {
        // keeping the first element alive, if it is there.
        auto element_it_0 = cache_container.check_cache(keys[0]);
        if (element_it_0 != cache_container.end()) {
            auto renewed_element = cache_container.get(element_it_0);
        }

        auto element_it = cache_container.check_cache(keys[value_index]);
        if (element_it == cache_container.end()) {
            cache_container.insert(keys[value_index],
                                   values_to_store[value_index]);
        }
    }

    cache_container.set_cache_size(7);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 7);
        REQUIRE(cache_container.capacity() == 7);
    }

    SECTION("Checking if we only kept the most recently updated elements.") {
        {
            size_t value_index = 0;
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }

        for (size_t value_index = 1;
             value_index < (number_of_values - cache_container.size() + 1);
             value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(element_it == cache_container.end());
        }

        for (size_t value_index =
                 (number_of_values - cache_container.size() + 1);
             value_index < number_of_values; value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }
    }
}

TEST_CASE("LRU_Cache for StateVector indices", "[Caching_Indices]") {

    LRU_cache<std::pair<const std::vector<size_t>, size_t>, std::vector<size_t>,
              hash_function>
        cache_container{10};

    REQUIRE(cache_container.size() == 0);
    REQUIRE(cache_container.capacity() == 10);

    std::vector<std::pair<const std::vector<size_t>, size_t>> keys = {
        {{1, 1, 1, 1}, 1},  {{1, 1, 1, 2}, 2},  {{1, 1, 2, 1}, 3},
        {{1, 2, 1, 1}, 4},  {{1, 2, 1, 3}, 5},  {{1, 2, 3, 1}, 6},
        {{1, 2, 3, 4}, 7},  {{1, 1, 1, 1}, 8},  {{1, 1, 1, 2}, 9},
        {{1, 1, 2, 1}, 10}, {{1, 2, 1, 1}, 11}, {{1, 2, 1, 3}, 12},
        {{1, 2, 3, 1}, 13}, {{1, 2, 3, 4}, 14}};

    std::vector<std::vector<size_t>> values_to_store = {
        {1, 1, 1, 1},     {3, 3, 3, 3},     {5, 5, 5, 5},     {7, 7, 7, 7},
        {9, 9, 9, 9},     {11, 11, 11, 11}, {13, 13, 13, 13}, {15, 15, 15, 15},
        {17, 17, 17, 17}, {19, 19, 19, 19}, {21, 21, 21, 21}, {23, 23, 23, 23},
        {25, 25, 25, 25}, {27, 27, 27, 27}};

    size_t number_of_values = values_to_store.size();

    size_t value_index = 0;
    for (auto &term : keys) {
        auto element_it = cache_container.check_cache(term);
        if (element_it == cache_container.end()) {
            cache_container.insert(term, values_to_store[value_index++]);
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
        REQUIRE(cache_container.get(element_it) == values_to_store.back());
    }

    SECTION("Checking if elements in the cache are the ones most recently "
            "introduced.") {
        for (size_t value_index = 0;
             value_index < (number_of_values - cache_container.size());
             value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(element_it == cache_container.end());
        }

        for (size_t value_index = (number_of_values - cache_container.size());
             value_index < number_of_values; value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }
    }

    cache_container.set_cache_size(5);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 5);
        REQUIRE(cache_container.capacity() == 5);
    }

    SECTION("After re-scaling, check if elements in the cache are the ones "
            "most recently introduced.") {
        for (size_t value_index = 0;
             value_index < (number_of_values - cache_container.size());
             value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(element_it == cache_container.end());
        }

        for (size_t value_index = (number_of_values - cache_container.size());
             value_index < number_of_values; value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }
    }

    cache_container.set_cache_size(7);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 5);
        REQUIRE(cache_container.capacity() == 7);
    }

    for (size_t value_index = 0; value_index < number_of_values;
         value_index++) {
        // keeping the first element alive, if it is there.
        auto element_it_0 = cache_container.check_cache(keys[0]);
        if (element_it_0 != cache_container.end()) {
            auto renewed_element = cache_container.get(element_it_0);
        }

        auto element_it = cache_container.check_cache(keys[value_index]);
        if (element_it == cache_container.end()) {
            cache_container.insert(keys[value_index],
                                   values_to_store[value_index]);
        }
    }

    cache_container.set_cache_size(7);
    SECTION("Checking new container capacity. ") {
        REQUIRE(cache_container.size() == 7);
        REQUIRE(cache_container.capacity() == 7);
    }

    SECTION("Checking if we only kept the most recently updated elements.") {
        {
            size_t value_index = 0;
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }

        for (size_t value_index = 1;
             value_index < (number_of_values - cache_container.size() + 1);
             value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(element_it == cache_container.end());
        }

        for (size_t value_index =
                 (number_of_values - cache_container.size() + 1);
             value_index < number_of_values; value_index++) {
            auto element_it = cache_container.check_cache(keys[value_index]);
            REQUIRE(cache_container.get_key(element_it) == keys[value_index]);
        }
    }
}
