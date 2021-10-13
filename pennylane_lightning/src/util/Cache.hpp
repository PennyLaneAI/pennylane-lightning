// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Least Recently Updated (LRU) cache policy class implementation.
 */

#pragma once

/* ======================================================= */
/* LRU Library Header */
/* ======================================================= */

#include <list>
#include <unordered_map>
#include <vector>

#include <limits>
#include <utility>

typedef std::pair<std::vector<size_t>, size_t> key_type;
typedef std::vector<size_t> stored_type;
typedef std::unordered_map<key_type, stored_type>::const_iterator
    const_iterator_type;

namespace Pennylane {

// Boost implementation of a hash combine:
// https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
struct pair_hash {
    std::size_t operator()(const key_type &key) const {
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

/**
 * @brief Least Recently Updated (LRU) cache policy class.
 *
 * This class define a container to store previously calculated indices.
 * Data is stored up to a maximum cache size, defined by the class constructor.
 * The key for accessing the cache storage is a vector and a scalar
 * representing respectively, the wires and number of qubits involved in the
 * calculation.
 *
 */
class LRU_cache {
  private:
    size_t max_size;
    std::list<key_type> lru_queue;
    std::unordered_map<key_type, stored_type, pair_hash> cache_map;
    std::unordered_map<key_type, std::list<key_type>::iterator, pair_hash>
        key_ring;

    void clear() {
        max_size = {};
        lru_queue.clear();
        cache_map.clear();
    }

  public:
    /**
     * @brief Construct a new LRU_cache object.
     *
     * The default cache size is 1.
     * If set to 0, the size will be reescaled to the maximum possible size.
     *
     * @param asked_max_size Maximum size of the cache object.
     */
    LRU_cache(size_t asked_max_size = 1) : max_size{asked_max_size} {
        if (max_size == 0) {
            max_size = std::numeric_limits<size_t>::max();
        }
    };

    /**
     * @brief Iterator for the map element.
     *
     * @param new_key key for the desired element.
     * @return an iterator to the stored element, if found, or a past-the-end
     * iterator if not.
     */
    typename std::unordered_map<key_type, stored_type>::const_iterator
    check_cache(const key_type &new_key) {
        return cache_map.find(new_key);
    };

    /**
     * @brief Insert a new element to the cache storage.
     *
     * @param new_key is a key for this element.
     * @param new_value is the element to be stored.
     */
    void insert(const key_type &new_key, const stored_type &new_value) {
        // opening a slot.
        if (cache_map.size() >= max_size) {
            auto lru_key = lru_queue.back();
            cache_map.erase(lru_key);
            lru_queue.pop_back();
        }
        // inserting new element.
        cache_map[new_key] = new_value;
        lru_queue.emplace_front(new_key);
        key_ring[new_key] = lru_queue.begin();
    };

    /**
     * @brief Renew the stay of an element already present at the cache.
     *
     * @param key is the key to the element.
     */
    void renew(const key_type &key) {
        lru_queue.splice(lru_queue.begin(), lru_queue, key_ring[key]);
    };

    /**
     * @brief Get the stored element.
     *
     * @param element_it iterator to the element.
     * @return stored element.
     */
    stored_type get(const_iterator_type element_it) {
        renew(element_it->first);
        return element_it->second;
    };

    /**
     * @brief Get the key of a stored element.
     *
     * @param element_it iterator to the element.
     * @return stored element key.
     */
    key_type get_key(const_iterator_type element_it) {
        return element_it->first;
    };

    /**
     * @brief Iterator for the first element in the storage.
     *
     */
    const_iterator_type begin() const { return cache_map.begin(); }

    /**
     * @brief Iterator for the last element in the storage.
     *
     */
    const_iterator_type end() const { return cache_map.end(); }

    /**
     * @brief Returns the occupancy of the cache storage.
     *
     */
    size_t size() const { return cache_map.size(); }

    ~LRU_cache() { clear(); }
};

} // namespace Pennylane
