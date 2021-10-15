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
/* Cache Library Header */
/* ======================================================= */

#include <list>
#include <unordered_map>
#include <vector>

#include <limits>
#include <utility>

namespace Pennylane {

namespace Util {
/**
 * @brief Least Recently Updated (LRU) cache policy class.
 *
 * This class define a container to store previously calculated indices.
 * Data is stored up to a maximum cache size, defined by the class constructor.
 * The key for accessing the cache storage is a vector and a scalar
 * representing respectively, the wires and number of qubits involved in the
 * calculation.
 *
 * @tparam key_type Type of the key associated with the value to be stored.
 * @tparam stored_type Type of the value to be stored.
 * @tparam hash_function Struct providing the hash function (optional
 * parameter).
 */
template <class key_type, class stored_type,
          typename hash_function = std::hash<key_type>>
class LRU_cache {
  private:
    using const_iterator_map_type =
        typename std::unordered_map<key_type, stored_type>::const_iterator;
    using const_iterator_list_type =
        typename std::list<key_type>::const_iterator;

    size_t max_size;
    std::list<key_type> lru_queue_;
    std::unordered_map<key_type, stored_type, hash_function> cache_map_;
    std::unordered_map<key_type, const_iterator_list_type, hash_function>
        key_ring_;

    void clear() {
        lru_queue_.clear();
        cache_map_.clear();
    }

    size_t cache_size_;

  public:
    /**
     * @brief Construct a new LRU_cache object.
     */
    LRU_cache(size_t cache_size = 0) : cache_size_{cache_size} {};

    /**
     * @brief Iterator for the map element.
     *
     * @param new_key key for the desired element.
     * @return an iterator to the stored element, if found, or a past-the-end
     * iterator if not.
     */
    typename std::unordered_map<key_type, stored_type>::const_iterator
    check_cache(const key_type &new_key) {
        return cache_map_.find(new_key);
    };

    /**
     * @brief Insert a new element to the cache storage.
     *
     * @param new_key is a key for this element.
     * @param new_value is the element to be stored.
     */
    void insert(const key_type &new_key, const stored_type &new_value) {
        // opening a slot.
        if (cache_size_ == 0)
            return;
        if (cache_map_.size() >= cache_size_) {
            auto lru_key = lru_queue_.back();
            cache_map_.erase(lru_key);
            lru_queue_.pop_back();
        }
        // inserting new element.
        cache_map_[new_key] = new_value;
        lru_queue_.emplace_front(new_key);
        key_ring_[new_key] = lru_queue_.begin();
    };

    /**
     * @brief Change the cache storage size.
     *
     * @param new_size is the new size desired.
     */
    void set_cache_size(size_t new_size) {
        cache_size_ = new_size;
        while (cache_map_.size() > cache_size_) {
            auto lru_key = lru_queue_.back();
            cache_map_.erase(lru_key);
            lru_queue_.pop_back();
        }
    };

    /**
     * @brief Renew the stay of an element already present at the cache.
     *
     * @param key is the key to the element.
     */
    void renew(const key_type &key) {
        lru_queue_.splice(lru_queue_.begin(), lru_queue_, key_ring_[key]);
    };

    /**
     * @brief Get the stored element.
     *
     * @param element_it iterator to the element.
     * @return stored element.
     */
    stored_type get(const_iterator_map_type element_it) {
        renew(element_it->first);
        return element_it->second;
    };

    /**
     * @brief Get the key of a stored element.
     *
     * @param element_it iterator to the element.
     * @return stored element key.
     */
    key_type get_key(const_iterator_map_type element_it) {
        return element_it->first;
    };

    /**
     * @brief Iterator for the first element in the storage.
     *
     */
    const_iterator_map_type begin() const { return cache_map_.begin(); }

    /**
     * @brief Iterator for the last element in the storage.
     *
     */
    const_iterator_map_type end() const { return cache_map_.end(); }

    /**
     * @brief Returns the occupancy of the cache storage.
     *
     */
    size_t size() const { return cache_map_.size(); }

    /**
     * @brief Returns the capacity of the cache storage.
     *
     */
    size_t capacity() const { return cache_size_; }

    ~LRU_cache() { clear(); }
};

} // namespace Util
} // namespace Pennylane
