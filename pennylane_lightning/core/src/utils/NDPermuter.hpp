// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is verbatim copied from JET's permuter module at:
// https://github.com/XanaduAI/jet/tree/v0.2.2/include/jet/permute and reserves
// all licensing and attributions to that repository's implementations,
// including inspiration from QFlex https://github.com/ngnrsaa/qflex.

#pragma once

#include <complex>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "Error.hpp"

// LCOV_EXCL_START
namespace Pennylane::Util {

/**
 * @brief Interface for tensor permutation backend.
 *
 * The Permuter class represents the front-end interface for calling
 * permutations, which are a generalization of transposition to high-rank
 * tensors. The class follows a composition-based approach, where we instantiate
 * with a given backend permuter, who makes available two `Transpose` methods,
 * one which returns the transform result, and another which modifies a
 * reference directly.
 *
 *   Example 1:
 *   const std::vector<size_t> data_in {0,1,2,3,4,5};
 *   std::vector<size_t> data_out(data_in.size(), 0);
 *   Permuter<DefaultPermuter<size_t>> p;
 *   p.Transpose(data_in, {2,3}, data_out, {"a","b"}, {"b","a"});
 *
 *   Example 2:
 *   const std::vector<size_t> data_in {0,1,2,3,4,5};
 *   Permuter<DefaultPermuter<size_t>> p;
 *   auto data_out = p.Transpose(data_in, {2,3}, {"a","b"}, {"b","a"});
 *
 * @tparam PermuteBackend
 */
template <class PermuterBackend> class Permuter {
  public:
    /**
     * @brief Reshape the given lexicographic data vector from old to new index
     * ordering.
     *
     * @tparam T Data participating in the permutation.
     * @param data_in Input data to be transposed.
     * @param shape Current shape of the tensor data in each dimension.
     * @param data_out Output data following the transpose.
     * @param current_order Current index ordering of the tensor.
     * @param new_order New index ordering of the tensor.
     */
    template <class T>
    void Transpose(const std::vector<T> &data_in,
                   const std::vector<size_t> &shape, std::vector<T> &data_out,
                   const std::vector<std::string> &current_order,
                   const std::vector<std::string> &new_order) {
        const std::set<std::string> idx_old(current_order.begin(),
                                            current_order.end());
        const std::set<std::string> idx_new(new_order.begin(), new_order.end());
        const std::size_t data_size =
            std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        PL_ABORT_IF_NOT(idx_old.size() == current_order.size(),
                        "Duplicate existing indices found. Please ensure "
                        "indices are unique.");
        PL_ABORT_IF_NOT(idx_new.size() == new_order.size(),
                        "Duplicate transpose indices found. Please ensure "
                        "indices are unique.");
        PL_ABORT_IF_NOT(shape.size() == new_order.size(),
                        "Tensor shape does not match number of indices.");
        PL_ABORT_IF_NOT(data_size == data_in.size(),
                        "Tensor shape does not match given input tensor data.");
        PL_ABORT_IF_NOT(
            data_size == data_out.size(),
            "Tensor shape does not match given output tensor data.");
        PL_ABORT_IF_NOT(
            idx_old == idx_new,
            "New indices are an invalid permutation of the existing indices");

        permuter_b_.Transpose(data_in, shape, data_out, current_order,
                              new_order);
    }

    /**
     * @brief Reshape the given lexicographic data vector from old to new index
     * ordering.
     *
     * @tparam T Data participating in the permutation.
     * @param data_in Input data to be transposed.
     * @param shape Current shape of the tensor data in each dimension.
     * @param current_order Current index ordering of the tensor.
     * @param new_order New index ordering of the tensor.
     * @return std::vector<T> Output data following the transpose.
     */
    template <class T>
    std::vector<T> Transpose(const std::vector<T> &data_in,
                             const std::vector<size_t> &shape,
                             const std::vector<std::string> &current_order,
                             const std::vector<std::string> &new_order) {
        const std::set<std::string> idx_old(current_order.begin(),
                                            current_order.end());
        const std::set<std::string> idx_new(new_order.begin(), new_order.end());
        const auto data_size =
            std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        PL_ABORT_IF_NOT(idx_old.size() == current_order.size(),
                        "Duplicate existing indices found. Please ensure "
                        "indices are unique.");
        PL_ABORT_IF_NOT(idx_new.size() == new_order.size(),
                        "Duplicate transpose indices found. Please ensure "
                        "indices are unique.");
        PL_ABORT_IF_NOT(shape.size() == new_order.size(),
                        "Tensor shape does not match number of indices.");
        PL_ABORT_IF_NOT(data_size == data_in.size(),
                        "Tensor shape does not match given tensor data.");
        PL_ABORT_IF_NOT(
            idx_old == idx_new,
            "New indices are an invalid permutation of the existing indices");

        PL_ABORT_IF(shape.empty(), "Tensor shape cannot be empty.");
        PL_ABORT_IF(new_order.empty(), "Tensor indices cannot be empty.");
        return permuter_b_.Transpose(data_in, shape, current_order, new_order);
    }

  protected:
    friend PermuterBackend;

  private:
    PermuterBackend permuter_b_;
};

/**
 * @brief Default Permuter backend class for generalised transforms. Adapted
 * from QFlex.
 *
 * @tparam blocksize Controls the internal data chunk size for cache blocking.
 */
template <size_t BLOCKSIZE = 1024> class DefaultPermuter {
  public:
    /**
     * @brief Reference-based transpose operation. See `Permuter` class for more
     * details.
     */
    template <class T>
    void Transpose(const std::vector<T> &data_,
                   const std::vector<size_t> &shape, std::vector<T> &data_out,
                   const std::vector<std::string> &old_indices,
                   const std::vector<std::string> &new_indices) {
        data_out = data_;

        if (new_indices == old_indices) {
            return;
        }

        const std::size_t num_indices = old_indices.size();
        const std::size_t total_dim = data_.size();
        std::size_t remaining_data = total_dim;

        if (num_indices == 0) {
            PL_ABORT("Number of indices cannot be zero.");
        }

        // Create map_old_to_new_idxpos from old to new indices, and
        // new_dimensions.
        std::vector<size_t> map_old_to_new_idxpos(num_indices);
        std::vector<size_t> new_dimensions(num_indices);
        for (size_t i = 0; i < num_indices; ++i) {
            for (size_t j = 0; j < num_indices; ++j) {
                if (old_indices[i] == new_indices[j]) {
                    map_old_to_new_idxpos[i] = j;
                    new_dimensions[j] = shape[i];
                    break;
                }
            }
        }

        std::vector<size_t> old_super_dimensions(num_indices, 1);
        std::vector<size_t> new_super_dimensions(num_indices, 1);

        const std::size_t old_dimensions_size = shape.size();
        for (size_t i = old_dimensions_size; --i;) {
            old_super_dimensions[i - 1] = old_super_dimensions[i] * shape[i];
            new_super_dimensions[i - 1] =
                new_super_dimensions[i] * new_dimensions[i];
        }

        std::vector<std::size_t> small_map_old_to_new_position(blocksize_);

        // Position old and new.
        std::size_t po = 0;
        std::size_t pn = 0;
        // Counter of the values of each indices in the iteration (old
        // ordering).
        std::vector<size_t> old_counter(num_indices, 0);
        // offset is important when doing this in blocks, as it's indeed
        // implemented.
        std::size_t offset = 0;
        // internal_po keeps track of interations within a block.
        // Blocks have size `blocksize`.
        std::size_t internal_po = 0;

        T *data = data_out.data();
        const T *scratch =
            data_.data(); // internal pointer offers better performance than
                          // pointer from argument

        std::size_t effective_max;

        while (true) {
            // If end of entire opration, break.
            if (po == total_dim - 1) {
                break;
            }

            internal_po = 0;
            // Each iteration of the while block goes through a new position.
            // Inside the while, j takes care of increasing indices properly.
            while (true) {
                po = 0;
                pn = 0;
                for (size_t i = 0; i < num_indices; i++) {
                    po += old_super_dimensions[i] * old_counter[i];
                    pn += new_super_dimensions[map_old_to_new_idxpos[i]] *
                          old_counter[i];
                }
                small_map_old_to_new_position[po - offset] = pn;

                bool complete{true};
                // NOLINTBEGIN
                for (size_t j = num_indices; j--;) {
                    if (++old_counter[j] < shape[j]) {
                        complete = false;
                        break;
                    } else {
                        old_counter[j] = 0;
                    }
                }
                // NOLINTEND
                // If end of block or end of entire operation, break.
                if ((++internal_po == blocksize_) || (po == total_dim - 1)) {
                    break;
                }
                // If last index (0) was increased, then go back to fastest
                // index.
                if (complete) {
                    break;
                }
            }
            // Copy data for this block, taking into account offset of
            // small_map...
            effective_max = std::min(blocksize_, remaining_data);
            for (size_t p = 0; p < effective_max; p++) {
                data[small_map_old_to_new_position[p]] = scratch[offset + p];
            }

            offset += blocksize_;
            remaining_data -= blocksize_;
        }
    }

    /**
     * @brief Return-based transpose operation. See `Permuter` class for more
     * details.
     */
    template <class T>
    std::vector<T> Transpose(std::vector<T> data_,
                             const std::vector<size_t> &shape,
                             const std::vector<std::string> &old_indices,
                             const std::vector<std::string> &new_indices) {
        std::vector<T> data_out(std::move(data_));
        Transpose(data_, shape, data_out, old_indices, new_indices);
        return data_out;
    }

  private:
    static constexpr std::size_t blocksize_ = BLOCKSIZE;
};

} // namespace Pennylane::Util
// LCOV_EXCL_END
