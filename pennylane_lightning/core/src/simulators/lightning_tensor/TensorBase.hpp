#pragma once

#include <vector>

#include "Error.hpp"

namespace Pennylane::LightningTensor {
//  column-major by default for the tensor discriptor
/**
 * @brief CRTP-enabled base class for tensor.
 *
 * @tparam Precision Floating point precision.
 * @tparam Derived Derived class to instantiate using CRTP.
 */

template <class PrecisionT, class Derived> class TensorBase {
  private:
    size_t rank_;                 // A rank N tensor has N modes
    size_t length_;               // Number of elements
    std::vector<size_t> modes_;   // modes for contraction identify
    std::vector<size_t> extents_; // Number of elements in each mode

  public:
    TensorBase(size_t rank, std::vector<size_t> &modes,
               std::vector<size_t> &extents)
        : rank_(rank), modes_(modes), extents_(extents) {
        PL_ABORT_IF(rank_ != extents_.size(),
                    "Please check if rank or extents are set correctly.");
        length_ = 1;
        for (auto extent : extents) {
            length_ *= extent;
        }
    };

    virtual ~TensorBase() {}

    auto getRank() -> size_t { return rank_; }

    auto getExtents() -> std::vector<size_t> { return extents_; }

    auto getModes() -> std::vector<size_t> { return modes_; };

    size_t getLength() const { return length_; }

    auto getData() { return static_cast<Derived *>(this)->getData(); }
};
} // namespace Pennylane::LightningTensor