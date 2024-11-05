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

/**
 * @file MPSTNCuda.hpp
 * MPS class with cuTensorNet backend. Note that current implementation only
 * support the open boundary condition.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "TNCudaBase.hpp"
#include "TensorCuda.hpp"
#include "TensornetBase.hpp"
#include "Util.hpp"
#include "cuda_helpers.hpp"
#include "tncudaError.hpp"
#include "tncuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda {

/**
 * @brief Managed memory Exact Tensor Network class using cutensornet high-level
 * APIs backed.
 *
 * @tparam Precision Floating-point precision type.
 */

template <class Precision>
class ExaTNCuda final : public TNCudaBase<Precision, ExaTNCuda<Precision>> {
  private:
    using BaseType = TNCudaBase<Precision, ExaTNCuda>;

    const std::string method_ = "exatn";

    const std::vector<std::vector<std::size_t>> sitesModes_;
    const std::vector<std::vector<std::size_t>> sitesExtents_;
    const std::vector<std::vector<int64_t>> sitesExtents_int64_;

  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  public:
    ExaTNCuda() = delete;

    explicit ExaTNCuda(const std::size_t numQubits)
        : BaseType(numQubits), sitesModes_(setSitesModes_()),
          sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
        BaseType::reset();
        BaseType::appendInitialMPSState(getSitesExtentsPtr().data());
    }

    explicit ExaTNCuda(const std::size_t numQubits, DevTag<int> dev_tag)
        : BaseType(numQubits, dev_tag), sitesModes_(setSitesModes_()),
          sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
        BaseType::reset();
        BaseType::appendInitialMPSState(getSitesExtentsPtr().data());
    }

    ~ExaTNCuda() = default;

    /**
     * @brief Get tensor network method name.
     *
     * @return std::string
     */
    [[nodiscard]] auto getMethod() const -> std::string { return method_; }

    /**
     * @brief Get a vector of pointers to extents of each site.
     *
     * @return std::vector<int64_t const *> Note int64_t const* is
     * required by cutensornet backend.
     */
    [[nodiscard]] auto getSitesExtentsPtr() -> std::vector<int64_t const *> {
        std::vector<int64_t const *> sitesExtentsPtr_int64(
            BaseType::getNumQubits());
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            sitesExtentsPtr_int64[i] = sitesExtents_int64_[i].data();
        }
        return sitesExtentsPtr_int64;
    }

    [[nodiscard]] auto getBondDims([[maybe_unused]] const std::size_t idx) const
        -> std::size_t {
        PL_ABORT("Not supported in Exact Tensor Network.");
        return 1;
    }

  private:
    /**
     * @brief Return siteModes to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<std::size_t>>
     */
    std::vector<std::vector<std::size_t>> setSitesModes_() {
        std::vector<std::vector<std::size_t>> localSitesModes;
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteModes = {i};
            localSitesModes.push_back(std::move(localSiteModes));
        }
        return localSitesModes;
    }

    /**
     * @brief Return sitesExtents to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<std::size_t>>
     */
    std::vector<std::vector<std::size_t>> setSitesExtents_() {
        std::vector<std::vector<std::size_t>> localSitesExtents;

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteExtents{
                BaseType::getQubitDims()[i]};
            localSitesExtents.push_back(std::move(localSiteExtents));
        }
        return localSitesExtents;
    }

    /**
     * @brief Return siteExtents_int64 to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<int64_t>>
     */
    std::vector<std::vector<int64_t>> setSitesExtents_int64_() {
        std::vector<std::vector<int64_t>> localSitesExtents_int64;

        for (const auto &siteExtents : sitesExtents_) {
            localSitesExtents_int64.push_back(
                std::move(Pennylane::Util::cast_vector<std::size_t, int64_t>(
                    siteExtents)));
        }
        return localSitesExtents_int64;
    }

    /**
     * @brief The tensors init helper function for ctor.
     */
    void initTensors_() {
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            // construct mps tensors reprensentation
            this->tensors_.emplace_back(sitesModes_[i].size(), sitesModes_[i],
                                        sitesExtents_[i],
                                        BaseType::getDevTag());

            this->tensors_out_.emplace_back(sitesModes_[i].size(),
                                            sitesModes_[i], sitesExtents_[i],
                                            BaseType::getDevTag());
        }
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
