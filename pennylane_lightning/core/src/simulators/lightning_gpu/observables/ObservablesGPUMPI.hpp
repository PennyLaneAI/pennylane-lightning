// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <functional>
#include <vector>

#include "CSRMatrix.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "LinearAlg.hpp"
#include "Observables.hpp"
#include "StateVectorCudaMPI.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Observables;
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU::MPI;
using Pennylane::LightningGPU::StateVectorCudaMPI;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Observables {

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <typename StateVectorT>
class NamedObsMPI final : public NamedObsBase<StateVectorT> {
  private:
    using BaseType = NamedObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObsMPI(std::string obs_name, std::vector<size_t> wires,
                std::vector<PrecisionT> params = {})
        : BaseType{obs_name, wires, params} {
        using Pennylane::Gates::Constant::gate_names;
        using Pennylane::Gates::Constant::gate_num_params;
        using Pennylane::Gates::Constant::gate_wires;

        const auto gate_op = lookup(reverse_pairs(gate_names),
                                    std::string_view{this->obs_name_});
        PL_ASSERT(lookup(gate_wires, gate_op) == this->wires_.size());
        PL_ASSERT(lookup(gate_num_params, gate_op) == this->params_.size());
    }
};

/**
 * @brief Class models arbitrary Hermitian observables
 *
 */

template <class StateVectorT>
class HermitianObsMPI final : public HermitianObsBase<StateVectorT> {
  private:
    using BaseType = HermitianObsBase<StateVectorT>;
    inline static const MatrixHasher mh;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using MatrixT = std::vector<std::complex<PrecisionT>>;
    using ComplexT = typename StateVectorT::ComplexT;

    /**
     * @brief Create Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObsMPI(MatrixT matrix, std::vector<size_t> wires)
        : BaseType{matrix, wires} {}

    auto getObsName() const -> std::string final {
        // To avoid collisions on cached GPU data, use matrix elements to
        // uniquely identify Hermitian
        // TODO: Replace with a performant hash function
        std::ostringstream obs_stream;
        obs_stream << "Hermitian" << mh(this->matrix_);
        return obs_stream.str();
    }
};

/**
 * @brief Class models Tensor product observables
 */
template <class StateVectorT>
class TensorProdObsMPI final : public TensorProdObsBase<StateVectorT> {
  private:
    using BaseType = TensorProdObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObsMPI(Ts &&...arg) : BaseType{arg...} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObsMPI<StateVectorT>> {
        return std::shared_ptr<TensorProdObsMPI<StateVectorT>>{
            new TensorProdObsMPI(std::move(obs))};
    }

    static auto
    create(std::vector<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObsMPI<StateVectorT>> {
        return std::shared_ptr<TensorProdObsMPI<StateVectorT>>{
            new TensorProdObsMPI(std::move(obs))};
    }
};

/**
 * @brief General Hamiltonian as a sum of observables.
 *
 */
template <class StateVectorT>
class HamiltonianMPI final : public HamiltonianBase<StateVectorT> {
  private:
    using BaseType = HamiltonianBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    template <typename T1, typename T2>
    explicit HamiltonianMPI(T1 &&coeffs, T2 &&obs) : BaseType{coeffs, obs} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param coeffs Argument to construct coefficients
     * @param obs Argument to construct terms
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<HamiltonianMPI<StateVectorT>> {
        return std::shared_ptr<HamiltonianMPI<StateVectorT>>(
            new HamiltonianMPI<StateVectorT>{std::move(coeffs),
                                             std::move(obs)});
    }

    // to work with
    void applyInPlace(StateVectorT &sv) const override {
        using CFP_t = typename StateVectorT::CFP_t;
        DataBuffer<CFP_t, int> buffer(sv.getDataBuffer().getLength(),
                                      sv.getDataBuffer().getDevTag());
        buffer.zeroInit();

        for (size_t term_idx = 0; term_idx < this->coeffs_.size(); term_idx++) {
            DevTag<int> dt_local(sv.getDataBuffer().getDevTag());
            dt_local.refresh();
            StateVectorT tmp(dt_local, sv.getNumGlobalQubits(),
                             sv.getNumLocalQubits(), sv.getData());
            this->obs_[term_idx]->applyInPlace(tmp);
            scaleAndAddC_CUDA(
                std::complex<PrecisionT>{this->coeffs_[term_idx], 0.0},
                tmp.getData(), buffer.getData(), tmp.getLength(),
                tmp.getDataBuffer().getDevTag().getDeviceID(),
                tmp.getDataBuffer().getDevTag().getStreamID(),
                tmp.getCublasCaller());
        }

        sv.CopyGpuDataToGpuIn(buffer.getData(), buffer.getLength());
    }
};

} // namespace Pennylane::LightningGPU::Observables
