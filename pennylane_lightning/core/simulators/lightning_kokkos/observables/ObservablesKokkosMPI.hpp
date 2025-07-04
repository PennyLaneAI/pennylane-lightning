// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Error.hpp"
#include "LinearAlgebraKokkos.hpp"
#include "Observables.hpp"
#include "StateVectorKokkosMPI.hpp"
#include "Util.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Observables;
using Pennylane::LightningKokkos::StateVectorKokkosMPI;
using Pennylane::LightningKokkos::Util::SparseMV_Kokkos;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Observables {
/**
 * @brief Final class for named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
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
    NamedObsMPI(std::string obs_name, std::vector<std::size_t> wires,
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
 * @brief Final class for Hermitian observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class HermitianObsMPI final : public HermitianObsBase<StateVectorT> {
  private:
    using BaseType = HermitianObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObsMPI(MatrixT matrix, std::vector<std::size_t> wires)
        : BaseType{matrix, wires} {}
};

/**
 * @brief Final class for TensorProdObs observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class TensorProdObsMPI final : public TensorProdObsBase<StateVectorT> {
  private:
    using BaseType = TensorProdObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
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
 * @brief Final class for a general Hamiltonian representation as a sum of
 * observables.
 *
 * @tparam StateVectorT State vector class.
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
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<HamiltonianMPI<StateVectorT>> {
        return std::shared_ptr<HamiltonianMPI<StateVectorT>>(
            new HamiltonianMPI<StateVectorT>{std::move(coeffs),
                                             std::move(obs)});
    }

    /**
     * @brief Updates the statevector sv:->sv'.
     * @param sv The statevector to update
     */
    void applyInPlace(StateVectorT &sv) const override {
        StateVectorT buffer(sv);
        buffer.initZeros();
        StateVectorT tmp(sv);

        for (std::size_t term_idx = 0; term_idx < this->coeffs_.size();
             term_idx++) {
            tmp.updateData(sv);
            this->obs_[term_idx]->applyInPlace(tmp);
            buffer.matchWires(tmp);
            LightningKokkos::Util::axpy_Kokkos<PrecisionT>(
                ComplexT{this->coeffs_[term_idx], 0.0}, tmp.getView(),
                buffer.getView(), tmp.getLocalBlockSize());
        }
        sv.updateData(buffer);
    }
};

} // namespace Pennylane::LightningKokkos::Observables
