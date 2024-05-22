// Copyright 2024 Xanadu Quantum Technologies Inc. and contributors.

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

#include <cuda.h>
#include <unordered_set>
#include <vector>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "TensorCuda.hpp"
#include "Util.hpp"

#include "cuError.hpp"
#include "cuGates_host.hpp"
#include "tncudaError.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor;

template <class T> using vector1D = std::vector<T>;
template <class T> using vector2D = std::vector<std::vector<T>>;
template <class T> using vector3D = std::vector<std::vector<std::vector<T>>>;

} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Observables {
/**
 * @brief A base class for observable classes of cutensornet backends.
 *
 * We note that the observable classes for cutensornet backends are designed to
 * be created in the same way as the observable classes for the statevector
 * backends across the lightning ecosystem. However, the main difference between
 * the observable objects for cutensornet backends and those for statevector
 * backends is that the former store the tensor data in the observable base
 * class. This is achieved by treating different types of observables as a
 * subset of a Hamiltonian observables. This design is to ensure that the easy
 * construction of an observable from the Python layer and its corresponding
 * cutensornet network operator object.
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT> class Observable {
  public:
    using CFP_t = typename StateTensorT::CFP_t;
    using PrecisionT = typename StateTensorT::PrecisionT;
    using ComplexT = typename StateTensorT::ComplexT;

  protected:
    vector1D<PrecisionT> coeffs_;      // coefficients of each term
    vector1D<std::size_t> numTensors_; // number of tensors in each term
    vector2D<std::size_t>
        numStateModes_; // number of state modes of each tensor in each term
    vector3D<std::size_t>
        stateModes_;       // state modes of each tensor in each term
    vector3D<CFP_t> data_; // data of each tensor in each term on host

  protected:
    Observable() = default;
    Observable(const Observable &) = default;
    Observable(Observable &&) noexcept = default;
    Observable &operator=(const Observable &) = default;
    Observable &operator=(Observable &&) noexcept = default;

  public:
    virtual ~Observable() = default;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<std::size_t> = 0;

    /**
     * @brief Get the number of tensors in each term (For non-Hamiltonian
     * observables, the size of std::vector return is 1).
     */
    [[nodiscard]] auto getNumTensors() const -> const vector1D<std::size_t> & {
        return numTensors_;
    }

    /**
     * @brief Get the number of state modes of each tensor in each term.
     */
    [[nodiscard]] auto getNumStateModes() const
        -> const vector2D<std::size_t> & {
        return numStateModes_;
    }

    /**
     * @brief Get the state modes of each tensor in each term.
     */
    [[nodiscard]] auto getStateModes() const -> const vector3D<std::size_t> & {
        return stateModes_;
    }

    /**
     * @brief Get the data of each tensor in each term on host.
     */
    [[nodiscard]] auto getData() const -> const vector3D<CFP_t> & {
        return data_;
    }

    /**
     * @brief Get the coefficients of a observable.
     */
    [[nodiscard]] auto getCoeffsPerTerm() const
        -> const vector1D<PrecisionT> & {
        return coeffs_;
    };
};

/**
 * @brief Named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT> class NamedObs : public Observable<StateTensorT> {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using CFP_t = typename StateTensorT::CFP_t;

  private:
    std::string obs_name_;
    std::vector<std::size_t> wires_;
    std::vector<PrecisionT> params_;

  public:
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObs(std::string obs_name, std::vector<std::size_t> wires,
             std::vector<PrecisionT> params = {})
        : obs_name_{obs_name}, wires_{wires}, params_{params} {
        this->coeffs_.push_back(PrecisionT{1.0});
        this->numTensors_.push_back(std::size_t{1});
        this->numStateModes_.push_back(
            vector1D<std::size_t>(std::size_t{1}, wires.size()));
        this->stateModes_.push_back(
            vector2D<std::size_t>(std::size_t{1}, wires));
        auto gateData =
            cuGates::DynamicGateDataAccess<PrecisionT>::getInstance()
                .getGateData(obs_name, params);
        this->data_.push_back(vector2D<CFP_t>(std::size_t{1}, gateData));
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return wires_;
    }
};

/**
 * @brief Hermitian observables
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT>
class HermitianObs : public Observable<StateTensorT> {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using CFP_t = typename StateTensorT::CFP_t;
    using ComplexT = typename StateTensorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

  private:
    MatrixT matrix_;
    std::vector<std::size_t> wires_;

  public:
    /**
     * @brief Construct a HermitianObs object, representing a given observable.
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObs(MatrixT matrix, std::vector<std::size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {
        this->coeffs_.push_back(PrecisionT{1.0});
        this->numTensors_.push_back(std::size_t{1});
        this->numStateModes_.push_back(
            vector1D<std::size_t>(std::size_t{1}, wires_.size()));
        this->stateModes_.push_back(
            vector2D<std::size_t>(std::size_t{1}, wires_));
        // Convert matrix to vector of vector
        std::vector<CFP_t> matrix_cu(matrix_.size());
        std::transform(
            matrix_.begin(), matrix_.end(), matrix_cu.begin(),
            [](const ComplexT &x) { return cuUtil::complexToCu<ComplexT>(x); });
        this->data_.push_back(vector2D<CFP_t>(std::size_t{1}, matrix_cu));
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        return "Hermitian";
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return wires_;
    }
};

/**
 * @brief Tensor product of observables.
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT>
class TensorProdObs : public Observable<StateTensorT> {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using CFP_t = typename StateTensorT::CFP_t;
    using ComplexT = typename StateTensorT::ComplexT;

  protected:
    std::vector<std::shared_ptr<Observable<StateTensorT>>> obs_;
    std::vector<std::size_t> all_wires_;

  public:
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObs(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        if (obs_.size() == 1 &&
            obs_[0]->getObsName().find('@') != std::string::npos) {
            // This would prevent the misuse of this constructor for creating
            // TensorProdObs(TensorProdObs).
            PL_ABORT("A new TensorProdObs observable cannot be created "
                     "from a single TensorProdObs.");
        }

        for (const auto &ob : obs_) {
            PL_ABORT_IF(ob->getObsName().find('@') != std::string::npos,
                        "A TensorProdObs observable cannot be created from a "
                        "TensorProdObs.");
        }

        for (const auto &ob : obs_) {
            PL_ABORT_IF(ob->getObsName().find("Hamiltonian") !=
                            std::string::npos,
                        "A TensorProdObs observable cannot be created from a "
                        "Hamiltonian.");
        }

        this->coeffs_.push_back(PrecisionT{1.0});
        this->numTensors_.push_back(obs_.size());

        vector1D<std::size_t> numStateModesLocal;
        vector2D<std::size_t> stateModesLocal;
        vector2D<CFP_t> dataLocal;
        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            numStateModesLocal.emplace_back(ob_wires.size());
            stateModesLocal.emplace_back(ob_wires);
            dataLocal.emplace_back(ob->getData().front().front());
        }

        this->numStateModes_.emplace_back(numStateModesLocal);
        this->stateModes_.emplace_back(stateModesLocal);
        this->data_.emplace_back(dataLocal);

        std::unordered_set<std::size_t> wires;
        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                PL_ABORT_IF(wires.contains(wire),
                            "All wires in observables must be disjoint.");
                wires.insert(wire);
            }
        }
        all_wires_ = std::vector<std::size_t>(wires.begin(), wires.end());
        std::sort(all_wires_.begin(), all_wires_.end());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     * @return std::shared_ptr<TensorProdObs<StateTensorT>>
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<StateTensorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateTensorT>> {
        return std::shared_ptr<TensorProdObs<StateTensorT>>{
            new TensorProdObs(std::move(obs))};
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     * @return std::shared_ptr<TensorProdObs<StateTensorT>>
     */
    static auto
    create(std::vector<std::shared_ptr<Observable<StateTensorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateTensorT>> {
        return std::shared_ptr<TensorProdObs<StateTensorT>>{
            new TensorProdObs(std::move(obs))};
    }

    /**
     * @brief Get the number of operations in observable.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getSize() const -> std::size_t { return obs_.size(); }

    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::size_t>&
     */
    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return all_wires_;
    }

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        const auto obs_size = obs_.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            obs_stream << obs_[idx]->getObsName();
            if (idx != obs_size - 1) {
                obs_stream << " @ ";
            }
        }
        return obs_stream.str();
    }
};

/**
 * @brief Hamiltonian representation as a sum of observables.
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT>
class Hamiltonian : public Observable<StateTensorT> {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using CFP_t = typename StateTensorT::CFP_t;
    using ComplexT = typename StateTensorT::ComplexT;

  private:
    std::vector<PrecisionT> coeffs_ham_;
    std::vector<std::shared_ptr<Observable<StateTensorT>>> obs_;

  public:
    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    template <typename T1, typename T2>
    Hamiltonian(T1 &&coeffs, T2 &&obs)
        : coeffs_ham_{std::forward<T1>(coeffs)}, obs_{std::forward<T2>(obs)} {
        PL_ASSERT(coeffs_ham_.size() == obs_.size());

        for (std::size_t term_idx = 0; term_idx < coeffs_ham_.size();
             term_idx++) {
            auto ob = obs_[term_idx];
            if (ob->getObsName().find("Hamiltonian") != std::string::npos) {
                for (std::size_t sub_term_idx = 0;
                     sub_term_idx < ob->getNumTensors().size();
                     sub_term_idx++) {
                    PrecisionT coeff = ob->getCoeffsPerTerm()[sub_term_idx];
                    coeff = coeff * coeffs_ham_[term_idx];
                    this->coeffs_.push_back(coeff);
                    this->numTensors_.emplace_back(
                        ob->getNumTensors()[sub_term_idx]);
                    this->numStateModes_.emplace_back(
                        ob->getNumStateModes()[sub_term_idx]);
                    this->stateModes_.emplace_back(
                        ob->getStateModes()[sub_term_idx]);
                    this->data_.emplace_back(ob->getData()[sub_term_idx]);
                }
            } else {
                this->coeffs_.push_back(coeffs_ham_[term_idx]);
                this->numTensors_.emplace_back(ob->getNumTensors().front());
                this->numStateModes_.emplace_back(
                    ob->getNumStateModes().front());
                this->stateModes_.emplace_back(ob->getStateModes().front());
                this->data_.emplace_back(ob->getData().front());
            }
        }
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     * @return std::shared_ptr<Hamiltonian<StateTensorT>>
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateTensorT>>> obs)
        -> std::shared_ptr<Hamiltonian<StateTensorT>> {
        return std::shared_ptr<Hamiltonian<StateTensorT>>(
            new Hamiltonian<StateTensorT>{std::move(coeffs), std::move(obs)});
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        std::unordered_set<std::size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            wires.insert(ob_wires.begin(), ob_wires.end());
        }
        auto all_wires = std::vector<std::size_t>(wires.begin(), wires.end());
        std::sort(all_wires.begin(), all_wires.end());
        return all_wires;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream ss;
        ss << "Hamiltonian: { 'coeffs' : " << coeffs_ham_
           << ", 'observables' : [";
        const auto term_size = coeffs_ham_.size();
        for (size_t t = 0; t < term_size; t++) {
            ss << obs_[t]->getObsName();
            if (t != term_size - 1) {
                ss << ", ";
            }
        }
        ss << "]}";
        return ss.str();
    }

    /**
     * @brief Get the coefficients of the observable.
     */
    [[nodiscard]] auto getCoeffs() const -> std::vector<PrecisionT> {
        return coeffs_ham_;
    };
};
} // namespace Pennylane::LightningTensor::TNCuda::Observables
