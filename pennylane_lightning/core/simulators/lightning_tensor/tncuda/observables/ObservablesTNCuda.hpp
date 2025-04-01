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

#include <tuple>
#include <unordered_set>
#include <vector>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Error.hpp"
#include "Util.hpp"

#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningGPU::Util;

template <class T> using vector1D = std::vector<T>;
template <class T> using vector2D = std::vector<vector1D<T>>;
template <class T> using vector3D = std::vector<vector2D<T>>;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Observables {
/**
 * @brief A base class for observable classes of cutensornet backends.
 *
 * We note that the observable classes for cutensornet backends have the same
 * user interface as the observable classes for the statevector backends across
 * the lightning ecosystem.
 *
 * @tparam TensorNetT tensor network class.
 */
template <class TensorNetT> class ObservableTNCuda {
  public:
    using PrecisionT = typename TensorNetT::PrecisionT;
    using ComplexT = typename TensorNetT::ComplexT;
    using MetaDataT = std::tuple<std::string, std::vector<PrecisionT>,
                                 std::vector<ComplexT>>; // name, params, matrix

  protected:
    vector1D<PrecisionT> coeffs_;      // coefficients of each term
    vector1D<std::size_t> numTensors_; // number of tensors in each term
    vector2D<std::size_t>
        numStateModes_; // number of state modes of each tensor in each term
    vector3D<std::size_t>
        stateModes_;               // state modes of each tensor in each term
    vector2D<MetaDataT> metaData_; // meta data of each tensor in each term

  protected:
    ObservableTNCuda() = default;
    ObservableTNCuda(const ObservableTNCuda &) = default;
    ObservableTNCuda(ObservableTNCuda &&) noexcept = default;
    ObservableTNCuda &operator=(const ObservableTNCuda &) = default;
    ObservableTNCuda &operator=(ObservableTNCuda &&) noexcept = default;

  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param other Instance of subclass of ObservableTNCuda<TensorNetT> to
     * compare.
     */
    [[nodiscard]] virtual bool
    isEqual(const ObservableTNCuda<TensorNetT> &other) const = 0;

  public:
    virtual ~ObservableTNCuda() = default;

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
     * @brief Get the meta data of each tensor in each term on host.
     */
    [[nodiscard]] auto getMetaData() const -> const vector2D<MetaDataT> & {
        return metaData_;
    }

    /**
     * @brief Get the coefficients of a observable.
     */
    [[nodiscard]] auto getCoeffs() const -> const vector1D<PrecisionT> & {
        return coeffs_;
    };

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] auto
    operator==(const ObservableTNCuda<TensorNetT> &other) const -> bool {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] auto
    operator!=(const ObservableTNCuda<TensorNetT> &other) const -> bool {
        return !(*this == other);
    }
};

/**
 * @brief Named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam TensorNetT tensor network class.
 */
template <class TensorNetT>
class NamedObsTNCuda : public ObservableTNCuda<TensorNetT> {
  public:
    using BaseType = ObservableTNCuda<TensorNetT>;
    using PrecisionT = typename TensorNetT::PrecisionT;
    using ComplexT = typename TensorNetT::ComplexT;

  private:
    std::string obs_name_;
    std::vector<std::size_t> wires_;
    std::vector<PrecisionT> params_;

    [[nodiscard]] auto isEqual(const ObservableTNCuda<TensorNetT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const NamedObsTNCuda<TensorNetT> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObsTNCuda object, representing a given
     * observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObsTNCuda(std::string obs_name, std::vector<std::size_t> wires,
                   std::vector<PrecisionT> params = {})
        : obs_name_{obs_name}, wires_{wires}, params_{params} {
        BaseType::coeffs_.emplace_back(PrecisionT{1.0});
        BaseType::numTensors_.emplace_back(std::size_t{1});
        BaseType::numStateModes_.emplace_back(
            vector1D<std::size_t>{wires_.size()});
        BaseType::stateModes_.emplace_back(vector2D<std::size_t>{wires_});

        BaseType::metaData_.push_back(
            {std::make_tuple(obs_name, params_, std::vector<ComplexT>{})});
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
 * @tparam TensorNetT tensor network class.
 */
template <class TensorNetT>
class HermitianObsTNCuda : public ObservableTNCuda<TensorNetT> {
  public:
    using BaseType = ObservableTNCuda<TensorNetT>;
    using PrecisionT = typename TensorNetT::PrecisionT;
    using ComplexT = typename TensorNetT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

  private:
    inline static const MatrixHasher mh;
    MatrixT matrix_;
    std::vector<std::size_t> wires_;

    [[nodiscard]] auto isEqual(const ObservableTNCuda<TensorNetT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const HermitianObsTNCuda<TensorNetT> &>(other);

        return (matrix_ == other_cast.matrix_) && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Construct a HermitianObs object, representing a given observable.
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObsTNCuda(MatrixT matrix, std::vector<std::size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {
        PL_ABORT_IF(wires_.size() != 1, "The number of Hermitian target wires "
                                        "must be 1 for Lightning-Tensor.");
        PL_ASSERT(matrix_.size() == Pennylane::Util::exp2(2 * wires_.size()));
        BaseType::coeffs_.emplace_back(PrecisionT{1.0});
        BaseType::numTensors_.emplace_back(std::size_t{1});
        BaseType::numStateModes_.emplace_back(
            vector1D<std::size_t>{wires_.size()});
        BaseType::stateModes_.emplace_back(vector2D<std::size_t>{wires_});

        BaseType::metaData_.push_back(
            {std::make_tuple("Hermitian", std::vector<PrecisionT>{}, matrix_)});
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        // To avoid collisions on cached GPU data, use matrix elements to
        // uniquely identify Hermitian
        // TODO: Replace with a performant hash function
        std::ostringstream obs_stream;
        obs_stream << "Hermitian" << mh(matrix_);
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return wires_;
    }

    /**
     * @brief Get the matrix of the Hermitian observable.
     */
    [[nodiscard]] auto getMatrix() const -> const MatrixT & { return matrix_; }
};

/**
 * @brief Tensor product of observables.
 *
 * @tparam TensorNetT tensor network class.
 */
template <class TensorNetT>
class TensorProdObsTNCuda : public ObservableTNCuda<TensorNetT> {
  public:
    using BaseType = ObservableTNCuda<TensorNetT>;
    using PrecisionT = typename TensorNetT::PrecisionT;
    using MetaDataT = BaseType::MetaDataT;

  private:
    std::vector<std::shared_ptr<ObservableTNCuda<TensorNetT>>> obs_;
    std::vector<std::size_t> all_wires_;

    [[nodiscard]] auto isEqual(const ObservableTNCuda<TensorNetT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const TensorProdObsTNCuda<TensorNetT> &>(other);

        if (obs_.size() != other_cast.obs_.size()) {
            return false;
        }

        for (std::size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect-forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObsTNCuda(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        if (obs_.size() == 1 &&
            obs_[0]->getObsName().find('@') != std::string::npos) {
            // This would prevent the misuse of this constructor for creating
            // TensorProdObs(TensorProdObs).
            PL_ABORT("A new TensorProdObs observable cannot be created "
                     "from a single TensorProdObs.");
        }

        for (const auto &ob : obs_) {
            PL_ABORT_IF(ob->getObsName().find("Hamiltonian") !=
                            std::string::npos,
                        "A TensorProdObs observable cannot be created from a "
                        "Hamiltonian.");
        }

        BaseType::coeffs_.push_back(PrecisionT{1.0});
        BaseType::numTensors_.push_back(obs_.size());

        vector1D<std::size_t> numStateModesLocal;
        vector2D<std::size_t> stateModesLocal;
        vector1D<MetaDataT> dataLocal;

        for (const auto &ob : obs_) {
            numStateModesLocal.insert(numStateModesLocal.end(),
                                      ob->getNumStateModes().front().begin(),
                                      ob->getNumStateModes().front().end());

            stateModesLocal.insert(stateModesLocal.end(),
                                   ob->getStateModes().front().begin(),
                                   ob->getStateModes().front().end());

            dataLocal.insert(dataLocal.end(), ob->getMetaData().front().begin(),
                             ob->getMetaData().front().end());
        }

        BaseType::numStateModes_.emplace_back(numStateModesLocal);
        BaseType::stateModes_.emplace_back(stateModesLocal);
        BaseType::metaData_.emplace_back(dataLocal);

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
     * @return std::shared_ptr<TensorProdObsTNCuda<TensorNetT>>
     */
    static auto
    create(std::initializer_list<std::shared_ptr<ObservableTNCuda<TensorNetT>>>
               obs) -> std::shared_ptr<TensorProdObsTNCuda<TensorNetT>> {
        return std::shared_ptr<TensorProdObsTNCuda<TensorNetT>>{
            new TensorProdObsTNCuda(std::move(obs))};
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     * @return std::shared_ptr<TensorProdObsTNCuda<TensorNetT>>
     */
    static auto
    create(std::vector<std::shared_ptr<ObservableTNCuda<TensorNetT>>> obs)
        -> std::shared_ptr<TensorProdObsTNCuda<TensorNetT>> {
        return std::shared_ptr<TensorProdObsTNCuda<TensorNetT>>{
            new TensorProdObsTNCuda(std::move(obs))};
    }

    /**
     * @brief Get the number of operations in the observable.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getSize() const -> std::size_t { return obs_.size(); }

    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::size_t>
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
        for (std::size_t idx = 0; idx < obs_size; idx++) {
            obs_stream << obs_[idx]->getObsName();
            if (idx != obs_size - 1) {
                obs_stream << " @ ";
            }
        }
        return obs_stream.str();
    }

    /**
     * @brief Get the observable.
     */
    [[nodiscard]] auto getObs() const
        -> std::vector<std::shared_ptr<ObservableTNCuda<TensorNetT>>> {
        return obs_;
    };
};

/**
 * @brief Hamiltonian representation as a sum of observables.
 *
 * @tparam TensorNetT tensor network class.
 */
template <class TensorNetT>
class HamiltonianTNCuda : public ObservableTNCuda<TensorNetT> {
  public:
    using BaseType = ObservableTNCuda<TensorNetT>;
    using PrecisionT = typename TensorNetT::PrecisionT;

  private:
    std::vector<PrecisionT> coeffs_ham_;
    std::vector<std::shared_ptr<ObservableTNCuda<TensorNetT>>> obs_;

    [[nodiscard]] bool
    isEqual(const ObservableTNCuda<TensorNetT> &other) const override {
        const auto &other_cast =
            static_cast<const HamiltonianTNCuda<TensorNetT> &>(other);

        if (coeffs_ham_ != other_cast.coeffs_ham_) {
            return false;
        }

        for (std::size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    template <typename T1, typename T2>
    HamiltonianTNCuda(T1 &&coeffs, T2 &&obs)
        : coeffs_ham_{std::forward<T1>(coeffs)}, obs_{std::forward<T2>(obs)} {
        BaseType::coeffs_ = coeffs_ham_;
        PL_ASSERT(BaseType::coeffs_.size() == obs_.size());

        for (std::size_t term_idx = 0; term_idx < BaseType::coeffs_.size();
             term_idx++) {
            auto ob = obs_[term_idx];
            // This is aligned with statevector backends
            PL_ABORT_IF(ob->getObsName().find("Hamiltonian") !=
                            std::string::npos,
                        "A Hamiltonian observable cannot be created from "
                        "another Hamiltonian.");
            BaseType::numTensors_.emplace_back(ob->getNumTensors().front());
            BaseType::numStateModes_.emplace_back(
                ob->getNumStateModes().front());
            BaseType::stateModes_.emplace_back(ob->getStateModes().front());
            BaseType::metaData_.emplace_back(ob->getMetaData().front());
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
     * @return std::shared_ptr<HamiltonianTNCuda<TensorNetT>>
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<ObservableTNCuda<TensorNetT>>>
               obs) -> std::shared_ptr<HamiltonianTNCuda<TensorNetT>> {
        return std::shared_ptr<HamiltonianTNCuda<TensorNetT>>(
            new HamiltonianTNCuda<TensorNetT>{std::move(coeffs),
                                              std::move(obs)});
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
        ss << "Hamiltonian: { 'coeffs' : " << BaseType::coeffs_
           << ", 'observables' : [";
        const auto term_size = BaseType::coeffs_.size();
        for (std::size_t t = 0; t < term_size; t++) {
            ss << obs_[t]->getObsName();
            if (t != term_size - 1) {
                ss << ", ";
            }
        }
        ss << "]}";
        return ss.str();
    }
    /**
     * @brief Get the observable.
     */
    [[nodiscard]] auto getObs() const
        -> std::vector<std::shared_ptr<ObservableTNCuda<TensorNetT>>> {
        return obs_;
    };

    /**
     * @brief Get the coefficients of the observable.
     */
    [[nodiscard]] auto getCoeffs() const -> std::vector<PrecisionT> {
        return BaseType::getCoeffs();
    };
};
} // namespace Pennylane::LightningTensor::TNCuda::Observables
