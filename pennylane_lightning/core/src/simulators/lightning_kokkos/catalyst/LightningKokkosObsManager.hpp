// Copyright 2023-2024 Xanadu Quantum Technologies Inc.

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

#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "Types.h"
#include "Utils.hpp"

#include "ObservablesKokkos.hpp"

namespace Catalyst::Runtime::Simulator {

/**
 * @brief The LightningKokkosObsManager caches observables of a program at
 * runtime and maps each one to a const unique index (`int64_t`) in the scope of
 * the global context manager.
 */
template <typename PrecisionT> class LightningKokkosObsManager final {
  private:
    using StateVectorT =
        Pennylane::LightningKokkos::StateVectorKokkos<PrecisionT>;
    using ObservableT = Pennylane::Observables::Observable<StateVectorT>;
    using ObservablePairType = std::pair<std::shared_ptr<ObservableT>, ObsType>;
    std::vector<ObservablePairType> observables_{};

  public:
    LightningKokkosObsManager() = default;
    ~LightningKokkosObsManager() = default;

    LightningKokkosObsManager(const LightningKokkosObsManager &) = delete;
    LightningKokkosObsManager &
    operator=(const LightningKokkosObsManager &) = delete;
    LightningKokkosObsManager(LightningKokkosObsManager &&) = delete;
    LightningKokkosObsManager &operator=(LightningKokkosObsManager &&) = delete;

    /**
     * @brief A helper function to clear constructed observables in the program.
     */
    void clear() { this->observables_.clear(); }

    /**
     * @brief Check the validity of observable keys.
     *
     * @param obsKeys The vector of observable keys
     * @return bool
     */
    [[nodiscard]] auto
    isValidObservables(const std::vector<ObsIdType> &obsKeys) const -> bool {
        return std::all_of(obsKeys.begin(), obsKeys.end(), [this](auto i) {
            return (i >= 0 &&
                    static_cast<std::size_t>(i) < this->observables_.size());
        });
    }

    /**
     * @brief Get the constructed observable instance.
     *
     * @param key The observable key
     * @return std::shared_ptr<ObservableT>
     */
    [[nodiscard]] auto getObservable(ObsIdType key)
        -> std::shared_ptr<ObservableT> {
        RT_FAIL_IF(!this->isValidObservables({key}), "Invalid observable key");
        return std::get<0>(this->observables_[key]);
    }

    /**
     * @brief Get the number of observables.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto numObservables() const -> std::size_t {
        return this->observables_.size();
    }

    /**
     * @brief Create and cache a new NamedObs instance.
     *
     * @param obsId The named observable id of type ObsId
     * @param wires The vector of wires the observable acts on
     * @return ObsIdType
     */
    [[nodiscard]] auto createNamedObs(ObsId obsId,
                                      const std::vector<std::size_t> &wires)
        -> ObsIdType {
        auto &&obs_str = std::string(
            Lightning::lookup_obs<Lightning::simulator_observable_support_size>(
                Lightning::simulator_observable_support, obsId));

        this->observables_.push_back(std::make_pair(
            std::make_shared<Pennylane::LightningKokkos::Observables::NamedObs<
                StateVectorT>>(obs_str, wires),
            ObsType::Basic));
        return static_cast<ObsIdType>(this->observables_.size() - 1);
    }

    /**
     * @brief Create and cache a new HermitianObs instance.
     *
     * @param matrix The row-wise Hermitian matrix
     * @param wires The vector of wires the observable acts on
     * @return ObsIdType
     */
    [[nodiscard]] auto
    createHermitianObs(const std::vector<std::complex<PrecisionT>> &matrix,
                       const std::vector<std::size_t> &wires) -> ObsIdType {
        std::vector<Kokkos::complex<PrecisionT>> matrix_k;
        matrix_k.reserve(matrix.size());
        for (const auto &elem : matrix) {
            matrix_k.push_back(static_cast<Kokkos::complex<PrecisionT>>(elem));
        }

        this->observables_.push_back(std::make_pair(
            std::make_shared<Pennylane::LightningKokkos::Observables::
                                 HermitianObs<StateVectorT>>(
                Pennylane::LightningKokkos::Observables::HermitianObs<
                    StateVectorT>{matrix_k, wires}),
            ObsType::Basic));

        return static_cast<ObsIdType>(this->observables_.size() - 1);
    }

    /**
     * @brief Create and cache a new TensorProd instance.
     *
     * @param obsKeys The vector of observable keys
     * @return ObsIdType
     */
    [[nodiscard]] auto
    createTensorProdObs(const std::vector<ObsIdType> &obsKeys) -> ObsIdType {
        const auto key_size = obsKeys.size();
        const auto obs_size = this->observables_.size();

        std::vector<std::shared_ptr<ObservableT>> obs_vec;
        obs_vec.reserve(key_size);

        for (const auto &key : obsKeys) {
            RT_FAIL_IF(static_cast<std::size_t>(key) >= obs_size || key < 0,
                       "Invalid observable key");

            auto &&[obs, type] = this->observables_[key];
            obs_vec.push_back(obs);
        }

        this->observables_.push_back(std::make_pair(
            Pennylane::LightningKokkos::Observables::TensorProdObs<
                StateVectorT>::create(obs_vec),
            ObsType::TensorProd));

        return static_cast<ObsIdType>(obs_size);
    }

    /**
     * @brief Create and cache a new HamiltonianObs instance.
     *
     * @param coeffs The vector of coefficients
     * @param obsKeys The vector of observable keys
     * @return ObsIdType
     */
    [[nodiscard]] auto
    createHamiltonianObs(const std::vector<PrecisionT> &coeffs,
                         const std::vector<ObsIdType> &obsKeys) -> ObsIdType {
        const auto key_size = obsKeys.size();
        const auto obs_size = this->observables_.size();

        RT_FAIL_IF(
            key_size != coeffs.size(),
            "Incompatible list of observables and coefficients; "
            "Number of observables and number of coefficients must be equal");

        std::vector<std::shared_ptr<ObservableT>> obs_vec;
        obs_vec.reserve(key_size);

        for (auto key : obsKeys) {
            RT_FAIL_IF(static_cast<std::size_t>(key) >= obs_size || key < 0,
                       "Invalid observable key");

            auto &&[obs, type] = this->observables_[key];
            obs_vec.push_back(obs);
        }

        this->observables_.push_back(std::make_pair(
            std::make_shared<Pennylane::LightningKokkos::Observables::
                                 Hamiltonian<StateVectorT>>(
                Pennylane::LightningKokkos::Observables::Hamiltonian<
                    StateVectorT>(coeffs, std::move(obs_vec))),
            ObsType::Hamiltonian));

        return static_cast<ObsIdType>(obs_size);
    }
};
} // namespace Catalyst::Runtime::Simulator
