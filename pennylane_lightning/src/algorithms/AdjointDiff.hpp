#pragma once

#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "StateVector.hpp"
#include "Util.hpp"

#include <iostream>

// Generators not needed outside this translation unit
namespace {

using namespace Pennylane::Util;

template <class T> static constexpr std::vector<std::complex<T>> getP00() {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getP11() {
    return {ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>()};
}

template <class T = double>
void applyGeneratorRX(Pennylane::StateVector<T> &sv,
                      const std::vector<size_t> &wires) {
    sv.applyOperation("PauliX", wires, false);
}

template <class T = double>
void applyGeneratorRY(Pennylane::StateVector<T> &sv,
                      const std::vector<size_t> &wires) {
    sv.applyOperation("PauliY", wires, false);
}

template <class T = double>
void applyGeneratorRZ(Pennylane::StateVector<T> &sv,
                      const std::vector<size_t> &wires) {
    sv.applyOperation("PauliZ", wires, false);
}

template <class T = double>
void applyGeneratorPhaseShift(Pennylane::StateVector<T> &sv,
                              const std::vector<size_t> &wires) {
    sv.applyOperation(getP11<T>(), wires, false);
}

template <class T = double>
void applyGeneratorCRX(Pennylane::StateVector<T> &sv,
                       const std::vector<size_t> &wires) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        shiftedState[internalIndices[0]] = shiftedState[internalIndices[1]] = 0;
        std::swap(shiftedState[internalIndices[2]],
                  shiftedState[internalIndices[3]]);
    }
}

template <class T = double>
void applyGeneratorCRY(Pennylane::StateVector<T> &sv,
                       const std::vector<size_t> &wires) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        std::complex<T> v0 = shiftedState[internalIndices[0]];
        shiftedState[internalIndices[0]] = shiftedState[internalIndices[1]] = 0;
        shiftedState[internalIndices[2]] =
            -IMAG<T>() * shiftedState[internalIndices[3]];
        shiftedState[internalIndices[3]] = IMAG<T>() * v0;
    }
}

template <class T = double>
void applyGeneratorCRZ(Pennylane::StateVector<T> &sv,
                       const std::vector<size_t> &wires) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        shiftedState[internalIndices[0]] = shiftedState[internalIndices[1]] = 0;
        shiftedState[internalIndices[3]] *= -1;
    }
}

template <class T = double>
void applyGeneratorControlledPhaseShift(Pennylane::StateVector<T> &sv,
                                        const std::vector<size_t> &wires) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        shiftedState[internalIndices[0]] = 0;
        shiftedState[internalIndices[1]] = 0;
        shiftedState[internalIndices[2]] = 0;
    }
}

template <class fp_t = double>
class SVUnique : public Pennylane::StateVector<fp_t> {
  private:
    std::unique_ptr<std::complex<fp_t>> arr_;
    size_t length_;
    size_t num_qubits_;

  public:
    SVUnique(size_t data_size)
        : arr_{new std::complex<fp_t>[data_size]}, length_{data_size},
          num_qubits_{log2(length_)}, Pennylane::StateVector<fp_t>{arr_.get(),
                                                                   data_size} {}

    SVUnique(const Pennylane::StateVector<fp_t> &sv)
        : SVUnique(sv.getLength()) {
        std::copy(sv.getData(), sv.getData() + sv.getLength(), arr_.get());
        length_ = sv.getLength();
        num_qubits_ = sv.getNumQubits();
    };

    SVUnique(const SVUnique<fp_t> &sv) : SVUnique(sv.getLength()) {
        std::copy(sv.getData(), sv.getData() + sv.getLength(), arr_.get());
    };

    std::complex<fp_t> *getData() { return arr_.get(); }
    std::complex<fp_t> *getData() const { return arr_.get(); }
};

} // namespace

namespace Pennylane {
namespace Algorithms {

/**
 * @brief Utility struct for a observable operations used by AdjointJacobian
 * class.
 *
 */
template <class T = double> struct ObsDatum {
    const std::vector<std::string> obs_name_;
    const std::vector<std::vector<T>> obs_params_;
    const std::vector<std::vector<size_t>> obs_wires_;
    ObsDatum(const std::vector<std::string> &obs_name,
             const std::vector<std::vector<T>> &obs_params,
             const std::vector<std::vector<size_t>> &obs_wires)
        : obs_name_{obs_name}, obs_params_{obs_params}, obs_wires_{
                                                            obs_wires} {};
    size_t getSize() const { return obs_name_.size(); }
    const std::vector<std::string> &getObsName() const { return obs_name_; }
    const std::vector<std::vector<T>> &getObsParams() const {
        return obs_params_;
    }
    const std::vector<std::vector<size_t>> &getObsWires() const {
        return obs_wires_;
    }
};

/**
 * @brief Utility class for encapsulating operations used by AdjointJacobian
 * class.
 *
 */
template <class T> struct OpsData {
    const std::vector<std::string> ops_name_;
    const std::vector<std::vector<T>> ops_params_;
    const std::vector<std::vector<size_t>> ops_wires_;
    const std::vector<bool> ops_inverses_;
    const std::vector<std::vector<std::complex<T>>> ops_matrices_;

    OpsData(
        const std::vector<std::string> &ops_name,
        const std::vector<std::vector<T>> &ops_params,
        const std::vector<std::vector<size_t>> &ops_wires,
        const std::vector<bool> &ops_inverses,
        const std::vector<std::vector<std::complex<T>>> &ops_matrices = {{}})
        : ops_name_{ops_name}, ops_params_{ops_params}, ops_wires_{ops_wires},
          ops_inverses_{ops_inverses}, ops_matrices_{ops_matrices} {};

    size_t getSize() const { return ops_name_.size(); }
    const std::vector<std::string> &getOpsName() const { return ops_name_; }
    const std::vector<std::vector<T>> &getOpsParams() const {
        return ops_params_;
    }
    const std::vector<std::vector<size_t>> &getOpsWires() const {
        return ops_wires_;
    }
    const std::vector<bool> &getOpsInverses() const { return ops_inverses_; }
    const std::vector<std::vector<std::complex<T>>> &getOpsMatrices() const {
        return ops_matrices_;
    }
};

template <class T = double> class AdjointJacobian {
  private:
    typedef void (*GeneratorFunc)(
        Pennylane::StateVector<T> &sv,
        const std::vector<size_t> &wires); // function pointer type

    const std::unordered_map<std::string, GeneratorFunc> generator_map{
        {"RX", &::applyGeneratorRX<T>},
        {"RY", &::applyGeneratorRY<T>},
        {"RZ", &::applyGeneratorRZ<T>},
        {"PhaseShift", &::applyGeneratorPhaseShift<T>},
        {"CRX", &::applyGeneratorCRX<T>},
        {"CRY", &::applyGeneratorCRY<T>},
        {"CRZ", &::applyGeneratorCRZ<T>},
        {"ControlledPhaseShift", &::applyGeneratorControlledPhaseShift<T>}};

    const std::unordered_map<std::string, T> scaling_factors{
        {"RX", -0.5},  {"RY", -0.5},
        {"RZ", -0.5},  {"PhaseShift", 1},
        {"CRX", -0.5}, {"CRY", -0.5},
        {"CRZ", -0.5}, {"ControlledPhaseShift", 1}};

  public:
    AdjointJacobian() {}

    const ObsDatum<T>
    createObs(const std::vector<std::string> &obs_name,
              const std::vector<std::vector<T>> &obs_params,
              const std::vector<std::vector<size_t>> &obs_wires) {
        return ObsDatum<T>(obs_name, obs_params, obs_wires);
    }

    const OpsData<T> createOpsData(
        const std::vector<std::string> &ops_name,
        const std::vector<std::vector<T>> &ops_params,
        const std::vector<std::vector<size_t>> &ops_wires,
        const std::vector<bool> &ops_inverses,
        const std::vector<std::vector<std::complex<T>>> &ops_matrices = {{}}) {
        return OpsData<T>(ops_name, ops_params, ops_wires, ops_inverses,
                          ops_matrices);
    }

    std::vector<std::complex<T>>
    copyStateData(const std::complex<T> *input_state, size_t state_length) {
        return {input_state, input_state + state_length};
    }

    void adjointJacobian(const std::complex<T> *psi, size_t num_elements,
                         std::vector<T> &jac,
                         const std::vector<ObsDatum<T>> &observables,
                         const OpsData<T> &operations,
                         const vector<size_t> &trainableParams,
                         size_t num_params) {

        const size_t num_observables = observables.size();

        auto lambda_data = copyStateData(psi, num_elements);
        StateVector<T> lambda(lambda_data.data(), num_elements);

        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            lambda.applyOperation(operations.getOpsName()[op_idx],
                                  operations.getOpsWires()[op_idx],
                                  operations.getOpsInverses()[op_idx],
                                  operations.getOpsParams()[op_idx]);
        }
        std::vector<std::vector<std::complex<T>>> H_lambda_data;
        std::vector<StateVector<T>> H_lambda;

        for (size_t h_i = 0; h_i < num_observables; h_i++) {
            H_lambda_data.push_back(lambda_data);
            H_lambda.push_back(
                StateVector<T>(H_lambda_data[h_i].data(), num_elements));

            for (size_t j = 0; j < observables[h_i].getSize(); j++) {
                H_lambda[h_i].applyOperation(
                    observables[h_i].getObsName()[j],
                    observables[h_i].getObsWires()[j], false,
                    observables[h_i].getObsParams()[j]);
            }
        }

        std::vector<std::complex<T>> phi_data(lambda_data);
        StateVector<T> phi(phi_data.data(), num_elements);

        for (int op_idx = operations.getOpsName().size() - 1; op_idx >= 0;
             op_idx--) {
            std::vector<std::complex<T>> mu_data(lambda_data);
            StateVector<T> mu(mu_data.data(), num_elements);

            phi.applyOperation(operations.getOpsName()[op_idx],
                               operations.getOpsWires()[op_idx],
                               !operations.getOpsInverses()[op_idx],
                               operations.getOpsParams()[op_idx]);

            generator_map.at(operations.getOpsName()[op_idx])(
                mu, operations.getOpsWires()[op_idx]);
            const T scalingFactor =
                scaling_factors.at(operations.getOpsName()[op_idx]);

            for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
                std::cout << "IDX=" << op_idx * num_observables + obs_idx
                          << "     Exp="
                          << -2 * scalingFactor *
                                 std::imag(
                                     innerProdC(H_lambda[obs_idx].getData(),
                                                mu.getData(), num_elements))
                          << std::endl;
                jac[op_idx * num_observables + obs_idx] =
                    -2 * scalingFactor *
                    std::imag(innerProdC(H_lambda[obs_idx].getData(),
                                         mu.getData(), num_elements));
                if (op_idx > 0) {
                    H_lambda[obs_idx].applyOperation(
                        operations.getOpsName()[op_idx],
                        operations.getOpsWires()[op_idx],
                        !operations.getOpsInverses()[op_idx],
                        operations.getOpsParams()[op_idx]);
                }
            }
        }
        // return jacobian;
    }
};

} // namespace Algorithms
} // namespace Pennylane

/*
    void adjointJacobian(StateVector<T> &psi, std::vector<T> &jac,
                         const vector<vector<string>> &observables,
                         const vector<vector<T>> &obsParams,
                         const vector<vector<size_t>> &obsWires,
                         const vector<string> &operations,
                         const vector<vector<T>> &opParams,
                         const vector<vector<size_t>> &opWires,
                         const vector<size_t> &trainableParams,
                         size_t num_params) {

        size_t numObservables = observables.size();
        int trainableParamNumber = trainableParams.size() - 1;
        int current_param_idx = num_params - 1;

        const size_t num_elements = psi.getLength();

        // 1. Copy the input state, create lambda
        std::unique_ptr<std::complex<T>[]> SV_lambda_data(
            new std::complex<T>[num_elements]);
        std::copy(psi.getData(), psi.getData() + num_elements,
                  SV_lambda_data.get());
        StateVector<T> SV_lambda(SV_lambda_data.get(), num_elements);

        // 2. Apply the unitaries (\hat{U}_{1:P}) to lambda
        std::vector<bool> inverses(operations.size(), false);
        SV_lambda.applyOperations(operations, opWires, inverses, opParams);

        // 3-4. Copy lambda and apply the observables
        // SV_lambda becomes |phi>

        std::unique_ptr<std::complex<T>[]> phi_data(
            new std::complex<T>[num_elements]);
        std::copy(SV_lambda.getData(), SV_lambda.getData() + num_elements,
                  phi_data.get());
        StateVector<T> phi_1(phi_data.get(), num_elements);

        std::vector<StateVector<T>> lambdas;
        lambdas.reserve(numObservables);
        std::vector<std::unique_ptr<std::complex<T>[]>> lambdas_data;
        lambdas_data.reserve(numObservables);
        for (size_t i = 0; i < numObservables; i++) {
            lambdas_data.emplace_back(new std::complex<T>[num_elements]);
            lambdas.emplace_back(
                StateVector<T>(lambdas_data[i].get(), num_elements));
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < numObservables; i++) {
            // copy |phi> and apply observables one at a time
            std::copy(SV_lambda_data.get(), SV_lambda_data.get() + num_elements,
                      lambdas_data[i].get());

            lambdas[i].applyOperation(observables[i], obsWires[i], false,
                                      obsParams[i]);
        }

        // replace with reverse iterator over values?
        for (int i = operations.size() - 1; i >= 0; i--) {

            if (opParams[i].size() > 1) {
                throw std::invalid_argument(
                    "The operation is not supported using "
                    "the adjoint differentiation method");
            } else if ((operations[i] != "QubitStateVector") &&
                       (operations[i] != "BasisState")) {

                std::unique_ptr<std::complex<T>[]> mu_data(
                    new std::complex<T>[num_elements]);
                std::copy(phi_1.getData(), phi_1.getData() + num_elements,
                          mu_data.get());

                StateVector<T> mu(mu_data.get(), num_elements);

                // create |phi'> = Uj*|phi>
                phi_1.applyOperation(operations[i], opWires[i], true,
                                     opParams[i]);

                // We have a parametrized gate
                if (!opParams[i].empty()) {

                    if (std::find(trainableParams.begin(),
                                  trainableParams.end(),
                                  current_param_idx) != trainableParams.end()) {

                        // create iH|phi> = d/d dUj/dtheta Uj* |phi> =
                        // dUj/dtheta|phi'>
                        const T scalingFactor =
                            scaling_factors.at(operations[i]);

                        generator_map.at(operations[i])(mu, opWires[i]);

                        for (size_t j = 0; j < lambdas.size(); j++) {

                            std::complex<T> sum =
                                innerProdC(lambdas[j].getData(), mu.getData(),
                                           num_elements);

                            // calculate 2 * shift * Real(i * sum) = -2 * shift
                            // * Imag(sum)
                            std::cout << "L[" << i << ", " << j
                                      << "]=" << lambdas[j] << std::endl;
                            std::cout << "mu[" << i << ", " << j << "]=" << mu
                                      << std::endl
                                      << std::endl;
                            jac[j * trainableParams.size() +
                                trainableParamNumber] =
                                //    2 * scalingFactor * std::real(sum);
                                -2 * scalingFactor * std::imag(sum);
                        }
                        trainableParamNumber--;
                    }
                    current_param_idx--;
                }

                for (size_t j = 0; j < lambdas.size(); j++) {
                    lambdas[j].applyOperation(operations[i], opWires[i], true,
                                              opParams[i]);
                }
            }
            /// missing else?
        }
    }
    */

/*


template <class T>
inline std::ostream &operator<<(std::ostream &out, const SVUnique<T> &sv) {
    const size_t num_qubits = sv.getNumQubits();
    const size_t length = sv.getLength();
    const auto data_ptr = sv.getData();
    out << "num_qubits=" << num_qubits << std::endl;
    out << "data=[";
    out << data_ptr[0];
    for (size_t i = 1; i < length - 1; i++) {
        out << "," << data_ptr[i];
    }
    out << "," << data_ptr[length - 1] << "]";

    return out;
}

*/
