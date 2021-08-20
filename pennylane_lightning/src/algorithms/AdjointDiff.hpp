#pragma once

#include <cmath>
#include <complex>
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
    sv.applyOperation("PauliY", wires, false);
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

} // namespace

namespace Pennylane {
namespace Algorithms {

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

    void adjointJacobian(StateVector<T> &phi, T *jac,
                         const vector<string> &observables,
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

        const size_t num_elements = phi.getLength();

        // 1. Copy the input state, create lambda
        std::unique_ptr<std::complex<T>[]> SV_lambda_data(
            new std::complex<T>[num_elements]);
        std::copy(phi.getData(), phi.getData() + num_elements,
                  SV_lambda_data.get());
        StateVector<T> SV_lambda(SV_lambda_data.get(), num_elements);

        // 2. Apply the unitaries (\hat{U}_{1:P}) to lambda
        std::vector<bool> inverses(operations.size(), false);
        SV_lambda.applyOperations(operations, opWires, inverses, opParams);

        // 3-4. Copy lambda and apply the observables
        // SV_lambda becomes |phi>
        std::vector<StateVector<T>> lambdas;
        lambdas.reserve(numObservables);
        std::vector<std::unique_ptr<std::complex<T>[]>> lambdas_data;
        lambdas_data.reserve(numObservables);
        for (int i = 0; i < numObservables; ++i) {
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
                // copy |phi> to |mu> before applying Uj*

                std::unique_ptr<std::complex<T>[]> phiCopyArr(
                    new std::complex<T>[num_elements]);
                std::copy(SV_lambda.getData(),
                          SV_lambda.getData() + num_elements, phiCopyArr.get());

                StateVector<T> mu(phiCopyArr.get(), num_elements);

                // create |phi'> = Uj*|phi>
                SV_lambda.applyOperation(operations[i], opWires[i], true,
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
                        std::cout << "mu::{\n\t" << mu << "\n}" << std::endl;

                        for (size_t j = 0; j < lambdas.size(); j++) {
                            std::cout << "lambdas[" << j << "]::{\n\t"
                                      << lambdas[j] << "\n}" << std::endl;

                            std::complex<T> sum =
                                innerProdC(lambdas[j].getData(), mu.getData(),
                                           num_elements);

                            // calculate 2 * shift * Real(i * sum) = -2 * shift
                            // * Imag(sum)
                            std::cout << "sum[" << current_param_idx
                                      << "]=" << sum << ", " << num_elements
                                      << ", "
                                      << j * trainableParams.size() +
                                             trainableParamNumber
                                      << std::endl;
                            jac[j * trainableParams.size() +
                                trainableParamNumber] =
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
};

} // namespace Algorithms
} // namespace Pennylane