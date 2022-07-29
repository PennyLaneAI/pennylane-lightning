// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "GateImplementationsLM.hpp"
namespace Pennylane::Gates {

template void GateImplementationsLM::applySingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliX<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliX<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliY<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliY<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliZ<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliZ<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyHadamard<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyHadamard<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyS<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);

template void GateImplementationsLM::applyS<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsLM::applyT<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);

template void GateImplementationsLM::applyT<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsLM::applyRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsLM::applyRot<float, float>(std::complex<float> *, size_t,
                                              const std::vector<size_t> &, bool,
                                              float, float, float);

template void
GateImplementationsLM::applyRot<double, double>(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool, double, double, double);

template void GateImplementationsLM::applyCY<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void
GateImplementationsLM::applyCY<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyCZ<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void
GateImplementationsLM::applyCZ<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyCNOT<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyCNOT<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applySWAP<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applySWAP<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyControlledPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyControlledPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsLM::applyCRot<float, float>(std::complex<float> *, size_t,
                                               const std::vector<size_t> &,
                                               bool, float, float, float);

template void
GateImplementationsLM::applyCRot<double, double>(std::complex<double> *, size_t,
                                                 const std::vector<size_t> &,
                                                 bool, double, double, double);

template void GateImplementationsLM::applyIsingXX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyIsingXX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyIsingYY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyIsingYY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyIsingZZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyIsingZZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyMultiRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);

template void GateImplementationsLM::applyMultiRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

} // namespace Pennylane::Gates
