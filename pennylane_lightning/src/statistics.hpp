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
/**
 * @file
 * \rst
 * Handles computing statistics using a statevector like basis state probabilities.
 * \endrst
 */
#pragma once

#include "lightning_qubit.hpp"


// Stopping template
//
template<int Dim, int M, int ValueIdx>
class DynamicWiresGenerator
{
public:
    template<typename... Shape>
    static inline VectorXcd marginal_probs(
        Ref<VectorXcd> state,
        const vector<int>& wires
        )
    {
        return DynamicWiresGenerator<Dim, M, ValueIdx -1>::marginal_probs(state, wires);
    }
};

// Valid stopping template: Dim>=M
template<int Dim, int M>
class DynamicWiresGenerator<Dim, M, M>
{

public:
    template<typename... Shape>
    static inline VectorXcd marginal_probs(
        Ref<VectorXcd> state,
        const vector<int>& wires
        )
    {
        // The correct size of wires have been generated
        return QubitOperations<Dim>::template marginal_probs<M>(state, wires);
    }
};

// Invalid stopping template: Dim<M
template<int Dim, int M>
class DynamicWiresGenerator<Dim, M, 0>
{

public:
    template<typename... Shape>
    static inline VectorXcd marginal_probs(
        Ref<VectorXcd> state,
        const vector<int>& wires
        )
    {
        // Return an empty vector
        return VectorXcd(0);
    }
};

// Init template
template<int Dim, int M>
class DynamicWiresInit
{

public:
    template<typename... Shape>
    static inline VectorXcd marginal_probs(
        Ref<VectorXcd> state,
        const vector<int>& wires
        )
    {
        return DynamicWiresGenerator<Dim, M, Dim>::marginal_probs(state, wires);
    }
};
