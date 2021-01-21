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

template<int X>
using Array_Xq = Eigen::array<int, X>;


std::unordered_set<int> get_inactive_wires(const vector<int>& wires, const int Dim){
    std::unordered_set<int> inactive_wires;

    // Set the consecutive wires
    for (int i=0; i<Dim; ++i){
        inactive_wires.insert(i);
    }

    std::unordered_set<int> wires_set;
    std::copy(wires.begin(),
            wires.end(),
            std::inserter(wires_set, wires_set.end()));

    for (const auto& elem : wires_set) {
        inactive_wires.erase(elem);
    }
    return inactive_wires;
}


vector<int> map_wires(const vector<int>& wires, const int num_qubits){
    vector<int> mapped_wires;
    for (int i=0; i<wires.size(); ++i) {
        int new_wire = (wires.at(i) -1) % num_qubits;
        mapped_wires.push_back(new_wire);
    }
    return mapped_wires;
}

template <int Dim, int M, typename... Shape>
VectorXcd compute_marginal(Ref<VectorXcd> state, const vector<int>& wires, Shape... shape){

        vector<int> mapped_wires = map_wires(wires, Dim);

        // Determine which subsystems are to be summed over
        std::unordered_set<int> inactive_wires = get_inactive_wires(mapped_wires, Dim);

        const int LenInactiveWires = Dim-M;
        Array_Xq<LenInactiveWires> dims;
        std::copy_n(std::make_move_iterator(inactive_wires.begin()), LenInactiveWires, dims.begin());

        // Faster not to store the intermediate tensor but compute with ref
        State_Xq<M> marginal_probs_tensor = TensorMap<State_Xq<Dim>>(state.data(), shape...).sum(dims);

        VectorXcd result = Map<VectorXcd> (marginal_probs_tensor.data(), marginal_probs_tensor.size(), 1);
        return result;
}

template<int Dim>
VectorXcd marginal_probs_aux(Ref<VectorXcd> state, const vector<int> &wires, 
        const int num_wires){

    if (Dim <= 0)
        throw std::invalid_argument("Must specify one or more qubits");

    switch(num_wires){
        case 1: return DynamicWiresInit<Dim, 1>::marginal_probs(state, wires);
        case 2: return DynamicWiresInit<Dim, 2>::marginal_probs(state, wires);
        case 3: return DynamicWiresInit<Dim, 3>::marginal_probs(state, wires);
        case 4: return DynamicWiresInit<Dim, 4>::marginal_probs(state, wires);
        case 5: return DynamicWiresInit<Dim, 5>::marginal_probs(state, wires);
        case 6: return DynamicWiresInit<Dim, 6>::marginal_probs(state, wires);
        case 7: return DynamicWiresInit<Dim, 7>::marginal_probs(state, wires);
        case 8: return DynamicWiresInit<Dim, 8>::marginal_probs(state, wires);
        case 9: return DynamicWiresInit<Dim, 9>::marginal_probs(state, wires);
        case 10: return DynamicWiresInit<Dim, 10>::marginal_probs(state, wires);
        case 11: return DynamicWiresInit<Dim, 11>::marginal_probs(state, wires);
        case 12: return DynamicWiresInit<Dim, 12>::marginal_probs(state, wires);
        case 13: return DynamicWiresInit<Dim, 13>::marginal_probs(state, wires);
        case 14: return DynamicWiresInit<Dim, 14>::marginal_probs(state, wires);
        case 15: return DynamicWiresInit<Dim, 15>::marginal_probs(state, wires);
        case 16: return DynamicWiresInit<Dim, 16>::marginal_probs(state, wires);
        case 17: return DynamicWiresInit<Dim, 17>::marginal_probs(state, wires);
        case 18: return DynamicWiresInit<Dim, 18>::marginal_probs(state, wires);
        case 19: return DynamicWiresInit<Dim, 19>::marginal_probs(state, wires);
        case 20: return DynamicWiresInit<Dim, 20>::marginal_probs(state, wires);
        case 21: return DynamicWiresInit<Dim, 21>::marginal_probs(state, wires);
        case 22: return DynamicWiresInit<Dim, 22>::marginal_probs(state, wires);
        case 23: return DynamicWiresInit<Dim, 23>::marginal_probs(state, wires);
        case 24: return DynamicWiresInit<Dim, 24>::marginal_probs(state, wires);
        case 25: return DynamicWiresInit<Dim, 25>::marginal_probs(state, wires);
        case 26: return DynamicWiresInit<Dim, 26>::marginal_probs(state, wires);
        case 27: return DynamicWiresInit<Dim, 27>::marginal_probs(state, wires);
        case 28: return DynamicWiresInit<Dim, 28>::marginal_probs(state, wires);
        case 29: return DynamicWiresInit<Dim, 29>::marginal_probs(state, wires);
        case 30: return DynamicWiresInit<Dim, 30>::marginal_probs(state, wires);
        case 31: return DynamicWiresInit<Dim, 31>::marginal_probs(state, wires);
        case 32: return DynamicWiresInit<Dim, 32>::marginal_probs(state, wires);
        case 33: return DynamicWiresInit<Dim, 33>::marginal_probs(state, wires);
        case 34: return DynamicWiresInit<Dim, 34>::marginal_probs(state, wires);
        case 35: return DynamicWiresInit<Dim, 35>::marginal_probs(state, wires);
        case 36: return DynamicWiresInit<Dim, 36>::marginal_probs(state, wires);
        case 37: return DynamicWiresInit<Dim, 37>::marginal_probs(state, wires);
        case 38: return DynamicWiresInit<Dim, 38>::marginal_probs(state, wires);
        case 39: return DynamicWiresInit<Dim, 39>::marginal_probs(state, wires);
        case 40: return DynamicWiresInit<Dim, 40>::marginal_probs(state, wires);
        case 41: return DynamicWiresInit<Dim, 41>::marginal_probs(state, wires);
        case 42: return DynamicWiresInit<Dim, 42>::marginal_probs(state, wires);
        case 43: return DynamicWiresInit<Dim, 43>::marginal_probs(state, wires);
        case 44: return DynamicWiresInit<Dim, 44>::marginal_probs(state, wires);
        case 45: return DynamicWiresInit<Dim, 45>::marginal_probs(state, wires);
        case 46: return DynamicWiresInit<Dim, 46>::marginal_probs(state, wires);
        case 47: return DynamicWiresInit<Dim, 47>::marginal_probs(state, wires);
        case 48: return DynamicWiresInit<Dim, 48>::marginal_probs(state, wires);
        case 49: return DynamicWiresInit<Dim, 49>::marginal_probs(state, wires);
        case 50: return DynamicWiresInit<Dim, 50>::marginal_probs(state, wires);
        default: throw std::invalid_argument("No support for > 50 qubits");
    }
}

VectorXcd marginal_probs(Ref<VectorXcd> state, const int qubits, const vector<int> &wires, 
        const int num_wires){

    if (qubits <= 0)
        throw std::invalid_argument("Must specify one or more qubits");

    switch(qubits){
        case 1: return marginal_probs_aux<1>(state, wires, num_wires);
        case 2: return marginal_probs_aux<2>(state, wires, num_wires);
        case 3: return marginal_probs_aux<3>(state, wires, num_wires);
        case 4: return marginal_probs_aux<4>(state, wires, num_wires);
        case 5: return marginal_probs_aux<5>(state, wires, num_wires);
        case 6: return marginal_probs_aux<6>(state, wires, num_wires);
        case 7: return marginal_probs_aux<7>(state, wires, num_wires);
        case 8: return marginal_probs_aux<8>(state, wires, num_wires);
        case 9: return marginal_probs_aux<9>(state, wires, num_wires);
        case 10: return marginal_probs_aux<10>(state, wires, num_wires);
        case 11: return marginal_probs_aux<11>(state, wires, num_wires);
        case 12: return marginal_probs_aux<12>(state, wires, num_wires);
        case 13: return marginal_probs_aux<13>(state, wires, num_wires);
        case 14: return marginal_probs_aux<14>(state, wires, num_wires);
        case 15: return marginal_probs_aux<15>(state, wires, num_wires);
        case 16: return marginal_probs_aux<16>(state, wires, num_wires);
        case 17: return marginal_probs_aux<17>(state, wires, num_wires);
        case 18: return marginal_probs_aux<18>(state, wires, num_wires);
        case 19: return marginal_probs_aux<19>(state, wires, num_wires);
        case 20: return marginal_probs_aux<20>(state, wires, num_wires);
        case 21: return marginal_probs_aux<21>(state, wires, num_wires);
        case 22: return marginal_probs_aux<22>(state, wires, num_wires);
        case 23: return marginal_probs_aux<23>(state, wires, num_wires);
        case 24: return marginal_probs_aux<24>(state, wires, num_wires);
        case 25: return marginal_probs_aux<25>(state, wires, num_wires);
        case 26: return marginal_probs_aux<26>(state, wires, num_wires);
        case 27: return marginal_probs_aux<27>(state, wires, num_wires);
        case 28: return marginal_probs_aux<28>(state, wires, num_wires);
        case 29: return marginal_probs_aux<29>(state, wires, num_wires);
        case 30: return marginal_probs_aux<30>(state, wires, num_wires);
        case 31: return marginal_probs_aux<31>(state, wires, num_wires);
        case 32: return marginal_probs_aux<32>(state, wires, num_wires);
        case 33: return marginal_probs_aux<33>(state, wires, num_wires);
        case 34: return marginal_probs_aux<34>(state, wires, num_wires);
        case 35: return marginal_probs_aux<35>(state, wires, num_wires);
        case 36: return marginal_probs_aux<36>(state, wires, num_wires);
        case 37: return marginal_probs_aux<37>(state, wires, num_wires);
        case 38: return marginal_probs_aux<38>(state, wires, num_wires);
        case 39: return marginal_probs_aux<39>(state, wires, num_wires);
        case 40: return marginal_probs_aux<40>(state, wires, num_wires);
        case 41: return marginal_probs_aux<41>(state, wires, num_wires);
        case 42: return marginal_probs_aux<42>(state, wires, num_wires);
        case 43: return marginal_probs_aux<43>(state, wires, num_wires);
        case 44: return marginal_probs_aux<44>(state, wires, num_wires);
        case 45: return marginal_probs_aux<45>(state, wires, num_wires);
        case 46: return marginal_probs_aux<46>(state, wires, num_wires);
        case 47: return marginal_probs_aux<47>(state, wires, num_wires);
        case 48: return marginal_probs_aux<48>(state, wires, num_wires);
        case 49: return marginal_probs_aux<49>(state, wires, num_wires);
        case 50: return marginal_probs_aux<50>(state, wires, num_wires);
        default: throw std::invalid_argument("No support for > 50 qubits");
    }
}
