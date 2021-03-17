#include "Optimize.hpp"
#include <vector>
#include <algorithm>

using std::unique_ptr;
using std::vector;
using std::string;
using std::tuple;

using Pennylane::CplxType;

typedef unsigned int UINT;

#include <iostream>

//TODO: input: 2 vectors of wires, outputs: 3 vectors: A, B and C as used in qulacs
vector<CplxType> Pennylane::create_identity(const unsigned int & dim){

    //TODO: throw if dim is not even
    vector<CplxType> identity(dim * dim);
    for (int i = 0; i< identity.size(); i+=(dim+1)){
        identity.at(i) = 1;
    }
    return identity;
}

bool wire_is_in(const unsigned int& wire, const INDICES& wires_to_check){
    return std::find(wires_to_check.begin(), wires_to_check.end(), wire) != wires_to_check.end();
}
/*
void update_target_wires(const string &opLabel, const INDICES& wires, INDICES& target_wires){
    // We assume that the last wire is the target

    const bool is_two_qubit_controlled = controlled_two_qubit_gates.find(opLabel1) != controlled_two_qubit_gates.end();
    const bool is_three_qubit_controlled = controlled_three_qubit_gates.find(opLabel1) != controlled_three_qubit_gates.end();

    if (is_two_qubit_controlled || is_three_qubit_controlled){
        target_wires.push_back(first_wires.back());
    }
    else{
        target_wires.insert(target_wires.end(), first_wires.begin(), first_wires.end());
    }
}
*/

tuple<INDICES, INDICES> Pennylane::separate_control_and_target(const string &opLabel, const INDICES& wires){
    std::set<std::string> controlled_two_qubit_gates {"CNOT", "CZ", "CRX", "CRY", "CRZ", "CRot"};
    std::set<std::string> controlled_three_qubit_gates {"Toffoli",  "CSWAP"};

    INDICES control_wires = {};
    INDICES target_wires = {};
    const bool is_two_qubit_controlled = controlled_two_qubit_gates.find(opLabel) != controlled_two_qubit_gates.end();
    const bool is_three_qubit_controlled = controlled_three_qubit_gates.find(opLabel) != controlled_three_qubit_gates.end();

    if (is_two_qubit_controlled){
        control_wires.push_back(wires.at(0));

        target_wires.push_back(wires.back());
    }
    else if (opLabel == "Toffoli"){
        control_wires.push_back(wires.at(0));
        control_wires.push_back(wires.at(1));

        target_wires.push_back(wires.back());
    }
    else if (opLabel == "CSWAP"){
        control_wires.push_back(wires.at(0));
        target_wires.push_back(wires.at(1));

        target_wires.push_back(wires.back());
    }
    else{
        target_wires.insert(target_wires.end(), wires.begin(), wires.end());
    }
    return std::make_tuple(control_wires, target_wires);
}

tuple<INDICES, INDICES> Pennylane::get_new_qubit_list(const string &opLabel1, const INDICES& first_wires, const string &opLabel2, const INDICES& second_wires){

    INDICES new_target_wires;
    INDICES new_control_wires;

    //1. operation
    auto res1 = separate_control_and_target(opLabel1, first_wires);
    const INDICES first_control_wires = std::get<0>(res1);
    const INDICES first_target_wires = std::get<1>(res1);

    //2. operation
    auto res2 = separate_control_and_target(opLabel2, second_wires);
    const INDICES second_control_wires = std::get<0>(res2);
    const INDICES second_target_wires = std::get<1>(res2);

    for (auto wire : first_target_wires) {
        //case 0-2:
        new_target_wires.push_back(wire);
    }

    for (auto wire : first_control_wires) {
        //case 4:
        // qubit belongs to first control and second control ->  if control_value is equal, goto new_control_set. If not, goto new_target_set
        if(wire_is_in(wire, second_control_wires)){
            new_control_wires.push_back(wire);
        }
        //case 3: qubit belongs to first control and second target
        //case 5: qubit belongs to first control and not in second
        else{
            new_target_wires.push_back(wire);
        }
    }

    for (auto wire : second_target_wires) {
        //case 6: qubit belongs to second target but not in first
        if(not wire_is_in(wire, first_target_wires) and not wire_is_in(wire, first_control_wires)){
            new_target_wires.push_back(wire);
        }

    }
    for (auto wire : second_control_wires) {
        //case 7: qubit belongs to second control but not in first
        if(not wire_is_in(wire, first_target_wires) and not wire_is_in(wire, first_control_wires)){
            new_target_wires.push_back(wire);
        }

    }
    return std::make_tuple(new_control_wires, new_target_wires);
}

/*
// Join new qubit indices to target_list according to a given first_target_wires, and set a new matrix to "matrix"
void Pennylane::get_extended_matrix(unique_ptr<Pennylane::AbstractGate> gate,
vector<CplxType>& matrix, INDICES& first_target_wires, INDICES& first_control_wires, vector<unsigned
int>& second_remaining_wires) {

    // can do QubitUnitary:
    Pennylane::QubitUnitary qubit_unitary(2, matrix);

    //Logic: determine the controlled and the target qubits

    // At first, qubit indices are ordered as (A,C,B)
    std::vector<UINT> unsorted_new_target_index_list;

    // TODO: is first_target_wires == first_target_wires from qulacs?
    unsorted_new_target_index_list.reserve( first_target_wires.size() + first_control_wires.size() + second_remaining_wires.size());
    unsorted_new_target_index_list.insert(unsorted_new_target_index_list.end(), first_target_wires.begin(), first_target_wires.end());
    unsorted_new_target_index_list.insert(unsorted_new_target_index_list.end(), first_control_wires.begin(), first_control_wires.end());
    unsorted_new_target_index_list.insert(unsorted_new_target_index_list.end(), second_remaining_wires.begin(), second_remaining_wires.end());

    // *** NOTE ***
    // Order of tensor product is reversed!!!
    // U_0 I_1 = I \tensor U = [[U,0], [0,U]]
    // 0-control-U_0 = |0><0| \tensor U + |1><1| \tensor I = [[U,0],[0,I]]
    // 1-control-U_0 = |0><0| \tensor I + |1><1| \tensor U = [[I,0],[0,U]]

    // *** Algorithm ***
    // The gate matrix corresponding to indices (A,C,B) has 2^|B| blocks of gate matrix with (A,C).
    // The (control_mask)-th block matrix is (A,C), and the others are Identity.
    // The gate matrix with (A,C) has 2^|C| blocks of gate matrix with A, which is equal to the original gate matrix.

    // Thus, the following steps work.
    // 1. Enumerate set B and C. -> Done
    // 2. Generate 2^{|A|+|B|+|C|}-dim identity matrix
    size_t new_matrix_qubit_count = (UINT)first_target_wires.size();
    size_t new_matrix_dim = 1ULL << new_matrix_qubit_count;
    auto matrix = create_identity(new_matrix_dim);

    // 3. Decide correct 2^{|A|+|C|}-dim block matrix from control values.
    ITYPE start_block_basis = (1ULL << (join_from_target.size() + join_from_other_gate.size())) * control_mask;

    // 4. Repeat 2^{|C|}-times paste of original gate matrix A .
    vector<CplxType> org_matrix = gate->asMatrix();
    //gate->set_matrix(org_matrix);
    size_t org_matrix_dim = 1ULL << gate->target_qubit_list.size();
    ITYPE repeat_count = 1ULL << join_from_other_gate.size();
    for (ITYPE repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
        size_t paste_start = (size_t)(start_block_basis + repeat_index * org_matrix_dim );

        // Set a block
        matrix.block( paste_start, paste_start, org_matrix_dim, org_matrix_dim) = org_matrix;
    }

}/*
}
    // 5. Since the order of (C,B,A) is different from that of the other gate, we sort (C,B,A) after generating matrix.
    // We do nothing if it is already sorted
    if (!std::is_sorted(unsorted_new_target_index_list.begin(), unsorted_new_target_index_list.end())) {

        // generate ascending index of the INDEX_NUMBER of unsorted_target_qubit_index_list.
        std::vector<std::pair<UINT,UINT>> sorted_element_position;
        for (UINT i = 0; i<unsorted_new_target_index_list.size();++i){
            sorted_element_position.push_back(std::make_pair( unsorted_new_target_index_list[i], i));
        }
        std::sort(sorted_element_position.begin(), sorted_element_position.end());
        std::vector<UINT> sorted_index(sorted_element_position.size(), -1);
        for (UINT i = 0; i < sorted_index.size(); ++i) sorted_index[ sorted_element_position[i].second ] = i;

        // If target qubit is not in the sorted position, we swap the element to the element in correct position. If not, go next index.
        // This sort will stop with n-time swap in the worst case, which is smaller than the cost of std::sort.
        // We cannot directly sort target qubit list in order to swap matrix rows and columns with respect to qubit ordering.
        UINT ind1 = 0;
        while (ind1 < sorted_index.size()) {
            if (sorted_index[ind1] != ind1) {
                UINT ind2 = sorted_index[ind1];

                // move to correct position
                std::swap(sorted_index[ind1], sorted_index[ind2]);
                std::swap(unsorted_new_target_index_list[ind1], unsorted_new_target_index_list[ind2]);

                // create masks
                const UINT min_index = std::min(ind1, ind2);
                const UINT max_index = std::max(ind1, ind2);
                const ITYPE min_mask = 1ULL << min_index;
                const ITYPE max_mask = 1ULL << max_index;

                const ITYPE loop_dim = new_matrix_dim >> 2;

                for (ITYPE state_index = 0; state_index < loop_dim; ++state_index) {
                    ITYPE basis_00 = state_index;
                    basis_00 = insert_zero_to_basis_index(basis_00, min_mask, min_index);
                    basis_00 = insert_zero_to_basis_index(basis_00, max_mask, max_index);
                    ITYPE basis_01 = basis_00 ^ min_mask;
                    ITYPE basis_10 = basis_00 ^ max_mask;

                    matrix.col((size_t)basis_01).swap(matrix.col((size_t)basis_10));
                    matrix.row((size_t)basis_01).swap(matrix.row((size_t)basis_10));
                }
            }
            else ind1++;
        }
    }

    //std::cout << "unsorted " << std::endl;
    //for (auto val : unsorted_target_list) std::cout << val << " "; std::cout << std::endl;
    //std::cout << matrix << std::endl;
    //sort_target_qubit(unsorted_new_target_index_list, matrix);
    //std::cout << "sorted " << std::endl;
    //for (auto val : unsorted_target_list) std::cout << val << " "; std::cout << std::endl;
    //std::cout << matrix << std::endl;

}
*/



/*

unique_ptr<AbstractGate> get_merged_op(unique_ptr<AbstractGate> gate1,
    unique_ptr<AbstractGate> gate2, const INDICES& wires1, const
    INDICES& wires2){
    //TODO: how to release the pointers to the previously created ops? Do we need to do that?

    //TODO: we assume that wires1 >= wires2
    gate->applyKernel(state, internalIndices, externalIndices);
    auto vector<CplxType> mx1 = gate1->asMatrix();
    auto vector<CplxType> mx2 = gate2->asMatrix();

    auto dim1 = gate1->length();
    auto dim2 = gate2->length();

    for (int i = 0; i<mx2.size(); ++i){
        StateVector state();
        gate->applyKernel(state, internalIndices, externalIndices);
    }

}
*/
