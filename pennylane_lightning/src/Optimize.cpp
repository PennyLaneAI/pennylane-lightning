#include "Optimize.hpp"
#include <vector>
#include <algorithm>

using std::unique_ptr;
using std::vector;
using std::string;
using std::tuple;

using Pennylane::CplxType;
using Pennylane::AbstractGate;

#include <iostream>

//TODO: input: 2 vectors of wires, outputs: 3 vectors: A, B and C as used in qulacs
vector<CplxType> Pennylane::create_identity(const UINT & dim){

    //TODO: throw if dim is not even
    vector<CplxType> identity(dim * dim);
    for (UINT i = 0; i< identity.size(); i+=(dim+1)){
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
/**
 * Insert 0 to qubit_index-th bit of basis_index. basis_mask must be 1ULL << qubit_index.
 */
inline static ITYPE insert_zero_to_basis_index(ITYPE basis_index, ITYPE basis_mask, UINT qubit_index){
    ITYPE temp_basis = (basis_index >> qubit_index) << (qubit_index+1);
    return temp_basis + basis_index % basis_mask;
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
        // qubit belongs to first control and second control ->  if control_wireue is equal, goto new_control_set. If not, goto new_target_set
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

void Pennylane::set_block(CplxType* mx, const size_t &dim, const size_t &start_ind, CplxType* block_mx, const size_t &block_dim){
    //TODO: validate

    size_t i = 0;
    for(size_t j = 0; j<block_dim*block_dim; j+=block_dim){
        for(size_t k = 0; k<block_dim; ++k){
            mx[start_ind + i + k] = block_mx[k+j];
        }
        i += dim;
    }
}

void Pennylane::swap_rows(CplxType* mx, const size_t &dim, const size_t row1, const size_t row2){
    //TODO: validate
    for(size_t i = 0; i<dim; ++i){
        std::swap(mx[row1 * dim +i], mx[row2 * dim +i]);
    }
}

void Pennylane::swap_cols(CplxType* mx, const size_t &dim, const size_t column1, const size_t column2){
    //TODO: validate
    for(size_t i=0; i<dim; ++i){
        auto row_num = i*dim;
        std::swap(mx[row_num +column1], mx[row_num +column2]);
    }
}


// Join new qubit indices to target_list according to a given first_target_wires, and set a new matrix to "matrix"
void Pennylane::get_extended_matrix(unique_ptr<AbstractGate> gate,
vector<CplxType>& matrix, INDICES& new_control_wires, INDICES&
new_target_wires, INDICES& first_control_wires, INDICES& first_target_wires) {

    // New qubits index may be in either gate_target_index, gate_control_index, or it comes from the other gate.
    // Case 0 : If qubit index is in gate_target_index -> named A
    std::vector<UINT> join_from_target = first_target_wires;

    //If multi-qubit gate, reverse qubit indices to conform to the qubit ordering
    std::reverse(join_from_target.begin(), join_from_target.end());

    // Case 1 : If qubit index is in gate_control_index -> named B
    // TODO: test if >1 join_from_control: do we need reversal?
    std::vector<UINT> join_from_control;
    ITYPE control_mask = 0;
    for (auto wire : new_target_wires) {
        if (wire_is_in(wire, first_control_wires)) {
            join_from_control.push_back(wire);

            // PennyLane only has all-up controls
            control_mask ^= (1ULL << (join_from_control.size()-1));
        }
    }

    // Case 2 : If qubit index is not in both -> named C
    // TODO: test if >1 join_from_other_gate: do we need reversal?
    std::vector<UINT> join_from_other_gate;
    for (auto wire : new_target_wires) {
        if (not wire_is_in(wire, first_target_wires) and not wire_is_in(wire, first_control_wires)){
            join_from_other_gate.push_back(wire);
        }
    }

    // At first, qubit indices are ordered as (A,C,B)
    std::vector<UINT> unsorted_new_target_index_list = join_from_target; // A
    unsorted_new_target_index_list.insert(unsorted_new_target_index_list.end(), join_from_other_gate.begin(), join_from_other_gate.end()); // C
    unsorted_new_target_index_list.insert(unsorted_new_target_index_list.end(), join_from_control.begin(), join_from_control.end()); // B

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
    size_t new_matrix_qubit_count = (UINT)new_target_wires.size();
    size_t new_matrix_dim = 1ULL << new_matrix_qubit_count;
    matrix = create_identity(new_matrix_dim);

    // 3. Decide correct 2^{|A|+|C|}-dim block matrix from control wires.
    ITYPE start_block_basis = (1ULL << (join_from_target.size() + join_from_other_gate.size())) * control_mask;
    /*
    std::cout << "The base: " << (1ULL << (join_from_target.size() + join_from_other_gate.size()));
    std::cout << "The control mask: " << control_mask;
    std::cout << "The start_block_basis: " << start_block_basis;
    */

    // 4. Repeat 2^{|C|}-times paste of original gate matrix A .
    vector<CplxType> org_matrix = gate->asMatrix();

    size_t org_matrix_dim = 1ULL << first_target_wires.size();
    ITYPE repeat_count = 1ULL << join_from_other_gate.size();

    // Identity is set well

    for (ITYPE repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
        size_t paste_start = (size_t)(start_block_basis + repeat_index * org_matrix_dim );

        // We would like to paste to the (paste_start, paste_start) coordinate of the matrix
        // Convert this to a single index of the vector
        auto paste_start_vector_like = paste_start * new_matrix_dim + paste_start;

        // Set a block
        Pennylane::set_block(matrix.data(), new_matrix_dim, paste_start_vector_like, org_matrix.data(), org_matrix_dim);
    }

    // Reverse due to qubit ordering
    // TODO: check: best option?
    //std::reverse(unsorted_new_target_index_list.begin(), unsorted_new_target_index_list.end());
    for (UINT i = 0; i<unsorted_new_target_index_list.size(); ++i)
        unsorted_new_target_index_list[i] = new_matrix_qubit_count - unsorted_new_target_index_list[i] - 1;

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

                // std::move to correct position
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

                    swap_cols(matrix.data(), new_matrix_dim, (size_t)basis_01, (size_t)basis_10);
                    swap_rows(matrix.data(), new_matrix_dim, (size_t)basis_01, (size_t)basis_10);
                }
            }
            else ind1++;
        }
    }

    //std::cout << "unsorted " << std::endl;
    //for (auto wire : unsorted_target_list) std::cout << val << " "; std::cout << std::endl;
    //std::cout << matrix << std::endl;
    //sort_target_qubit(unsorted_new_target_index_list, matrix);
    //std::cout << "sorted " << std::endl;
    //for (auto wire : unsorted_target_list) std::cout << val << " "; std::cout << std::endl;
    //std::cout << matrix << std::endl;

}

void matmul(CplxType* mx1, CplxType* mx2, CplxType* res, size_t dim) {

    CplxType* dstPtr = res;
    const CplxType* leftPtr = mx1;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            const CplxType* rightPtr = mx2 + j;
            CplxType sum = leftPtr[0] * rightPtr[0];
            for (size_t n = 1; n < dim; ++n) {
                rightPtr += dim;
                sum += leftPtr[n] * rightPtr[0];
            }
            *dstPtr++ = sum;
        }
        leftPtr += dim;
    }
}

unique_ptr<AbstractGate> Pennylane::merge(unique_ptr<AbstractGate> gate_first,
const string& label1, const INDICES & wires1, unique_ptr<AbstractGate>
gate_second, const string& label2, const INDICES & wires2) {

        for (auto it : gate_first->asMatrix()){
            std::cout << it << " ";
        }
        vector<CplxType> orgmat1 = gate_first->asMatrix();
        vector<CplxType> orgmat2 = gate_second->asMatrix();

        // obtain updated qubit information
        auto all_res = Pennylane::get_new_qubit_list(label1, wires1, label2, wires2);
        INDICES new_control_list = std::get<0>(all_res);
        INDICES new_target_list = std::get<1>(all_res);
        /*
        std::cout << "New control list: ";
        for (auto it : new_control_list){
            std::cout << it << " ";
        }
        */

        std::sort(new_target_list.begin(), new_target_list.end());
        std::sort(new_control_list.begin(), new_control_list.end());

        //TODO: revisit separate control and target
        // This is considering all wires as target with control wires in the beginning
        new_target_list.insert(new_target_list.begin(), new_control_list.begin(), new_control_list.end());
        new_control_list = {};


        // extend gate matrix to whole qubit list
        vector<CplxType> matrix_first, matrix_second;

        //TODO: revisit separate control and target
        /*
        auto res_wires = Pennylane::separate_control_and_target(label1, wires1);
        INDICES first_control = std::get<0>(res_wires);
        INDICES first_target = std::get<1>(res_wires);
        */
        INDICES first_control = {};
        auto first_target = wires1;
        get_extended_matrix(std::move(gate_first), matrix_first, new_control_list, new_target_list, first_control, first_target);

        /*
        //TODO: revisit separate control and target
        auto res_wires2 = Pennylane::separate_control_and_target(label2, wires2);
        INDICES second_control = std::get<0>(res_wires2);
        INDICES second_target = std::get<1>(res_wires2);
        */

        INDICES second_control = {};
        auto second_target = wires2;
        get_extended_matrix(std::move(gate_second), matrix_second, new_control_list, new_target_list, second_control, second_target );

        /*
        std::cout << "first gate is extended from \n";
        for (auto it : orgmat1){
            std::cout << it << " ";
        }
        std::cout << "\n\nto\n\n";
        for (auto it : matrix_first){
            std::cout << it << " ";
        }
        std::cout << "second gate is extended from \n";
        for (auto it : orgmat2){
            std::cout << it << " ";
        }
        std::cout << "\n\nto\n\n";
        for (auto it : matrix_second){
            std::cout << it << " ";
        }
        */

        vector<CplxType> new_matrix(matrix_first.size());
        auto new_dim = int(sqrt(matrix_first.size()));
        //std::cout << "dim: " << new_dim;
        matmul(matrix_second.data(), matrix_first.data(), new_matrix.data(), new_dim);

        // generate new matrix gate
        // can do QubitUnitary:
        auto gate = Pennylane::constructGate(new_matrix);

        /*
        std::cout << "result matrix is " << "\n\n";

        for (int i =0; i<new_dim*new_dim; ++i){
            std::cout << new_matrix[i] << " ";
        }
        std::cout << "result matrix size " << new_matrix.size() << "\n\n";
        auto gate = Pennylane::constructGate({1,0,0,1});
        */
        return gate;
    }

void Pennylane::optimize_light(vector<unique_ptr<AbstractGate>> && gate_list, const vector<string>& labels, vector<vector<unsigned int>>& wires, const UINT qubit_count) {
    std::vector<std::pair<int, std::vector<UINT>>> current_step(qubit_count, std::make_pair(-1, std::vector<UINT>()));
    std::cout << "Things in the current_step: \n";
    for (auto item : current_step) {
        std::cout << "elso, UINT: " << item.first;
        for (auto ui : item.second) {
            std::cout << ui << " ";
        }
    }

    UINT ind1 = 0;
    for (auto && gate : gate_list) {
        //unique_ptr<AbstractGate> gate = std::move(gate_list[ind1]);
        std::vector<UINT> target_qubits;
        std::vector<UINT> parent_qubits;

        for (auto val : wires[ind1]) target_qubits.push_back(val);
        //for (auto val : gate->get_control_index_list()) target_qubits.push_back(val);
        std::sort(target_qubits.begin(), target_qubits.end());

        int pos = -1;
        int hit = -1;
        for (UINT target_qubit : target_qubits) {
            if (current_step[target_qubit].first > pos) {
                pos = current_step[target_qubit].first;
                hit = target_qubit;
            }
        }
        if(hit!=-1)
            parent_qubits = current_step[hit].second;
        if (std::includes(parent_qubits.begin(), parent_qubits.end(), target_qubits.begin(), target_qubits.end())) {
            std::cout << " bent vagyok a merge reszeben, pos: " << pos << " ind1: " << ind1;


            //
            auto merged_gate = Pennylane::merge(std::move(gate_list[pos]), labels[pos], wires[pos], std::move(gate), labels[ind1], wires[ind1]);

            // Remove first merged gate
            vector<unique_ptr<AbstractGate>>::iterator first_gate_it = gate_list.begin() + ind1;
            std::cout << "before first erase: " << gate_list.size();
            gate_list.erase(first_gate_it);
            std::cout << "after first erase: " << gate_list.size();

            vector<unique_ptr<AbstractGate>>::iterator insert_here = gate_list.begin() + pos + 1;
            gate_list.insert(insert_here, std::move(merged_gate));
            std::cout << "Pushing merged gate: ";

            // Remove second merged gate
            vector<unique_ptr<AbstractGate>>::iterator second_gate_it = gate_list.begin() + pos;
            std::cout << "before second erase: " << gate_list.size();
            gate_list.erase(second_gate_it);
            std::cout << "after second erase: " << gate_list.size();

            ind1--;

            //std::cout << "merge ";
            //for (auto val : target_qubits) std::cout << val << " ";
            //std::cout << " into ";
            //for (auto val : parent_qubits) std::cout << val << " ";
            //std::cout << std::endl;
        }
        else {
            std::cout << "bent az else agban";
            for (auto target_qubit : target_qubits) {
                auto pair = std::make_pair(ind1, target_qubits);
                std::cout << "putting the follwoing pair: ";
                std::cout << pair.first << std::endl;
                for (auto item : pair.second){
                    std::cout << item << "  ";
                }
                std::cout << "target qubit: " << target_qubit;
                current_step[target_qubit] = pair;
            }
        }
        ++ind1;
    }
}
