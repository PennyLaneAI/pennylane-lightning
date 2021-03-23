#pragma once

#include <tuple>
#include <set>

#include "Gates.hpp"

using std::unique_ptr;
using std::vector;
using std::tuple;
using std::string;


using Pennylane::CplxType;

typedef vector<unsigned int> INDICES;
typedef size_t ITYPE;
typedef unsigned int UINT;


namespace Pennylane {

vector<CplxType> create_identity(const unsigned int & dim);

tuple<INDICES, INDICES> separate_control_and_target(const string &opLabel, const INDICES& wires);
tuple<INDICES, INDICES> get_new_qubit_list(const string &opLabel1, const INDICES& first_wires, const string &opLabel2, const INDICES& second_wires);
void set_block(CplxType* mx, const size_t &dim, const size_t &start_ind, CplxType* block_mx, const size_t &block_dim);
void swap_rows(CplxType* mx, const size_t &dim, const size_t row1, const size_t row2);
void swap_cols(CplxType* mx, const size_t &dim, const size_t column1, const size_t column2);


void get_extended_matrix(unique_ptr<Pennylane::AbstractGate> gate,
    vector<CplxType>& matrix, INDICES& new_target_wires, INDICES&
    new_control_wires,INDICES& first_target_wires, INDICES& first_control_wires);

unique_ptr<AbstractGate> merge(unique_ptr<AbstractGate> gate_first, const string& label1,
const INDICES & wires1, unique_ptr<AbstractGate> gate_second, const string&
label2, const INDICES & wires2);

//void optimize_light(vector<unique_ptr<AbstractGate>> gate_list, const vector<INDICES>& wires, const UINT qubit_count);
void optimize_light(vector<unique_ptr<AbstractGate>> gate_list, const vector<string>& labels, const vector<vector<unsigned int>>& wires, const UINT qubit_count);

}
