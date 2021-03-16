#pragma once

#include "Gates.hpp"
using std::unique_ptr;
using std::vector;

using Pennylane::CplxType;



namespace Pennylane {

vector<CplxType> create_identity(const unsigned int & dim);

void get_extended_matrix(unique_ptr<Pennylane::AbstractGate> gate,
vector<CplxType>& matrix, vector<unsigned int>& first_target_wires,
vector<unsigned int>& first_control_wires, vector<unsigned int>&
second_remaining_wires); 
}
