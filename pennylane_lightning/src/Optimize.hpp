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

namespace Pennylane {

vector<CplxType> create_identity(const unsigned int & dim);
/*
void get_extended_matrix(unique_ptr<Pennylane::AbstractGate> gate,
vector<CplxType>& matrix, vector<unsigned int>& first_target_wires,
vector<unsigned int>& first_control_wires, vector<unsigned int>&
second_remaining_wires);
*/

tuple<INDICES, INDICES> separate_control_and_target(const string &opLabel, const INDICES& wires);

}
