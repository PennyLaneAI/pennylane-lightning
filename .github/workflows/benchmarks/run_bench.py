# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# Generate data
import pennylane as qml
import timeit
import json
import sys

if len(sys.argv)!=3:
    raise ValueError("Please provide the device name and the filename as the only arguments.")

device_string, filename = sys.argv[1], sys.argv[2]

ops = ["PauliX", "T", "Hadamard"]

op_res = {o:[] for o in ops}

for num_q in range(1,10):
    dev = qml.device(device_string, wires=num_q)
    for gate in ops:
        @qml.qnode(dev)
        def circuit():
            getattr(qml, gate)(wires=0)
            return qml.expval(qml.PauliZ(0))

        number = 1000
        res = timeit.timeit(circuit, number =number)/number
        op_res[gate].append(res)

with open(filename, 'w') as fp:
    json.dump(op_res, fp)
