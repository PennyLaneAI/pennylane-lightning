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

# The MIT License (MIT)
#
# Copyright (c) 2009-2018 Xiuzhe (Roger) Luo,
# and other contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Acknowledging the approach for plotting from the quantum-benchmarks repository
# at https://github.com/Roger-luo/quantum-benchmarks.

#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from parameters import qubits, ops
import numpy as np

colors = ["darkblue", "tab:orange", "tab:olive"]
projects = [
    "lightning_master.json",
    "lightning_pr.json",
    "default_qubit.json",
]

COLOR = dict(zip(projects, colors))

op_results = {o: [] for o in ops}

for p in projects:
    with open(p) as f:
        data = json.load(f)
        for k in data.keys():
            op_results[k].append(data[k])

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
((ax1, ax2), (ax3, ax4)) = ax

axes = ax.flatten()

for op, a in zip(ops, ax.flatten()):
    a.set_xlabel("nqubits", size=16)
    a.set_ylabel("ns", size=16)
    a.set_title(op + " gate")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

for a, op in zip(axes, op_results.keys()):
    for k, v in enumerate(projects):
        data = op_results[op][k]
        data = np.array(data) * 1e9
        a.semilogy(qubits, data, "-o", markersize=4, color=COLOR[v], linestyle="None")

plots = []
plt.tight_layout()
plt.subplots_adjust(top=0.85)

lgd = fig.legend(
    plots,
    labels=[p.split(".")[0] for p in projects],
    loc="upper center",
    ncol=4,
    frameon=False,
    prop={"size": 15},
    borderaxespad=-0.4,
    bbox_to_anchor=(0.5, 0.97),
)

plt.savefig("gates.png")
