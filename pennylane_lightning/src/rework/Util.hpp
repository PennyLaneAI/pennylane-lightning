// Copyright 2020 Xanadu Quantum Technologies Inc.

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
 * Contains uncategorised utility functions.
 */
#pragma once

namespace Pennylane {

    /**
     * Calculates 2^n for some integer n > 0 using bitshifts.
     * 
     * @param n the exponent
     * @return value of 2^pow
     */
    inline size_t exp2(const unsigned int& n) {
        return (size_t)1 << n;
    }

}
