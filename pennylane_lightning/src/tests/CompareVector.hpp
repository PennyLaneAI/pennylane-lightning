#pragma once

#include <cstdlib>
#include <vector>

template <class T, class AllocA, class AllocB>
bool operator==(const std::vector<T, AllocA> &lhs,
                const std::vector<T, AllocB> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t idx = 0; idx < lhs.size(); idx++) {
        if (lhs[idx] != rhs[idx]) {
            return false;
        }
    }
    return true;
}
