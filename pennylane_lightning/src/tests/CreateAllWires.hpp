#pragma once
#include "BitUtil.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "GateOperation.hpp"

#include <cstdlib>
#include <vector>

namespace Pennylane {

class WiresGenerator {
  public:
    [[nodiscard]] virtual auto all_perms() const
        -> const std::vector<std::vector<size_t>> & = 0;
};
class CombinationGenerator : public WiresGenerator {
  private:
    std::vector<size_t> v_;
    std::vector<std::vector<size_t>> all_perms_;

  public:
    void comb(size_t n, size_t r) {
        if (r == 0) {
            all_perms_.push_back(v_);
            return;
        }
        if (n < r) {
            return;
        }

        v_[r - 1] = n - 1;
        comb(n - 1, r - 1);

        comb(n - 1, r);
    }

    CombinationGenerator(size_t n, size_t r) {
        v_.resize(r);
        comb(n, r);
    }

    [[nodiscard]] auto all_perms() const
        -> const std::vector<std::vector<size_t>> & override {
        return all_perms_;
    }
};
class PermutationGenerator : public WiresGenerator {
  private:
    std::vector<std::vector<size_t>> all_perms_;
    std::vector<size_t> available_elts_;
    std::vector<size_t> v;

  public:
    void perm(size_t n, size_t r) {
        if (r == 0) {
            all_perms_.push_back(v);
            return;
        }
        for (size_t i = 0; i < n; i++) {
            v[r - 1] = available_elts_[i];
            std::swap(available_elts_[n - 1], available_elts_[i]);
            perm(n - 1, r - 1);
            std::swap(available_elts_[n - 1], available_elts_[i]);
        }
    }

    PermutationGenerator(size_t n, size_t r) {
        v.resize(r);

        available_elts_.resize(n);
        std::iota(available_elts_.begin(), available_elts_.end(), 0);
        perm(n, r);
    }

    [[nodiscard]] auto all_perms() const
        -> const std::vector<std::vector<size_t>> & override {
        return all_perms_;
    }
};

/**
 * @brief Create all possible combination of wires
 * for a given number of qubits and gate operation
 *
 * @param n_qubits Number of qubits
 * @param gate_op Gate operation
 * @param order Whether the ordering matters (if true, permutation is used)
 */
auto crateAllWires(size_t n_qubits, Gates::GateOperation gate_op, bool order)
    -> std::vector<std::vector<size_t>>;
} // namespace Pennylane
