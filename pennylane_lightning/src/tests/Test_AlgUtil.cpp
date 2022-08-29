#include "AlgUtil.hpp"
#include "Observables.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;
using namespace Pennylane::Algorithms;
using namespace Pennylane::Simulators;

class TestException : public std::exception {};

template <typename T> class TestObservable : public Observable<T> {
  public:
    void
    applyInPlace([[maybe_unused]] StateVectorManagedCPU<T> &sv) const override {
        throw TestException();
    }

    [[nodiscard]] auto
    isEqual([[maybe_unused]] const Observable<T> &other) const
        -> bool override {
        return true;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        return "TestObservable";
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return {};
    }
};

TEMPLATE_TEST_CASE("applyObservables", "[Algorithms]", float, double) {
    using PrecisionT = TestType;

    const size_t num_qubits = 8;

    SECTION("Exceptions are rethrown correctly") {
        std::vector<StateVectorManagedCPU<PrecisionT>> states(
            8, StateVectorManagedCPU<PrecisionT>(num_qubits));

        StateVectorManagedCPU<PrecisionT> ref_state(num_qubits);

        std::vector<std::shared_ptr<Observable<PrecisionT>>> observables{
            std::make_shared<TestObservable<PrecisionT>>(),
            std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                   std::vector<size_t>{0}),
            std::make_shared<TestObservable<PrecisionT>>(),
            std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                   std::vector<size_t>{0}),
            std::make_shared<TestObservable<PrecisionT>>(),
            std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                   std::vector<size_t>{0}),
            std::make_shared<TestObservable<PrecisionT>>(),
            std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                   std::vector<size_t>{0}),
        };

        REQUIRE_THROWS_AS(
            applyObservables<PrecisionT>(states, ref_state, observables),
            TestException);
    }
}

TEMPLATE_TEST_CASE("applyOperationsAdj", "[Algorithms]", float, double) {
    using PrecisionT = TestType;

    const size_t num_qubits = 8;

    SECTION("Exceptions are rethrown correctly") {
        std::vector<StateVectorManagedCPU<PrecisionT>> states(
            8, StateVectorManagedCPU<PrecisionT>(num_qubits));

        OpsData<PrecisionT> ops_data{{"InvalidOpsName"}, {{}}, {{0, 1}}, {{}}};

        REQUIRE_THROWS(applyOperationsAdj<PrecisionT>(states, ops_data, 0));
    }
}
