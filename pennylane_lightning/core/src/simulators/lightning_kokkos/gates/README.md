# Implementation

## Expval kernels

The general multi-qubit operator kernel requires a private variable `coeffs_in` to store state vector coefficients.
In the Kokkos framework, this variable cannot be a private member of the functor, say, because all threads will overwrite it.
One must then use `TeamPolicy`s together with `scratch_memory_space` which allows creating and manipulating thread-local variables.
This implementation however appears suboptimal compared with the straightforward `RangePolicy` with bit-injection one.

The last being more verbose, it is only implemented for 1- and 2-qubit observables.
It is however possible to generate the code automatically for higher qubit counts with the following Python script. 

```python
name = "getExpValFourQubitOpFunctor"
n_wires = 4
nU = 2**n_wires

print(f"""template <class PrecisionT> struct {name} {{
      
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    std::size_t dim;
    std::size_t num_qubits;

    {name}(const KokkosComplexVector &arr_,
                                 std::size_t num_qubits_,
                                 const KokkosComplexVector &matrix_,
                                 KokkosIntVector &wires_) {{
        dim = 1U << wires_.size();
        num_qubits = num_qubits_;
        wires = wires_;
        arr = arr_;
        matrix = matrix_;
    }}
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {{
    const std::size_t n_wires = wires.size();
    const std::size_t kdim = k * dim;
""")

for k in range(nU):
    print(f"""
    std::size_t i{k:04b} = kdim | {k};
    for (std::size_t pos = 0; pos < n_wires; pos++) {{
        std::size_t x =
            ((i{k:04b} >> (n_wires - pos - 1)) ^
                (i{k:04b} >> (num_qubits - wires(pos) - 1))) &
            1U;
        i{k:04b} = i{k:04b} ^ ((x << (n_wires - pos - 1)) |
                        (x << (num_qubits - wires(pos) - 1)));
    }}
    """)

for k in range(nU):
    tmp = f"expval += real(conj(arr(i{k:04b})) * ("
    tmp += f"matrix(0B{k:04b}{0:04b}) * arr(i{0:04b})"
    for j in range(1, nU):
        tmp += f" + matrix(0B{k:04b}{j:04b}) * arr(i{j:04b})"
    print(tmp)
    print("));")
print("}")
print("};")
```