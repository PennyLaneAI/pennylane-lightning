AVX2/512 kernel implementation
##############################

AVX2 and AVX512 are extensions to the x86 instruction set providing multiple data processing within a single instruction (SIMD; single instruction, multiple data), supported by modern Intel and AMD CPUs.
AVX2 (AVX512) works with 256-bit (512-bit) registers and naturally supports different integer types (4 or 8 bytes) as well as floating point types (single or double precision).
Those instructions are accessible within C++ using intrinsic functions provided by compilers (Intel/GCC/Clang) [#f1]_.
As an introduction to AVX2/512 intrinsic is out of the scope of this document, we recommend e.g. `this website <https://chryswoods.com/vector_c++/immintrin.html>`_ for an introduction and `Intel intrinsics guide <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>`_ for a reference.

We now discuss how we use SIMD to implement quantum gate operations in Lightning Qubit.
When using SIMD, it is natural to consider data as a list of packed arrays, i.e. two-dimensional array where each row has ``packed_size`` numbers.
For example, let us consider a four-qubit quantum state :math:`C` (which has 16 complex numbers) and ``packed_size=8`` (when we use AVX512 with a double precision floating point or AVX2 with a single precision floating point). In this case, we can write down the statevector (coefficients in the computational basis) as

+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|C[0000].real    |C[0000].imag    |C[0001].real    |C[0001].imag    |C[0010].real    |C[0010].imag    |C[0011].real    |C[0011].imag    |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|C[0100].real    |C[0100].imag    |C[0101].real    |C[0101].imag    |C[0110].real    |C[0110].imag    |C[0111].real    |C[0111].imag    |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|C[1000].real    |C[1000].imag    |C[1001].real    |C[1001].imag    |C[1010].real    |C[1010].imag    |C[1011].real    |C[1011].imag    |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
|C[1100].real    |C[1100].imag    |C[1101].real    |C[1101].imag    |C[1110].real    |C[1110].imag    |C[1111].real    |C[1111].imag    |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+

where we use the binary representation for indices, i.e. :math:`C[i_0 i_1 i_2 i_3] = \langle i_0 i_1 i_2 i_3 |C \rangle`. We can use SIMD to load a whole row (8 numbers) from memory to the register and manipulate multiple data within a row (e.g. swap the first 4 values to the last 4 values).


Implementing single qubit gates
-------------------------------

We now consider an implementation of a single qubit gate. For simplicity, we use the Pauli-X gate where the operation just flips the qubit, e.g. applying :math:`X_2` gives

.. math::

   C'[i_0i_1i_2i_3] = \langle i_0 i_1 i_2 i_3 |X_2| C \rangle = \langle i_0 i_1 \neg i_2 i_3 | C \rangle = C[i_0 i_1 \neg i_2 i_3]

where :math:`\neg 0 = 1` and :math:`\neg 1 = 0` is the bit negation operator.

We see that the gates (1) on the second and the last wires (:math:`X_2` and :math:`X_3`) act within a row (internally in a packed array) but (2) on the zeroth and the first wires (:math:`X_0` and :math:`X_1`) act between rows (externally).
Specifically, the Pauli-X gate just permutes the elements within a row in the former case (1) whereas it swaps the whole two rows in the latter case (2).
This implies that we need two independent implementations of the same gate depending on where the gate applies to.

The following simple (C++ style) pseudocode shows how the algorithm is implemented.

.. code-block::

   template<typename PrecisionT, std::size_t packed_size>
   class ApplyPauliX {
      template<std::size_t wire>
      void applyInternal(...) {
         // Within a row
         permutation = compute a permutation within a row for a given wire
         for every row index {
            row = load a row from the memory
            permute elements in row using permutation
            save row to the memory
         }
      }
      void applyExternal(std::size_t wire, ...) {
         // Between rows
         for proper index k {
            row1 = load k-th row
            row2 = load [k + (1 << wire)]-th row
            save row2 to k-th row
            save row1 to [k + (1 << wire)]-th row
         }
      }
   }

Note that this is a general high-level code and does not care about details such as how many elements are in a row.
However, one can translate it to C++ code without runtime overhead with modern techniques (``constexpr`` and ``template`` classes/functions). One tricky part is how we implement permutations efficiently,
which can be found in :ref:`namespace_Pennylane__LightningQubit__Gates__AVXCommon__Permutation`.
As some AVX2/512 permutation functions require its permutation data to be a compile-time constant, we require ``wire`` to be a compile-time parameter for ``applyInternal``.
The full implementation of the functions can be found in
:cpp:class:`Pennylane::LightningQubit::Gates::AVXCommon::ApplyPauliX`.


Choosing an appropriate function
--------------------------------

When such a function is given, we can choose a proper function to call for a given statevector and a wire. For example, we can simply write as follows.

.. code-block:: cpp

   void applyPauliX(num_qubits, wire, ...) {
      if (2**num_qubits < packed_size / 2) {
         // data size is smaller than the size of a row
         call a fallback function
      }

      if (packed_size/2 < 2**wire) {
         switch(wire) {
         case 0:
            call ApplyPauliX<PrecisionT, packed_size>::applyInternal<0>(...)
         case 1:
            call ApplyPauliX<PrecisionT, packed_size>::applyInternal<1>(...)
         ...
         }
      } else {
         call ApplyPauliX<PrecisionT, packed_size>::applyExternal(wire, ...)
      }
   }

Note that we used a switch-case statement for calling internal functions as the wire index must be a compile-time template parameter for ``applyInternal``.
Since all single-qubit gate functions share the same structure,
it might be beneficial to make a simple helper function that automatically finds a target function depending on the given information.
Two classes :cpp:class:`Pennylane::LightningQubit::Gates::AVXCommon::SingleQubitGateWithParamHelper` and :cpp:class:`Pennylane::LightningQubit::Gates::AVXCommon::SingleQubitGateWithoutParamHelper` provide such functionality for a single-qubit gate with and without parameters, respectively.


Two-qubit gates
---------------

Two qubit gates are also implemented in the same way.
It is slightly more involved as there are four different cases depending on the wires.
So we implement 4 (or 3 when the gate acts symmetrically on the wires) functions.
See :cpp:class:`Pennylane::LightningQubit::Gates::AVXCommon::ApplyCNOT` for example.


.. rubric:: Footnotes

.. [#f1] Still, note that each intrinsic function will not necessarily be a single instruction after compilation, as the number of SIMD registers is limited. Thus compilers handle these optimizations.
