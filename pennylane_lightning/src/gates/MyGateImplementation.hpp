#include <vector>
#include <thread>
#include <algorithm>
#include "BitUtil.hpp"
#include "GateUtil.hpp"
#include "Gates.hpp"
#include "GateOperation.hpp"
#include "KernelType.hpp"
#include "PauliGenerator.hpp"

using namespace Pennylane;
using namespace Pennylane::Gates;

class MyGateImplementation: public PauliGenerator<MyGateImplementation>{
  public:
    constexpr static KernelType kernel_id = KernelType::MyKernel; // Will be discussed below
    constexpr static std::string_view name = "MyGateImpl"; // Name of your kernel

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,
        GateOperation::PauliZ
    }; // List of implemented gates


    [[maybe_unused]] constexpr static std::array<GeneratorOperation,1> implemented_generators{};
    
    static inline 
    std::vector<size_t> bounds(size_t parts, size_t mem){
         std::vector<size_t> bnd;
        size_t delta = mem/parts;
        size_t reminder = mem%parts;
        size_t N1=0, N2=0;
        bnd.push_back(N1);
        for(size_t i=0;i<parts;++i){
            N2 = N1 + delta;
            if(i == parts -1)
                 N2+=reminder;
            bnd.push_back(N2);
            N1 = N2;
        }
        return bnd;
    }


    template <class PrecisionT>
    static inline void PauliX_thread(std::complex<PrecisionT>* data, size_t L, size_t R, size_t wire_parity_inv, size_t wire_parity, size_t rev_wire_shift){
        
        size_t n0;
        size_t i00;
        size_t i10;
    
        for(size_t k=L; k<R;k=k+2){
            #pragma unroll
            for(size_t l=0;l<2;l++){
                n0 = k + l;
                i00 = (((n0 << 1U) & wire_parity_inv )| (wire_parity & n0));
                i10 = i00 | rev_wire_shift;
                std::swap(data[i00],data[i10]);
            }
        }
    }

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT>* data,
                            size_t num_qubits,
                            const std::vector<size_t>& wires,
                            [[maybe_unused]] bool inverse) {
        /* Write your implementation */
        using Util::fillLeadingOnes, Util::fillTrailingOnes;
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const size_t N = Util::exp2(num_qubits-1);

        //size_t num_threads = std::thread::hardware_concurrency();//Get the number of threads available

        size_t num_threads = 2;

        std::vector<std::thread> workers;
        std::vector<size_t> limits = bounds(num_threads,N);

        //lauch a group of threads
        for(size_t i=0;i<num_threads;i++){
            workers.push_back(std::thread(PauliX_thread<PrecisionT>,data, limits[i], limits[i+1],wire_parity_inv, wire_parity, rev_wire_shift));
        }
        //Join the threads with the main thread
    
        for(auto &t:workers){
            t.join();
        }
    }

    template <class PrecisionT>
    static inline void PauliZ_thread(std::complex<PrecisionT> *data, size_t L, size_t R, size_t wire_parity_inv, size_t wire_parity, size_t rev_wire_shift){
     size_t i0;
     for(size_t k=L; k<R;k=k+2){
         #pragma unroll
         for(size_t l=0;l<2;l++){
             size_t n = k + l;
            i0 = (((n << 1U) & wire_parity_inv )| (wire_parity & n)) | rev_wire_shift;
            data[i0]*=-1;

         }
     } 
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT>* data,
                            size_t num_qubits,
                            const std::vector<size_t>& wires,
                            [[maybe_unused]] bool inverse) {
        /* Write your implementation */

        using Util::fillLeadingOnes, Util::fillTrailingOnes;
        const size_t rev_wire = num_qubits - wires[0] -1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const size_t N = Util::exp2(num_qubits-1);
 
        //size_t num_threads = std::thread::hardware_concurrency();//Get the number of threads available 
        size_t num_threads = 2;
 
        std::vector<std::thread> workers;
        std::vector<size_t> limits = bounds(num_threads,N);
 
        //lauch a group of threads
        for(size_t i=0;i<num_threads;i++)
        {
            workers.push_back(std::thread(PauliZ_thread<PrecisionT>,data, limits[i], limits[i+1], wire_parity_inv, wire_parity, rev_wire_shift));
        }
        //Join the threads with the main thread
        for(auto &t:workers)
        {
            t.join();
        }
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorPhaseShift(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool adj) -> PrecisionT {
        using Util::fillLeadingOnes, Util::fillTrailingOnes;
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            arr[i0] = std::complex<PrecisionT>{0.0, 0.0};
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1.0);
    }
};
