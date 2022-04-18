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


    [[maybe_unused]] constexpr static std::array<GeneratorOperation,1> implemented_generators{};//{
    //GeneratorOperation::MultiRZ,
   //};

    /*
    constexpr static std::array implemented_generators = { 
        GeneratorOperation::PauliX,
        GeneratorOperation::PauliZ
    };*/
    
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
        
        /*
        std::complex<PrecisionT> tmp;
        for(size_t k=L; k<R;k++)
        {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            tmp = data[i0];
            data[i0] = data[i1];
            data[i1] = tmp;
            //std::swap(data[i0],data[i1]);
        }
        */

    std::complex<PrecisionT> tmp0,tmp1,tmp2,tmp3;
    size_t n0;//,n1,n2,n3;
    size_t i00,i01,i02,i03;
    size_t i10,i11,i12,i13;
    
    for(size_t k=L; k<R;k=k+4)
    {
        //version 1
      
        n0 = k;
        i00 = (((n0 << 1U) & wire_parity_inv )| (wire_parity & n0));
        
        n0 = k + 1;
        i01 = (((n0 << 1U) & wire_parity_inv )| (wire_parity & n0));

        n0 = k + 2;
        i02 = (((n0 << 1U) & wire_parity_inv )| (wire_parity & n0));

        n0 = k + 3;
        i03 = (((n0 << 1U) & wire_parity_inv) | (wire_parity & n0));
         
        tmp0 = data[i00];
        tmp1 = data[i01];
        tmp2 = data[i02];
        tmp3 = data[i03];

        i10 = i00 | rev_wire_shift;
        i11 = i01 | rev_wire_shift;
        i12 = i02 | rev_wire_shift;
        i13 = i03 | rev_wire_shift;

        data[i00] = data[i10];
        data[i01] = data[i11];
        data[i02] = data[i12];
        data[i03] = data[i13];


        data[i10] = tmp0;
        data[i11] = tmp1;
        data[i12] = tmp2;
        data[i13] = tmp3;
        

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
        for(size_t i=0;i<num_threads;i++)
        {
            workers.push_back(std::thread(PauliX_thread<PrecisionT>,data, limits[i], limits[i+1],wire_parity_inv, wire_parity, rev_wire_shift));
        }
        //Join the threads with the main thread
        //*
        for(auto &t:workers)
        {
            t.join();
        }
        //
    }

    template <class PrecisionT>
    static inline void PauliZ_thread(std::complex<PrecisionT> *data, size_t L, size_t R, size_t wire_parity_inv, size_t wire_parity, size_t rev_wire_shift){
    
    /*
     for(size_t k=L; k<R;k++)
     {
         size_t i0 = (((k << 1U) & wire_parity_inv) | (wire_parity & k)) | rev_wire_shift;
         data[i0] *= -1;
         //T tmp = arr[i0];
         //arr[i0]=tmp;
         //arr[i1]*=-1;
         //arr[k]*=-1;
     }
     */
    
     std::complex<PrecisionT> tmp0,tmp1,tmp2,tmp3;
     size_t i0,i1,i2,i3;
     for(size_t k=L; k<R;k=k+4)
     {
         size_t n = k;
         i0 = (((n << 1U) & wire_parity_inv )| (wire_parity & n)) | rev_wire_shift;
         tmp0 = data[i0];
         n=n+1;
         i1 = (((n << 1U) & wire_parity_inv )| (wire_parity & n)) | rev_wire_shift;
         tmp1 = data[i1];

         data[i0]=-tmp0;


         n=n+1;
         i2 = (((n << 1U) & wire_parity_inv )| (wire_parity & n)) | rev_wire_shift;
         tmp2 = data[i2];

         data[i1]=-tmp1;

         n=n+1;
         i3 = (((n << 1U) & wire_parity_inv )| (wire_parity & n)) | rev_wire_shift;
         tmp3 = data[i3];

         //arr[i0]=-tmp0;
     //arr[i1]=-tmp1;
         data[i2]=-tmp2;
         data[i3]=-tmp3;

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
        //
        for(auto &t:workers)
        {
            t.join();
        }
        //
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
