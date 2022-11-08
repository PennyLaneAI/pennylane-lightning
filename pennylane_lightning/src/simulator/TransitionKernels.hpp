#pragma once

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <stack>
#include <unordered_map>
#include <vector>
#include <type_traits>

#include "StateVectorManagedCPU.hpp"
#include "StateVectorRawCPU.hpp"

namespace Pennylane {

  enum class TransitionKernelType {Local, NonRandom};

  template<typename fp_t>
  class TransitionKernel {
  public:
    virtual size_t init_state () = 0;
    //outputs the next state and the qratio
    virtual std::pair<size_t,fp_t> operator() (size_t) = 0;
  };

  template <typename fp_t>
  class LocalTransitionKernel : public TransitionKernel<fp_t> {
  private:

    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<size_t> distrib_num_qubits_;
    std::uniform_int_distribution<size_t> distrib_binary_;
    size_t num_qubits_;
  
  public:
    
    LocalTransitionKernel(size_t num_qubits) {
      num_qubits_ = num_qubits;
      gen_ = std::mt19937(rd_());
      distrib_num_qubits_ = std::uniform_int_distribution<size_t>(0,num_qubits-1);
      distrib_binary_ = std::uniform_int_distribution<size_t>(0,1);
    }

    size_t init_state(){
      return 0;
    }
    
    std::pair<size_t,fp_t> operator() (size_t s1) {
      size_t qubit_site = distrib_num_qubits_(gen_);
      size_t qubit_value = distrib_binary_(gen_);
      size_t current_bit = (s1 >> qubit_site) & 1;
    
      if (qubit_value == current_bit)
	return std::pair<size_t,fp_t>(s1,1);
      else if (current_bit == 0){
	return std::pair<size_t,fp_t>(s1+std::pow(2,qubit_site),1);
      }
      else {
	return std::pair<size_t,fp_t>(s1-std::pow(2,qubit_site),1);
      }
    }
  };

  template <typename fp_t>
  class NonZeroRandomTransitionKernel : public TransitionKernel<fp_t> {
    // : public TransitionKernel<fp_t> {
  private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<size_t> distrib_;
    size_t sv_length_;
    std::vector<size_t> non_zeros_;
    
  public:
    size_t init_state(){
      return 0;
    }
    
    NonZeroRandomTransitionKernel(const std::complex<fp_t>* sv, size_t sv_length, fp_t min_error) {
      auto data = sv;
      sv_length_ = sv_length;

      //find nonzero candidates
      for (size_t i = 0; i < sv_length_; i++){
	if (fabs(data[i].real()) > min_error ||
	    fabs(data[i].imag()) > min_error){
	  non_zeros_.push_back(i);
	}
      }
      gen_ = std::mt19937(rd_());
      distrib_ = std::uniform_int_distribution<size_t>(0,non_zeros_.size()-1);
    }
  
    std::pair<size_t,fp_t> operator() ([[maybe_unused]] size_t s1) {
      auto s2 = distrib_(gen_);
      return std::pair<size_t,fp_t>(non_zeros_[s2],1);
    }
  };
  

  template <typename fp_t>
  std::unique_ptr<TransitionKernel<fp_t>> kernel_factory
  (
   const TransitionKernelType kernel_type,
   const std::complex<fp_t>* sv,
   size_t num_qubits
  )
  {

    auto sv_length = Util::exp2(num_qubits);
    if (kernel_type == TransitionKernelType::Local){
      return std::unique_ptr<TransitionKernel<fp_t>>(new NonZeroRandomTransitionKernel<fp_t>(sv,sv_length,std::numeric_limits<fp_t>::epsilon()));
    }
    else { //local is the default 
      return std::unique_ptr<TransitionKernel<fp_t>>(new LocalTransitionKernel<fp_t>(num_qubits));       
    }
  }
}
