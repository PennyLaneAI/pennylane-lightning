
#include <algorithm>
#include <bitset>
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <type_traits>

#include <cuda_runtime.h>
#include <cutensornet.h>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "cuTensorNetError.hpp"
#include "cuda_helpers.hpp"

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;

// Function to convert a size_t value to a binary string
std::string size_t_to_binary_string(const size_t &numQubits, size_t val) {
    std::string str;
    for (size_t i = 0; i < numQubits; i++) {
        str = (val >> i) & 1 ? ('1' + str) : ('0' + str);
    }
    return str;
}
} // namespace

// namespace Pennylane::LightningTensor {
//  column-major by default for the tensor discriptor
template <class PrecisionT, class Derived> class TensorBase {
  private:
    size_t rank_;   // A rank N tensor has N modes
    size_t length_; // Number of elements in a
    std::vector<size_t> modes_;
    std::vector<size_t> extents_;

  public:
    TensorBase(size_t rank, std::vector<size_t> &modes,
               std::vector<size_t> &extents)
        : rank_(rank), modes_(modes), extents_(extents) {
        PL_ABORT_IF(rank_ != extents_.size(),
                    "Please check if rank or extents are set correctly.");
        length_ = 1;
        for (auto extent : extents) {
            length_ *= extent;
        }
    };

    virtual ~TensorBase() {}

    auto getRank() -> size_t { return rank_; }

    auto getExtents() -> std::vector<size_t> { return extents_; }

    auto getModes() -> std::vector<size_t> { return modes_; };

    size_t getLength() const { return length_; }

    auto getData() { return static_cast<Derived *>(this)->getData(); }
};

template <class PrecisionT>
class cuDeviceTensor
    : public TensorBase<PrecisionT, cuDeviceTensor<PrecisionT>> {
  public:
    // using BaseType = TensorBase<PrecisionT, cuDeviceTensor<PrecisionT>>;
    using BaseType = TensorBase<PrecisionT, cuDeviceTensor>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

    cuDeviceTensor(size_t rank, std::vector<size_t> &modes,
                   std::vector<size_t> &extents, int device_id = 0,
                   cudaStream_t stream_id = 0, bool device_alloc = true)
        : TensorBase<PrecisionT, cuDeviceTensor<PrecisionT>>(rank, modes,
                                                             extents),
          data_buffer_{
              std::make_shared<Pennylane::LightningGPU::DataBuffer<CFP_t>>(
                  BaseType::getLength(), device_id, stream_id, device_alloc)} {}

    cuDeviceTensor(size_t rank, std::vector<size_t> &modes,
                   std::vector<size_t> &extents,
                   Pennylane::LightningGPU::DevTag<int> dev_tag,
                   bool device_alloc = true)
        : TensorBase<PrecisionT, cuDeviceTensor<PrecisionT>>(rank, modes,
                                                             extents),
          data_buffer_{
              std::make_shared<Pennylane::LightningGPU::DataBuffer<CFP_t>>(
                  BaseType::getLength(), dev_tag, device_alloc)} {}

    // cuDeviceTensor() = delete;
    // cuDeviceTensor(const cuDeviceTensor &other) = delete;
    // cuDeviceTensor(cuDeviceTensor &&other) = delete;

    ~cuDeviceTensor() {}

    /**
     * @brief Return a pointer to the GPU data.
     *
     * @return const CFP_t* Complex device pointer.
     */
    [[nodiscard]] auto getData() const -> const CFP_t * {
        return data_buffer_->getData();
    }
    /**
     * @brief Return a pointer to the GPU data.
     *
     * @return CFP_t* Complex device pointer.
     */
    [[nodiscard]] auto getData() -> CFP_t * { return data_buffer_->getData(); }

    /**
     * @brief Get the CUDA stream for the given object.
     *
     * @return cudaStream_t&
     */
    inline auto getStream() -> cudaStream_t {
        return data_buffer_->getStream();
    }
    /**
     * @brief Get the CUDA stream for the given object.
     *
     * @return const cudaStream_t&
     */
    inline auto getStream() const -> cudaStream_t {
        return data_buffer_->getStream();
    }

    void setStream(const cudaStream_t &s) { data_buffer_->setStream(s); }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param sv StateVector host data class.
     */
    inline void
    CopyHostDataToGpu(const std::vector<std::complex<PrecisionT>> &sv,
                      bool async = false) {
        PL_ABORT_IF_NOT(BaseType::getLength() == sv.size(),
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyHostDataToGpu(sv.data(), sv.size(), async);
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param host_sv Complex data pointer to array.
     * @param length Number of complex elements.
     */
    inline void CopyGpuDataToGpuIn(const CFP_t *gpu_sv, std::size_t length,
                                   bool async = false) {
        PL_ABORT_IF_NOT(BaseType::getLength() == length,
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyGpuDataToGpu(gpu_sv, length, async);
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param host_sv Complex data pointer to array.
     * @param length Number of complex elements.
     */
    inline void CopyHostDataToGpu(const std::complex<PrecisionT> *host_sv,
                                  std::size_t length, bool async = false) {
        PL_ABORT_IF_NOT(BaseType::getLength() == length,
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyHostDataToGpu(
            reinterpret_cast<const CFP_t *>(host_sv), length, async);
    }

    /**
     * @brief Explicitly copy data from GPU device to host memory.
     *
     * @param sv Complex data pointer to receive data from device.
     */
    inline void CopyGpuDataToHost(std::complex<PrecisionT> *host_sv,
                                  size_t length, bool async = false) const {
        PL_ABORT_IF_NOT(BaseType::getLength() == length,
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyGpuDataToHost(host_sv, length, async);
    }

    const Pennylane::LightningGPU::DataBuffer<CFP_t> &getDataBuffer() const {
        return *data_buffer_;
    }

    Pennylane::LightningGPU::DataBuffer<CFP_t> &getDataBuffer() {
        return *data_buffer_;
    }

    /**
     * @brief Move and replace DataBuffer for statevector.
     *
     * @param other Source data to copy from.
     */
    void updateData(
        std::unique_ptr<Pennylane::LightningGPU::DataBuffer<CFP_t>> &&other) {
        data_buffer_ = std::move(other);
    }

  private:
    std::shared_ptr<Pennylane::LightningGPU::DataBuffer<CFP_t>> data_buffer_;
};

template <class PrecisionT> class MPS_cuDevice {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

  private:
    cutensornetHandle_t handle_{nullptr};
    cudaDataType_t typeData_;
    cutensornetComputeType_t typeCompute_;
    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

    size_t numQubits_;
    size_t maxExtent_;
    std::vector<size_t> qubitDims_;

    Pennylane::LightningGPU::DevTag<int> dev_tag_;

    std::vector<cuDeviceTensor<PrecisionT>> d_mpsTensors_;

  public:
    MPS_cuDevice(size_t &numQubits, size_t &maxExtent,
                 std::vector<size_t> &qubitDims,
                 Pennylane::LightningGPU::DevTag<int> &dev_tag)
        : numQubits_(numQubits), maxExtent_(maxExtent), qubitDims_(qubitDims),
          dev_tag_(dev_tag) {

        if constexpr (std::is_same_v<PrecisionT, double>) {
            typeData_ = CUDA_C_64F;
            typeCompute_ = CUTENSORNET_COMPUTE_64F;
        } else {
            typeData_ = CUDA_C_32F;
            typeCompute_ = CUTENSORNET_COMPUTE_32F;
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreate(&handle_));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateState(
            /* const cutensornetHandle_t */ handle_,
            /* cutensornetStatePurity_t */ purity_,
            /* int32_t numStateModes */ numQubits_,
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(qubitDims_.data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetState_t * */ &quantumState_));

        for (size_t i = 0; i < numQubits_; i++) {
            // size_t siteRank;
            std::vector<size_t> modes;
            std::vector<size_t> siteExtents;
            if (i == 0) {
                // L
                modes = std::vector<size_t>({i, i + numQubits_});
                siteExtents = std::vector<size_t>({qubitDims[i], maxExtent_});
            } else if (i == numQubits_ - 1) {
                // R
                modes = std::vector<size_t>({i + numQubits_, i});
                siteExtents = std::vector<size_t>({qubitDims[i], maxExtent_});
            } else {
                // M
                modes = std::vector<size_t>(
                    {i + numQubits_ - 1, i, i + numQubits_});
                siteExtents =
                    std::vector<size_t>({maxExtent_, qubitDims[i], maxExtent_});
            }
            d_mpsTensors_.push_back(cuDeviceTensor<PrecisionT>(
                modes.size(), modes, siteExtents, dev_tag_));
        }
    }

    // Set a zero state for d_mpsTensors
    void reset() {
        size_t index = 0;
        this->setBasisState(index);
    }

    void setBasisState(size_t index) {
        // Assuming the site vector is [1,0] or [0,1] and bond vector is
        // [1,0,0...].
        std::string str = size_t_to_binary_string(numQubits_, index);

        std::cout << str << std::endl;

        CFP_t value_cu = Pennylane::LightningGPU::Util::complexToCu<
            std::complex<PrecisionT>>({1.0, 0.0});

        for (size_t i = 0; i < d_mpsTensors_.size(); i++) {
            d_mpsTensors_[i].getDataBuffer().zeroInit();

            size_t target = 0;

            if (i == 0) {
                target = str.at(numQubits_ - 1 - i) == '0' ? 0 : 1;
            } else if (i == numQubits_ - 1) {
                target = str.at(numQubits_ - 1 - i) == '0' ? 0 : maxExtent_;
            } else {
                target = str.at(numQubits_ - 1 - i) == '0' ? 0 : maxExtent_;
            }

            PL_CUDA_IS_SUCCESS(
                cudaMemcpy(&d_mpsTensors_[i].getDataBuffer().getData()[target],
                           &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));
        }

        std::vector<std::vector<int64_t>> extents;
        std::vector<int64_t *> extentsPtr(numQubits_);
        std::vector<void *> mpsTensorsDataPtr(numQubits_, nullptr);

        for (size_t i = 0; i < numQubits_; i++) {
            std::vector<int64_t> localExtents(
                d_mpsTensors_[i].getExtents().size());

            for (size_t j = 0; j < d_mpsTensors_[i].getExtents().size(); j++) {
                localExtents[j] =
                    static_cast<int64_t>(d_mpsTensors_[i].getExtents()[j]);
            }

            extents.push_back(localExtents);

            extentsPtr[i] = extents[i].data();
            mpsTensorsDataPtr[i] =
                static_cast<void *>(d_mpsTensors_[i].getDataBuffer().getData());
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateInitializeMPS(
            /*const cutensornetHandle_t*/ handle_,
            /*cutensornetState_t*/ quantumState_,
            /*cutensornetBoundaryCondition_t*/
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /*const int64_t *const*/ extentsPtr.data(),
            /*const int64_t *const*/ nullptr,
            /*void **/ mpsTensorsDataPtr.data()));
    };

    auto getStateVector() -> std::vector<std::complex<PrecisionT>> {
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateWorkspaceDescriptor(handle_, &workDesc));
        // 1D representation of mpsTensor
        std::vector<size_t> modes(1, 1);
        std::vector<size_t> extent(1, (1 << numQubits_));
        cuDeviceTensor<PrecisionT> d_mpsTensor(modes.size(), modes, extent,
                                               dev_tag_);

        std::vector<void *> d_mpsTensorsPtr(
            1, static_cast<void *>(d_mpsTensor.getDataBuffer().getData()));

        std::size_t freeBytes{0}, totalBytes{0};
        PL_CUDA_IS_SUCCESS(cudaMemGetInfo(&freeBytes, &totalBytes));

        // Both 4096 and 1024 are magic numbers here
        const std::size_t scratchSize =
            (freeBytes - (totalBytes % 4096)) / 1024;

        const std::size_t d_scratch_length = scratchSize / sizeof(size_t);

        DataBuffer<size_t, int> d_scratch(d_scratch_length, dev_tag_, true);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            handle_, quantumState_, scratchSize, workDesc, 0x0));

        int64_t worksize{0};
        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
            handle_, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
            CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
            &worksize));

        if (static_cast<std::size_t>(worksize) <= scratchSize) {
            PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceSetMemory(
                handle_, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                reinterpret_cast<void *>(d_scratch.getData()), worksize));
        } else {
            std::cout << "ERROR: Insufficient workspace size on Device!\n";
            std::abort();
        }

        std::vector<int64_t *> extentsPtr;
        std::vector<int64_t> extent_int64(1, (1 << numQubits_));
        extentsPtr.emplace_back(extent_int64.data());

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            handle_, quantumState_, workDesc, extentsPtr.data(), nullptr,
            d_mpsTensorsPtr.data(), 0));

        std::vector<std::complex<double>> results(extent.front());

        d_mpsTensor.CopyGpuDataToHost(results.data(), results.size());

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        return results;
    }
};

int main() {
    size_t numQubits = 3;
    size_t maxExtent = 2;
    std::vector<size_t> qubitDims(numQubits, 2);
    std::cout << "Quantum circuit: " << numQubits << " qubits\n";
    Pennylane::LightningGPU::DevTag<int> dev_tag(0, 0);

    MPS_cuDevice<double> mps(numQubits, maxExtent, qubitDims, dev_tag);

    size_t index = 7;
    mps.setBasisState(index);
    auto finalState = mps.getStateVector();

    for (auto &element : finalState) {
        std::cout << element << std::endl;
    }

    return 0;
}
