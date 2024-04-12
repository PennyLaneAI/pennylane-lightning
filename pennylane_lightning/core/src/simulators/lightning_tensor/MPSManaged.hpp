#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>

#include <vector>

#include <cuda_runtime.h>
#include <cutensornet.h>

#include "cuTensorNetError.hpp"

/****************************************************************
 *                   Basic Matrix Product State (MPS) Algorithm
 *
 *  Input:
 *    1. A-J are MPS tensors
 *    2. XXXXX are rank-4 gate tensors:
 *
 *     A---B---C---D---E---F---G---H---I---J          MPS tensors
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 0
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 1
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 2
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 3
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 4
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 5
 *     |   |   |   |   |   |   |   |   |   |
 *     XXXXX   XXXXX   XXXXX   XXXXX   XXXXX          gate cycle 6
 *     |   |   |   |   |   |   |   |   |   |
 *     |   XXXXX   XXXXX   XXXXX   XXXXX   |          gate cycle 7
 *     |   |   |   |   |   |   |   |   |   |
 *
 *
 *  Output:
 *    1. maximal virtual extent of the bonds (===) is `maxVirtualExtent` (set by
 *user).
 *
 *     A===B===C===D===E===F===G===H===I===J          MPS tensors
 *     |   |   |   |   |   |   |   |   |   |
 *
 *
 *  Algorithm:
 *    Iterative over the gate cycles, within each cycle, perform gate split
 *operation below for all relevant tensors
 *              ---A---B----
 *                 |   |       GateSplit     ---A===B---
 *                 XXXXX       ------->         |   |
 *                 |   |
 ******************************************************************/

// Sphinx: #2
class MPSHelper {
  public:
    /**
     * \brief Construct an MPSHelper object for gate splitting algorithm.
     *        i       j       k
     *     -------A-------B-------                      i        j        k
     *           p|       |q            ------->     -------A`-------B`-------
     *            GGGGGGGGG                                r|        |s
     *           r|       |s
     * \param[in] numSites The number of sites in the MPS
     * \param[in] physExtent The extent for the physical mode where the gate
     * tensors are acted on. \param[in] maxVirtualExtent The maximal extent
     * allowed for the virtual mode shared between adjacent MPS tensors.
     * \param[in] initialVirtualExtents A vector of size \p numSites-1 where the
     * ith element denotes the extent of the shared mode for site i and site i+1
     * in the beginning of the simulation. \param[in] typeData The data type for
     * all tensors and gates \param[in] typeCompute The compute type for all
     * gate splitting process
     */
    MPSHelper(int32_t numSites, int64_t physExtent, int64_t maxVirtualExtent,
              const std::vector<int64_t> &initialVirtualExtents,
              cudaDataType_t typeData, cutensornetComputeType_t typeCompute)
        : numSites_(numSites), physExtent_(physExtent), typeData_(typeData),
          typeCompute_(typeCompute) {
        // initialize vectors to store the modes and extents for physical and
        // virtual bond
        for (int32_t i = 0; i < numSites + 1; i++) {
            int64_t e =
                (i == 0 || i == numSites) ? 1 : initialVirtualExtents.at(i - 1);
            extentsPerSite_.push_back(e);
            virtualModes_.push_back(nextMode_++);
            if (i != numSites) {
                physModes_.push_back(nextMode_++);
            }
        }
        int64_t untruncatedMaxExtent = (int64_t)std::pow(
            physExtent_, numSites_ / 2); // maximal virtual extent for the MPS
        maxVirtualExtent_ =
            maxVirtualExtent == 0
                ? untruncatedMaxExtent
                : std::min(maxVirtualExtent, untruncatedMaxExtent);
    }

    /**
     * \brief Initialize the MPS metadata and cutensornet library.
     */
    void initialize() {
        // initialize workDesc, svdInfo and input tensor descriptors
        assert(!inited_);
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreate(&handle_));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateWorkspaceDescriptor(handle_, &workDesc_));
        for (int32_t i = 0; i < numSites_; i++) {
            cutensornetTensorDescriptor_t descTensor;
            const int64_t extents[]{extentsPerSite_[i], physExtent_,
                                    extentsPerSite_[i + 1]};
            const int32_t modes[]{virtualModes_[i], physModes_[i],
                                  virtualModes_[i + 1]};
            PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
                handle_,
                /*numModes=*/3, extents,
                /*strides=*/nullptr, // fortran layout
                modes, typeData_, &descTensor));
            descTensors_.push_back(descTensor);
        }
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateTensorSVDConfig(handle_, &svdConfig_));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateTensorSVDInfo(handle_, &svdInfo_));
        inited_ = true;
    }

    /**
     * \brief Compute the maximal number of elements for each site.
     */
    std::vector<size_t> getMaxTensorElements() const {
        // compute the maximal tensor sizes for all sites during MPS simulation
        std::vector<size_t> maxTensorElements(numSites_);
        int64_t maxLeftExtent = 1;
        for (int32_t i = 0; i < numSites_; i++) {
            int64_t maxRightExtent =
                std::min({(int64_t)std::pow(physExtent_, i + 1),
                          (int64_t)std::pow(physExtent_, numSites_ - i - 1),
                          maxVirtualExtent_});
            maxTensorElements[i] = physExtent_ * maxLeftExtent * maxRightExtent;
            maxLeftExtent = maxRightExtent;
        }
        return maxTensorElements;
        // return std::move(maxTensorElements);
    }

    /**
     * \brief Update the SVD truncation setting.
     * \param[in] absCutoff The cutoff value for absolute singular value
     * truncation. \param[in] relCutoff The cutoff value for relative singular
     * value truncation. \param[in] renorm The option for renormalization of the
     * truncated singular values. \param[in] partition The option for
     * partitioning of the singular values.
     */
    cutensornetStatus_t
    setSVDConfig(double absCutoff, double relCutoff,
                 cutensornetTensorSVDNormalization_t renorm,
                 cutensornetTensorSVDPartition_t partition) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetTensorSVDConfigSetAttribute(
            handle_, svdConfig_, CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
            &absCutoff, sizeof(absCutoff)));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetTensorSVDConfigSetAttribute(
            handle_, svdConfig_, CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
            &relCutoff, sizeof(relCutoff)));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetTensorSVDConfigSetAttribute(
            handle_, svdConfig_, CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION,
            &renorm, sizeof(renorm)));

        if (partition != CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL) {
            std::cout
                << "This helper class currently only supports "
                   "\"parititon=CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL\""
                << std::endl;
            exit(-1);
        }
        PL_CUTENSORNET_IS_SUCCESS(cutensornetTensorSVDConfigSetAttribute(
            handle_, svdConfig_, CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION,
            &partition, sizeof(partition)));
        return CUTENSORNET_STATUS_SUCCESS;
    }

    /**
     * \brief Update the algorithm to use for the gating process.
     * \param[in] gateAlgo The gate algorithm to use for MPS simulation.
     */
    void setGateAlgorithm(cutensornetGateSplitAlgo_t gateAlgo) {
        gateAlgo_ = gateAlgo;
    }

    /**
     * \brief Compute the maximal workspace needed for MPS gating algorithm.
     * \param[out] workspaceSize The required workspace size on the device.
     */
    cutensornetStatus_t computeMaxWorkspaceSizes(int64_t *workspaceSize) {
        cutensornetTensorDescriptor_t descTensorInA;
        cutensornetTensorDescriptor_t descTensorInB;
        cutensornetTensorDescriptor_t descTensorInG;
        cutensornetTensorDescriptor_t descTensorOutA;
        cutensornetTensorDescriptor_t descTensorOutB;

        const int64_t maxExtentsAB[]{maxVirtualExtent_, physExtent_,
                                     maxVirtualExtent_};
        const int64_t extentsInG[]{physExtent_, physExtent_, physExtent_,
                                   physExtent_};

        // const int32_t modesInA[] = {'i','p','j'};
        // const int32_t modesInB[] = {'j','q','k'};
        // const int32_t modesInG[] = {'p', 'q', 'r', 's'};
        // const int32_t modesOutA[] = {'i','r','j'};
        // const int32_t modesOutB[] = {'j','s','k'};

        const int32_t modesInA[] = {1, 2, 3};
        const int32_t modesInB[] = {3, 4, 5};
        const int32_t modesInG[] = {2, 4, 6, 7};
        const int32_t modesOutA[] = {1, 6, 3};
        const int32_t modesOutB[] = {3, 7, 5};

        // create tensor descriptors for largest gate split process
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/3, maxExtentsAB,
            /*strides=*/nullptr, // fortran layout
            modesInA, typeData_, &descTensorInA));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/3, maxExtentsAB,
            /*strides=*/nullptr, // fortran layout
            modesInB, typeData_, &descTensorInB));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/4, extentsInG,
            /*strides=*/nullptr, // fortran layout
            modesInG, typeData_, &descTensorInG));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/3, maxExtentsAB,
            /*strides=*/nullptr, // fortran layout
            modesOutA, typeData_, &descTensorOutA));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/3, maxExtentsAB,
            /*strides=*/nullptr, // fortran layout
            modesOutB, typeData_, &descTensorOutB));
        // query workspace size
        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceComputeGateSplitSizes(
            handle_, descTensorInA, descTensorInB, descTensorInG,
            descTensorOutA, descTensorOutB, gateAlgo_, svdConfig_, typeCompute_,
            workDesc_));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
            handle_, workDesc_, CUTENSORNET_WORKSIZE_PREF_MIN,
            CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
            workspaceSize));
        // free the tensor descriptors
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorInA));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorInB));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorInG));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorOutA));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorOutB));
        return CUTENSORNET_STATUS_SUCCESS;
    }

    /**
     * \brief Compute the maximal workspace needed for MPS gating algorithm.
     * \param[in] work Pointer to the allocated workspace.
     * \param[in] workspaceSize The required workspace size on the device.
     */
    cutensornetStatus_t setWorkspace(void *work, int64_t workspaceSize) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceSetMemory(
            handle_, workDesc_, CUTENSORNET_MEMSPACE_DEVICE,
            CUTENSORNET_WORKSPACE_SCRATCH, work, workspaceSize));
        return CUTENSORNET_STATUS_SUCCESS;
    }

    /**
     * \brief In-place execution of the apply gate algorithm on \p siteA and \p
     * siteB. \param[in] siteA The first site where the gate is applied to.
     * \param[in] siteB The second site where the gate is applied to. Must be
     * adjacent to \p siteA. \param[in,out] dataInA The data for the MPS tensor
     * at \p siteA. The input will be overwritten with output mps tensor data.
     * \param[in,out] dataInB The data for the MPS tensor at \p siteB. The input
     * will be overwritten with output mps tensor data. \param[in] dataInG The
     * input data for the gate tensor. \param[in] verbose Whether to print out
     * the runtime information regarding truncation. \param[in] stream The CUDA
     * stream on which the computation is performed.
     */
    cutensornetStatus_t applyGate(uint32_t siteA, uint32_t siteB, void *dataInA,
                                  void *dataInB, const void *dataInG,
                                  bool verbose, cudaStream_t stream) {
        if ((siteB - siteA) != 1) {
            std::cout << "SiteB must be the right site of siteA" << std::endl;
            return CUTENSORNET_STATUS_INVALID_VALUE;
        }
        if (siteB >= static_cast<uint32_t>(numSites_)) {
            std::cout << "Site index can not exceed maximal number of sites"
                      << std::endl;
            return CUTENSORNET_STATUS_INVALID_VALUE;
        }

        auto descTensorInA = descTensors_[siteA];
        auto descTensorInB = descTensors_[siteB];

        cutensornetTensorDescriptor_t descTensorInG;

        /*********************************
         * Create output tensor descriptors
         **********************************/
        int32_t physModeInA = physModes_[siteA];
        int32_t physModeInB = physModes_[siteB];
        int32_t physModeOutA = nextMode_++;
        int32_t physModeOutB = nextMode_++;
        const int32_t modesG[]{physModeInA, physModeInB, physModeOutA,
                               physModeOutB};
        const int64_t extentG[]{physExtent_, physExtent_, physExtent_,
                                physExtent_};
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/4, extentG,
            /*strides=*/nullptr, // fortran layout
            modesG, typeData_, &descTensorInG));

        int64_t leftExtentA = extentsPerSite_[siteA];
        int64_t extentABIn = extentsPerSite_[siteA + 1];
        int64_t rightExtentB = extentsPerSite_[siteA + 2];
        // Compute the expected shared extent of output tensor A and B.
        int64_t combinedExtentLeft =
            std::min(leftExtentA, extentABIn * physExtent_) * physExtent_;
        int64_t combinedExtentRight =
            std::min(rightExtentB, extentABIn * physExtent_) * physExtent_;
        int64_t extentABOut = std::min(
            {combinedExtentLeft, combinedExtentRight, maxVirtualExtent_});

        cutensornetTensorDescriptor_t descTensorOutA;
        cutensornetTensorDescriptor_t descTensorOutB;
        const int32_t modesOutA[]{virtualModes_[siteA], physModeOutA,
                                  virtualModes_[siteA + 1]};
        const int32_t modesOutB[]{virtualModes_[siteB], physModeOutB,
                                  virtualModes_[siteB + 1]};
        const int64_t extentOutA[]{leftExtentA, physExtent_, extentABOut};
        const int64_t extentOutB[]{extentABOut, physExtent_, rightExtentB};

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/3, extentOutA,
            /*strides=*/nullptr, // fortran layout
            modesOutA, typeData_, &descTensorOutA));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateTensorDescriptor(
            handle_,
            /*numModes=*/3, extentOutB,
            /*strides=*/nullptr, // fortran layout
            modesOutB, typeData_, &descTensorOutB));

        /**********
         * Execution
         ***********/
        PL_CUTENSORNET_IS_SUCCESS(cutensornetGateSplit(
            handle_, descTensorInA, dataInA, descTensorInB, dataInB,
            descTensorInG, dataInG, descTensorOutA,
            dataInA,       // overwrite in place
            /*s=*/nullptr, // we partition s equally onto A and B, therefore s
                           // is not needed
            descTensorOutB, dataInB, // overwrite in place
            gateAlgo_, svdConfig_, typeCompute_, svdInfo_, workDesc_, stream));

        /**************************
         * Query runtime information
         ***************************/
        if (verbose) {
            int64_t fullExtent;
            int64_t reducedExtent;
            double discardedWeight;
            PL_CUTENSORNET_IS_SUCCESS(cutensornetTensorSVDInfoGetAttribute(
                handle_, svdInfo_, CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT,
                &fullExtent, sizeof(fullExtent)));
            PL_CUTENSORNET_IS_SUCCESS(cutensornetTensorSVDInfoGetAttribute(
                handle_, svdInfo_, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                &reducedExtent, sizeof(reducedExtent)));
            PL_CUTENSORNET_IS_SUCCESS(cutensornetTensorSVDInfoGetAttribute(
                handle_, svdInfo_, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT,
                &discardedWeight, sizeof(discardedWeight)));
            std::cout << "virtual bond truncated from " << fullExtent << " to "
                      << reducedExtent << " with a discarded weight "
                      << discardedWeight << std::endl;
        }

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorInA));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorInB));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyTensorDescriptor(descTensorInG));

        // update pointer to the output tensor descriptor and the output shared
        // extent
        physModes_[siteA] = physModeOutA;
        physModes_[siteB] = physModeOutB;
        descTensors_[siteA] = descTensorOutA;
        descTensors_[siteB] = descTensorOutB;

        int32_t numModes = 3;
        std::vector<int64_t> extentAOut(numModes);
        PL_CUTENSORNET_IS_SUCCESS(cutensornetGetTensorDetails(
            handle_, descTensorOutA, &numModes, nullptr, nullptr,
            extentAOut.data(), nullptr));
        // update the shared extent of output A and B which can potentially get
        // reduced if absCutoff and relCutoff is non-zero.
        extentsPerSite_[siteA + 1] =
            extentAOut[2]; // mode label order is always (left_virtual,
                           // physical, right_virtual)
        return CUTENSORNET_STATUS_SUCCESS;
    }

    /**
     * \brief Free all the tensor descriptors in mpsHelper.
     */
    ~MPSHelper() {
        if (inited_) {
            for (auto &descTensor : descTensors_) {
                cutensornetDestroyTensorDescriptor(descTensor);
            }
            cutensornetDestroy(handle_);
            cutensornetDestroyWorkspaceDescriptor(workDesc_);
        }
        if (svdConfig_ != nullptr) {
            cutensornetDestroyTensorSVDConfig(svdConfig_);
        }
        if (svdInfo_ != nullptr) {
            cutensornetDestroyTensorSVDInfo(svdInfo_);
        }
    }

  private:
    int32_t numSites_;   ///< Number of sites in the MPS
    int64_t physExtent_; ///< Extent for the physical index
    int64_t maxVirtualExtent_{
        0}; ///< The maximal extent allowed for the virtual dimension
    cudaDataType_t typeData_;
    cutensornetComputeType_t typeCompute_;

    bool inited_{false};
    std::vector<int32_t>
        physModes_; ///< A vector of length \p numSites_ storing
                    ///< the physical mode of each site.
    std::vector<int32_t>
        virtualModes_; ///< A vector of length \p numSites_+1; For site i,
                       ///< virtualModes_[i] and virtualModes_[i+1] represents
                       ///< the left and right virtual mode.
    std::vector<int64_t>
        extentsPerSite_; ///< A vector of length \p numSites_+1; For site i,
                         ///< extentsPerSite_[i] and extentsPerSite_[i+1]
                         ///< represents the left and right virtual extent.

    cutensornetHandle_t handle_{nullptr};
    std::vector<cutensornetTensorDescriptor_t>
        descTensors_; /// A vector of length \p numSites_ storing the
                      /// cutensornetTensorDescriptor_t for each site
    cutensornetWorkspaceDescriptor_t workDesc_{nullptr};
    cutensornetTensorSVDConfig_t svdConfig_{nullptr};
    cutensornetTensorSVDInfo_t svdInfo_{nullptr};
    cutensornetGateSplitAlgo_t gateAlgo_{CUTENSORNET_GATE_SPLIT_ALGO_DIRECT};
    int32_t nextMode_{
        0}; /// The next mode label to use for labelling site tensors and gates.
};