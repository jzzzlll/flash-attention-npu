#ifndef FAG_KERNEL_MLA_950_H
#define FAG_KERNEL_MLA_950_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v4 MLA backward device-kernel.
 *
 * This is an MLA-specialised variant of FlashAttentionScoreGrad950 with the
 * memory layout from MLA设计.md A5:
 *   - L1: Q 3x128x192, K 2x128x192, dY 3x128x128, V 1x128x128,
 *         dS 2x128x128, P 2x128x128
 *   - L0C: 128x192 (QK/dY·V/dQ), 128x192 (dK), 128x128 (dV)
 *   - UB:  mm1/mm2 2x64x128 fp32, mask, sftg, lse, dS, P
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "../flash_attn_npu_3/fag_kernel.cpp"
#include "kernel_operator.h"

using Catlass::Gemm::Ascend950FagMlaL1Layout;

static_assert(Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH == MLA_QDY_BUFFER_DEPTH,
              "MLA Q/dY buffer depth must match shared sync-flag array size");

/*
 * MLA A5 memory layout (from MLA设计.md):
 *
 * L1 (496 KiB total for fp16/bf16):
 *   - P       : 2 slots x 128 x 128  =  64 KiB
 *   - dS      : 2 slots x 128 x 128  =  64 KiB
 *   - Q       : 3 slots x 128 x 192  = 144 KiB (resident for C34)
 *   - K       : 2 slots x 128 x 192  =  96 KiB (resident across S2 sweep)
 *   - dY      : 3 slots x 128 x 128  =  96 KiB (resident for C5)
 *   - V       : 1 slot  x 128 x 128  =  32 KiB
 *
 * L0C (256 KiB total, three fixed slots):
 *   - slot 0 : 128 x 192 fp32 =  96 KiB  (QK / dY·V / dQ)
 *   - slot 1 : 128 x 192 fp32 =  96 KiB  (dK)
 *   - slot 2 : 128 x 128 fp32 =  64 KiB  (dV)
 *
 * UB (per AIV, ~213.5 KiB total):
 *   - mm1Res  : 2 x 64 x 128 fp32 =  64 KiB  (dY·V)
 *   - mm2Res  : 2 x 64 x 128 fp32 =  64 KiB  (Q·K^T)
 *   - mask    : 2 x 64 x 128 u8   =   8 KiB
 *   - sftg    : 2 x 64 x 8  fp32 =   4 KiB  (softmax_grad, named delta in code)
 *   - lse     : 2 x 64     fp32 = 512  B
 *   - dS      : 2 x (64+1) x 128 fp16 = 32.5 KiB
 *   - P       : 2 x (64+1) x 128 fp16 = 32.5 KiB
 *
 * Task pipeline:
 *   C1 (Q·K^T) / C2 (dY·V^T)  --AIC-->
 *   V1 (scaledMaskSoftmax) / V2 (dS) --AIV-->
 *   C5 (P^T·dY) / C34 (dS·K / dS^T·Q) --AIC-->
 *   steady state: C1/C2 of task i overlap V1/V2/C5/C34 of task i-1.
 */

template <typename DataType, class BlockMmadSdP_, class BlockMmaddQKV_, class EpilogueScaledMaskSoftmax_,
          class EpilogueSubMul_, FAGTiling950::Layout INPUT_LAYOUT, bool IS_ATTEN_MASK, bool IS_SOFTCAP>
class FlashAttentionScoreGradMla950 {
  public:
    using BlockMmadSdP = BlockMmadSdP_;
    using BlockMmaddQKV = BlockMmaddQKV_;
    using EpilogueScaledMaskSoftmax = EpilogueScaledMaskSoftmax_;
    using EpilogueSubMul = EpilogueSubMul_;

    // Methods
    CATLASS_DEVICE
    FlashAttentionScoreGradMla950() {}

    CATLASS_DEVICE
    ~FlashAttentionScoreGradMla950() {}

    CATLASS_DEVICE
    void Init(FAGKernelParams const& params)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData*>(params.tiling);
        inputStrides_.q = {tiling_->inputStrides.q.batch, tiling_->inputStrides.q.sequence,
                           tiling_->inputStrides.q.head};
        inputStrides_.k = {tiling_->inputStrides.k.batch, tiling_->inputStrides.k.sequence,
                           tiling_->inputStrides.k.head};
        inputStrides_.v = {tiling_->inputStrides.v.batch, tiling_->inputStrides.v.sequence,
                           tiling_->inputStrides.v.head};
        inputStrides_.dout = {tiling_->inputStrides.dout.batch, tiling_->inputStrides.dout.sequence,
                              tiling_->inputStrides.dout.head};
        inputStrides_.out = {tiling_->inputStrides.out.batch, tiling_->inputStrides.out.sequence,
                             tiling_->inputStrides.out.head};
        inputStrides_.enabled = tiling_->inputStrides.enabled;

        doutGm_.SetGlobalBuffer((__gm__ DataType*)params.dout);
        qGm_.SetGlobalBuffer((__gm__ DataType*)params.q);
        kGm_.SetGlobalBuffer((__gm__ DataType*)params.k);
        vGm_.SetGlobalBuffer((__gm__ DataType*)params.v);
        attenMaskGm_.SetGlobalBuffer((__gm__ uint8_t*)params.attenMask);
        softmaxLseGm_.SetGlobalBuffer((__gm__ float*)params.softmaxLse);
        cuSeqQGm_.SetGlobalBuffer((__gm__ int32_t*)params.cuSeqQlen);
        cuSeqKvGm_.SetGlobalBuffer((__gm__ int32_t*)params.cuSeqKvlen);
        // *Used* lengths (seqused); cu tensors above stay the region layout.
        seqUsedQGm_.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedQ);
        seqUsedKvGm_.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedKv);
        hasSeqUsedQ_ = params.seqUsedQ != nullptr;
        hasSeqUsedKv_ = params.seqUsedKv != nullptr;

        dqWorkspaceGm_.SetGlobalBuffer((__gm__ float*)(params.workspace + tiling_->dqOffset));
        dkWorkspaceGm_.SetGlobalBuffer((__gm__ float*)(params.workspace + tiling_->dkOffset));
        dvWorkspaceGm_.SetGlobalBuffer((__gm__ float*)(params.workspace + tiling_->dvOffset));

        batchNum_ = static_cast<uint32_t>(tiling_->batch);
        qSeqlen_ = static_cast<uint32_t>(tiling_->qSeqlen);
        kvSeqlen_ = static_cast<uint32_t>(tiling_->kvSeqlen);
        qHeadNum_ = tiling_->qHeadNum;
        kvHeadNum_ = tiling_->kvHeadNum;
        groupNum_ = static_cast<uint32_t>(tiling_->groupSize);
        qkHeadDim_ = tiling_->qkHeadDim;
        vHeadDim_ = tiling_->vHeadDim;
        qBlockSize_ = tiling_->qTile;
        kvBlockSize_ = tiling_->kvTile;
        coreNum_ = tiling_->usedCoreNum;
        continuousBlockNum_ = tiling_->continuousBlockNum;
        waveSize_ = static_cast<uint64_t>(coreNum_) * continuousBlockNum_;
        scaleValue_ = tiling_->scaleValue;
        softcapValue_ = tiling_->softcapValue;
        softcapInputScale_ = IS_SOFTCAP ? scaleValue_ / softcapValue_ : scaleValue_;

        // L1 layout (MLA A5):
        //   [P ping][P pong][dS ping][dS pong] | cube working via Mm12L1Offset
        //   cube: [RES_KT|RES_K half][RES_VT][Q0][Q1][Q2][K0][K1][DY0][DY1][DY2]
        //   Q 3-slot, K 2-slot, dY 3-slot; V 1-slot loaded per KV step.
        //   D=192 keeps K resident in two slices (128+64) per L0A/L0B 32 KiB ping-pong.
        const uint32_t l1TileBytes = qBlockSize_ * kvBlockSize_ * sizeof(DataType);
        const uint32_t l1PBaseOffset = 0;
        const uint32_t l1dSBaseOffset = TASK_PINGPONG * l1TileBytes;
        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<DataType>(l1PBaseOffset + i * l1TileBytes);
            l1dSTensor[i] = resource.l1Buf.template GetBufferByByte<DataType>(l1dSBaseOffset + i * l1TileBytes);
        }

        // Per-AIV UB layout (MLA A5):
        //   ping/pong each: [mm1Res][mm2Res][attenMask][sftg(=delta)][lse][P][dS]
        //   mm1Res/mm2Res: 64 x 128 fp32; attenMask: 64 x 128 u8;
        //   sftg(=delta): 64 x 8 fp32; lse: 64 fp32;
        //   P/dS: (64+1) x 128 fp16 NZ.
        // Fixpipe SPLIT_M sends half of the logical M rows to each AIV.
        const uint32_t mAligned = RoundUp(qBlockSize_, 16);
        const uint32_t rowsPerSubBlock = mAligned / 2;
        const uint32_t mmResTileBytes = rowsPerSubBlock * kvBlockSize_ * sizeof(float);
        const uint32_t attenMaskTileBytes = rowsPerSubBlock * kvBlockSize_ * sizeof(uint8_t);
        const uint32_t lseTileBytes = RoundUp(rowsPerSubBlock, 8) * 8U * sizeof(float);
        const uint32_t pTileBytes = (rowsPerSubBlock + 1) * kvBlockSize_ * sizeof(DataType);
        const uint32_t dSTileBytes = pTileBytes;
        const uint32_t deltaTileBytes = rowsPerSubBlock * 8U * sizeof(float);

        const uint32_t mm1SlotOffset = 0;
        const uint32_t mm2SlotOffset = mm1SlotOffset + mmResTileBytes;
        const uint32_t attenMaskSlotOffset = mm2SlotOffset + mmResTileBytes;
        const uint32_t lseSlotOffset = RoundUp(attenMaskSlotOffset + attenMaskTileBytes, 32);
        const uint32_t pSlotOffset = RoundUp(lseSlotOffset + lseTileBytes, 32);
        const uint32_t dSSlotOffset = RoundUp(pSlotOffset + pTileBytes, 32);
        const uint32_t deltaSlotOffset = RoundUp(dSSlotOffset + dSTileBytes, 32);
        const uint32_t pingPongHalfBytes = RoundUp(deltaSlotOffset + deltaTileBytes, 32);

        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            const uint32_t halfBaseOffset = i * pingPongHalfBytes;
            ubMm1ResTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + mm1SlotOffset);
            ubMm2ResTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + mm2SlotOffset);
            attenMaskUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<uint8_t>(halfBaseOffset + attenMaskSlotOffset);
            lseUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + lseSlotOffset);
            ubPTensor[i] = resource.ubBuf.template GetBufferByByte<DataType>(halfBaseOffset + pSlotOffset);
            ubDSTensor[i] = resource.ubBuf.template GetBufferByByte<DataType>(halfBaseOffset + dSSlotOffset);
            deltaUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + deltaSlotOffset);
        }

        if (batchNum_ != 0 && qBlockSize_ != 0 && kvBlockSize_ != 0) {
            LoadDecoderBatch();
        }
        InitEvents();
    }

    CATLASS_DEVICE
    void operator()(FAGKernelParams const& params)
    {
        Init(params);
        uint32_t coreIdx = 0;
        uint32_t subBlockIdx = 0;
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_VEC__
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t vectorBlockIdx = AscendC::GetBlockIdx();
        coreIdx = vectorBlockIdx / subBlockNum;
        subBlockIdx = vectorBlockIdx % subBlockNum;
        epilogueScaledMaskSoftmax_.Init(resource, params.tiling);
        epilogueSubMul_.Init(resource, params.workspace, params.tiling);
        epiloguePre_.Init(resource, params.workspace, params.tiling);
        epilogueSoftmaxGradFront_.Init(resource, params.dout, params.out, params.workspace, params.tiling);
        epiloguePost_.Init(resource, params.dq, params.dk, params.dv, params.workspace, params.tiling);
        const uint32_t vectorCoreNum = coreNum_ * subBlockNum;
        epiloguePre_(vectorBlockIdx, vectorCoreNum, vWaitMte3Ping);
        epilogueSoftmaxGradFront_(vectorBlockIdx, vectorCoreNum, mte3WaitMte2Ping, mte3WaitMte2Pong, vWaitMte2Ping,
                                  vWaitMte2Pong, vWaitMte3Ping, vWaitMte3Pong);
#endif
        AscendC::SyncAll<false>();
        RunTasks(coreIdx, subBlockIdx);
        AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
        epiloguePost_(vectorBlockIdx, vectorCoreNum, mte3WaitMte2Ping, mte3WaitMte2Pong, vWaitMte2Ping, vWaitMte2Pong,
                      vWaitMte3Ping, vWaitMte3Pong);
#endif
    }

  private:
    using TilingData = FAGTiling950::FAGTilingData;

    CATLASS_DEVICE
    void InitEvents()
    {
        vWaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        vWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        mte3WaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
        mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
        lseMte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        lseMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        deltaMte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        deltaMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        mte3WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        mte3WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        pVWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
        pVWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
        dSVWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
        dSVWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
    }

    CATLASS_DEVICE
    void GetBatchShape(uint32_t batchIdx, uint64_t& qBatchStart, uint64_t& kvBatchStart, uint32_t& s1Length,
                       uint32_t& s2Length)
    {
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            const uint64_t qEnd = static_cast<uint64_t>(cuSeqQGm_.GetValue(batchIdx));
            const uint64_t kvEnd = static_cast<uint64_t>(cuSeqKvGm_.GetValue(batchIdx));
            qBatchStart = batchIdx == 0 ? 0 : static_cast<uint64_t>(cuSeqQGm_.GetValue(batchIdx - 1));
            kvBatchStart = batchIdx == 0 ? 0 : static_cast<uint64_t>(cuSeqKvGm_.GetValue(batchIdx - 1));
            // *Used* lengths from seqused when provided; batch starts stay on cu.
            s1Length = hasSeqUsedQ_ ? static_cast<uint32_t>(seqUsedQGm_.GetValue(batchIdx))
                                    : static_cast<uint32_t>(qEnd - qBatchStart);
            s2Length = hasSeqUsedKv_ ? static_cast<uint32_t>(seqUsedKvGm_.GetValue(batchIdx))
                                     : static_cast<uint32_t>(kvEnd - kvBatchStart);
        } else {
            s1Length = qSeqlen_;
            s2Length = kvSeqlen_;
            qBatchStart = static_cast<uint64_t>(batchIdx) * s1Length;
            kvBatchStart = static_cast<uint64_t>(batchIdx) * s2Length;
        }
    }

    CATLASS_DEVICE
    void GetValidS1BlockRange(uint32_t s1Length, uint32_t s2Length, uint32_t s2BlockIdx,
                              uint32_t& firstValidS1, uint32_t& validS1Num)
    {
        validS1Num = 0;
        // No mask: every S1 block is valid.
        if constexpr (!IS_ATTEN_MASK) {
            validS1Num = static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_));
            return;
        }
        firstValidS1 = 0;
        const int64_t s2Start = static_cast<int64_t>(s2BlockIdx) * kvBlockSize_;
        const int64_t delta = static_cast<int64_t>(s2Length) - s1Length;
        // Causal: only the first S1 block is restricted; the range extends
        // to the final Q block, without a left-window or KV-tail calculation.
        if (tiling_->maskType == static_cast<uint32_t>(FAGTiling950::MaskType::CAUSAL)) {
            const int64_t s1IdxUp = Max(static_cast<int64_t>(0), s2Start - delta);
            if (s1IdxUp >= s1Length) {
                return;
            }
            firstValidS1 = static_cast<uint32_t>(s1IdxUp / qBlockSize_);
            validS1Num = static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_)) - firstValidS1;
            return;
        }

        // BAND: both S1 boundaries depend on the window and actual KV tile size.
        const int64_t s1IdxUp = tiling_->s2Token == INT32_MAX ? 0 :
            Max(static_cast<int64_t>(0), s2Start - delta - tiling_->s2Token);
        const int64_t s2RealSize = Min(static_cast<int64_t>(kvBlockSize_), static_cast<int64_t>(s2Length) - s2Start);
        const int64_t s1IdxDown = tiling_->s1Token == INT32_MAX ? s1Length :
            Min(static_cast<int64_t>(s1Length), s2Start + s2RealSize - delta + tiling_->s1Token);
        if (s1IdxUp >= s1IdxDown) {
            return;
        }
        firstValidS1 = static_cast<uint32_t>(s1IdxUp / qBlockSize_);
        validS1Num = static_cast<uint32_t>((s1IdxDown + qBlockSize_ - 1) / qBlockSize_) - firstValidS1;
    }

    CATLASS_DEVICE
    uint64_t CountValidS1Blocks(uint32_t s1Length, uint32_t s2Length, uint32_t s2BlockNum)
    {
        uint64_t count = 0;
        for (uint32_t s2BlockIdx = 0; s2BlockIdx < s2BlockNum; ++s2BlockIdx) {
            uint32_t firstValidS1 = 0;
            uint32_t validS1Num = 0;
            GetValidS1BlockRange(s1Length, s2Length, s2BlockIdx, firstValidS1, validS1Num);
            count += validS1Num;
        }
        return count;
    }

    CATLASS_DEVICE
    void LoadDecoderBatch()
    {
        GetBatchShape(decoderBatchIdx_, decoderQBatchStart_, decoderKvBatchStart_, decoderS1Length_, decoderS2Length_);
        decoderS1BlockNum_ = static_cast<uint32_t>(CeilDiv(decoderS1Length_, qBlockSize_));
        decoderS2BlockNum_ = static_cast<uint32_t>(CeilDiv(decoderS2Length_, kvBlockSize_));
        decoderValidS1PerN2_ = CountValidS1Blocks(decoderS1Length_, decoderS2Length_, decoderS2BlockNum_);
        decoderBatchBlockNum_ = kvHeadNum_ * groupNum_ * decoderValidS1PerN2_;
        decoderN2Idx_ = 0;
        decoderS2BlockIdx_ = 0;
        decoderS2BlockBegin_ = 0;
    }

    CATLASS_DEVICE
    void FillBlockInfo(uint64_t blockId, uint32_t n2Idx, uint32_t groupIdx, uint32_t s1BlockIdx, uint32_t s2BlockIdx,
                       FAGBlockInfo& block)
    {
        const uint32_t s1Start = s1BlockIdx * qBlockSize_;
        const uint32_t s2Start = s2BlockIdx * kvBlockSize_;
        const uint64_t totalS1Start = decoderQBatchStart_ + s1Start;
        const uint64_t totalS2Start = decoderKvBatchStart_ + s2Start;
        const uint64_t qHeadIdx = static_cast<uint64_t>(n2Idx) * groupNum_ + groupIdx;

        block.blockId = blockId;
        block.batchIdx = decoderBatchIdx_;
        block.n2Idx = n2Idx;
        block.groupIdx = groupIdx;
        block.s1BlockIdx = s1BlockIdx;
        block.s2BlockIdx = s2BlockIdx;
        block.s1Start = s1Start;
        block.s2Start = s2Start;
        block.curBatchS1 = decoderS1Length_;
        block.curBatchS2 = decoderS2Length_;
        block.s1Extend = Min(qBlockSize_, decoderS1Length_ - s1Start);
        block.s2Extend = Min(kvBlockSize_, decoderS2Length_ - s2Start);
        block.totalS1Start = totalS1Start;
        block.totalS2Start = totalS2Start;
        block.qOffset = (totalS1Start * qHeadNum_ + qHeadIdx) * qkHeadDim_;
        block.kOffset = (totalS2Start * kvHeadNum_ + n2Idx) * qkHeadDim_;
        block.vOffset = (totalS2Start * kvHeadNum_ + n2Idx) * vHeadDim_;
        block.doutOffset = (totalS1Start * qHeadNum_ + qHeadIdx) * vHeadDim_;
        // TODO: M方向划分和MM1输出一致
        block.firstHalfRealS1 = CeilDiv(block.s1Extend, 2U);
    }

    CATLASS_DEVICE uint64_t InputOffset(const FagTensorStrides& strides, const FAGBlockInfo& block, bool isQ) const
    {
        uint64_t token = isQ ? block.s1Start : block.s2Start;
        uint64_t batchOffset = block.batchIdx * strides.batch;
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            token = isQ ? block.totalS1Start : block.totalS2Start;
            batchOffset = 0;
        }
        const uint64_t head = isQ ? block.n2Idx * groupNum_ + block.groupIdx : block.n2Idx;
        return batchOffset + token * strides.sequence + head * strides.head;
    }

    CATLASS_DEVICE
    bool DecodeBlock(uint64_t blockId, FAGBlockInfo& block)
    {
        // ------------bIdx------------
        while (decoderBatchIdx_ < batchNum_ && blockId >= decoderBatchBlockBegin_ + decoderBatchBlockNum_) {
            decoderBatchBlockBegin_ += decoderBatchBlockNum_;
            ++decoderBatchIdx_;
            if (decoderBatchIdx_ < batchNum_) {
                LoadDecoderBatch();
            }
        }
        if (decoderBatchIdx_ >= batchNum_) {
            return false;
        }
        // ------------n2Idx------------
        const uint64_t blockNumPerN2 =
            static_cast<uint64_t>(groupNum_) * (decoderValidS1PerN2_);
        const uint64_t blockInBatch = blockId - decoderBatchBlockBegin_;
        const uint32_t n2Idx = static_cast<uint32_t>(blockInBatch / blockNumPerN2);
        const uint64_t blockInN2 = blockInBatch % blockNumPerN2;
        if (n2Idx != decoderN2Idx_) {
            decoderN2Idx_ = n2Idx;
            decoderS2BlockIdx_ = 0;
            decoderS2BlockBegin_ = 0;
        }
        // ------------s2Idx------------
        uint32_t firstValidS1 = 0;
        uint32_t validS1Num = 0;
        while (decoderS2BlockIdx_ < decoderS2BlockNum_) {
            GetValidS1BlockRange(decoderS1Length_, decoderS2Length_, decoderS2BlockIdx_, firstValidS1, validS1Num);
            const uint64_t s2BlockTasks = static_cast<uint64_t>(groupNum_) * validS1Num;
            if (blockInN2 < decoderS2BlockBegin_ + s2BlockTasks) {
                break;
            }
            decoderS2BlockBegin_ += s2BlockTasks;
            ++decoderS2BlockIdx_;
        }
        if (decoderS2BlockIdx_ >= decoderS2BlockNum_) {
            return false;
        }
        // ------------gIdx & s1Idx------------
        const uint64_t blockInS2 = blockInN2 - decoderS2BlockBegin_;
        const uint32_t groupIdx = static_cast<uint32_t>(blockInS2 / validS1Num);
        const uint32_t s1BlockIdx = firstValidS1 + static_cast<uint32_t>(blockInS2 % validS1Num);
        FillBlockInfo(blockId, n2Idx, groupIdx, s1BlockIdx, decoderS2BlockIdx_, block);
        return true;
    }

    // Return the next task assigned to coreIdx after `cur`, following the
    // global iteration order batch -> n2 -> s2Block -> group -> s1Ord and
    // respecting (kvGroupId % coreNum_ == coreIdx).  Fills `next` with the
    // same fields as FillBlockInfoFrom plus loadKv/initDkv/flushDkv.
    CATLASS_DEVICE
    bool FindNextCoreTask(uint32_t coreIdx, const FAGBlockInfo& cur, uint64_t curKvGroupId,
                          FAGBlockInfo& next, uint64_t& nextKvGroupId)
    {
        uint32_t firstValidS1 = 0;
        uint32_t validS1Num = 0;
        GetValidS1BlockRange(cur.curBatchS1, cur.curBatchS2, cur.s2BlockIdx, firstValidS1, validS1Num);
        if (validS1Num == 0) {
            return false;
        }
        const uint32_t s1Ord = cur.s1BlockIdx - firstValidS1;
        const uint64_t qBatchStart = cur.totalS1Start - cur.s1Start;
        const uint64_t kvBatchStart = cur.totalS2Start - cur.s2Start;

        // Same (batch, n2, s2Block) group: advance group/s1Ord.
        if (s1Ord + 1 < validS1Num) {
            FillBlockInfoFrom(cur.batchIdx, cur.n2Idx, cur.groupIdx, cur.s1BlockIdx + 1, cur.s2BlockIdx,
                              qBatchStart, kvBatchStart, cur.curBatchS1, cur.curBatchS2, next);
            const uint32_t taskInGroup = cur.groupIdx * validS1Num + (s1Ord + 1);
            const uint32_t tasksInGroup = groupNum_ * validS1Num;
            next.loadKv = (taskInGroup == 0);
            next.initDkv = (taskInGroup == 0);
            next.flushDkv = (taskInGroup + 1 == tasksInGroup);
            nextKvGroupId = curKvGroupId;
            return true;
        }
        if (cur.groupIdx + 1 < groupNum_) {
            FillBlockInfoFrom(cur.batchIdx, cur.n2Idx, cur.groupIdx + 1, firstValidS1, cur.s2BlockIdx,
                              qBatchStart, kvBatchStart, cur.curBatchS1, cur.curBatchS2, next);
            const uint32_t taskInGroup = (cur.groupIdx + 1) * validS1Num;
            const uint32_t tasksInGroup = groupNum_ * validS1Num;
            next.loadKv = (taskInGroup == 0);
            next.initDkv = (taskInGroup == 0);
            next.flushDkv = (taskInGroup + 1 == tasksInGroup);
            nextKvGroupId = curKvGroupId;
            return true;
        }

        // Cross s2Block/n2/batch to the next s2Block owned by this core.
        uint32_t batchIdx = cur.batchIdx;
        uint32_t n2Idx = cur.n2Idx;
        uint32_t s2BlockIdx = cur.s2BlockIdx + 1;
        uint64_t kvGroupId = curKvGroupId + 1;
        uint32_t s1Length = cur.curBatchS1;
        uint32_t s2Length = cur.curBatchS2;
        uint64_t qbs = qBatchStart;
        uint64_t kvbs = kvBatchStart;
        uint32_t s2BlockNum = static_cast<uint32_t>(CeilDiv(s2Length, kvBlockSize_));

        while (batchIdx < batchNum_) {
            while (n2Idx < kvHeadNum_) {
                while (s2BlockIdx < s2BlockNum) {
                    if ((kvGroupId % coreNum_) == coreIdx) {
                        GetValidS1BlockRange(s1Length, s2Length, s2BlockIdx, firstValidS1, validS1Num);
                        if (validS1Num != 0) {
                            FillBlockInfoFrom(batchIdx, n2Idx, 0, firstValidS1, s2BlockIdx,
                                              qbs, kvbs, s1Length, s2Length, next);
                            next.loadKv = true;
                            next.initDkv = true;
                            next.flushDkv = (groupNum_ == 1 && validS1Num == 1);
                            nextKvGroupId = kvGroupId;
                            return true;
                        }
                    }
                    ++s2BlockIdx;
                    ++kvGroupId;
                }
                ++n2Idx;
                s2BlockIdx = 0;
            }
            ++batchIdx;
            n2Idx = 0;
            s2BlockIdx = 0;
            if (batchIdx < batchNum_) {
                GetBatchShape(batchIdx, qbs, kvbs, s1Length, s2Length);
                s2BlockNum = static_cast<uint32_t>(CeilDiv(s2Length, kvBlockSize_));
            }
        }
        return false;
    }

    CATLASS_DEVICE
    void RunTasks(uint32_t coreIdx, uint32_t subBlockIdx)
    {
        if (coreIdx >= coreNum_ || coreNum_ == 0 || qBlockSize_ == 0 || kvBlockSize_ == 0) {
            return;
        }
#ifdef __DAV_CUBE__
        BlockMmadSdP mm12(resource, Mm12L1Offset());
        BlockMmaddQKV mm345(resource, Mm12L1Offset());
#endif
#ifdef __DAV_VEC__
        SetVecEvents();
#endif
        // Assign each (batch, n2, s2Block) group to exactly one AIC so that
        // contiguous s1 ranges stay on the same core (KV L1 residency + local
        // dk/dv accumulate). Round-robin over kvGroupId.
        uint64_t taskId = 0;
        uint64_t kvGroupId = 0;
        bool hasPrev = false;
        bool ktResident = false;
        bool vtResident = false;
        FAGBlockInfo prefetchedBlock{};
        bool hasPrefetched = false;

        for (uint32_t batchIdx = 0; batchIdx < batchNum_; ++batchIdx) {
            uint64_t qBatchStart = 0;
            uint64_t kvBatchStart = 0;
            uint32_t s1Length = 0;
            uint32_t s2Length = 0;
            GetBatchShape(batchIdx, qBatchStart, kvBatchStart, s1Length, s2Length);
            const uint32_t s2BlockNum = static_cast<uint32_t>(CeilDiv(s2Length, kvBlockSize_));

            for (uint32_t n2Idx = 0; n2Idx < kvHeadNum_; ++n2Idx) {
                for (uint32_t s2BlockIdx = 0; s2BlockIdx < s2BlockNum; ++s2BlockIdx, ++kvGroupId) {
                    if ((kvGroupId % coreNum_) != coreIdx) {
                        continue;
                    }

                    uint32_t firstValidS1 = 0;
                    uint32_t validS1Num = 0;
                    GetValidS1BlockRange(s1Length, s2Length, s2BlockIdx, firstValidS1, validS1Num);
                    const uint32_t tasksInGroup = groupNum_ * validS1Num;
                    if (tasksInGroup == 0) {
                        continue;
                    }

                    uint32_t taskInGroup = 0;
                    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
                        for (uint32_t s1Ord = 0; s1Ord < validS1Num; ++s1Ord, ++taskInGroup) {
                            FAGBlockInfo block{};
                            const uint32_t s1BlockIdx = firstValidS1 + s1Ord;
                            FillBlockInfoFrom(batchIdx, n2Idx, groupIdx, s1BlockIdx, s2BlockIdx, qBatchStart,
                                              kvBatchStart, s1Length, s2Length, block);
                            block.taskId = taskId;
                            block.loadKv = (taskInGroup == 0);
                            block.initDkv = (taskInGroup == 0);
                            block.flushDkv = (taskInGroup + 1 == tasksInGroup);

                            // Use prefetched Q/dY when this task was preloaded in the
                            // previous iteration; otherwise load from GM.
                            bool loadQ = true;
                            bool loadDY = true;
                            if (hasPrefetched &&
                                prefetchedBlock.batchIdx == block.batchIdx &&
                                prefetchedBlock.n2Idx == block.n2Idx &&
                                prefetchedBlock.groupIdx == block.groupIdx &&
                                prefetchedBlock.s1BlockIdx == block.s1BlockIdx &&
                                prefetchedBlock.s2BlockIdx == block.s2BlockIdx) {
                                loadQ = false;
                                loadDY = false;
                                hasPrefetched = false;
                            }

#ifdef __DAV_CUBE__
                            if (block.loadKv) {
                                // New s2 group: switch K ping-pong slot. The previous
                                // group's K slot stays resident until its last task's
                                // C34 finishes; release it after that C34 below.
                                currentKSlot_ = 1 - currentKSlot_;
                                releaseOldKSlot_ = ktResident;
                            }
                            block.kSlot = currentKSlot_;

                            ProcessC1Stage(block, loadQ, mm12);
                            ProcessC2Stage(block, loadDY, mm12);
                            if (block.loadKv) {
                                ktResident = true;
                                vtResident = true;
                            }

                            // Look ahead and prefetch Q/dY for the next task on this core.
                            FAGBlockInfo next{};
                            uint64_t nextKvGroupId = 0;
                            if (FindNextCoreTask(coreIdx, block, kvGroupId, next, nextKvGroupId)) {
                                next.taskId = taskId + 1;
                                PrefetchQDY(next, mm12);
                                prefetchedBlock = next;
                                hasPrefetched = true;
                            } else {
                                hasPrefetched = false;
                            }

                            if (hasPrev) {
                                ProcessC5Stage(previousBlock_, true, mm345);
                                ProcessC34Stage(previousBlock_, true, mm345);
                            }
                            if (releaseOldKSlot_ && hasPrev) {
                                mm12.ReleaseResidentKT(previousBlock_.kSlot);
                                releaseOldKSlot_ = false;
                            }
#endif
#ifdef __DAV_VEC__
                            if (hasPrev) {
                                ProcessV1Stage(previousBlock_, subBlockIdx);
                                ProcessV2Stage(previousBlock_, subBlockIdx);
                            }
#endif
                            previousBlock_ = block;
                            hasPrev = true;
                            ++taskId;
                        }
                    }
                }
            }
        }

        if (hasPrev) {
#ifdef __DAV_CUBE__
            ProcessC5Stage(previousBlock_, false, mm345);
            ProcessC34Stage(previousBlock_, false, mm345);
            if (ktResident) {
                mm12.ReleaseResidentKT(currentKSlot_);
            }
            if (vtResident) {
                mm12.ReleaseResidentVT();
            }
#endif
#ifdef __DAV_VEC__
            ProcessV1Stage(previousBlock_, subBlockIdx);
            ProcessV2Stage(previousBlock_, subBlockIdx);
#endif
        }
#ifdef __DAV_VEC__
        WaitVecEvents();
#endif
    }

    CATLASS_DEVICE
    void FillBlockInfoFrom(uint32_t batchIdx, uint32_t n2Idx, uint32_t groupIdx, uint32_t s1BlockIdx,
                           uint32_t s2BlockIdx, uint64_t qBatchStart, uint64_t kvBatchStart, uint32_t s1Length,
                           uint32_t s2Length, FAGBlockInfo& block)
    {
        const uint32_t s1Start = s1BlockIdx * qBlockSize_;
        const uint32_t s2Start = s2BlockIdx * kvBlockSize_;
        const uint64_t totalS1Start = qBatchStart + s1Start;
        const uint64_t totalS2Start = kvBatchStart + s2Start;
        const uint64_t qHeadIdx = static_cast<uint64_t>(n2Idx) * groupNum_ + groupIdx;

        block.batchIdx = batchIdx;
        block.n2Idx = n2Idx;
        block.groupIdx = groupIdx;
        block.s1BlockIdx = s1BlockIdx;
        block.s2BlockIdx = s2BlockIdx;
        block.s1Start = s1Start;
        block.s2Start = s2Start;
        block.curBatchS1 = s1Length;
        block.curBatchS2 = s2Length;
        block.s1Extend = Min(qBlockSize_, s1Length - s1Start);
        block.s2Extend = Min(kvBlockSize_, s2Length - s2Start);
        block.totalS1Start = totalS1Start;
        block.totalS2Start = totalS2Start;
        block.qOffset = (totalS1Start * qHeadNum_ + qHeadIdx) * qkHeadDim_;
        block.kOffset = (totalS2Start * kvHeadNum_ + n2Idx) * qkHeadDim_;
        block.vOffset = (totalS2Start * kvHeadNum_ + n2Idx) * vHeadDim_;
        block.doutOffset = (totalS1Start * qHeadNum_ + qHeadIdx) * vHeadDim_;
        block.firstHalfRealS1 = CeilDiv(block.s1Extend, 2U);
    }

#ifdef __DAV_CUBE__
    CATLASS_DEVICE
    void ProcessC1Stage(FAGBlockInfo const& block, bool loadQ, BlockMmadSdP& mm12)
    {
        const uint32_t pdsSlot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint32_t qdySlot = static_cast<uint32_t>(block.taskId % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH);
        const uint16_t flagId = MLA_SYNC_C1_TO_V1_FLAG[qdySlot];
        auto q = MakeGmTensor(qGm_, InputOffset(inputStrides_.q, block, true), block.s1Extend, qkHeadDim_,
                              inputStrides_.q.sequence);
        auto k = MakeGmTensor(kGm_, InputOffset(inputStrides_.k, block, false), block.s2Extend, qkHeadDim_,
                              inputStrides_.k.sequence);
        auto v = MakeGmTensor(vGm_, InputOffset(inputStrides_.v, block, false), block.s2Extend, vHeadDim_,
                              inputStrides_.v.sequence);
        auto sLayout = tla::MakeLayout(tla::MakeShape(block.s1Extend, block.s2Extend),
                                       tla::MakeStride(kvBlockSize_, tla::Int<1>{}));
        auto s = tla::MakeTensor(ubMm1ResTensor[pdsSlot], sLayout, Catlass::Arch::PositionUB{});
        // Prefetch V^T on the group’s first task so MTE2 overlaps S cube.
        mm12.ComputeS(q, k, s, Catlass::GemmCoord(block.s1Extend, block.s2Extend, qkHeadDim_), block.loadKv,
                      block.taskId, loadQ, block.loadKv, v, block.s2Extend, vHeadDim_, block.kSlot);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId + V0_V1_FLAG_ID_OFFSET);
    }

    CATLASS_DEVICE
    void ProcessC2Stage(FAGBlockInfo const& block, bool loadDY, BlockMmadSdP& mm12)
    {
        const uint32_t pdsSlot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint32_t qdySlot = static_cast<uint32_t>(block.taskId % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH);
        const uint16_t flagId = MLA_SYNC_C2_TO_V2_FLAG[qdySlot];
        auto dy = MakeGmTensor(doutGm_, InputOffset(inputStrides_.dout, block, true), block.s1Extend, vHeadDim_,
                               inputStrides_.dout.sequence);
        auto v = MakeGmTensor(vGm_, InputOffset(inputStrides_.v, block, false), block.s2Extend, vHeadDim_,
                              inputStrides_.v.sequence);
        auto dpLayout = tla::MakeLayout(tla::MakeShape(block.s1Extend, block.s2Extend),
                                        tla::MakeStride(kvBlockSize_, tla::Int<1>{}));
        auto dp = tla::MakeTensor(ubMm2ResTensor[pdsSlot], dpLayout, Catlass::Arch::PositionUB{});
        // VT already prefetched in C1 when loadKv; only wait on first use.
        mm12.ComputeDP(dy, v, dp, Catlass::GemmCoord(block.s1Extend, block.s2Extend, vHeadDim_),
                       /*loadV=*/false, /*waitV=*/block.loadKv, block.taskId, loadDY);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId + V0_V1_FLAG_ID_OFFSET);
    }

#ifdef __DAV_CUBE__
    CATLASS_DEVICE
    void PrefetchQDY(FAGBlockInfo const& block, BlockMmadSdP& mm12)
    {
        auto q = MakeGmTensor(qGm_, InputOffset(inputStrides_.q, block, true), block.s1Extend, qkHeadDim_,
                              inputStrides_.q.sequence);
        auto dy = MakeGmTensor(doutGm_, InputOffset(inputStrides_.dout, block, true), block.s1Extend, vHeadDim_,
                               inputStrides_.dout.sequence);
        mm12.PrefetchQ(q, Catlass::GemmCoord(block.s1Extend, block.s2Extend, qkHeadDim_), block.taskId);
        mm12.PrefetchDY(dy, Catlass::GemmCoord(block.s1Extend, block.s2Extend, vHeadDim_), block.taskId);
    }
#endif

    CATLASS_DEVICE
    void ProcessC5Stage(FAGBlockInfo const& block, bool returnL1, BlockMmaddQKV& mm345)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V1_TO_C5_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V1_TO_C5_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t pdsSlot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto dv = MakeGmTensor(dvWorkspaceGm_, block.vOffset, block.s2Extend, vHeadDim_, kvHeadNum_ * vHeadDim_);

        // MLA dv=128 fits in the dedicated DV L0C slot; accumulate until flush.
        mm345.ComputeDv(l1PTensor[pdsSlot], dv, Catlass::GemmCoord(block.s1Extend, vHeadDim_, block.s2Extend),
                        block.taskId,
                        block.initDkv, block.flushDkv, /*enAtomicDv=*/false);
        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V1_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V1_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }

    CATLASS_DEVICE
    void ProcessC34Stage(FAGBlockInfo const& block, bool returnL1, BlockMmaddQKV& mm345)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V2_TO_C34_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V2_TO_C34_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t pdsSlot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto k = MakeGmTensor(kGm_, InputOffset(inputStrides_.k, block, false), block.s2Extend, qkHeadDim_,
                              inputStrides_.k.sequence);
        auto dq = MakeGmTensor(dqWorkspaceGm_, block.qOffset, block.s1Extend, qkHeadDim_, qHeadNum_ * qkHeadDim_);
        auto dk = MakeGmTensor(dkWorkspaceGm_, block.kOffset, block.s2Extend, qkHeadDim_, kvHeadNum_ * qkHeadDim_);

        // MLA qk=192: dQ is flushed per D-slice; dK accumulates in the
        // dedicated DK L0C slot and flushes at the group end.
        mm345.ComputeDqDk(l1dSTensor[pdsSlot], k, dq, dk,
                          Catlass::GemmCoord(block.s1Extend, qkHeadDim_, block.s2Extend),
                          block.taskId,
                          block.kSlot,
                          /*enAtomicDq=*/true, block.initDkv, block.flushDkv,
                          /*enAtomicDk=*/true);
        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C34_TO_V2_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C34_TO_V2_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }

    template <class GlobalTensor>
    CATLASS_DEVICE auto MakeGmTensor(GlobalTensor& gm, uint64_t offset, uint32_t rows, uint64_t cols,
                                     uint64_t rowStride)
    {
        auto layout = tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(rowStride, tla::Int<1>{}));
        return tla::MakeTensor(gm[offset], layout, Catlass::Arch::PositionGM{});
    }

    CATLASS_DEVICE
    uint32_t Mm12L1Offset() const
    {
        // P/dS ping-pong occupies the front of L1; cube working tiles
        // (RES_KT[+K half]/VT + Q0/Q1 + DY0/DY1) start after that.
        return 2U * TASK_PINGPONG * qBlockSize_ * kvBlockSize_ * sizeof(DataType);
    }
#endif

#ifdef __DAV_VEC__
    CATLASS_DEVICE
    void SetVecEvents()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPing);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPong);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPing);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPong);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Ping);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Pong);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Ping);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Pong);
    }

    CATLASS_DEVICE
    void WaitVecEvents()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPing);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPong);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPing);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPong);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Ping);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Pong);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Ping);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Pong);
    }

    CATLASS_DEVICE
    void ProcessV1Stage(FAGBlockInfo const& block, uint32_t subBlockIdx)
    {
        const uint32_t flagSlot = static_cast<uint32_t>(block.taskId % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH);
        const uint32_t bufferId = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(MLA_SYNC_C1_TO_V1_FLAG[flagSlot]);

        // Before task 1 and later overwrite P in L1, C5 must return ownership
        // of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V1_FLAG);
        }

        auto attenMaskLayout = tla::MakeLayout<uint8_t, Catlass::layout::RowMajor>(256, 256);
        auto attenMaskGmTensor = tla::MakeTensor(attenMaskGm_, attenMaskLayout, Catlass::Arch::PositionGM{});

        uint64_t lseRows = 0;
        uint64_t lseColumns = 0;
        uint64_t lseRowOffset = 0;
        uint64_t lseColOffset = 0;
        const uint64_t n1Idx = static_cast<uint64_t>(block.n2Idx) * groupNum_ + block.groupIdx;
        const uint64_t subBlockS1Offset = subBlockIdx * block.firstHalfRealS1;
        const uint64_t subBlockRealS1 = subBlockIdx ? block.s1Extend - block.firstHalfRealS1 : block.firstHalfRealS1;
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            // TND:  [N1, totalS1]
            lseRows = qHeadNum_;
            lseColumns = tiling_->totalQ;
            lseRowOffset = n1Idx;
            lseColOffset = block.totalS1Start + subBlockS1Offset;
        } else {
            // BSND: [B * N1, S1]
            lseRows = static_cast<uint64_t>(batchNum_) * qHeadNum_;
            lseColumns = qSeqlen_;
            lseRowOffset = static_cast<uint64_t>(block.batchIdx) * qHeadNum_ + n1Idx;
            lseColOffset = block.s1Start + subBlockS1Offset;
        }
        auto softmaxLseLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(lseRows, lseColumns);
        auto softmaxLseGmTensor = tla::MakeTensor(
            softmaxLseGm_, softmaxLseLayout, tla::MakeCoord(lseRowOffset, lseColOffset), Catlass::Arch::PositionGM{});

        event_t mte2ToVEvent = bufferId ? vWaitMte2Pong : vWaitMte2Ping;
        event_t vToMte2Event = bufferId ? lseMte2WaitVPong : lseMte2WaitVPing;
        event_t vToMte3Event = bufferId ? mte3WaitVPong : mte3WaitVPing;
        event_t mte3ToVEvent = bufferId ? pVWaitMte3Pong : pVWaitMte3Ping;
        if (subBlockRealS1 != 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
            epilogueScaledMaskSoftmax_(block, attenMaskGmTensor, softmaxLseGmTensor, attenMaskUbTensor[bufferId],
                                       lseUbTensor[bufferId], ubMm1ResTensor[bufferId], ubPTensor[bufferId],
                                       l1PTensor[bufferId], mte2ToVEvent, vToMte2Event, vToMte3Event, mte3ToVEvent,
                                       softcapInputScale_, softcapValue_, subBlockRealS1, subBlockIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
        }

        // Notify C5 only after this vector sub-block has finished writing P.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_V1_TO_C5_FLAG);
    }

    CATLASS_DEVICE
    void ProcessV2Stage(FAGBlockInfo const& block, uint32_t subBlockIdx)
    {
        const uint32_t flagSlot = static_cast<uint32_t>(block.taskId % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH);
        const uint32_t bufferId = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(MLA_SYNC_C2_TO_V2_FLAG[flagSlot]);

        // Before task 1 and later overwrite dS in L1, C3/C4 must return
        // ownership of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_C34_TO_V2_FLAG);
        }
        event_t mte2ToVEvent = bufferId ? vWaitMte2Pong : vWaitMte2Ping;
        event_t vToMte2Event = bufferId ? deltaMte2WaitVPong : deltaMte2WaitVPing;
        event_t vToMte3Event = bufferId ? mte3WaitVPong : mte3WaitVPing;
        event_t mte3ToVEvent = bufferId ? dSVWaitMte3Pong : dSVWaitMte3Ping;

        const uint32_t subBlockRealS1 = subBlockIdx ? block.s1Extend - block.firstHalfRealS1 : block.firstHalfRealS1;
        if (subBlockRealS1 != 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
            epilogueSubMul_(block, ubMm2ResTensor[bufferId], ubPTensor[bufferId], ubMm1ResTensor[bufferId],
                            ubDSTensor[bufferId], deltaUbTensor[bufferId], l1dSTensor[bufferId], mte2ToVEvent,
                            vToMte3Event, mte3ToVEvent, subBlockIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
        }

        // Notify C3/C4 only after this vector sub-block has finished writing dS.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_V2_TO_C34_FLAG);
    }

#endif

  private:
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
#ifdef __DAV_VEC__
    Catlass::Epilogue::Block::FagPre<Catlass::Arch::Ascend950, TilingData> epiloguePre_;
    Catlass::Epilogue::Block::FagSoftmaxGradFront<DataType, Catlass::Arch::Ascend950, TilingData>
        epilogueSoftmaxGradFront_;
    EpilogueScaledMaskSoftmax epilogueScaledMaskSoftmax_;
    EpilogueSubMul epilogueSubMul_;
    Catlass::Epilogue::Block::FagPost<DataType, Catlass::Arch::Ascend950, TilingData> epiloguePost_;
#endif

    const __gm__ TilingData* tiling_ = nullptr;

    AscendC::GlobalTensor<DataType> doutGm_;
    AscendC::GlobalTensor<DataType> qGm_;
    AscendC::GlobalTensor<DataType> kGm_;
    AscendC::GlobalTensor<DataType> vGm_;
    AscendC::GlobalTensor<uint8_t> attenMaskGm_;
    AscendC::GlobalTensor<float> softmaxLseGm_;
    AscendC::GlobalTensor<int32_t> cuSeqQGm_;
    AscendC::GlobalTensor<int32_t> cuSeqKvGm_;
    AscendC::GlobalTensor<int32_t> seqUsedQGm_;
    AscendC::GlobalTensor<int32_t> seqUsedKvGm_;
    bool hasSeqUsedQ_ = false;
    bool hasSeqUsedKv_ = false;
    AscendC::GlobalTensor<float> dqWorkspaceGm_;
    AscendC::GlobalTensor<float> dkWorkspaceGm_;
    AscendC::GlobalTensor<float> dvWorkspaceGm_;

    AscendC::LocalTensor<DataType> l1PTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> l1dSTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> ubMm1ResTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> ubMm2ResTensor[TASK_PINGPONG];
    AscendC::LocalTensor<uint8_t> attenMaskUbTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> lseUbTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> ubPTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> ubDSTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> deltaUbTensor[TASK_PINGPONG];

    event_t vWaitMte2Ping;
    event_t vWaitMte2Pong;
    event_t vWaitMte3Ping;
    event_t vWaitMte3Pong;
    event_t mte3WaitMte2Ping;
    event_t mte3WaitMte2Pong;
    event_t lseMte2WaitVPing;
    event_t lseMte2WaitVPong;
    event_t deltaMte2WaitVPing;
    event_t deltaMte2WaitVPong;
    event_t mte3WaitVPing;
    event_t mte3WaitVPong;
    event_t pVWaitMte3Ping;
    event_t pVWaitMte3Pong;
    event_t dSVWaitMte3Ping;
    event_t dSVWaitMte3Pong;

    uint32_t batchNum_ = 0;
    uint32_t qSeqlen_ = 0;
    uint32_t kvSeqlen_ = 0;
    FagInputStrides inputStrides_;
    uint64_t qHeadNum_ = 0;
    uint64_t kvHeadNum_ = 0;
    uint32_t groupNum_ = 0;
    uint64_t qkHeadDim_ = 0;
    uint64_t vHeadDim_ = 0;
    uint32_t qBlockSize_ = 0;
    uint32_t kvBlockSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t continuousBlockNum_ = 0;
    uint64_t waveSize_ = 0;
    float scaleValue_ = 1.0f;
    float softcapValue_ = 0.0f;
    float softcapInputScale_ = 1.0f;

    uint32_t decoderBatchIdx_ = 0;
    uint64_t decoderBatchBlockBegin_ = 0;
    uint64_t decoderBatchBlockNum_ = 0;
    uint64_t decoderQBatchStart_ = 0;
    uint64_t decoderKvBatchStart_ = 0;
    uint32_t decoderS1Length_ = 0;
    uint32_t decoderS2Length_ = 0;
    uint32_t decoderS1BlockNum_ = 0;
    uint32_t decoderS2BlockNum_ = 0;
    uint64_t decoderValidS1PerN2_ = 0;

    uint32_t decoderN2Idx_ = 0;
    uint32_t decoderS2BlockIdx_ = 0;
    uint64_t decoderS2BlockBegin_ = 0;

    FAGBlockInfo previousBlock_{};
    uint32_t currentKSlot_ = 0;
    bool releaseOldKSlot_ = false;
};

#endif  // FAG_KERNEL_MLA_950_H
