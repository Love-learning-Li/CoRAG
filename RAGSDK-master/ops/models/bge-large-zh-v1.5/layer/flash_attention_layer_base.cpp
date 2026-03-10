/*
 * -------------------------------------------------------------------------
 *  This file is part of the RAGSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * RAGSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
*/
#include "flash_attention_layer_base.h"
#include "feed_forward.h"
#include "operations/plugin_op/aclnn_addmm.h"

namespace atb_speed {
namespace bge_large {
enum FlashAttentionLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_Q_LINEARWEIGHT,
    IN_QUERY_BIAS,
    IN_K_LINEARWEIGHT,
    IN_KEY_BIAS,
    IN_V_LINEARWEIGHT,
    IN_VALUE_BIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_DENSE_BIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_LAYERNORM_BIAS,
    IN_FEEDFORWARDWEIGHT,
    IN_FEEDFORWARD_BIAS,
    IN_FEEDOUT_WEIGHT,
    IN_OUT_DENSE_BIAS,
    IN_LASTLAYERWEIGHT,
    IN_OUT_LAYERNORM_BIAS,
    IN_ATTENTIONMASK,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    OUT_LAYEROUT,
    INTERMIDATE_Q_MIXEDLINEAROUT,
    INTERMIDATE_K_MIXEDLINEAROUT,
    INTERMIDATE_V_MIXEDLINEAROUT,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_FEEDOUT,
    INTERMIDATE_DENSEOUT,
    INTERMIDATE_ADDDENLAYDOUT
};

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 11;
int64_t reshape_seqLen;

void AddQLinearNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &qLinearNode = opGraph.nodes.at(nodeId++);
    qLinearNode.operation = new atb_speed::common::AclnnAddmm("aclnnAddmm");
    qLinearNode.inTensorIds = { IN_HIDDENSTATES, IN_Q_LINEARWEIGHT, IN_QUERY_BIAS };
    qLinearNode.outTensorIds = { INTERMIDATE_Q_MIXEDLINEAROUT };
    qLinearNode.inTensorReshapeFuncs.resize(1);
    qLinearNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2 表示输出的维度
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        reshape_seqLen = oldShape.dims[oldShapeDimIndex + 1];
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex] * oldShape.dims[oldShapeDimIndex + 1];
        oldShapeDimIndex += 2; // 2 表示输出的维度偏移
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    };
}

void AddKLinearNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &kLinearNode = opGraph.nodes.at(nodeId++);
    kLinearNode.operation = new atb_speed::common::AclnnAddmm("aclnnAddmm");
    kLinearNode.inTensorIds = { IN_HIDDENSTATES, IN_K_LINEARWEIGHT, IN_KEY_BIAS };
    kLinearNode.outTensorIds = { INTERMIDATE_K_MIXEDLINEAROUT };
    kLinearNode.inTensorReshapeFuncs.resize(1);
    kLinearNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2 表示输出的维度
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex] * oldShape.dims[oldShapeDimIndex + 1];
        oldShapeDimIndex += 2; // 2 表示输出的维度偏移
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    };
}

void AddVLinearNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &vLinearNode = opGraph.nodes.at(nodeId++);
    vLinearNode.operation = new atb_speed::common::AclnnAddmm("aclnnAddmm");
    vLinearNode.inTensorIds = { IN_HIDDENSTATES, IN_V_LINEARWEIGHT, IN_VALUE_BIAS };
    vLinearNode.outTensorIds = { INTERMIDATE_V_MIXEDLINEAROUT };
    vLinearNode.inTensorReshapeFuncs.resize(1);
    vLinearNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2 表示输出的维度
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex] * oldShape.dims[oldShapeDimIndex + 1];
        oldShapeDimIndex += 2; // 2 表示输出的维度偏移
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    };
}

atb::Status AddSelfAttentionKvCacheNode(atb::GraphParam& opGraph, size_t& nodeId, const FlashAttentionLayerParam &param)
{
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    selfAttentionParam.headNum = param.headNum;
    if (param.dk == 0) {
        return atb::ERROR_INVALID_PARAM;
    }
    selfAttentionParam.qkScale = 1.0 / sqrt(param.dk);
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {
        INTERMIDATE_Q_MIXEDLINEAROUT,
        INTERMIDATE_K_MIXEDLINEAROUT,
        INTERMIDATE_V_MIXEDLINEAROUT,
        IN_ATTENTIONMASK,
        IN_SEQLEN,
    };
    selfAttentionKvCacheNode.outTensorIds = { INTERMIDATE_SELFOUT };

#ifdef Ascend310P
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(3);
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3 表示输出的维度偏移
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = param.headNum;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++]/param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3 表示输出的维度偏移
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = param.headNum;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++]/param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3 表示输出的维度偏移
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = param.headNum;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++]/param.headNum;
    };
#endif

    return atb::NO_ERROR;
}

void AddSelfOutLinearNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    selfOutLinearNode.operation = new atb_speed::common::AclnnAddmm("aclnnAddmm");
    selfOutLinearNode.inTensorIds = { INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_DENSE_BIAS };
    selfOutLinearNode.outTensorIds = { INTERMIDATE_SELFLINEAROUT };

#ifdef Ascend310P
    selfOutLinearNode.inTensorReshapeFuncs.resize(1);
    selfOutLinearNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2 表示输出的维度偏移
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex]*oldShape.dims[oldShapeDimIndex+1];
    };
#endif
}

atb::Status AddSelfResidualAddNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT };
    selfResidualAddNode.inTensorReshapeFuncs.resize(2); // 2 表示输入tensor个数
    selfResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3 表示输出的维度偏移
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++] / reshape_seqLen;
        newShape.dims[newShapeDimIndex++] = reshape_seqLen;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    };

    return atb::NO_ERROR;
}

atb::Status AddSelfNormNode(atb::GraphParam& opGraph, size_t& nodeId, const FlashAttentionLayerParam &param)
{
    const int32_t beginParamsAxis = 2;  // 2 表示参数起始维度
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::infer::LayerNormParam selfInterNormParam;
    selfInterNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    selfInterNormParam.normParam.epsilon = param.layerNormEps;
    selfInterNormParam.normParam.beginNormAxis = beginParamsAxis;
    selfInterNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(selfInterNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_LAYERNORM_BIAS };
    selfNormNode.outTensorIds = { INTERMIDATE_SELFNORMOUT };
    return atb::NO_ERROR;
}

void AddFeedNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &feedNode = opGraph.nodes.at(nodeId++);
    atb_speed::bge_large::FeedForwardParam feedParam;

#ifndef Ascend310P
    feedParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
#endif

    feedParam.transposeB = true;
    feedParam.isBias = true;
    atb_speed::bge_large::FeedForwardLayer(feedParam, &feedNode.operation);
    feedNode.inTensorIds = { INTERMIDATE_SELFNORMOUT, IN_FEEDFORWARDWEIGHT, IN_FEEDFORWARD_BIAS };
    feedNode.outTensorIds = { INTERMIDATE_FEEDOUT };
}

void AddFeedOutLinearNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &feedOutLinearNode = opGraph.nodes.at(nodeId++);
    feedOutLinearNode.operation = new atb_speed::common::AclnnAddmm("aclnnAddmm");
    feedOutLinearNode.inTensorIds = { INTERMIDATE_FEEDOUT, IN_FEEDOUT_WEIGHT, IN_OUT_DENSE_BIAS };
    feedOutLinearNode.outTensorIds = { INTERMIDATE_DENSEOUT };
}


atb::Status AddAfterDenseAddNode(atb::GraphParam& opGraph, size_t& nodeId)
{
    atb::Node &afterDenseAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addTwoParam;
    addTwoParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addTwoParam, &afterDenseAddNode.operation);
    afterDenseAddNode.inTensorIds = { INTERMIDATE_DENSEOUT, INTERMIDATE_SELFNORMOUT };
    afterDenseAddNode.outTensorIds = { INTERMIDATE_ADDDENLAYDOUT };
    afterDenseAddNode.inTensorReshapeFuncs.resize(1);
    afterDenseAddNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3 表示输出的维度偏移
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++] / reshape_seqLen;
        newShape.dims[newShapeDimIndex++] = reshape_seqLen;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    };

    return atb::NO_ERROR;
}

atb::Status AddLayerNormNode(atb::GraphParam& opGraph, size_t& nodeId, const FlashAttentionLayerParam &param)
{
    atb::Node &layerNormNode = opGraph.nodes.at(nodeId++);
    const int32_t beginParamsAxis = 2;
    atb::infer::LayerNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    selfNormParam.normParam.epsilon = param.layerNormEps;
    selfNormParam.normParam.beginNormAxis = beginParamsAxis;
    selfNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(selfNormParam, &layerNormNode.operation);
    layerNormNode.inTensorIds = { INTERMIDATE_ADDDENLAYDOUT, IN_LASTLAYERWEIGHT, IN_OUT_LAYERNORM_BIAS };
    layerNormNode.outTensorIds = { OUT_LAYEROUT };
    return atb::NO_ERROR;
}

void FreeNodeResource(atb::GraphParam &opGraph)
{
    for (size_t i = 0; i < opGraph.nodes.size(); i++) {
        if (opGraph.nodes[i].operation != nullptr) {
            delete opGraph.nodes[i].operation;
            opGraph.nodes[i].operation = nullptr;
        }
    }
}

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph = {GetFuncNameAndNameSpace(__PRETTY_FUNCTION__), IN_TENSOR_COUNT,
        OUT_TENSOR_COUNT, INTERMEDIATE_TENSOR_COUNT, std::vector<atb::Node>(NODE_COUNT), nullptr};

    size_t nodeId = 0;

    AddQLinearNode(opGraph, nodeId);

    AddKLinearNode(opGraph, nodeId);

    AddVLinearNode(opGraph, nodeId);
    
    // attention
    atb::Status status = AddSelfAttentionKvCacheNode(opGraph, nodeId, param);
    if (status != atb::NO_ERROR) {
        FreeNodeResource(opGraph);
        return status;
    }

    AddSelfOutLinearNode(opGraph, nodeId);

    // hiddenStates + afterDense
    status = AddSelfResidualAddNode(opGraph, nodeId);
    if (status != atb::NO_ERROR) {
        FreeNodeResource(opGraph);
        return status;
    }

    // layerNorm
    status = AddSelfNormNode(opGraph, nodeId, param);
    if (status != atb::NO_ERROR) {
        FreeNodeResource(opGraph);
        return status;
    }

    // feedForward
    AddFeedNode(opGraph, nodeId);

    AddFeedOutLinearNode(opGraph, nodeId);

    // add
    status = AddAfterDenseAddNode(opGraph, nodeId);
    if (status != atb::NO_ERROR) {
        FreeNodeResource(opGraph);
        return status;
    }
    // layerNorm
    status = AddLayerNormNode(opGraph, nodeId, param);
    if (status != atb::NO_ERROR) {
        FreeNodeResource(opGraph);
        return status;
    }
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    
    atb::Status atbStatus = atb::CreateOperation(opGraph, operation);
    if (atbStatus != atb::NO_ERROR) {
        FreeNodeResource(opGraph);
        return atbStatus;
    }

    return atb::NO_ERROR;
}

FlashAttentionLayerBase::FlashAttentionLayerBase() = default;

FlashAttentionLayerBase::~FlashAttentionLayerBase() = default;

void FlashAttentionLayerBase::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}

void from_json(const nlohmann::json &paramJson, FlashAttentionLayerParam &param)
{
    paramJson.at("layerNormEps").get_to(param.layerNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);
    if (paramJson.contains("rank")) {
        paramJson.at("rank").get_to(param.rank);
    }
    if (paramJson.contains("rankSize")) {
        paramJson.at("rankSize").get_to(param.rankSize);
    }
}

atb::Operation *CreateFlashAttentionLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::bge_large::FlashAttentionLayer(paramJson.get<FlashAttentionLayerParam>(), &op);
    return op;
}
} // namespace bge_large
} // namespace atb_speed