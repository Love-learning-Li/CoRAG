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
#include "flash_attention_model.h"
#include "atb/atb_infer.h"
#include "atb_speed/utils/model_factory.h"
#include "atb_speed/utils/operation_util.h"
#include "models/bge-large-zh-v1.5/layer/flash_attention_layer_base.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace bge_large {
REGISTER_MODEL(bge_large, FlashAttentionModel);

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONIDS,
    IN_TENSOR_TOKENTYPEIDS,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

// embedding
const int IN_TENSOR_INPUTIDS_ID = 0;
const int WORDEMBEDDINGNODE_WEIGHT_ID = 0;
const int WORDEMBEDDING_OUT_TENSORS = 0;

const int IN_TENSOR_POSITIONEIDS_ID = 1;
const int POSITIONEMBEDDINGNODE_WEIGHT_ID = 1;
const int POSITIONEMBEDDING_OUT_TENSORS = 1;

const int IN_TENSOR_TOKENTYPEIDS_ID = 2;
const int TOKENTYPEEMBEDDINGNODE_WEIGHT_ID = 2;
const int TOKENTYPEEMBEDDING_OUT_TENSORS = 2;

const int EMBEDDINGNODE_WEIGHT_COUNT = 3;
const int BIAS_COUNT = 1;
const int FIRST_ADD_OUT_TENSORS = 3;
const int SECOND_ADD_OUT_TENSORS = 4;

const int LAYER_FIRST_OUT_TENSORS = 5;

const int MASK_TRANSDATA_OUT_TENSORS = 6;
// layers
const int WEIGHT_COUNT_PER_LAYER = 8;
const int BIAS_COUNT_PER_LAYER = 8;
#ifdef Ascend310P
const int OPERATION_COUNT_BEFORE_LAYER = 7;
#else
const int OPERATION_COUNT_BEFORE_LAYER = 6;
#endif

const int OPERATION_COUNT_AFTER_LAYER = 0;

#ifdef Ascend310P
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 7;
#else
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 6;
#endif

// pooler
const int OUT_TENSOR_HIDDENSTATES_ID = 0;
const int OUT_TENSOR_HIDDENSTATES_DIM_NUM = 3;
const int FA_LAYER_IN_TOKENOFFSET_ID = 18;
const int FA_LAYER_IN_SEQLEN_ID = 19;

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        throw std::runtime_error(ss.str());
    }

    if (!paramJson.contains("layerNormEps") || !paramJson.contains("headNum") ||
        !paramJson.contains("dk") || !paramJson.contains("layerNum")) {
        std::stringstream ss;
        ss << "json param must be contain layerNormEps, headNum, dk, layerNum" << std::endl;
        throw std::runtime_error(ss.str());
    }

    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = CheckPositive(paramJson["headNum"].get<int>());
    dk = CheckPositive(paramJson["dk"].get<int>());
    layerNum = CheckPositive(paramJson["layerNum"].get<int>());
    if (layerNum > MAX_LAYER_NUM) {
        std::stringstream ss;
        ss << "layerNum must be less than or equal to "<< MAX_LAYER_NUM << std::endl;
        throw std::runtime_error(ss.str());
    }

    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = CheckPositive(paramJson["rankSize"].get<int>());
    }
}

FlashAttentionModel::FlashAttentionModel(const std::string &param) : Model("FlashAttentionModel", param)
{
    param_.FromString(param);
}

FlashAttentionModel::~FlashAttentionModel() = default;

uint32_t FlashAttentionModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t FlashAttentionModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status FlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID) = graph_.weightTensors.at(WORDEMBEDDINGNODE_WEIGHT_ID).desc;
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dimNum = OUT_TENSOR_HIDDENSTATES_DIM_NUM;

    size_t inTensorShapeDimIndex = 0;
    size_t outTensorShapeDimIndex = 0;

    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
        inTensorDescs.at(IN_TENSOR_INPUTIDS_ID).shape.dims[inTensorShapeDimIndex++];
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
        inTensorDescs.at(IN_TENSOR_INPUTIDS_ID).shape.dims[inTensorShapeDimIndex++];
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] = outDim;

    return atb::NO_ERROR;
}

void FlashAttentionModel::BuildLayerNode(int& nodeId)
{
    atb::Operation *op = nullptr;
    atb::Tensor *firstInTensor = &graph_.internalTensors.at(LAYER_FIRST_OUT_TENSORS);

    // N × Layers 24 node = 5
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::bge_large::FlashAttentionLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        atb_speed::bge_large::FlashAttentionLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER + WEIGHT_COUNT_PER_LAYER;
            ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(layerId * (WEIGHT_COUNT_PER_LAYER +
                WEIGHT_COUNT_PER_LAYER) + weightTensorId + EMBEDDINGNODE_WEIGHT_COUNT + BIAS_COUNT + 1);
        }
#ifdef Ascend310P
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(MASK_TRANSDATA_OUT_TENSORS);
#else
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
#endif
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);

        if (layerId == param_.layerNum - 1) {
            layerNode.outTensors = { &graph_.outTensors.at(0) };
        } else {
            layerNode.outTensors = { &graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId) };
        }
        firstInTensor = layerNode.outTensors.at(0);
    }
}

atb::Status FlashAttentionModel::BuildLayerNormNode(int& nodeId)
{
    atb::Operation *op = nullptr;
        // Layer Norm
    auto &embNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam embNormParam;

    embNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    const int32_t beginParamsAxis = 2;
    embNormParam.normParam.epsilon = param_.layerNormEps;
    embNormParam.normParam.beginNormAxis = beginParamsAxis;
    embNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(embNormParam, &op);
    embNormNode.operation.reset(op);
    embNormNode.inTensors = { &graph_.internalTensors.at(SECOND_ADD_OUT_TENSORS),
        &graph_.weightTensors.at(EMBEDDINGNODE_WEIGHT_COUNT),
        &graph_.weightTensors.at(EMBEDDINGNODE_WEIGHT_COUNT + BIAS_COUNT) };
    embNormNode.outTensors = { &graph_.internalTensors.at(LAYER_FIRST_OUT_TENSORS) };

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BuildPositionIdsNode(int& nodeId)
{
    atb::Operation *op = nullptr;
    auto &embAddSecNode = graph_.nodes.at(nodeId++);
    atb::infer::ElewiseParam addSecParam;
    addSecParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addSecParam, &op);
    embAddSecNode.operation.reset(op);
    embAddSecNode.inTensors = { &graph_.internalTensors.at(FIRST_ADD_OUT_TENSORS),
        &graph_.internalTensors.at(POSITIONEMBEDDING_OUT_TENSORS) };
    embAddSecNode.outTensors = { &graph_.internalTensors.at(SECOND_ADD_OUT_TENSORS) };

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BuildAddNode(int& nodeId)
{
    atb::Operation *op = nullptr;
    auto &embAddNode = graph_.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &op);
    embAddNode.operation.reset(op);
    embAddNode.inTensors = { &graph_.internalTensors.at(WORDEMBEDDING_OUT_TENSORS),
        &graph_.internalTensors.at(TOKENTYPEEMBEDDING_OUT_TENSORS) };
    embAddNode.outTensors = { &graph_.internalTensors.at(FIRST_ADD_OUT_TENSORS) };

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BuildTokentypeEmbeddingNode(int& nodeId)
{
    atb::Operation *op = nullptr;
    auto &tokentypeEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam tokentypeEmbeddingParam;
    CREATE_OPERATION(tokentypeEmbeddingParam, &op);
    tokentypeEmbeddingNode.operation.reset(op);
    tokentypeEmbeddingNode.inTensors = { &graph_.weightTensors.at(TOKENTYPEEMBEDDINGNODE_WEIGHT_ID),
        &graph_.inTensors.at(IN_TENSOR_TOKENTYPEIDS) };
    tokentypeEmbeddingNode.outTensors = { &graph_.internalTensors.at(TOKENTYPEEMBEDDING_OUT_TENSORS) };

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BuildPositionEmbeddingNode(int& nodeId)
{
    atb::Operation *op = nullptr;
    auto &positionEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam positionEmbeddingParam;
    CREATE_OPERATION(positionEmbeddingParam, &op);
    positionEmbeddingNode.operation.reset(op);
    positionEmbeddingNode.inTensors = { &graph_.weightTensors.at(POSITIONEMBEDDINGNODE_WEIGHT_ID),
        &graph_.inTensors.at(IN_TENSOR_POSITIONIDS) };
    positionEmbeddingNode.outTensors = { &graph_.internalTensors.at(POSITIONEMBEDDING_OUT_TENSORS) };

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BuildWordEmbeddingNode(int& nodeId)
{
    atb::Operation *op = nullptr;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = { &graph_.weightTensors.at(WORDEMBEDDINGNODE_WEIGHT_ID),
        &graph_.inTensors.at(IN_TENSOR_INPUTIDS) };
    wordEmbeddingNode.outTensors = { &graph_.internalTensors.at(WORDEMBEDDING_OUT_TENSORS) };

    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    const int weightTensorSize =
        EMBEDDINGNODE_WEIGHT_COUNT + 1 + BIAS_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum * 2;
    graph_.weightTensors.resize(weightTensorSize);
    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const size_t internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Status atbStatus;
    // Word Embedding
    atbStatus = BuildWordEmbeddingNode(nodeId);
    if (atbStatus != atb::NO_ERROR) {
        return atbStatus;
    }

    // Position Embedding
    atbStatus = BuildPositionEmbeddingNode(nodeId);
    if (atbStatus != atb::NO_ERROR) {
        return atbStatus;
    }

    // Token Type Embedding
    atbStatus = BuildTokentypeEmbeddingNode(nodeId);
    if (atbStatus != atb::NO_ERROR) {
        return atbStatus;
    }

    // Add
    atbStatus = BuildAddNode(nodeId);
    if (atbStatus != atb::NO_ERROR) {
        return atbStatus;
    }
    // Add position ids
    atbStatus = BuildPositionIdsNode(nodeId);
    if (atbStatus != atb::NO_ERROR) {
        return atbStatus;
    }
    // Layer Norm
    atbStatus = BuildLayerNormNode(nodeId);
    if (atbStatus != atb::NO_ERROR) {
        return atbStatus;
    }
#ifdef Ascend310P
    // transdata IN_TENSOR_ATTENTIONMASK from nd to nz
    atb::Operation *op = nullptr;
    auto &transdataNode = graph_.nodes.at(nodeId++);
    atb::infer::TransdataParam transdataParam;
    transdataParam.transdataType = atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;
    CREATE_OPERATION(transdataParam, &op);
    transdataNode.operation.reset(op);
    transdataNode.inTensors = { &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK)};
    transdataNode.outTensors = { &graph_.internalTensors.at(MASK_TRANSDATA_OUT_TENSORS) };
#endif

    BuildLayerNode(nodeId);
    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::ParseParam(const std::string &param)
{
    CHECK_PARAM_LT(param.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) <<"parse param fail, please check param's format,error: " << e.what() << param;
        return atb::ERROR_INVALID_PARAM;
    }

    if (!paramJson.contains("tokenOffset") || !paramJson.contains("seqLen")) {
        ATB_LOG(ERROR) <<"json param must contain tokenOffset and seqLen "<< param;
        return atb::ERROR_INVALID_PARAM;
    }

    std::vector<int> tokenOffsets = paramJson["tokenOffset"].template get<std::vector<int>>();
    std::vector<int> seqLens = paramJson["seqLen"].template get<std::vector<int>>();
    if (tokenOffsets.size() > MAX_BATCH_SIZE  || seqLens.size() > MAX_BATCH_SIZE) {
        ATB_LOG(ERROR) <<"tokenOffset and seqLen size must be less than or equal to "<<MAX_BATCH_SIZE;
        return atb::ERROR_INVALID_PARAM;
    }

    tokenOffset_.clear();
    for (const auto &tokenOffset : tokenOffsets) {
        CHECK_PARAM_GE(tokenOffset, 0);
        CHECK_PARAM_LT(tokenOffset, MAX_PARAM_VALUE);
        tokenOffset_.push_back(tokenOffset);
    }
    seqLen_.clear();
    for (const auto &seqLen : seqLens) {
        CHECK_PARAM_GE(seqLen, 0);
        CHECK_PARAM_LT(seqLen, MAX_PARAM_VALUE);
        seqLen_.push_back(seqLen);
    }
    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER ||
        nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t tokenOffsetTensorId = FA_LAYER_IN_TOKENOFFSET_ID;
    const uint32_t seqLenTensorId = FA_LAYER_IN_SEQLEN_ID;

    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace bge_large
} // namespace atb_speed