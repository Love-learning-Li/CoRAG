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

#include <atb/atb_infer.h>

#include "atb_speed/log.h"

#include "operations/plugin_op/aclnn_addmm.h"
#include "operations/plugin_op/aclnn_gelu_operation.h"
#include "feed_forward.h"

namespace atb_speed {
namespace bge_large {
enum FeedForwardId : int {
    IN_HIDDENSIZE_ID = 0,
    INTERMEDIATE_SIZE_ID,
    IN_BIAS_UP_ID,
    ACTIVATION_OUT_ID,
    INTERMEDIATE_DENSE_OUT_ID
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

atb::Status FeedForwardLayer(const FeedForwardParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "FeedForwardLayer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;

    auto &linearNode = opGraph.nodes.at(nodeId++);
    linearNode.operation = new atb_speed::common::AclnnAddmm("aclnnAddmm");
    linearNode.inTensorIds = { IN_HIDDENSIZE_ID, INTERMEDIATE_SIZE_ID, IN_BIAS_UP_ID };
    linearNode.outTensorIds = { INTERMEDIATE_DENSE_OUT_ID };
    linearNode.inTensorReshapeFuncs.resize(1);
    linearNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2 表示dim 维度为2
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex] * oldShape.dims[oldShapeDimIndex + 1];
        oldShapeDimIndex += 2; // 2 表示dim 维度为2
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    };

    auto &actNode = opGraph.nodes.at(nodeId++);

    atb::infer::ActivationParam actParam;
    actParam.activationType = param.activationType;

    // 310P
#ifdef Ascend310P
    atb_speed::common::AclNNGeluParam aclNNGeluParam;
    actNode.operation = new atb_speed::common::GeluOperation("actNode", aclNNGeluParam);
#endif

#ifdef Ascend910B
    CREATE_OPERATION(actParam, &actNode.operation);
#endif

    actNode.inTensorIds = { INTERMEDIATE_DENSE_OUT_ID };
    actNode.outTensorIds = { ACTIVATION_OUT_ID };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace bge_large
} // namespace atb_speed
