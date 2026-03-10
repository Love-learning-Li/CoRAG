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

#ifndef ATB_SPEED_LAYERS_FEED_FORWARD_H
#define ATB_SPEED_LAYERS_FEED_FORWARD_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace bge_large {
struct FeedForwardParam {
    void *hcclComm = nullptr;
    atb::infer::ActivationType activationType;
    int64_t geluApproximate = -1;
    bool transposeB = true;
    bool isBias = true;
    std::string backend = "hccl";
    bool isBF16 = false;
};

atb::Status FeedForwardLayer(const FeedForwardParam &param, atb::Operation **operation);
} // namespace bge_large
} // namespace atb_speed
#endif
