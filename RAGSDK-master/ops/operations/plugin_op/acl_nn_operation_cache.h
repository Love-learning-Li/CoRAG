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
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_LOCAL_CACHE_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_LOCAL_CACHE_H
#include <acl/acl.h>
#include <aclnn/opdev/common_types.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include "acl_nn_tensor.h"
#include "executor_manager.h"

namespace atb_speed {
namespace common {

struct AclNNVariantPack {
    atb::SVector<std::shared_ptr<AclNNTensor>> aclInTensors;
    atb::SVector<std::shared_ptr<AclNNTensor>> aclOutTensors;
    atb::SVector<aclTensorList *> aclInTensorList;
    atb::SVector<aclTensorList *> aclOutTensorList;
};

struct AclNNOpCache {
    AclNNVariantPack aclnnVariantPack;
    aclOpExecutor *aclExecutor = nullptr;
    uint64_t workspaceSize;
    atb::Status UpdateAclNNVariantPack(const atb::VariantPack &variantPack);
    void Destory();
};

} // namespace common
} // namespace atb_speed
#endif