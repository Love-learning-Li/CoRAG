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
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_GLOBAL_CACHE_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_GLOBAL_CACHE_H

#include <vector>
#include <string>
#include <map>
#include <atb/atb_infer.h>
#include "acl_nn_operation_cache.h"
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

const uint16_t DEFAULT_ACLNN_GLOBAL_CACHE_SIZE = 16;
constexpr int32_t DECIMAL = 10;

class AclNNGlobalCache {
public:
    explicit AclNNGlobalCache();
    std::shared_ptr<AclNNOpCache> GetGlobalCache(std::string opName, atb::VariantPack variantPack);
    atb::Status UpdateGlobalCache(std::string opName, std::shared_ptr<AclNNOpCache> cache);
    std::string PrintGlobalCache();

private:
    int nextUpdateIndex_ = 0;
    uint16_t globalCacheCountMax_ = 16;
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>> aclnnGlobalCache_;
};

} // namespace common
} // namespace atb_speed
#endif