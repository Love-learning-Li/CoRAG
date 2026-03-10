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
#include <sstream>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "utils.h"
#include "acl_nn_global_cache.h"

namespace atb_speed {
namespace common {

AclNNGlobalCache::AclNNGlobalCache()
{
    this->globalCacheCountMax_ =  DEFAULT_ACLNN_GLOBAL_CACHE_SIZE;
}

std::shared_ptr<AclNNOpCache> AclNNGlobalCache::GetGlobalCache(std::string opName, atb::VariantPack variantPack)
{
    // 获取Op对应的Global Cache列表
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it = \
        this->aclnnGlobalCache_.find(opName);
    if (it == this->aclnnGlobalCache_.end()) {
        ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << opName << "] not found in AclNNGlobalCache";
        return nullptr;
    }
    std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;

    // 在Global Cache列表中基于variantPack找到匹配的Cache
    for (size_t i = 0; i < opGlobalCacheList.size(); i++) {
        ATB_LOG(INFO) << "Plugin Op Cache: Global Cache index " << i << " call isVariankPackEqual";
        if (isVariankPackEqual(opGlobalCacheList[i]->aclnnVariantPack, variantPack)) {
            // Global Cache命中
            return opGlobalCacheList[i];
        }
    }

    return nullptr;
}

atb::Status AclNNGlobalCache::UpdateGlobalCache(std::string opName, std::shared_ptr<AclNNOpCache> cache)
{
    // 获取Op对应的Global Cache列表
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it = \
        this->aclnnGlobalCache_.find(opName);
    if (it == this->aclnnGlobalCache_.end()) {
        // 不存在opName对应的Cache列表
        ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << opName << "] not found in AclNNGlobalCache, add one";
        this->aclnnGlobalCache_[opName] = {cache};
        return atb::NO_ERROR;
    }
    std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;

    // Cache未已满
    if (opGlobalCacheList.size() < this->globalCacheCountMax_) {
        ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << opName << "] global cache is not full, add one";
        opGlobalCacheList.push_back(cache);
        return atb::NO_ERROR;
    }

    // Cache已满
    ATB_LOG(INFO) << "Plugin Op Cache: Op name["
                  << opName << "] global cache is full, update index " << nextUpdateIndex_;
    opGlobalCacheList[nextUpdateIndex_] = cache;
    CHECK_PARAM_GT(globalCacheCountMax_, 0);
    nextUpdateIndex_ = (nextUpdateIndex_ + 1) % globalCacheCountMax_;
    return atb::NO_ERROR;
}

std::string AclNNGlobalCache::PrintGlobalCache()
{
    std::stringstream ss;
    ss << "Plugin Op Cache: Global Cache Summary ";
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it;
    for (it = this->aclnnGlobalCache_.begin(); it != this->aclnnGlobalCache_.end(); it++) {
        ss << "Op name[" << it->first << "] ";
        std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;
        for (size_t i = 0; i < opGlobalCacheList.size(); i++) {
            ss << "Cache Addr[" << opGlobalCacheList[i].get() << "] ";
        }
    }
    return ss.str();
}

} // namespace common
} // namespace atb_speed
