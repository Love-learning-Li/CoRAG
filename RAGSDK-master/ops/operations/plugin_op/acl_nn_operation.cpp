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
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "atb_speed/utils/statistic.h"
#include "utils.h"
#include "singleton.h"
#include "executor_manager.h"
#include "acl_nn_global_cache.h"
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

AclNNOperation::AclNNOperation(const std::string &opName) : opName_(opName)
{
    this->aclnnOpCache_ = std::make_shared<AclNNOpCache>();
}

AclNNOperation::~AclNNOperation()
{
    ATB_LOG(INFO) << "AclNNOperation deconstructor";
    this->DestroyOperation();
}

std::string AclNNOperation::GetName() const { return this->opName_; }

void AclNNOperation::DestroyOperation() const
{
    this->aclnnOpCache_->Destory();
}

atb::Status AclNNOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context)
{
    ATB_LOG(INFO) << this->opName_ << " setup start";

    // 1. 检查Context是否为空
    if (context == nullptr) {
        ATB_LOG(ERROR) << this->opName_ << " setup context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    // 2. 获取Executor和Workspace
    int ret = UpdateAclNNOpCache(variantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << this->opName_ << " call UpdateAclNNOpCache, error:" << ret;
        this->aclnnOpCache_->Destory();
        return ret;
    }

    // 3. 更新传入的workspaceSize
    workspaceSize = this->aclnnOpCache_->workspaceSize;

    ATB_LOG(INFO) << "setup end";
    ATB_LOG(INFO) << GetSingleton<AclNNGlobalCache>().PrintGlobalCache();
    ATB_LOG(INFO) << GetSingleton<ExecutorManager>().PrintExecutorCount();
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::UpdateAclNNOpCache(const atb::VariantPack &variantPack)
{
    // 此方法会准备好Execute时所需的Executor和workspace
    // 前提条件：GlobalCache中的executor要保证LocalCache里面一定也要有引用；仅对LocalCache进行释放

    // 1. 查看Local Cache中Executor是否可以复用
    ATB_LOG(INFO) << "Plugin Op Cache: Local Cache call isVariankPackEqual";
    if (isVariankPackEqual(this->aclnnOpCache_->aclnnVariantPack, variantPack)) {
        // Local Cache命中
        ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" << (this)
                      << "] Cache addr[" << this->aclnnOpCache_.get() << "] Executor addr["
                      << this->aclnnOpCache_->aclExecutor << "] Local Cache Hit";
        return atb::NO_ERROR;
    }

    // 2. 查看Global Cache中Executor是否可以复用
    std::shared_ptr<AclNNOpCache> globalCache = \
        GetSingleton<AclNNGlobalCache>().GetGlobalCache(this->opName_, variantPack);
    if (globalCache != nullptr) {
        // Global Cache命中
        ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" << (this) << "] Cache addr["
                      << globalCache.get() << "] Executor addr[" << globalCache->aclExecutor << "] Global Cache Hit";
        // 2.1 释放旧的Local Cache
        ATB_LOG(INFO) << "Plugin Op Cache: destory local cache before switching to global cache";
        this->aclnnOpCache_->Destory();
        // 2.2 更新Local Cache
        this->aclnnOpCache_ = globalCache;
        // 2.3 更新ExecutorManager
        int count = GetSingleton<ExecutorManager>().IncreaseReference(this->aclnnOpCache_->aclExecutor);
        ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << this->opName_ << "] Executor addr["
                      << this->aclnnOpCache_->aclExecutor << "] count update to " << count;
        return atb::NO_ERROR;
    }

    // 3. Local Cache和Global Cache都未命中
    // 3.1 释放Local Cache
    ATB_LOG(INFO) << "Plugin Op Cache: destory local cache before create a new one";
    this->aclnnOpCache_->Destory();
    // 3.2 根据variantPack，更新aclnnOpCache_，获取WorkSpace和Executor
    this->aclnnOpCache_ = std::make_shared<AclNNOpCache>();
    int ret = CreateAclNNOpCache(variantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << this->opName_ << " call CreateAclNNOpCache fail, error:" << ret;
        return ret;
    }
    ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << this->opName_ << "] Op addr["
                  << (this) << "] Cache addr[" << this->aclnnOpCache_.get() << "] Executor addr["
                  << this->aclnnOpCache_->aclExecutor << "] create Local Cache";
    // 3.3 更新ExecutorManager，新增Executor，count为1
    int count = GetSingleton<ExecutorManager>().IncreaseReference(this->aclnnOpCache_->aclExecutor);
    ATB_LOG(INFO) << "Plugin Op Cache: Op name[" << this->opName_ << "] increase Executor addr["
                  << this->aclnnOpCache_->aclExecutor << "] count update to " << count;
    // 3.4 更新Global Cache（旧的Global Cache直接替换指针就行）
    GetSingleton<AclNNGlobalCache>().UpdateGlobalCache(this->opName_, this->aclnnOpCache_);

    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNOpCache(const atb::VariantPack &variantPack)
{
    int ret = CreateAclNNVariantPack(variantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << this->opName_ << " call CreateAclNNVariantPack fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
    }

    ret = SetAclNNWorkspaceExecutor();
    if (ret != 0) {
        ATB_LOG(ERROR) << this->opName_ << " call SetAclNNWorkspaceExecutor fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
    }

    ret = aclSetAclOpExecutorRepeatable(this->aclnnOpCache_->aclExecutor);
    if (ret != 0) {
        ATB_LOG(ERROR) << this->opName_ << " call aclSetAclOpExecutorRepeatable fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
    }

    ATB_LOG(INFO) << "Plugin Op Cache: create Executor addr[" << this->aclnnOpCache_->aclExecutor << "]";

    return atb::NO_ERROR;
}

atb::Status AclNNOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                    atb::Context *context)
{
    ATB_LOG(INFO) << this->opName_ << " execute start";
    if (!context) {
        ATB_LOG(ERROR) << this->opName_ << " execute fail, context param is null";
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtStream stream = context->GetExecuteStream();
    if (!stream) {
        ATB_LOG(ERROR) << this->opName_ << " execute fail, execute stream in context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    // 更新数据传入的地址
    int ret = this->aclnnOpCache_->UpdateAclNNVariantPack(variantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << this->opName_ << " call UpdateAclNNVariantPack fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
    }

    ATB_LOG(INFO) << "Input workspaceSize " << workspaceSize << " localCache workspaceSize "
                  << this->aclnnOpCache_->workspaceSize;
    ret = ExecuteAclNNOp(workspace, stream);
    if (ret != 0) {
        ATB_LOG(ERROR) << this->opName_ << " call ExecuteAclNNOp fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
    }

    ATB_LOG(INFO) << this->opName_ << " execute end";

    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed
