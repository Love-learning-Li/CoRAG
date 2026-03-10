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
#include "singleton.h"
#include "acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

void AclNNOpCache::Destory()
{
    ATB_LOG(INFO) << "Plugin Op Cache: AclNNOpCache addr [" << (this) << "]destory";
    if (this->aclExecutor == nullptr) {
        return;
    }

    // ExecutorManager中的引用减1
    int count = GetSingleton<ExecutorManager>().DecreaseReference(this->aclExecutor);
    if (count != 0) {
        // 如果executor的引用不为0，则不删除executor及其对应的aclTensor
        return;
    }

    // 如果aclExecutor存在且引用为0，则destory
    ATB_LOG(INFO) << "Plugin Op Cache: destory Executor addr[" << this->aclExecutor << "]";
    aclDestroyAclOpExecutor(this->aclExecutor);
    this->aclExecutor = nullptr;

    // 清空aclTensor
    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensors.size(); ++i) {
        if (this->aclnnVariantPack.aclInTensors[i]->tensorListidx == AclNNTensor::NOT_IN_TENSORLIST) {
            aclDestroyTensor(this->aclnnVariantPack.aclInTensors[i]->tensor);
        }
    }
    this->aclnnVariantPack.aclInTensors.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensors.size(); ++i) {
        if (this->aclnnVariantPack.aclOutTensors[i]->tensorListidx == AclNNTensor::NOT_IN_TENSORLIST) {
            aclDestroyTensor(this->aclnnVariantPack.aclOutTensors[i]->tensor);
        }
    }
    this->aclnnVariantPack.aclOutTensors.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensorList.size(); ++i) {
        aclDestroyTensorList(this->aclnnVariantPack.aclInTensorList[i]);
    }
    this->aclnnVariantPack.aclInTensorList.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensorList.size(); ++i) {
        aclDestroyTensorList(this->aclnnVariantPack.aclOutTensorList[i]);
    }
    this->aclnnVariantPack.aclOutTensorList.clear();
}

atb::Status AclNNOpCache::UpdateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "call UpdateAclNNVariantPack ";
    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensors.size(); ++i) {
        int ret = -1;
        this->aclnnVariantPack.aclInTensors[i]->atbTensor = variantPack.inTensors.at(i);
        if (!this->aclnnVariantPack.aclInTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        if (this->aclnnVariantPack.aclInTensors[i]->tensorListidx == AclNNTensor::NOT_IN_TENSORLIST) {
            ret = aclSetInputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclInTensors[i]->tensor,
                this->aclnnVariantPack.aclInTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicInputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclInTensors[i]->tensorListidx,
                this->aclnnVariantPack.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclInTensorList[this->aclnnVariantPack.aclInTensors[i]->tensorListidx],
                this->aclnnVariantPack.aclInTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << "inTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensors.size(); ++i) {
        int ret = -1;
        this->aclnnVariantPack.aclOutTensors[i]->atbTensor = variantPack.outTensors.at(i);
        if (!this->aclnnVariantPack.aclOutTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        if (this->aclnnVariantPack.aclOutTensors[i]->tensorListidx == AclNNTensor::NOT_IN_TENSORLIST) {
            ret = aclSetOutputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclOutTensors[i]->tensor,
                this->aclnnVariantPack.aclOutTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicOutputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclOutTensors[i]->tensorListidx,
                this->aclnnVariantPack.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclOutTensorList[this->aclnnVariantPack.aclOutTensors[i]->tensorListidx],
                this->aclnnVariantPack.aclOutTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << "outTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed