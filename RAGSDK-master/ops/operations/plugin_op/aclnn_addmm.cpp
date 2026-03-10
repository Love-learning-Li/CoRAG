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
#include "aclnn_addmm.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "utils.h"

#include "aclnnop/aclnn_addmm.h"

namespace atb_speed {
    namespace common {

        AclnnAddmm::AclnnAddmm(const std::string &name) : AclNNOperation(name) {}

        AclnnAddmm::~AclnnAddmm() {}

        atb::Status AclnnAddmm::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                           atb::SVector<atb::TensorDesc> &outTensorDescs) const
        {
            outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
            outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
            outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
            outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
            outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[DIM1];
            return 0;
        }

        uint32_t AclnnAddmm::GetInputNum() const
        {
            return DIM3;
        }

        uint32_t AclnnAddmm::GetOutputNum() const
        {
            return DIM1;
        }

        int AclnnAddmm::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
        {
            ATB_LOG(INFO) << opName_ << " CreateAclNNVariantPack start";
            int ret = 0;
            ret = CreateAclNNInTensorVariantPack(variantPack);
            if (ret != 0) {
                ATB_LOG(ERROR) << this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail";
                return ret;
            }
            ret = CreateAclNNOutTensorVariantPack(variantPack);
            if (ret != 0) {
                ATB_LOG(ERROR) << this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail";
                return ret;
            }
            ATB_LOG(INFO) << opName_ << " CreateAclNNVariantPack end";
            return atb::NO_ERROR;
        }

        int AclnnAddmm::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
        {
            AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
            aclnnVariantPack.aclInTensors.resize(GetInputNum());
            for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
                std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
                aclnnTensor->tensorIdx = static_cast<int>(i);
                aclnnTensor->needUpdateTensorDataPtr = true;
                aclnnTensor->atbTensor = variantPack.inTensors.at(i);
                atb::Tensor squeezedAtbTensor = variantPack.inTensors.at(i);

                aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
                aclnnTensor->tensor = aclCreateTensor(squeezedAtbTensor.desc.shape.dims,
                                                      squeezedAtbTensor.desc.shape.dimNum,
                                                      squeezedAtbTensor.desc.dtype,
                                                      aclnnTensor->strides.data(),
                                                      0,
                                                      squeezedAtbTensor.desc.format,
                                                      squeezedAtbTensor.desc.shape.dims,
                                                      squeezedAtbTensor.desc.shape.dimNum,
                                                      squeezedAtbTensor.deviceData);
                if (aclnnTensor->tensor == nullptr) {
                    ATB_LOG(ERROR) << this->opName_ << " InTensor aclCreateTensor index " << i << " fail";
                    return atb::ERROR_INTERNAL_ERROR;
                }
                aclnnVariantPack.aclInTensors[i] = aclnnTensor;
            }
            return atb::NO_ERROR;
        }

        int AclnnAddmm::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
        {
            AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
            aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
            for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
                std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
                aclnnTensor->tensorIdx = static_cast<int>(i);
                aclnnTensor->needUpdateTensorDataPtr = true;
                aclnnTensor->atbTensor = variantPack.outTensors.at(i);
                atb::Tensor squeezedAtbTensor = variantPack.outTensors.at(i);
                aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
                aclnnTensor->tensor = aclCreateTensor(squeezedAtbTensor.desc.shape.dims,
                                                      squeezedAtbTensor.desc.shape.dimNum,
                                                      squeezedAtbTensor.desc.dtype,
                                                      aclnnTensor->strides.data(),
                                                      0,
                                                      squeezedAtbTensor.desc.format,
                                                      squeezedAtbTensor.desc.shape.dims,
                                                      squeezedAtbTensor.desc.shape.dimNum,
                                                      squeezedAtbTensor.deviceData);
                if (aclnnTensor->tensor == nullptr) {
                    ATB_LOG(ERROR) << this->opName_ << " OutTensor aclCreateTensor index " << i << " fail";
                    return atb::ERROR_INTERNAL_ERROR;
                }
                aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
            }
            return atb::NO_ERROR;
        }

        int AclnnAddmm::SetAclNNWorkspaceExecutor()
        {
            ATB_LOG(INFO) << opName_ << " SetAclNNWorkspaceExecutor start";
            AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
            int ret = aclnnAddmmGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(2)->tensor,
                                                 aclnnVariantPack.aclInTensors.at(0)->tensor,
                                                 aclnnVariantPack.aclInTensors.at(1)->tensor,
                                                 alpha,
                                                 beta,
                                                 aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
                                                 0,
                                                 &this->aclnnOpCache_->workspaceSize,
                                                 &this->aclnnOpCache_->aclExecutor);
            ATB_LOG(INFO) << opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                          << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                          << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor;
            return ret;
        }

        int AclnnAddmm::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
        {
            ATB_LOG(INFO) << opName_ << " aclnnAddmm start";

            int ret = aclnnAddmm(workspace, this->aclnnOpCache_->workspaceSize,
                                 this->aclnnOpCache_->aclExecutor, stream);
            ATB_LOG(INFO) << opName_ << " aclnnAddmm end, ret:" << ret;
            return ret;
        }
    }
}