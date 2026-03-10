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

#include "utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_gelu.h"
#include "aclnnop/aclnn_gelu_v2.h"
#include "aclnn_gelu_operation.h"


namespace atb_speed::common {

    GeluOperation::GeluOperation(
        const std::string &name,
        atb_speed::common::AclNNGeluParam param
    ) : AclNNOperation(name), param_(param), opName_(name)
    {
    }

    GeluOperation::~GeluOperation()
    {
        ATB_LOG(ERROR) <<"GeluOperation deconstruct";
        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @param[in] outTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @return atb::Status
     */
    atb::Status GeluOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const
    {
        ATB_LOG(ERROR) <<opName_ << " InferShape start";
        outTensorDesc.at(0).format = inTensorDesc.at(0).format;
        outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
        outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;

        ATB_LOG(ERROR) <<"Check " << opName_ << " input dimNum=" << inTensorDesc.at(0).shape.dimNum;
        for (uint64_t dim = 0; dim < inTensorDesc.at(0).shape.dimNum; ++dim) {
            ATB_LOG(ERROR) <<"input dim" << dim << " shape=" << inTensorDesc.at(0).shape.dims[dim];
            outTensorDesc.at(0).shape.dims[dim] = inTensorDesc.at(0).shape.dims[dim];
        }

        ATB_LOG(ERROR) <<opName_ << " InferShape end";
        return atb::NO_ERROR;
    }

    uint32_t GeluOperation::GetInputNum() const
    {
        return NUM1;  // inputTensorNum = 1
    }

    uint32_t GeluOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    atb::Status GeluOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
    {
        ATB_LOG(ERROR) <<opName_ << " CreateAclNNVariantPack start";
        int ret;

        ret = CreateAclNNInTensorVariantPack(variantPack);
        if (ret != 0) {
            std::stringstream ss;
            ss << this->opName_ << " AclnnTensor CreateAclNNInTensorVariantPack fail, error: " << ret;
            ATB_LOG(ERROR) << this->opName_ << " AclnnTensor CreateAclNNInTensorVariantPack fail, error: " << ret;
            throw std::runtime_error(ss.str());
        }

        ret = CreateAclNNOutTensorVariantPack(variantPack);
        if (ret != 0) {
            std::stringstream ss;
            ss << this->opName_ << " AclnnTensor CreateAclNNOutTensorVariantPack fail, error: " << ret;
            ATB_LOG(ERROR) << this->opName_ << " AclnnTensor CreateAclNNOutTensorVariantPack fail, error: " << ret;
            throw std::runtime_error(ss.str());
        }

        ATB_LOG(ERROR) <<opName_ << " CreateAclNNVariantPack end";
        return atb::NO_ERROR;
    }

    atb::Status GeluOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
            if (aclnnTensor->tensor == nullptr) {
                ATB_LOG(ERROR) << this->opName_ << " InTensor aclCreateTensor index " << i << " fail";
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclnnVariantPack.aclInTensors[i] = aclnnTensor;
        }
        return atb::NO_ERROR;
    }

    atb::Status GeluOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclNnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclNnVariantPack.aclOutTensors.resize(GetOutputNum());
        for (size_t i = 0; i < aclNnVariantPack.aclOutTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
            if (aclnnTensor->tensor == nullptr) {
                ATB_LOG(ERROR) << this->opName_ << " outTensor aclCreateTensor index " << i << " fail";
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclNnVariantPack.aclOutTensors[i] = aclnnTensor;
        }
        return atb::NO_ERROR;
    }

    std::shared_ptr<AclNNTensor> GeluOperation::CreateTensor(atb::Tensor atbTensor, size_t tensorIdx)
    {
        ATB_LOG(ERROR) <<opName_ << " CreateTensor start";
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = static_cast<int>(tensorIdx);
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = atbTensor;
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(atbTensor);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype,
            aclnnTensor->strides.data(),
            0,
            squeezedAtbTensor.desc.format,
            squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.deviceData);
        ATB_LOG(ERROR) <<opName_ << " CreateTensor end";
        return aclnnTensor;
    }

    int GeluOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_LOG(ERROR) <<
            opName_ << " SetAclNNWorkspaceExecutor start, geluApproximate: " << param_.geluApproximate;
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        if (param_.geluApproximate == -1) {
            int ret = aclnnGeluGetWorkspaceSize(
                aclnnVariantPack.aclInTensors.at(0)->tensor,   // self
                aclnnVariantPack.aclOutTensors.at(0)->tensor,  // out
                &this->aclnnOpCache_->workspaceSize,
                &this->aclnnOpCache_->aclExecutor);
            ATB_LOG(ERROR) <<
                opName_ << " SetAclNNWorkspaceExecutor end"
                        << ", ret: " << ret
                        << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                        << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor;
            return ret;
        } else {
            int ret = aclnnGeluV2GetWorkspaceSize(
                aclnnVariantPack.aclInTensors.at(0)->tensor,   // x
                param_.geluApproximate,                        // approximate
                aclnnVariantPack.aclOutTensors.at(0)->tensor,  // y
                &this->aclnnOpCache_->workspaceSize,
                &this->aclnnOpCache_->aclExecutor);
            ATB_LOG(ERROR) <<
                opName_ << " SetAclNNWorkspaceExecutor end"
                        << ", ret: " << ret
                        << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                        << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor;
            return ret;
        }
    }

    int GeluOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_LOG(ERROR) <<opName_ << " ExecuteAclNNOp start";
        if (param_.geluApproximate == -1) {
            int ret = aclnnGelu(
                workspace,
                this->aclnnOpCache_->workspaceSize,
                this->aclnnOpCache_->aclExecutor,
                stream);
            ATB_LOG(ERROR) <<opName_ << " ExecuteAclNNOp end, ret: " << ret;
            return ret;
        } else {
            int ret = aclnnGeluV2(
                workspace,
                this->aclnnOpCache_->workspaceSize,
                this->aclnnOpCache_->aclExecutor,
                stream);
            ATB_LOG(ERROR) <<opName_ << " ExecuteAclNNOp end, ret: " << ret;
            return ret;
        }
    }

}  // namespace atb_speed::common
