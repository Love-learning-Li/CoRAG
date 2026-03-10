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
#ifndef MXRAGEMBFRAMEWORK_ACLNN_ADDMM_H
#define MXRAGEMBFRAMEWORK_ACLNN_ADDMM_H
#include "acl_nn_operation.h"

namespace atb_speed {
    namespace common {
        class AclnnAddmm : public AclNNOperation {
        public:
            explicit AclnnAddmm(const std::string &name);

            ~AclnnAddmm() override;

            atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                   atb::SVector<atb::TensorDesc> &outTensorDescs) const override;

            uint32_t GetInputNum() const override;

            uint32_t GetOutputNum() const override;

        private:
            int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
            int SetAclNNWorkspaceExecutor() override;
            int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

            int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack);
            int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack);

            bool alphaValue = true;
            bool betaValue = true;
            aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_BOOL);
            aclScalar* beta = aclCreateScalar(&betaValue, ACL_BOOL);
        };
    }
}


#endif // MXRAGEMBFRAMEWORK_ACLNN_ADDMM_H
