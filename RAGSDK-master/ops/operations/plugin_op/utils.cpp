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
#include <cstring>
#include "atb_speed/log.h"
#include "utils.h"

namespace atb_speed {
namespace common {

atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims)
{
    atb::SVector<int64_t> tmpStrides(tensorDims.dimNum, 1);
    for (int64_t i = static_cast<int64_t>(tensorDims.dimNum) - 2; i >= 0; i--) {
        tmpStrides[i] = tensorDims.dims[i + 1] * tmpStrides[i + 1];
    }
    return tmpStrides;
}

atb::SVector<int64_t> GetTransposeTensorStride(atb::Dims &tensorDims)
{
    atb::SVector<int64_t> tmptransposeStrides(tensorDims.dimNum, 1);
    tmptransposeStrides[0] = 1;
    tmptransposeStrides[1] = tensorDims.dims[1];
    return tmptransposeStrides;
}

bool Is910B()
{
    return std::string(aclrtGetSocName()).find("Ascend910B") != std::string::npos;
}

atb::Tensor SqueezeBatchSeq(atb::Tensor atbTensor)
{
    if (atbTensor.desc.shape.dimNum == DIM3) {
        atbTensor.desc.shape.dimNum = DIM2;
        atbTensor.desc.shape.dims[DIM0] = atbTensor.desc.shape.dims[DIM0] * atbTensor.desc.shape.dims[DIM1];
        atbTensor.desc.shape.dims[DIM1] = atbTensor.desc.shape.dims[DIM2];
    }
    return atbTensor;
}

std::string PrintAclNNVariankPack(const AclNNVariantPack &aclnnVariantPack)
{
    std::stringstream ss;
    ss << "Plugin Op Cache: AclNNVariantPack ";
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
        const atb::TensorDesc &tensorDesc = aclnnVariantPack.aclInTensors[i]->atbTensor.desc;
        ss << "index " << i << " dtype " << tensorDesc.dtype
           << " format " << tensorDesc.format << " dimNum " << tensorDesc.shape.dimNum;
        for (uint64_t j = 0; j < tensorDesc.shape.dimNum; j++) {
            ss << "dim[" << j << "]=" << tensorDesc.shape.dims[j] << " ";
        }
    }
    return ss.str();
}

std::string PrintATBVariankPack(const atb::VariantPack &atbVariantPack)
{
    std::stringstream ss;
    ss << "Plugin Op Cache: ATBVariantPack ";
    for (size_t i = 0; i < atbVariantPack.inTensors.size(); i++) {
        const atb::TensorDesc &tensorDesc = atbVariantPack.inTensors[i].desc;
        ss << "index " << i << " dtype " << tensorDesc.dtype
           << " format " << tensorDesc.format << " dimNum " << tensorDesc.shape.dimNum;
        for (uint64_t j = 0; j < tensorDesc.shape.dimNum; j++) {
            ss << "dim[" << j << "]=" << tensorDesc.shape.dims[j] << " ";
        }
    }
    return ss.str();
}

bool isHostDataEqual(const atb::Tensor &tensorA, const atb::Tensor &tensorB, int tensorIdx)
{
    if (tensorA.hostData != nullptr && tensorB.hostData == nullptr) {
        ATB_LOG(INFO) << "Plugin Op Cache: tensor index " << tensorIdx
                        << " aclnnVariantPack hostData is not null but atbVariantPack hostData is";
        return false;
    }
    if (tensorA.hostData == nullptr && tensorB.hostData != nullptr) {
        ATB_LOG(INFO) << "Plugin Op Cache: tensor index " << tensorIdx
                        << " aclnnVariantPack hostData is null but atbVariantPack hostData is not";
        return false;
    }
    if (tensorA.hostData != nullptr && tensorB.hostData != nullptr) {
        if (tensorA.dataSize != tensorB.dataSize) {
            ATB_LOG(INFO) << "Plugin Op Cache: tensor index " << tensorIdx << " dataSize not equal";
            return false;
        }
        if (memcmp(tensorA.hostData, tensorB.hostData, tensorA.dataSize) != 0) {
            ATB_LOG(INFO) << "Plugin Op Cache: tensor index " << tensorIdx << " hostData not equal";
            return false;
        }
    }
    return true;
}

bool isTensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB, int tensorIdx)
{
    if (tensorDescA.dtype != tensorDescB.dtype) {
        ATB_LOG(INFO) << "Plugin Op Cache: tensor index " << tensorIdx
                        << " dtype not equal, aclnnVariantPack dtype " << tensorDescA.dtype
                        << " atbVariantPack dtype " << tensorDescB.dtype;
        return false;
    }
    if (tensorDescA.format != tensorDescB.format) {
        ATB_LOG(INFO) << "Plugin Op Cache: tensor index " << tensorIdx
                        << " format not equal, aclnnVariantPack format " << tensorDescA.format
                        << " atbVariantPack format " << tensorDescB.format;
        return false;
    }
    if (tensorDescA.shape.dimNum != tensorDescB.shape.dimNum) {
        ATB_LOG(INFO) << "Plugin Op Cache: tensor index " << tensorIdx
                        << " dimNum not equal, aclnnVariantPack dimNum " << tensorDescA.shape.dimNum
                        << " atbVariantPack dimNum " << tensorDescB.shape.dimNum;
        return false;
    }
    for (uint64_t j = 0; j < tensorDescA.shape.dimNum; j++) {
        if (tensorDescA.shape.dims[j] != tensorDescB.shape.dims[j]) {
            ATB_LOG(INFO) << "Plugin Op Cache: : tensor index " << tensorIdx
                            << " shape.dims " << j << " not equal, aclnnVariantPack value "
                            << tensorDescA.shape.dims[j] << " atbVariantPack value " << tensorDescB.shape.dims[j];
            return false;
        }
    }
    return true;
}

bool isVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const atb::VariantPack &atbVariantPack)
{
    ATB_LOG(INFO) << PrintAclNNVariankPack(aclnnVariantPack);
    ATB_LOG(INFO) << PrintATBVariankPack(atbVariantPack);

    // 判断InTensor数量是否一致
    if (aclnnVariantPack.aclInTensors.size() != atbVariantPack.inTensors.size()) {
        ATB_LOG(INFO) << "Plugin Op Cache: size not equal, aclnnVariantPack size "
                      << aclnnVariantPack.aclInTensors.size() << " atbVariantPack size "
                      << atbVariantPack.inTensors.size();
        return false;
    }

    // 判断每个InTensor的dtype，format，shape和host_data是否一致
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
        const atb::Tensor &tensorA = aclnnVariantPack.aclInTensors[i]->atbTensor;
        const atb::Tensor &tensorB = atbVariantPack.inTensors[i];

        if (!isHostDataEqual(tensorA, tensorB, i)) {
            return false;
        }

        if (!isTensorDescEqual(tensorA.desc, tensorB.desc, i)) {
            return false;
        }
    }

    ATB_LOG(INFO) << "Plugin Op Cache: TensorDesc match";
    return true;
}

} // namespace common
} // namespace atb_speed