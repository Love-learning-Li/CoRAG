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

#include "buffer_device.h"
#include <acl/acl.h>
#include <atb_speed/utils/timer.h>
#include <atb/types.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/statistic.h"
#include "adapter/utils/utils.h"

namespace atb_speed {
constexpr int KB_1 = 1024;
constexpr int MB_1 = 1024 * 1024;
constexpr int GB_1 = 1024 * 1024 * 1024;
constexpr int DIM_NUM_2 = 2;
BufferDevice::BufferDevice(uint64_t bufferSize) : bufferSize_(bufferSize)
{
    ATB_LOG(INFO) << "BufferDevice::BufferDevice called, bufferSize:" << bufferSize;
    bufferSize_ = bufferSize;
    if (bufferSize_ > 0) {
        ATB_LOG(FATAL) << "BufferDevice::GetBuffer bufferSize:" << bufferSize_;
        atTensor_ = CreateAtTensor(bufferSize_);
        buffer_ = atTensor_.data_ptr();
    }
}

BufferDevice::~BufferDevice() {}

void *BufferDevice::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= bufferSize_) {
        ATB_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize << "<= bufferSize_:" << bufferSize_
                        << ", not new device mem.";
        return atTensor_.data_ptr();
    }
   
    if (aclrtSynchronizeDevice() != 0) {
        return nullptr;
    }

    atTensor_.reset();
    atTensor_ = CreateAtTensor(bufferSize);
    bufferSize_ = uint64_t(atTensor_.numel());
    ATB_LOG(INFO) << "BufferDevice::GetBuffer new bufferSize:" << bufferSize;
    buffer_ = atTensor_.data_ptr();
    return atTensor_.data_ptr();
}

torch::Tensor BufferDevice::CreateAtTensor(const uint64_t bufferSize) const
{
    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;

    tensorDesc.shape.dimNum = DIM_NUM_2;
    tensorDesc.shape.dims[0] = KB_1;
    tensorDesc.shape.dims[1] = bufferSize / KB_1 + int(1);

    return Utils::CreateAtTensorFromTensorDesc(tensorDesc);
}
} // namespace atb_speed
