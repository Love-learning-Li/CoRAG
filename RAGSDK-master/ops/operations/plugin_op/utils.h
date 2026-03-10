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

#ifndef ATB_SPEED_PLUGIN_UTILS_H
#define ATB_SPEED_PLUGIN_UTILS_H
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;
const int NUM5 = 5;

atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims);

atb::SVector<int64_t> GetTransposeTensorStride(atb::Dims &tensorDims);

bool Is910B();

atb::Tensor SqueezeBatchSeq(atb::Tensor atbTensor);

bool isVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const atb::VariantPack &atbVariantPack);

} // namespace common
} // namespace atb_speed
#endif