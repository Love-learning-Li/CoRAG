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
#ifndef ATB_SPEED_UTILS_CONFIG_H
#define ATB_SPEED_UTILS_CONFIG_H
#include <string>
#include <set>

namespace atb_speed {
class Config {
public:
    Config();
    ~Config();
    static std::string GetSaveTensorDir();
    bool IsSaveTensor() const;
    void DisableSaveTensor();
    uint64_t GetSaveTensorMaxNum() const;
    bool IsConvertNCHWToND() const;
    bool IsSaveTensorForRunner(const std::string &runnerName) const;
    bool IsTorchTensorFormatCast() const;
    bool IsUseTilingCopyStream() const;
    bool IsLayerInternalTensorReuse() const;

private:
    static bool IsEnable(bool enable = false);
    void InitSaveTensor() const;

private:
    bool isSaveTensor_ = false;
    uint64_t saveTensorMaxNum_ = 1;
    bool isConvertNCHWToND_ = false;
    bool isTorchTensorFormatCast_ = true;
    bool isUseTilingCopyStream_ = false;
    std::set<std::string> saveTensorRunnerNameSet_;
    bool isLayerInternalTensorReuse_ = false;
};
} // namespace atb_speed
#endif