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
#include "atb_speed/utils/operation_factory.h"
#include "atb_speed/log.h"

namespace atb_speed {
bool OperationFactory::Register(const std::string &operationName, CreateOperationFuncPtr createOperation)
{
    auto it = OperationFactory::GetRegistryMap().find(operationName);
    if (it != OperationFactory::GetRegistryMap().end()) {
        ATB_LOG(WARN) << operationName << " operation already exists, but the duplication doesn't matter.";
        return false;
    }
    OperationFactory::GetRegistryMap()[operationName] = createOperation;
    return true;
}

atb::Operation *OperationFactory::CreateOperation(const std::string &operationName, const nlohmann::json &param)
{
    auto it = OperationFactory::GetRegistryMap().find(operationName);
    if (it != OperationFactory::GetRegistryMap().end()) {
        ATB_LOG(INFO) << "find operation: " << operationName;
        return it->second(param);
    }
    ATB_LOG(WARN) << "OperationName: " << operationName << " not find in operation factory map";
    return nullptr;
}

std::unordered_map<std::string, CreateOperationFuncPtr> &OperationFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateOperationFuncPtr> operationRegistryMap;
    return operationRegistryMap;
}
} // namespace atb_speed
