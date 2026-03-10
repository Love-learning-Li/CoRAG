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

#include "operation_creator.h"

#include <functional>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"
#include "atb_speed/utils/operation_factory.h"


namespace atb_speed {

using OperationCreateFunc = std::function<atb::Operation *(const nlohmann::json &paramJson)>;

std::map<std::string, OperationCreateFunc> g_funcMap = {

};

atb::Operation *CreateOperation(const std::string &opName, const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) <<"parse param fail, please check param's format,error: " << e.what() << param;
        return nullptr;
    }

    auto operation = atb_speed::OperationFactory::CreateOperation(opName, paramJson);
    if (operation != nullptr) {
        ATB_LOG(INFO) << "Get Op from the OperationFactory, opName: " << opName;
        return operation;
    }

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ATB_LOG(ERROR) << "not support opName:" << opName;
        return nullptr;
    }

    try {
        return it->second(paramJson);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return nullptr;
}
} /* namespace atb */