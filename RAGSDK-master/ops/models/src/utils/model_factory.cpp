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
#include "atb_speed/utils/model_factory.h"
#include "atb_speed/log.h"

namespace atb_speed {
bool ModelFactory::Register(const std::string &modelName, CreateModelFuncPtr createModel)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        ATB_LOG(WARN) << modelName << " model already exists, but the duplication doesn't matter.";
        return false;
    }
    ModelFactory::GetRegistryMap()[modelName] = createModel;
    return true;
}

std::shared_ptr<atb_speed::Model> ModelFactory::CreateInstance(const std::string &modelName, const std::string &param)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        ATB_LOG(INFO) << "find model: " << modelName;
        return it->second(param);
    }
    ATB_LOG(WARN) << "ModelName: " << modelName << " not find in model factory map";
    return nullptr;
}

std::unordered_map<std::string, CreateModelFuncPtr> &ModelFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateModelFuncPtr> modelRegistryMap;
    return modelRegistryMap;
}
} // namespace atb_speed
