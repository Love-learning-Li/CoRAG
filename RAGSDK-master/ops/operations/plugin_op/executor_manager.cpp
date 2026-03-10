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
#include "atb_speed/log.h"
#include "executor_manager.h"

namespace atb_speed {
namespace common {


int ExecutorManager::IncreaseReference(aclOpExecutor *executor)
{
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it == this->executorCount_.end()) {
        ATB_LOG(INFO) << "Plugin Op Cache: Executor addr[" << executor << "] not found in ExecutorManager, add one";
        this->executorCount_[executor] = 1;
        return 1;
    }

    int &count = it->second;
    count += 1;
    ATB_LOG(INFO) << "Plugin Op Cache: ExecutorManager Executor addr["
                  << executor << "] increase reference to " << count;
    return count;
}

int ExecutorManager::DecreaseReference(aclOpExecutor *executor)
{
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it == this->executorCount_.end()) {
        ATB_LOG(ERROR) << "Plugin Op Cache: Executor addr[" << executor << "] not found in ExecutorManager";
        return 0;
    }
    int &count = it->second;
    if (count == 1) {
        ATB_LOG(INFO) << "Plugin Op Cache: delete Executor addr[" << executor << "]";
        this->executorCount_.erase(executor);
        return 0;
    }

    count -= 1;
    ATB_LOG(INFO) << "Plugin Op Cache: ExecutorManager Executor addr["
                  << executor << "] decrease reference to " << count;
    return count;
}

int ExecutorManager::GetReference(aclOpExecutor *executor)
{
    ATB_LOG(INFO) << "Plugin Op Cache: ExecutorManager Executor addr[" << executor << "] get reference";
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it == this->executorCount_.end()) {
        ATB_LOG(ERROR) << "Plugin Op Cache: Executor addr[" << executor << "] not found in ExecutorManager";
        return 0;
    }
    return it->second;
}

std::string ExecutorManager::PrintExecutorCount()
{
    std::stringstream ss;
    ss << "Plugin Op Cache: Executor Summary ";
    std::map<aclOpExecutor *, int>::iterator it;
    for (it = this->executorCount_.begin(); it != this->executorCount_.end(); it++) {
        ss << "Executor Addr[" << it->first << "] count " << it->second << " ";
    }
    return ss.str();
}

} // namespace common
} // namespace atb_speed
