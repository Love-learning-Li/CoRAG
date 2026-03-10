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
#include "atb_speed/log/log_core.h"
#include <cstdlib>
#include <string>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "atb_speed/log/log_sink_stdout.h"
#include "atb_speed/log/log_sink_file.h"

namespace atb_speed {
static bool GetLogToStdoutFromEnv()
{
    const char *envLogToStdout = std::getenv("ATB_LOG_TO_STDOUT");
    return ((envLogToStdout != nullptr) && (std::strlen(envLogToStdout) == 1) &&
        (strcmp(envLogToStdout, "1") == 0));
}

static bool GetLogToFileFromEnv()
{
    const char *envLogToFile = std::getenv("ATB_LOG_TO_FILE");
    return ((envLogToFile != nullptr) && (std::strlen(envLogToFile) == 1) &&
        (strcmp(envLogToFile, "1") == 0));
}

static LogLevel GetLogLevelFromEnv()
{
    const char *env = std::getenv("ATB_LOG_LEVEL");
    if (env == nullptr || std::strlen(env) > 5) { // 5, 限制环境变量入参长度
        return LogLevel::WARN;
    }
    std::string envLogLevel(env);
    std::transform(envLogLevel.begin(), envLogLevel.end(), envLogLevel.begin(), ::toupper);
    static std::unordered_map<std::string, LogLevel> levelMap{
        { "TRACE", LogLevel::TRACE }, { "DEBUG", LogLevel::DEBUG }, { "INFO", LogLevel::INFO },
        { "WARN", LogLevel::WARN }, { "ERROR", LogLevel::ERROR }, { "FATAL", LogLevel::FATAL }
    };
    auto levelIt = levelMap.find(envLogLevel);
    return levelIt != levelMap.end() ? levelIt->second : LogLevel::WARN;
}

LogCore::LogCore()
{
    level_ = GetLogLevelFromEnv();
    if (GetLogToStdoutFromEnv()) {
        AddSink(std::make_shared<LogSinkStdout>(level_));
    }
    if (GetLogToFileFromEnv()) {
        AddSink(std::make_shared<LogSinkFile>(level_));
    }
    levelCounts_.resize(static_cast<int>(LogLevel::FATAL) + 1);
    for (size_t i = 0; i < levelCounts_.size(); ++i) {
        levelCounts_.at(i) = 0;
    }
}

LogCore &LogCore::Instance()
{
    static LogCore logCore;
    return logCore;
}

LogLevel LogCore::GetLogLevel() const
{
    return level_;
}

void LogCore::SetLogLevel(LogLevel level)
{
    level_ = level;
}

void LogCore::Log(const LogEntity &logEntity)
{
    levelCounts_.at(static_cast<int>(logEntity.level)) += 1;
    for (auto &sink : sinks_) {
        sink->Log(logEntity);
    }
}

void LogCore::AddSink(const std::shared_ptr<LogSink> sink)
{
    sinks_.push_back(sink);
}

const std::vector<std::shared_ptr<LogSink>> &LogCore::GetAllSinks() const
{
    return sinks_;
}

atb::SVector<uint64_t> LogCore::GetLogLevelCount() const
{
    return levelCounts_;
}
} // namespace atb