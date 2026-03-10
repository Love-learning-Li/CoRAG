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
#ifndef ATB_SPEED_LOG_LOGCORE_H
#define ATB_SPEED_LOG_LOGCORE_H
#include <memory>
#include <vector>
#include "atb_speed/log/log_entity.h"
#include "atb_speed/log/log_sink.h"
#include "atb/svector.h"

namespace atb_speed {
class LogCore {
public:
	LogCore();
    ~LogCore() = default;
    static LogCore &Instance();
    LogLevel GetLogLevel() const;
    void SetLogLevel(LogLevel level);
    void Log(const LogEntity &logEntity);
    void AddSink(const std::shared_ptr<LogSink> sink);
    const std::vector<std::shared_ptr<LogSink>> &GetAllSinks() const;
    atb::SVector<uint64_t> GetLogLevelCount() const;

private:
	std::vector<std::shared_ptr<LogSink>> sinks_;
    LogLevel level_ = LogLevel::INFO;
    atb::SVector<uint64_t> levelCounts_;
};
} // namespace atb_speed
#endif
