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
#ifndef ATB_SPEED_LOG_LOGSINK_H
#define ATB_SPEED_LOG_LOGSINK_H
#include "atb_speed/log/log_entity.h"

namespace atb_speed {
class LogSink {
public:
    explicit LogSink(LogLevel level = LogLevel::INFO);
    virtual ~LogSink() = default;
    void Log(const LogEntity &logEntity);

private:
    virtual void LogImpl(const LogEntity &logEntity) = 0;
    LogLevel level_;
};
} // namespace atb_speed
#endif