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
#include "atb_speed/log/log_sink_stdout.h"
#include <iostream>
#include <iomanip>

namespace atb_speed {
LogSinkStdout::LogSinkStdout(LogLevel level) : LogSink(level) {}
const int MICROSECOND = 1000000;
void LogSinkStdout::LogImpl(const LogEntity &logEntity)
{
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(logEntity.time);
    int us =
        std::chrono::duration_cast<std::chrono::microseconds>(logEntity.time.time_since_epoch()).count() % MICROSECOND;
    std::cout << "[" << std::put_time(std::localtime(&tmpTime), "%F %T") << "." << us << "] [" <<
        LogLevelToString(logEntity.level) << "] [" << logEntity.processId << "] [" << logEntity.threadId << "] [" <<
        logEntity.fileName << ":" << logEntity.line << "]" << logEntity.content << std::endl;
}
} // namespace atb