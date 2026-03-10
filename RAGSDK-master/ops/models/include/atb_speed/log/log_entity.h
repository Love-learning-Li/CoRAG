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
#ifndef ATB_SPEED_LOG_LOGENTITY_H
#define ATB_SPEED_LOG_LOGENTITY_H
#include <chrono>
#include <string>

namespace atb_speed {
enum class LogLevel {
    TRACE = 0,
	DEBUG,
	INFO,
	WARN,
	ERROR,
	FATAL
};

std::string LogLevelToString(LogLevel level);

struct LogEntity {
    std::chrono::system_clock::time_point time;
    size_t processId = 0;
    size_t threadId = 0;
    LogLevel level = LogLevel::TRACE;
    const char *fileName = nullptr;
    int line = 0;
    const char *funcName = nullptr;
    std::string content;
};
} // namespace atb_speed
#endif