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
#include "atb_speed/log/log_stream.h"
#include <atb_speed/log.h>
#include <thread>
#include <iostream>
#include <cstring>
#include <cstdarg>
#include <securec.h>
#include <unistd.h>
#include <syscall.h>
#include "atb_speed/log/log_core.h"

namespace atb_speed {
LogStream::LogStream(const char *filePath, int line, const char *funcName, LogLevel level)
{
    if (filePath == nullptr) {
        std::stringstream ss;
        ss << "filePath of atb_speed::LogStream can't be nullptr, please check" << std::endl;
        throw std::runtime_error(ss.str());
    }
    const char *str = strrchr(filePath, '/');
    if (str) {
        logEntity_.fileName = str + 1;
    } else {
        logEntity_.fileName = filePath;
    }
    logEntity_.time = std::chrono::system_clock::now();
    logEntity_.level = level;
    logEntity_.processId = static_cast<uint32_t>(syscall(SYS_getpid));
    logEntity_.threadId = static_cast<uint32_t>(syscall(SYS_gettid));
    logEntity_.funcName = funcName;
    logEntity_.line = line;
}

void LogStream::Format(const char *format, ...)
{
    useStream_ = false;
    const int maxBufferLenth = 1024;
    std::string content;
    va_list args;
    va_start(args, format);
    char buffer[maxBufferLenth + 1] = {0};
    int ret = vsnprintf_s(buffer, maxBufferLenth, maxBufferLenth, format, args);
    if (ret < 0) {
        ATB_LOG(ERROR) << "vsnprintf_s Error! Error Code:" << ret;
        return;
    }
    va_end(args);
    content.resize(ret + 1);
    va_start(args, format);
    int ref = vsnprintf_s(&content.front(), content.size(), maxBufferLenth, format, args);
    if (ref < 0) {
        ATB_LOG(ERROR) << "vsnprintf_s Error! Error Code:" << ref;
        return;
    }
    va_end(args);
    logEntity_.content = content;
}

LogStream::~LogStream()
{
    if (useStream_) {
        logEntity_.content = stream_.str();
    }
    LogCore::Instance().Log(logEntity_);
}
} // namespace atb