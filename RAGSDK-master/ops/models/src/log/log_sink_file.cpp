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
#include "atb_speed/log/log_sink_file.h"
#include <string>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <syscall.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <linux/limits.h>
#include "atb_speed/utils/filesystem.h"

namespace atb_speed {
const int64_t MAX_LOG_FILE_SIZE = 1073741824; // 1G
const size_t MAX_LOG_FILE_COUNT = 5;

LogSinkFile::LogSinkFile(LogLevel level) : LogSink(level)
{
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream curTime;
    curTime << std::put_time(std::localtime(&tmpTime), "%Y%m%d%H%M%S");
    curTime_ = curTime.str();
    if (!FileSystem::Exists(fileDir_)) {
        FileSystem::Makedirs(fileDir_, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    std::stringstream filePath;
    filePath << fileDir_ << std::string("atb_speed_") << std::to_string(syscall(SYS_gettid)) << "_" << curTime_ <<
        "_" << fileCount_ << ".log";

    auto resolvedPath = std::make_unique<char[]>(PATH_MAX);
    std::string filePathString = filePath.str();
    char *resolvedPathPtr = realpath(filePathString.c_str(), resolvedPath.get());
    if (resolvedPathPtr == nullptr) {
        std::cout << "WARNING: Failed to canonicalize log directory: " << filePath.str() << std::endl;
    }

    fileHandle_.open(resolvedPath.get(), std::ios_base::out);
    if (!fileHandle_.is_open()) {
        std::cerr << "Failed to open file: " << filePath.str() << std::endl;
        return;
    }
}

LogSinkFile::~LogSinkFile()
{
    fileHandle_.close();
    fileHandle_.clear();
}

void LogSinkFile::LogImpl(const LogEntity &logEntity)
{
    const int microsecond = 1000000;
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(logEntity.time);
    int us = std::chrono::duration_cast<std::chrono::microseconds>(logEntity.time.time_since_epoch()).count() %
    microsecond;
    std::stringstream content;
    content << "[" << std::put_time(std::localtime(&tmpTime), "%F %T") << "." << us << "] [" <<
        LogLevelToString(logEntity.level) << "] [" << logEntity.processId << "] [" << logEntity.threadId << "] [" <<
        logEntity.fileName << ":" << logEntity.line << "]" << logEntity.content << std::endl;

    fileHandle_ << content.str();
    if (isFlush_) {
        fileHandle_.flush();
    }
    int64_t fileSize = static_cast<int64_t>(fileHandle_.tellp());
    if (fileSize >= MAX_LOG_FILE_SIZE) {
        fileHandle_.close();
        fileCount_++;
        if (fileCount_ == MAX_LOG_FILE_COUNT) {
            std::cout << "WARNING: Log file has rolled over. Old logs are being overwritten." << std::endl;
            fileCount_ = 0;
        }

        if (!FileSystem::IsPathValid(fileDir_)) {
            std::cerr << "path:" << fileDir_ << " is invalid";
            return;
        }
        std::stringstream filePath;
        filePath << fileDir_ << std::string("atb_speed_") << std::to_string(syscall(SYS_gettid)) << "_" << curTime_ <<
            "_" <<fileCount_ << ".log";

        auto resolvedPath = std::make_unique<char[]>(PATH_MAX);
        std::string filePathString = filePath.str();
        char *resolvedPathPtr = realpath(filePathString.c_str(), resolvedPath.get());
        if (resolvedPathPtr == nullptr) {
            std::cout << "WARNING: Failed to canonicalize log directory: " << filePath.str() << std::endl;
        }

        fileHandle_.open(resolvedPath.get(), std::ios_base::out);
        if (!fileHandle_.is_open()) {
            std::cerr << "Failed to open file: " << filePath.str() << std::endl;
            return;
        }
    }
}
} // namespace AsdOps