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

#ifndef ATB_LOG_H
#define ATB_LOG_H
#include "atb_speed/log/log_stream.h"
#include "atb_speed/log/log_core.h"
#include "atb_speed/log/log_sink.h"
#include "atb_speed/log/log_entity.h"

#define ATB_LOG(level) ATB_LOG_##level

#define ATB_FLOG(level, format, ...) ATB_FLOG_##level(format, __VA_ARGS__)

#define ATB_LOG_IF(condition, level) \
    if (condition)                   \
    ATB_LOG(level)

#define ATB_LOG_TRACE                                                   \
    if (atb_speed::LogLevel::TRACE >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::TRACE)
#define ATB_LOG_DEBUG                                                   \
    if (atb_speed::LogLevel::DEBUG >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::DEBUG)
#define ATB_LOG_INFO                                                   \
    if (atb_speed::LogLevel::INFO >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::INFO)
#define ATB_LOG_WARN                                                   \
    if (atb_speed::LogLevel::WARN >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::WARN)
#define ATB_LOG_ERROR                                                   \
    if (atb_speed::LogLevel::ERROR >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::ERROR)
#define ATB_LOG_FATAL                                                   \
    if (atb_speed::LogLevel::FATAL >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::FATAL)

#define ATB_FLOG_TRACE(format, ...)                                     \
    if (atb_speed::LogLevel::TRACE >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::TRACE).Format(format, __VA_ARGS__)
#define ATB_FLOG_DEBUG(format, ...)                                     \
    if (atb_speed::LogLevel::DEBUG >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::DEBUG).Format(format, __VA_ARGS__)
#define ATB_FLOG_INFO(format, ...)                                     \
    if (atb_speed::LogLevel::INFO >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::INFO).Format(format, __VA_ARGS__)
#define ATB_FLOG_WARN(format, ...)                                     \
    if (atb_speed::LogLevel::WARN >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::WARN).Format(format, __VA_ARGS__)
#define ATB_FLOG_ERROR(format, ...)                                     \
    if (atb_speed::LogLevel::ERROR >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::ERROR).Format(format, __VA_ARGS__)
#define ATB_FLOG_FATAL(format, ...)                                     \
    if (atb_speed::LogLevel::FATAL >= atb_speed::LogCore::Instance().GetLogLevel()) \
    atb_speed::LogStream(__FILE__, __LINE__, __FUNCTION__, atb_speed::LogLevel::FATAL).Format(format, __VA_ARGS__)
#endif