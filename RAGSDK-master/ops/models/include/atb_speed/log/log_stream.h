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
#ifndef ATB_SPEED_LOG_LOGSTREAM_H
#define ATB_SPEED_LOG_LOGSTREAM_H
#include <sstream>
#include <vector>
#include <iostream>
#include "atb_speed/log/log_entity.h"


namespace atb_speed {
    
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (auto& el : vec) {
        os << el << ',';
    }
    return os;
}

class LogStream {
public:
    LogStream(const char *filePath, int line, const char *funcName, LogLevel level);
    ~LogStream();
    friend std::ostream& operator<<(std::ostream& os, const LogStream& obj);
    template <typename T> LogStream &operator << (const T &value)
    {
        stream_ << value;
        return *this;
    }
    void Format(const char *format, ...);

private:
    LogEntity logEntity_;
    std::stringstream stream_;
    bool useStream_ = true;
};
} // namespace atb_speed
#endif
