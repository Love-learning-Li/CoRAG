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
#ifndef ATB_SPEED_LOG_SINKFILE_H
#define ATB_SPEED_LOG_SINKFILE_H

#include <fstream>
#include "atb_speed/log/log_sink.h"

namespace atb_speed {
class LogSinkFile : public LogSink {
public:
	explicit LogSinkFile(LogLevel level);
    ~LogSinkFile() override;

private:
	void LogImpl(const LogEntity &logEntity) override;

private:
	std::ofstream fileHandle_;
    int32_t fileCount_ = 0;
    bool isFlush_ = false;
    std::string curTime_;
    std::string fileDir_ = "atb_temp/log/";
};
} // namespace atb_speed
#endif
