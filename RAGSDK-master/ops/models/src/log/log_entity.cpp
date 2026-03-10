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
#include "atb_speed/log/log_entity.h"
#include <vector>
#include <string>

namespace atb_speed {
std::string LogLevelToString(LogLevel level)
{
	static std::vector<std::string> levelStrs = { "trace", "debug", "info", "warn", "error", "fatal" };
    size_t levelInt = static_cast<size_t>(level);
    return levelInt < levelStrs.size() ? levelStrs[levelInt] : "unknown";
}
} // namespace atb