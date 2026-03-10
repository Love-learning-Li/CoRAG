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
#ifndef ATB_SPEED_UTILS_WORKSPACE_H
#define ATB_SPEED_UTILS_WORKSPACE_H
#include <cstdint>
#include <memory>
#include <vector>
#include "buffer_base.h"

namespace atb_speed {

class Workspace {
public:
    Workspace();
    ~Workspace();
    void *GetWorkspaceBuffer(uint64_t bufferSize);

private:
    uint64_t GetWorkspaceBufferRing() const;
    uint64_t GetWorkspaceBufferSize() const;

private:
    std::vector<std::unique_ptr<BufferBase>> workspaceBuffers_;
    size_t workspaceBufferOffset_ = 0;
};
} // namespace atb_speed
#endif