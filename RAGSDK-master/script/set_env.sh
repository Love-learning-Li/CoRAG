#!/bin/bash
# -------------------------------------------------------------------------
# This file is part of the RAGSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# RAGSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
# Description: SDK uninstallation tool.
# Author: Mind SDK
# Create: 2025
# History: NA

export RAG_SDK_HOME=/home/HwHiAiUser/Ascend
export PYTHONPATH=$RAG_SDK_HOME/ops/transformer_adapter:$PYTHONPATH
export LD_LIBRARY_PATH=$RAG_SDK_HOME/ops/lib/:$LD_LIBRARY_PATH