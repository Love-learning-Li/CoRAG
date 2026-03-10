#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os
import sys

from mx_rag.version import __version__

sys.tracebacklimit = 0

# 默认关闭ragas的track
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
# 默认HF_HUB离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
# 默认datasets离线模式
os.environ["HF_DATASETS_OFFLINE"] = "1"
# 默认不自动下载nltk资源
os.environ["AUTO_DOWNLOAD_NLTK"] = "false"
