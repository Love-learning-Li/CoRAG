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

import ssl

from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, STR_TYPE_CHECK_TIP, MB


class ClientParam:
    @validate_params(
        use_http=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        ca_file=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        crl_file=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        timeout=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 600,
                     message="param must be int and value range (0, 600]"),
        response_limit_size=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 10 * MB,
                                 message="param must be int and value range (0, 10MB]"),
    )
    def __init__(self,
                 use_http: bool = False,
                 ca_file: str = "",
                 crl_file: str = "",
                 timeout: int = 60,
                 response_limit_size: int = MB):
        self.use_http = use_http
        self.ca_file = ca_file
        self.crl_file = crl_file
        self.timeout: int = timeout
        self.response_limit_size: int = response_limit_size
