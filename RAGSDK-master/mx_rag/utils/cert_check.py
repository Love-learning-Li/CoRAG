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

from datetime import datetime
from .url_checker import StringLengthChecker, CheckResult

from .common import PubkeyType, ParseCertInfo


# 证书最大限制1M
MAX_CERT_LIMIT = 1 * 1024 * 1024


class CertContentsChecker(StringLengthChecker):
    X509_V3 = 3
    RSA_LEN_LIMIT = 3072
    # 椭圆曲线密钥长度
    EC_LEN_LIMIT = 256
    # 允许的签名算法
    SAFE_SIGNATURE_ALGORITHM = ("sha256WithRSAEncryption", "sha512WithRSAEncryption", "ecdsa-with-SHA256")

    def __init__(self, attr_name=None, min_len: int = 1, max_len: int = MAX_CERT_LIMIT):
        super().__init__(attr_name, min_len, max_len)

    def check_cert_info(self, cert_buffer) -> CheckResult:
        cert_info = ParseCertInfo(cert_buffer)
        time_now = datetime.utcnow()
        if time_now <= cert_info.start_time or time_now >= cert_info.end_time:
            msg_format = f"Cert contents checker: invalid cert validity period."
            return CheckResult.make_failed(msg_format)

        if cert_info.cert_version != self.X509_V3:
            msg_format = f"Cert contents checkers: check cert version '{cert_info.cert_version}' is not safe."
            return CheckResult.make_failed(msg_format)

        if cert_info.pubkey_type not in (PubkeyType.EVP_PKEY_RSA.value, PubkeyType.EVP_PKEY_EC.value):
            msg_format = "Cert contents checkers: check cert pubkey type is not safe."
            return CheckResult.make_failed(msg_format)

        if cert_info.pubkey_type == PubkeyType.EVP_PKEY_RSA.value and cert_info.signature_len < self.RSA_LEN_LIMIT:
            msg_format = "Cert contents checkers: check cert pubkey length is not safe."
            return CheckResult.make_failed(msg_format)

        if cert_info.pubkey_type == PubkeyType.EVP_PKEY_EC.value and cert_info.signature_len < self.EC_LEN_LIMIT:
            msg_format = "Cert contents checkers: check cert pubkey length is not safe."
            return CheckResult.make_failed(msg_format)

        if cert_info.signature_algorithm not in self.SAFE_SIGNATURE_ALGORITHM:
            msg_format = "Cert contents checkers: check signature algorithm is not safe."
            return CheckResult.make_failed(msg_format)

        basic_constraints = cert_info.extensions.get("basicConstraints", "")
        if "CA:" not in basic_constraints:
            msg_format = "Cert contents checkers: 'CA' not found in basic constraints."
            return CheckResult.make_failed(msg_format)

        key_usage = cert_info.extensions.get("keyUsage", "")
        if "Digital Signature" not in key_usage:
            msg_format = "Cert contents checkers: 'Digital Signature' not found in key usage."
            return CheckResult.make_failed(msg_format)

        return CheckResult.make_success()

    def check_dict(self, data: dict) -> CheckResult:
        result = super().check_dict(data)
        if not result.success:
            return result

        cert_buffer = self.raw_value(data)
        if not cert_buffer:
            return CheckResult.make_success()

        return self.check_cert_info(cert_buffer)
