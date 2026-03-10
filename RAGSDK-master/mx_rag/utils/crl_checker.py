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

import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from cryptography import x509
from cryptography.exceptions import InvalidSignature


class CRLCheckError(Exception):
    pass


class CRLChecker:
    def __init__(
        self,
        crl_path: str,
        issuer_cert_path: str,
        allow_no_crl: bool = False,
        allow_expired_crl: bool = False,
    ):
        self.crl_path = Path(crl_path)
        self.issuer_cert_path = Path(issuer_cert_path)
        self.allow_no_crl = allow_no_crl
        self.allow_expired_crl = allow_expired_crl
        self._crl: Optional[x509.CertificateRevocationList] = None
        self._issuer_cert: Optional[x509.Certificate] = None

    @property
    def crl(self) -> Optional[x509.CertificateRevocationList]:
        if self._crl is None:
            try:
                with self.crl_path.open('rb') as f:
                    self._crl = x509.load_pem_x509_crl(f.read())
                    logger.info("Loaded local CRL successfully.")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load local CRL: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
        return self._crl

    @property
    def issuer_cert(self) -> Optional[x509.Certificate]:
        if self._issuer_cert is None:
            try:
                with self.issuer_cert_path.open('rb') as f:
                    self._issuer_cert = x509.load_pem_x509_certificate(f.read())
                    logger.info("Loaded CRL issuer certificate successfully.")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load CA certificate: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
        return self._issuer_cert

    def check_crl(self) -> bool:
        return self._check_crl_format() and self._check_crl_signature() and self._check_crl_time()

    def verify(self, peer_cert_path: str) -> bool:
        if not self.crl:
            if self.allow_no_crl:
                logger.warning("No local CRL found. Connection allowed per configuration. Security event logged.")
                return True
            logger.error("No local CRL found. Connection denied.")
            return False
        if not self.check_crl():
            return False
        if self._is_certificate_revoked(peer_cert_path):
            return False
        return True

    def _check_crl_format(self) -> bool:
        crl = self.crl
        if not crl or not getattr(crl, "extensions", None):
            logger.error("CRL does not have extensions and may not be in X.509 v2 format.")
            return False
        return True

    def _check_crl_signature(self) -> bool:
        crl = self.crl
        issuer_cert = self.issuer_cert
        if not crl or not issuer_cert:
            logger.error("CRL or CA certificate not loaded, cannot verify CRL signature.")
            return False
        try:
            if crl.is_signature_valid(issuer_cert.public_key()):
                return True
            logger.error("CRL signature is invalid.")
            return False
        except InvalidSignature as e:
            logger.error(f"CRL signature is invalid: {e}")
        except Exception as e:
            logger.error(f"CRL signature verification failed: {e}")
        return False

    def _check_crl_time(self) -> bool:
        crl = self.crl
        if not crl:
            return False
        now = datetime.datetime.now(datetime.timezone.utc)
        if not crl.last_update_utc <= now < crl.next_update_utc:
            logger.warning(f"CRL has expired")
            if not self.allow_expired_crl:
                logger.error("Connection denied due to expired CRL.")
                return False
            logger.warning("Expired CRL is allowed by configuration. Security event logged.")
        return True

    def _is_certificate_revoked(self, peer_cert_path: str) -> bool:
        crl = self.crl
        if not crl:
            logger.error("CRL not loaded, cannot check revocation.")
            return True
        try:
            with Path(peer_cert_path).open('rb') as f:
                cert = x509.load_pem_x509_certificate(f.read())
        except (IOError, ValueError) as e:
            logger.error(f"Failed to load peer certificate: {e}")
            # If we can't check the cert, treat it as revoked
            return True

        revoked = crl.get_revoked_certificate_by_serial_number(cert.serial_number)
        if revoked is not None:
            logger.error("Peer certificate has been revoked. Connection denied.")
            return True
        
        return False
