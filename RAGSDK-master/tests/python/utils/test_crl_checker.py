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

import unittest
import tempfile
from pathlib import Path
from mx_rag.utils.crl_checker import CRLChecker
import datetime
from unittest.mock import patch, MagicMock, PropertyMock
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


# Helper function to generate crypto materials for tests
def _generate_test_crypto(temp_path):
    """Generates a CA key, CA cert, peer key, peer cert, and a CRL."""
    # Generate CA private key
    ca_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_public_key = ca_private_key.public_key()

    # Generate self-signed CA certificate
    ca_subject = ca_issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"Test CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_issuer)
        .public_key(ca_public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=10))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_private_key, hashes.SHA256())
    )
    ca_cert_path = temp_path / "ca.pem"
    ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))

    # Generate peer private key
    peer_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    peer_public_key = peer_private_key.public_key()

    # Generate peer certificate signed by CA
    peer_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"Test Peer")])
    peer_cert = (
        x509.CertificateBuilder()
        .subject_name(peer_subject)
        .issuer_name(ca_issuer)
        .public_key(peer_public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=5))
        .sign(ca_private_key, hashes.SHA256())
    )
    peer_cert_path = temp_path / "peer.pem"
    peer_cert_path.write_bytes(peer_cert.public_bytes(serialization.Encoding.PEM))

    # Generate another peer certificate to be revoked
    revoked_peer_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"Revoked Peer")])
    revoked_peer_cert = (
        x509.CertificateBuilder()
        .subject_name(revoked_peer_subject)
        .issuer_name(ca_issuer)
        .public_key(rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=5))
        .sign(ca_private_key, hashes.SHA256())
    )
    revoked_peer_cert_path = temp_path / "revoked_peer.pem"
    revoked_peer_cert_path.write_bytes(revoked_peer_cert.public_bytes(serialization.Encoding.PEM))

    # Generate CRL
    revoked_cert = (
        x509.RevokedCertificateBuilder()
        .serial_number(revoked_peer_cert.serial_number)
        .revocation_date(datetime.datetime.now(datetime.timezone.utc))
        .build()
    )
    crl_builder = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(ca_issuer)
        .last_update(datetime.datetime.now(datetime.timezone.utc))
        .next_update(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1))
        .add_extension(x509.CRLNumber(1024), critical=False)
        .add_revoked_certificate(revoked_cert)
    )
    crl = crl_builder.sign(ca_private_key, hashes.SHA256())
    crl_path = temp_path / "ca.crl"
    crl_path.write_bytes(crl.public_bytes(serialization.Encoding.PEM))

    return ca_cert_path, peer_cert_path, revoked_peer_cert_path, crl_path


class TestCRLCheckerWithCrypto(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        (
            self.ca_cert_path,
            self.peer_cert_path,
            self.revoked_peer_cert_path,
            self.crl_path,
        ) = _generate_test_crypto(self.tmp_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_crl_and_issuer_cert_properties_load_successfully(self):
        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        self.assertIsNotNone(checker.crl)
        self.assertIsNotNone(checker.issuer_cert)
        self.assertIsInstance(checker.crl, x509.CertificateRevocationList)
        self.assertIsInstance(checker.issuer_cert, x509.Certificate)

    def test_crl_property_handles_load_failure(self):
        invalid_path = self.tmp_path / "nonexistent.crl"
        checker = CRLChecker(
            crl_path=str(invalid_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        self.assertIsNone(checker.crl)

    def test_issuer_cert_property_handles_load_failure(self):
        invalid_path = self.tmp_path / "nonexistent.pem"
        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(invalid_path),
        )
        self.assertIsNone(checker.issuer_cert)

    def test_check_crl_succeeds_with_valid_crl(self):
        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        self.assertTrue(checker.check_crl())

    def test_check_crl_fails_with_invalid_signature(self):
        # Create a different CA to sign the CRL, making the signature invalid
        other_ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        crl_builder = x509.CertificateRevocationListBuilder().issuer_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"Test CA")])
        ).last_update(
            datetime.datetime.now(datetime.timezone.utc)
        ).next_update(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        )
        bad_crl = crl_builder.sign(other_ca_key, hashes.SHA256())
        bad_crl_path = self.tmp_path / "bad_sig.crl"
        bad_crl_path.write_bytes(bad_crl.public_bytes(serialization.Encoding.PEM))

        checker = CRLChecker(
            crl_path=str(bad_crl_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        self.assertFalse(checker._check_crl_signature())
        self.assertFalse(checker.check_crl())

    @patch('mx_rag.utils.crl_checker.datetime')
    def test_check_crl_time_expired_crl_denied(self, mock_datetime):
        # Set time to be after the CRL's next_update
        mock_datetime.datetime.now.return_value = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(days=2)
        mock_datetime.timezone.utc = datetime.timezone.utc

        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
            allow_expired_crl=False,
        )
        self.assertFalse(checker._check_crl_time())
        self.assertFalse(checker.check_crl())

    @patch('mx_rag.utils.crl_checker.datetime')
    def test_check_crl_time_expired_crl_allowed(self, mock_datetime):
        mock_datetime.datetime.now.return_value = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(days=2)
        mock_datetime.timezone.utc = datetime.timezone.utc

        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
            allow_expired_crl=True,
        )
        self.assertTrue(checker._check_crl_time())

    def test_verify_no_crl_allowed(self):
        checker = CRLChecker(
            crl_path="nonexistent.crl",
            issuer_cert_path=str(self.ca_cert_path),
            allow_no_crl=True,
        )
        self.assertTrue(checker.verify(str(self.peer_cert_path)))

    def test_verify_no_crl_denied(self):
        checker = CRLChecker(
            crl_path="nonexistent.crl",
            issuer_cert_path=str(self.ca_cert_path),
            allow_no_crl=False,
        )
        self.assertFalse(checker.verify(str(self.peer_cert_path)))

    def test_verify_valid_peer_cert(self):
        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        self.assertTrue(checker.verify(str(self.peer_cert_path)))

    def test_verify_revoked_peer_cert(self):
        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        self.assertFalse(checker.verify(str(self.revoked_peer_cert_path)))

    def test_is_certificate_revoked_handles_invalid_peer_cert_path(self):
        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        # Expect True because it's treated as revoked if it can't be loaded
        self.assertTrue(checker._is_certificate_revoked("nonexistent.pem"))

    def test_check_crl_format_fails_if_no_extensions(self):
        checker = CRLChecker(
            crl_path=str(self.crl_path),
            issuer_cert_path=str(self.ca_cert_path),
        )
        # Mock the loaded CRL to have no extensions
        mock_crl = MagicMock()
        delattr(mock_crl, "extensions")
        with patch.object(CRLChecker, 'crl', new_callable=PropertyMock, return_value=mock_crl):
            self.assertFalse(checker._check_crl_format())


if __name__ == "__main__":
    unittest.main()
