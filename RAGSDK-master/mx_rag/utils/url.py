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
import stat
from typing import Dict, Iterator

import urllib3
from urllib3.exceptions import TimeoutError as urllib3_TimeoutError, HTTPError
from loguru import logger

from mx_rag.utils.client_param import ClientParam
from .url_checker import HttpUrlChecker, HttpsUrlChecker
from .tlsconfing import TlsConfig
from .cert_check import CertContentsChecker
from .common import MB
from .file_check import SecFileCheck, FileCheckError, PathNotFileException
from .crl_checker import CRLChecker, CRLCheckError

HTTP_SUCCESS = 200
MAX_CERT_FILE_SIZE = MB
MIN_PASSWORD_LENGTH = 8
PASSWORD_REQUIREMENT = 2


class Result:
    def __init__(self, success: bool, data):
        self.success = success
        self.data = data


def is_url_valid(url, use_http) -> bool:
    if url.startswith("http:") and not use_http:
        return False
    check_key = "url"
    if use_http and HttpUrlChecker(check_key).check({check_key: url}):
        return True
    elif not use_http and HttpsUrlChecker(check_key).check({check_key: url}):
        return True
    return False


class RequestUtils:

    def __init__(self,
                 retries=3,
                 num_pools=200,
                 maxsize=200,
                 client_param: ClientParam = ClientParam()
                 ):

        self.use_http = client_param.use_http
        self.response_limit_size = client_param.response_limit_size

        if client_param.use_http:
            ssl_ctx = TlsConfig._get_init_context()
        else:
            # Use https, check certificate and crl
            self._check_https_para(client_param)
            success, ssl_ctx = TlsConfig.get_client_ssl_context(client_param.ca_file, client_param.crl_file)
            if not success:
                # When failed, ssl_ctx is the error message
                raise ValueError(f'{ssl_ctx}')

        self.pool = urllib3.PoolManager(ssl_context=ssl_ctx,
                                        retries=retries,
                                        timeout=client_param.timeout,
                                        num_pools=num_pools,
                                        maxsize=maxsize)

    @staticmethod
    def _check_ca_content(ca_file: str):
        try:
            R_FLAGS = os.O_RDONLY
            MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            with os.fdopen(os.open(ca_file, R_FLAGS, MODES), 'r') as f:
                ca_data = f.read()
        except FileNotFoundError as e:
            logger.error(f"Certificate file '{ca_file}' not found.")
            raise ValueError(f"Certificate file '{ca_file}' not found.") from e
        except PermissionError as e:
            logger.error(f"Permission denied when reading certificate file: '{ca_file}'")
            raise ValueError(f"Permission denied for certificate file: {ca_file}") from e
        except Exception as e:
            logger.error(f"read cert file failed, find exception: {e}")
            raise ValueError('read cert file failed') from e

        ret = CertContentsChecker("cert").check_dict({"cert": ca_data})
        if not ret:
            logger.error(f"invalid ca cert content: '{ret.reason}'")
            raise ValueError('invalid cert content')

    def post(self, url: str, body: str, headers: Dict):
        if not is_url_valid(url, self.use_http):
            logger.error("url check failed")
            return Result(False, "")

        try:
            response = self.pool.request(method='POST',
                                         url=url,
                                         body=body,
                                         headers=headers,
                                         preload_content=False)
        except urllib3_TimeoutError:
            logger.error("The request timed out")
            return Result(False, "")
        except HTTPError:
            logger.error("Request failed due to HTTP error")
            return Result(False, "")
        except Exception:
            logger.error("request failed")
            return Result(False, "")

        try:
            content_length = int(response.headers.get("Content-Length"))
        except ValueError as e:
            logger.error(f"Invalid Content-Length header in response: {e}")
            return Result(False, "")
        except Exception as e:
            logger.error(f"get content length failed, find exception: {e}")
            return Result(False, "")

        if content_length > self.response_limit_size:
            logger.error("content length exceed limit")
            return Result(False, "")

        if response.status == HTTP_SUCCESS:
            try:
                response_data = response.read(amt=self.response_limit_size)
            except urllib3.exceptions.TimeoutError as e:
                logger.error(f"Timeout error while reading response: {e}")
                return Result(False, "")
            except urllib3.exceptions.HTTPError as e:
                logger.error(f"HTTP error while reading response: {e}")
                return Result(False, "")
            except Exception as e:
                logger.error(f"An unexpected error occurred while reading response: {e}")
                return Result(False, "")

            return Result(True, response_data)
        else:
            logger.error(f"request failed with status code {response.status}")
            return Result(False, "")

    def post_streamly(self, url: str, body: str, headers: Dict, chunk_size: int = 1024):
        if not is_url_valid(url, self.use_http):
            logger.error("url check failed")
            yield Result(False, "")

        try:
            response = self.pool.request(method='POST', url=url, body=body, headers=headers, preload_content=False)
        except urllib3_TimeoutError:
            logger.error("The request timed out")
            yield Result(False, "")
            return
        except HTTPError:
            logger.error("Request failed due to HTTP error")
            yield Result(False, "")
            return
        except Exception:
            logger.error(f"request failed")
            yield Result(False, "")
            return

        try:
            content_type = response.headers.get("Content-Type")
            if content_type is None:
                raise ValueError("Invalid Content-Type header")
            content_type = str(content_type)
        except KeyError as e:
            logger.error(f"Content-Type header is missing: {e}")
            yield Result(False, "")
            return
        except ValueError as e:
            logger.error(f"Invalid Content-Type header: {e}")
            yield Result(False, "")
            return
        except Exception as e:
            logger.error(f"Failed to get Content-Type, unexpected error: {e}")
            yield Result(False, "")
            return

        if 'text/event-stream' not in content_type:
            logger.error("content type is not stream")
            yield Result(False, "")
            return

        if response.status == HTTP_SUCCESS:
            for result in self._iter_lines(response, chunk_size):
                yield result
        else:
            logger.error(f"request failed with status code {response.status}")
            yield Result(False, "")

    def _iter_lines(self, response, chunk_size=1024) -> Iterator[Result]:
        buffer = b''
        total_length = 0
        try:
            for chunk in response.stream(chunk_size):
                total_length += len(chunk)
                if total_length > self.response_limit_size:
                    logger.error("content length exceed limit")
                    yield Result(False, "")
                    return

                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    yield Result(True, line + b'\n')

            if buffer:
                yield Result(True, buffer)
        except urllib3.exceptions.HTTPError as e:
            logger.error(f"HTTP error while reading response: {e}")
            yield Result(False, "")
        except Exception as e:
            logger.error(f"read response failed, find exception: {e}")
            yield Result(False, "")

    def _check_https_para(self, client_param: ClientParam):
        try:
            SecFileCheck(client_param.ca_file, MAX_CERT_FILE_SIZE).check()
        except (FileCheckError, PathNotFileException) as e:
            logger.error(f"check ca file failed: {e}")
            raise ValueError('check ca file failed') from e

        self._check_ca_content(client_param.ca_file)

        if not client_param.crl_file:
            logger.info("No CRL file provided; skipping CRL checks.")
            return

        try:
            SecFileCheck(client_param.crl_file, MAX_CERT_FILE_SIZE).check()
        except (FileCheckError, PathNotFileException) as e:
            logger.error(f"check crl file failed: {e}")
            raise ValueError('check crl file failed') from e

        checker = CRLChecker(crl_path=client_param.crl_file, issuer_cert_path=client_param.ca_file)
        if not checker.check_crl():
            logger.error(f"CRL check failed for file: {client_param.crl_file}")
            raise CRLCheckError("CRL check error")
