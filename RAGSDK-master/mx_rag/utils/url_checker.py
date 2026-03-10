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

import re
import sys
import abc
from abc import ABC
from typing import Optional, Union, Any


class CheckResult:
    def __init__(self, success: bool, reason: str, checker_description: Any = None):
        self.success: bool = success
        self.reason: str = reason
        self.check_description: Any = checker_description

    def __str__(self):
        return "%s" % self.__dict__

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return self.success

    @classmethod
    def make_success(cls):
        return cls(True, "")

    @classmethod
    def make_failed(cls, msg: str, checker_description=None):
        return cls(False, msg, checker_description)


class AttrCheckerInterface:
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def required(self) -> bool:
        pass

    @abc.abstractmethod
    def check(self, data: Optional[Union[object, dict, int, str, float, bool, bytes, list, tuple]]) -> CheckResult:
        pass

    @abc.abstractmethod
    def raw_value(self, data):
        pass

    @abc.abstractmethod
    def check_dict(self, data: Optional[Union[dict, int, str, float, bool, bytes, list, tuple]]) -> CheckResult:
        pass


class AttrCheckerBase(AttrCheckerInterface):
    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def name(self) -> str:
        return self.attr_name

    @abc.abstractmethod
    def required(self) -> bool:
        pass

    def check(self, data: Optional[Union[object, dict, int, str, float, bool, bytes, list, tuple]]) -> CheckResult:
        if isinstance(data, dict):
            return self.check_dict(data)
        elif isinstance(data, (int, str, float, bool, bytes, list, tuple)) or data is None:
            return self.check_dict(data)
        else:
            return self.check_dict(data.__dict__)

    def raw_value(self, data):
        if self.name() is None:
            return data
        return data.get(self.name())

    @abc.abstractmethod
    def check_dict(self, data: Optional[Union[dict, int, str, float, bool, bytes, list, tuple]]) -> CheckResult:
        pass


class ExistsChecker(AttrCheckerBase, ABC):
    def __init__(self, attr_name=None, required: bool = True):
        super().__init__(attr_name)
        self._required = required

    def required(self) -> bool:
        return self._required

    def check_dict(self, data: dict) -> CheckResult:
        if not self.required():
            return CheckResult.make_success()
        if data is None:
            return CheckResult.make_failed("Exists checker: input is null while check {}".format(self.name()))
        value = self.raw_value(data)
        if value is not None:
            return CheckResult.make_success()
        return CheckResult.make_failed("Exists checker: {} not exists".format(self.name()))


class StringLengthChecker(ExistsChecker, ABC):
    def __init__(
        self,
        attr_name=None,
        min_len: int = 0,
        max_len: int = 32,
        required: bool = True,
    ):
        super().__init__(attr_name, required)
        self.min_len: int = min_len
        self.max_len: int = max_len

    def check_dict(self, data: dict) -> CheckResult:
        result = super().check_dict(data)
        if not result.success:
            return result
        value = self.raw_value(data)
        if value is None:
            return CheckResult.make_success()
        if not isinstance(value, str):
            msg_format = "String length checker: invalid value type '{}' of {}"
            return CheckResult.make_failed(msg_format.format(type(value), self.name()))
        if len(value) < self.min_len or len(value) > self.max_len:
            msg_format = "String length checker: invalid length of {}"
            return CheckResult.make_failed(msg_format.format(self.name()))
        return CheckResult.make_success()


class RegexStringChecker(StringLengthChecker, ABC):
    def __init__(
        self,
        attr_name: str = None,
        match_str: str = "",
        min_len: int = 0,
        max_len: int = sys.maxsize,
        required: bool = True,
    ):
        super().__init__(attr_name, min_len, max_len, required)
        self.match_str = match_str

    def check_dict(self, data: dict) -> CheckResult:
        result = super().check_dict(data)
        if not result.success:
            return result
        value = self.raw_value(data)
        if value is None:
            return CheckResult.make_success()
        if not isinstance(value, str):
            msg_format = "Regex string checker: invalid value type '{}' of {}"
            return CheckResult.make_failed(msg_format.format(type(value), self.name()))
        find_iter = re.fullmatch(self.match_str, value)
        if find_iter is None or find_iter.group(0) != value:
            msg_format = "Regex string checker: invalid format of {}"
            return CheckResult.make_failed(msg_format.format(self.name()))
        return CheckResult.make_success()


class HttpUrlChecker(RegexStringChecker):
    def __init__(self, attr_name: str = None, min_len: int = 0, max_len: int = 2048, required: bool = True):
        super().__init__(attr_name, "(http|HTTP)://[-A-Za-z0-9+&/%=~_|!:,.;]*[-A-Za-z0-9+&/%=~_|]", min_len,
                         max_len, required)


class HttpsUrlChecker(RegexStringChecker):
    def __init__(self, attr_name: str = None, min_len: int = 0, max_len: int = 2048, required: bool = True):
        super().__init__(attr_name, "(https|HTTPS)://[-A-Za-z0-9+&/%=~_|!:,.;]*[-A-Za-z0-9+&/%=~_|]", min_len,
                         max_len, required)
