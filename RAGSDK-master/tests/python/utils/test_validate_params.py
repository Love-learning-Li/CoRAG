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
from typing import Optional, List, Any, Dict

from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field, BaseModel, ValidationError, field_validator, ConfigDict

from mx_rag.utils.common import validate_params, validate_list_str


class Person():
    MAX_LIMIT = 100
    name: str
    number: int = 10

    @validate_params(
        age=dict(validator=lambda x: x > 10),
        weight=dict(validator=lambda x: 90 <= x <= 150)
    )
    def __init__(self, age: int, weight: int, ranker: int = 1):
        self.age = age
        self.weight = weight
        self.ranker = ranker

    def call_back_fun(self, func, *args, **kwargs):
        func(*args, **kwargs)

    @validate_params(
        param1=dict(validator=lambda x: 0.0 < x < 1.0),
    )
    def validate_call_back_fun(self, param1: float, func, *args):
        func(*args)

    @validate_params(
        param1=dict(validator=lambda x: x < Person.MAX_LIMIT),
    )
    def validate_self_var(self, param1):
        pass


@validate_params(
    name=dict(validator=lambda x: isinstance(x, str)),
    weight=dict(validator=lambda x: 0 <= x <= 50)
)
class Animal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    master: Person
    weight: int = 10

    @property
    def _llm_type(self) -> str:
        pass

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        pass


class Cat(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    age: int = Field(ge=0, le=10)
    gender: str
    color: str = Field(min_length=0, max_length=10, default="white")
    # 或者master: Person = None，如果Person未继承BaseModel，必须设置arbitrary_types_allowed = True
    master: Optional[Person] = None
    attribute: Optional[str] = None
    # min_length支持iterables；多层只校验最外层
    multidict: List[Dict[str, str]] = Field(min_length=0, max_length=1, default=None)

    @field_validator('gender')
    def validate_gender(cls, input_value):
        if input_value not in ["male", "female"]:
            raise ValueError(f"month value must be in [male, female]")
        return input_value


@validate_params(
    param1=dict(validator=lambda x: isinstance(x, str) and len(x) >= 5),
    param2=dict(validator=lambda x: x > 0)
)
def non_class_funciton(param1, param2: int, param3=None):
    pass


@validate_params(
    param3=dict(validator=lambda x: x > 0)
)
def non_class_funciton1(param1, param2: int, param3: int = 50):
    pass


@validate_params(
    param1=dict(validator=lambda x: isinstance(x, int)),
    param2=dict(validator=lambda x: x > 0),
    param3=dict(validator=
                lambda x: all(len(item) > 1 for item in x),
                message="check rule: lambda x: all(len(item) > 1 for item in x)")
)
def non_class_funciton2(param1: int, param2: int, param3: List[dict]):
    pass


@validate_params(
    param1={"validator": lambda x: validate_list_str(x, [1, 3], [5, 10]),
            "message": "param type is not List[str] or length of list not in [1, 3] "
                       "or length of str in list not in [5, 10]"}
)
def non_class_funciton3(param1: List[str]):
    pass


class TestValidateParams(unittest.TestCase):
    def test_class_scope(self):
        Person(18, 140, 2)
        Person(18, weight=140, ranker=-1)
        with self.assertRaises(ValueError):
            Person(18, 85, 10)

    def test_non_calss_funciton(self):
        non_class_funciton("hello", 1)
        with self.assertRaises(ValueError):
            non_class_funciton("hello", param2=-1, param3=5)
        with self.assertRaises(ValueError):
            non_class_funciton(1, param2=-1, param3=5)

    def test_call_back_function(self):
        person = Person(18, 140, 2)
        person.call_back_fun(non_class_funciton, "world!", param2=3, param3=5)
        person.call_back_fun(non_class_funciton, param1="world!", param2=3, param3=5)
        person.validate_call_back_fun(0.5, non_class_funciton, "world!", 5)
        with self.assertRaises(ValueError):
            person.validate_call_back_fun(1.1, non_class_funciton, "world!", 5)

    def test_default_parm_validation(self):
        non_class_funciton1(1, 2)

    def test_validate_self_var(self):
        person = Person(18, 140, 2)
        person.validate_self_var(80)
        with self.assertRaises(ValueError):
            person.validate_self_var(110)

    # 类继承BaseModel或者langchain的LLM等，也支持类变量的校验
    # 如果类继承BaseModel，需要设置arbitrary_types_allowed为true，否则校验的类型也要继承BaseModel
    def test_class_variable(self):
        person = Person(18, 140, 2)
        Animal(name="panda", master=person)
        with self.assertRaises(ValidationError):
            Animal(name="panda", master=123)
        Animal(name="panda", master=person, weight=0)
        with self.assertRaises(ValueError):
            Animal(name="panda", master=person, weight=-1)

    def test_pydantic(self):
        person = Person(18, 140, 2)
        cat = Cat(age=10, gender="male", master=person, attribute="attribute")
        self.assertEqual(cat.color, "white")
        # gender不符合要求
        with self.assertRaises(ValidationError):
            Cat(age=10, gender="median", attribute=None)
        # age不符合要求
        with self.assertRaises(ValidationError):
            Cat(age=20, gender="male", attribute=None)
        # 传入attribute
        Cat(age=0, gender="male", attribute=None)
        # 不传入attribute
        Cat(age=0, gender="male")
        # color不满足要求
        with self.assertRaises(ValidationError):
            Cat(age=10, gender="male", master=person, attribute="attribute", color="hello world!")
        # multidict不满足要求
        with self.assertRaises(ValidationError):
            Cat(age=10, gender="male", master=person, attribute="attribute", multidict=[{"1": "a"}, {"2": "b"}])
        # 只校验了multidict最外层格式
        Cat(age=10, gender="male", master=person, attribute="attribute", multidict=[])

    def test_log_info(self):
        try:
            non_class_funciton2("1", 1, [{1: "a", 2: "b", 3: "c"}, {4: "a", 5: "b", 6: "c"}])
        except Exception as e:
            self.assertTrue(str(e).find("'param1' of function 'non_class_funciton2' is invalid") > -1)
        try:
            non_class_funciton2(1, -1, [{1: "a", 2: "b", 3: "c"}, {4: "a", 5: "b", 6: "c"}])
        except Exception as e:
            self.assertTrue(str(e).find("'param2' of function 'non_class_funciton2' is invalid") > -1)
        try:
            non_class_funciton2(1, 1, [{1: "a"}, {4: "a"}])
        except Exception as e:
            self.assertTrue(str(e).find("lambda x: all(len(item) > 1 for item in x)") > -1)

    def test_validate_list_str(self):
        non_class_funciton3(["hello!", "world!", "beautiful"])
        with self.assertRaises(ValueError):
            non_class_funciton3([123, "world!", "beautiful"])
        with self.assertRaises(ValueError):
            non_class_funciton3(["hello!", "world!", "beautiful", "xxxxx"])
        with self.assertRaises(ValueError):
            non_class_funciton3(["hi!", "world!", "beautiful"])
