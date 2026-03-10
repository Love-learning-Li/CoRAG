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
import json
import shutil
import stat
import unittest
from unittest.mock import patch, MagicMock

import pytest

from mx_rag.utils.common import MAX_FILE_SIZE
from mx_rag.utils.file_check import FileCheckError, FileCheck, SecDirCheck
from mx_rag.utils.file_operate import read_jsonl_from_file, write_jsonl_to_file


class TestFileOperate(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(current_dir, "../../data/finetune.jsonl")
        self.datas = [{"query": "question1", "pos": ["doc content1"]}, {"query": "question2", "pos": ["doc content2"]}]

    def test_read_jsonl_from_file_exception(self):
        with self.assertRaises(FileCheckError):
            read_jsonl_from_file(self.file_path)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("mx_rag.utils.file_check.FileCheck.check_input_path_valid")
    def test_write_and_read_jsonl_file(self, dir_check_mock, check_input_mock):
        W_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        with os.fdopen(os.open(self.file_path, W_FLAGS, MODES), "w") as f:
            for data in self.datas:
                data_str = json.dumps(data, ensure_ascii=False)
                f.write(data_str)
                f.write("\n")
        path = os.path.realpath(self.file_path)
        datas = read_jsonl_from_file(path)
        self.assertEqual(self.datas, datas)


class TestCheckFileOwner(unittest.TestCase):
    def setUp(self):
        self.current_uid = 1000
        self.other_uid = 2000
        self.file_path = '/path/to/your/file'
        self.dir_path = '/path/to/your'

        # Patch os.stat 和 os.getuid
        patcher_getuid = patch('os.getuid', return_value=self.current_uid)
        patcher_stat = patch('os.stat')
        self.addCleanup(patcher_getuid.stop)
        self.addCleanup(patcher_stat.stop)
        self.mock_getuid = patcher_getuid.start()
        self.mock_stat = patcher_stat.start()

    def configure_stat_mock(self, file_owner_uid=None, dir_owner_uid=None, file_exists=True, dir_exists=True):
        file_stat = MagicMock()
        dir_stat = MagicMock()

        if file_owner_uid is not None:
            file_stat.st_uid = file_owner_uid
        else:
            file_stat.st_uid = self.current_uid  # 默认为当前用户

        if dir_owner_uid is not None:
            dir_stat.st_uid = dir_owner_uid
        else:
            dir_stat.st_uid = self.current_uid  # 默认为当前用户

        def side_effect(path):
            if path == self.file_path:
                if file_exists:
                    return file_stat
                else:
                    raise FileNotFoundError
            elif path == self.dir_path:
                if dir_exists:
                    return dir_stat
                else:
                    raise FileNotFoundError
            else:
                raise FileNotFoundError

        self.mock_stat.side_effect = side_effect

    def test_file_and_dir_owned_by_current_user(self):
        # 文件以及所在目录的属主均为当前用户
        self.configure_stat_mock()

        try:
            FileCheck.check_file_owner(self.file_path)
            print("Test passed: File and directory owned by current user.")
        except FileCheckError:
            self.fail("check_file_owner raised FileCheckError unexpectedly!")

    def test_file_owned_by_other_user(self):
        # 文件的属主是另一个用户
        self.configure_stat_mock(file_owner_uid=self.other_uid)

        # 调用函数，预期返回一个异常
        with self.assertRaises(FileCheckError):
            FileCheck.check_file_owner(self.file_path)
        print("Test passed: Detected file owned by another user.")

    def test_directory_owned_by_other_user(self):
        # 文件所在目录的属主为另一个用户
        self.configure_stat_mock(dir_owner_uid=self.other_uid)

        # 调用函数，预期返回一个异常
        with self.assertRaises(FileCheckError):
            FileCheck.check_file_owner(self.file_path)
        print("Test passed: Detected directory owned by another user.")

    def test_file_not_found(self):
        # 文件不存在
        self.configure_stat_mock(file_exists=False)

        # Run the function and expect an exception
        with self.assertRaises(FileCheckError):
            FileCheck.check_file_owner(self.file_path)
        print("Test passed: Detected file not found.")

    def test_directory_not_found(self):
        # 文件所在目录不存在
        self.configure_stat_mock(dir_exists=False)

        # Run the function and expect an exception
        with self.assertRaises(FileCheckError):
            FileCheck.check_file_owner(self.file_path)
        print("Test passed: Detected directory not found.")


class TestSecDirCheck(unittest.TestCase):
    dir_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../test_dir"))

    def setUp(self) -> None:
        os.makedirs(self.dir_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir_path)

    # 创建临时文件的辅助函数
    def create_temp_file(self, path: str, mode: int):
        with open(path, 'w') as f:
            f.write("test")
        os.chmod(path, mode)

    # 测试用例1: 文件权限等于mode_limit，不应抛出异常
    def test_check_mode_equal(self):
        temp_file = "temp_file.txt"
        self.create_temp_file(temp_file, 0o755)
        try:
            FileCheck.check_mode(temp_file, 0o755)
        except FileCheckError:
            pytest.fail("Should not raise an exception when file mode is equal to mode_limit")
        finally:
            os.remove(temp_file)

    # 测试用例2: 文件权限小于mode_limit，不应抛出异常
    def test_check_mode_less(self):
        temp_file = "temp_file.txt"
        self.create_temp_file(temp_file, 0o750)
        try:
            FileCheck.check_mode(temp_file, 0o755)
        except FileCheckError:
            pytest.fail("Should not raise an exception when file mode is less than mode_limit")
        finally:
            os.remove(temp_file)

    # 测试用例3: 文件权限大于mode_limit，应抛出异常
    def test_check_mode_greater(self):
        temp_file = "temp_file.txt"
        self.create_temp_file(temp_file, 0o760)
        with pytest.raises(FileCheckError) as e:
            FileCheck.check_mode(temp_file, 0o755)
        os.remove(temp_file)
        assert "greater than" in str(e.value)

    # 测试用例4: 文件不存在，应抛出异常
    def test_check_mode_file_not_found(self):
        temp_file = "non_existent_file.txt"
        with pytest.raises(FileCheckError) as e:
            FileCheck.check_mode(temp_file, 0o755)
        assert "File not found" in str(e.value)

    # 测试用例5: 文件权限大于mode_limit，应抛出异常（检查所有位）
    def test_check_mode_greater_all_bits(self):
        temp_file = "temp_file.txt"
        self.create_temp_file(temp_file, 0o777)
        with pytest.raises(FileCheckError) as e:
            FileCheck.check_mode(temp_file, 0o755)
        os.remove(temp_file)
        assert "greater than" in str(e.value)

    def test_dir_file_num_1(self):
        # 测试当前目录下有两个文件，期望存在一个文件
        with open(os.path.join(self.dir_path, "1.txt"), "w+") as f:
            f.write("11111")

        with open(os.path.join(self.dir_path, "2.txt"), "w+") as f:
            f.write("2222")

        with self.assertRaises(ValueError) as cm:
            SecDirCheck(self.dir_path, max_size=1024, max_file_num=1).check()

        self.assertTrue("file nums" in cm.exception.__str__())

    def test_dir_file_num_2(self):
        # 测试当前目录及子目录总文件数2个文件，期望存在一个文件
        with open(os.path.join(self.dir_path, "1.txt"), "w+") as f:
            f.write("11111")

        os.makedirs(os.path.join(self.dir_path, "test"))
        with open(os.path.join(self.dir_path, "2.txt"), "w+") as f:
            f.write("2222")

        with self.assertRaises(ValueError) as cm:
            SecDirCheck(self.dir_path, max_size=1024, max_file_num=1).check()

        self.assertTrue("file nums" in cm.exception.__str__())

        with self.assertRaises(ValueError) as cm:
            SecDirCheck(self.dir_path, max_size=1024, max_depth=1).check()

        self.assertTrue("max_depth" in cm.exception.__str__())

    def test_dir_file_size(self):
        # 测试当前目录下文件大小不超过1024
        with open(os.path.join(self.dir_path, "1.txt"), "w+") as f:
            f.write("11111")

        os.makedirs(os.path.join(self.dir_path, "test"))
        with open(os.path.join(self.dir_path, "2.txt"), "w+") as f:
            f.write("2" * 1025)

        with self.assertRaises(FileCheckError) as cm:
            SecDirCheck(self.dir_path, max_size=1024).check()

        self.assertTrue("FileSizeLimit" in cm.exception.__str__())

    def create_jsonl_file(self, path: str):
        datas = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]
        with open(path, "w") as f:
            for data in datas:
                f.write(json.dumps(data) + '\n')

    def data(self):
        return [
            {"id": 3, "name": "Ali"},
            {"id": 4, "name": "Lucy"}
        ]

    def test_write_jsonl_to_file(self):
        self.create_jsonl_file("test.jsonl")
        write_jsonl_to_file(self.data(), "test.jsonl")

    def test_write_jsonl_to_file_exception(self):

        self.create_temp_file("test.txt", 0o755)
        with self.assertRaises(Exception):
            write_jsonl_to_file(self.data(), "test.txt")

    def test_write_jsonl_to_file_io_error(self):
        """测试IO错误的情况"""
        self.create_jsonl_file("test.jsonl")
        with patch('json.dumps', side_effect=IOError("Simulated IO Error")):
            with pytest.raises(Exception, match=r"write jsonl to file IO Error"):
                write_jsonl_to_file(self.data(), "test.jsonl")

        with patch('json.dumps', side_effect=Exception("Exception")):
            with pytest.raises(Exception, match=r"write jsonl to file failed"):
                write_jsonl_to_file(self.data(), "test.jsonl")


if __name__ == '__main__':
    unittest.main()

