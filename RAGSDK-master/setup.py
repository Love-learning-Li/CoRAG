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
import glob
import logging
import os
import shutil
import stat
from pathlib import Path

import yaml
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as build_py_orig

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
build_folder = ('build/bdist*', 'build/lib')
cache_folder = ('mx_rag.egg-info', '_package_output')
pwd = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(pwd, "build/lib")


def get_ci_version_info():
    """
    Get version information from ci config file
    :return: version number
    """
    ci_version_file = this_directory.joinpath('ci', 'config', 'config.ini')
    version = '7.0.RC1'
    logging.info("get version from %s", ci_version_file)
    try:
        R_FLAGS = os.O_RDONLY
        MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        with os.fdopen(os.open(ci_version_file, R_FLAGS, MODES), 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                else:
                    continue
                if key.strip() == 'version':
                    version = value.strip()
                    break
    except Exception as ex:
        logging.warning("get version failed, %s", str(ex))
    return version


def build_dependencies():
    """generate python file"""
    version_file = os.path.join(pkg_dir, 'mx_rag', 'version.py')
    version_file_dir = os.path.join(pkg_dir, 'mx_rag')
    if not os.path.exists(version_file_dir):
        os.makedirs(version_file_dir, exist_ok=True)

    with os.fdopen(os.open(version_file, os.O_WRONLY | os.O_CREAT, mode=stat.S_IRUSR | stat.S_IWUSR), 'w') as f:
        f.write(f"__version__ = '{get_ci_version_info()}'\n")
        f.write(f"__build_time__ = '{datetime.date.today()}'\n")


def clean():
    for folder in cache_folder:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for pattern in build_folder:
        for name in glob.glob(pattern):
            if os.path.exists(name):
                shutil.rmtree(name)


def get_all_py_files():
    src_dir = f"{pwd}/mx_rag"
    file_list = []
    for name in glob.glob(f"{src_dir}/**/*.py", recursive=True):
        file = name[len(src_dir) + 1:]
        file_list.append(file)
    return file_list


def get_all_json_files():
    src_dir = f"{pwd}/mx_rag"
    file_list = []
    for name in glob.glob(f"{src_dir}/evaluate/**/*.json", recursive=True):
        file = name[len(src_dir) + 1:]
        file_list.append(file)

    for name in glob.glob(f"{src_dir}/tools/**/*.json", recursive=True):
        file = name[len(src_dir) + 1:]
        file_list.append(file)
    return file_list


clean()

build_dependencies()

required_package = []

package_data = {'': get_all_py_files() + get_all_json_files()}


class BuildBy(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        res = []
        for pkg, mod, file in modules:
            res.append((pkg, mod, file))
        return res


setup(
    name='mx_rag',
    version=get_ci_version_info(),
    platforms=['linux', ],
    description='RAGSDK is library to build RAG system',
    python_requires='>= 3.10, <3.12',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required_package,
    package_data=package_data,
    packages=find_packages(exclude=["*test*"]),
    include_package_data=True,
    cmdclass={
        'build_py': BuildBy,
    },
)

clean()
