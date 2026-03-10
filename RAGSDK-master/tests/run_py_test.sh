#!/bin/bash
# The script to run unit test case.
# -------------------------------------------------------------------------
# This file is part of the RAGSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# RAGSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run_py_test.sh" ; exit ; } ; pwd)
TOP_PATH="${CUR_PATH}"/../
FAKE_PACKAGE_PATH="${CUR_PATH}"/fake_package
export PYTHONPATH=$TOP_PATH:$PYTHONPATH:$FAKE_PACKAGE_PATH

export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libgomp.so.1:$LD_PRELOAD
export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD

mkdir test_results

function run_test_cases() {
    echo "Get testcases final result."
    pytest --cov="${CUR_PATH}"/../mx_rag --cov-report=html --cov-report=xml --junit-xml=./final.xml --html=./final.html --self-contained-html --durations=5 -vs --cov-branch  --cov-config=.coveragerc
    coverage xml -i --omit="build/*,cust_op/*,src/*,*/libs/*,*/evaluate/*,*/train_data_generator.py,*/ops/*"
    cp coverage.xml final.xml final.html ./test_results
    cp -r htmlcov ./test_results
    rm -rf coverage.xml final.xml final.html htmlcov
}

pip3 install pytest pytest-cov pytest-html langchain_opengauss==0.1.5
pip3.11 install -r ../requirements.txt --exists-action i
echo "************************************* Start mxRAG LLT Test *************************************"
start=$(date +%s)
run_test_cases
ret=$?
end=$(date +%s)
echo "*************************************  End  mxRAG LLT Test *************************************"
echo "LLT running take: $(expr "${end}" - "${start}") seconds"


exit "${ret}"
