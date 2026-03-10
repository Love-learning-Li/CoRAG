#!/bin/bash
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

CUR_DIR=$(dirname "$(readlink -f "$0")")

TARGET_PLATFORM="$1"
PY_VERSION="$2"

if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi
mkdir "build"

if [ -d "$CUR_DIR/$TARGET_PLATFORM" ]; then
    echo "Removing existing build directory..."
    rm -rf $CUR_DIR/$TARGET_PLATFORM
fi
mkdir -p $CUR_DIR/$TARGET_PLATFORM

cd build

export PYTHON_INCLUDE_PATH="$($PY_VERSION -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$($PY_VERSION  -c 'from sysconfig import get_paths; print(get_paths()["platlib"])')"
python_location=$(pip3 show torch |grep Location | awk -F ' ' '{print $2}')
export PYTORCH_INSTALL_PATH="$python_location/torch"
export PYTORCH_NPU_INSTALL_PATH="$python_location/torch_npu"

if [ -e "/opt/rh/devtoolset-7/root/usr/bin/gcc" ]; then
    cmake -DCMAKE_C_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/gcc  -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/g++ -DTARGET_PLATFORM:string=$TARGET_PLATFORM  -DCMAKE_INSTALL_PREFIX=$CUR_DIR/$TARGET_PLATFORM ..
else
    cmake -DTARGET_PLATFORM:string=$TARGET_PLATFORM  -DCMAKE_INSTALL_PREFIX=$CUR_DIR/$TARGET_PLATFORM ..
fi

make -j"$(nproc)"
make install
cd ..