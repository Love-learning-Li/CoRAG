#!/bin/bash
# CI一键构建脚本.
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

warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH=$(uname -m)
PY311_VER=py$(python3.11 -c "import platform; print(platform.python_version().replace('.','')[0:3])")
CUR_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=$(readlink -f "${CUR_PATH}"/..)
SO_OUTPUT_DIR="${ROOT_PATH}"/mx_rag/lib
TRANSFOMER_ADAPTER_OUTPUT_DIR="${ROOT_PATH}"/ops/transformer_adapter/output

export CFLAGS="-fstack-protector-strong -fPIC -fPIE -O2 -std=c11 -ftrapv -Wall -Wextra -Werror -fno-common"
export LDFLAGS="-s -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack"

function clean()
{
    [ -n "$SO_OUTPUT_DIR" ] && rm -rf "$SO_OUTPUT_DIR"
    [ -n "$TRANSFOMER_ADAPTER_OUTPUT_DIR" ] && rm -rf "$TRANSFOMER_ADAPTER_OUTPUT_DIR"
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/dist
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/mx_rag/build
    find "${ROOT_PATH}/mx_rag" -name "*.so" -exec rm {} \;
    echo "clean .so output dir"
    find "${ROOT_PATH}/mx_rag" -name "*.c" -exec rm {} \;
    echo "clean .c output dir"
}

function build_so_package()
{
    local py=$1
    find "${ROOT_PATH}/mx_rag"  \( -name "*.so" -o -name "*.c" \) -exec  rm -f {} \;

    cd "${ROOT_PATH}/mx_rag"
    ${py} ./setup.py build_ext -j"$(nproc)"
    mkdir -p "${SO_OUTPUT_DIR}"
    cp -arfv build/lib.linux-*/mx_rag/* .
    rm ./setup*.so
    rm ./version*.so
    rm -rf build
}

function build_wheel_package()
{
    local py=$1
    tag=$2
    cd "${ROOT_PATH}"
    ${py} ./setup.py bdist_wheel --plat-name linux_"${ARCH}" --python-tag "${tag}"
    echo "prepare resource"
}

function package()
{
  bash "${CUR_PATH}"/package.sh "$1"
}

function build_ops()
{
    platform=$1
    local py=$2
    echo "perpare ops build"
    cd "${ROOT_PATH}/ops"
    dos2unix build.sh
    bash build.sh $platform "${py}"
    echo "build $platform ops success"
}

function get_nlohmann() {
  if [ ! -e "${ROOT_PATH}/opensource/json" ]; then
    mkdir -p "${ROOT_PATH}/opensource"
    cd "${ROOT_PATH}/opensource"
    wget https://github.com/nlohmann/json/archive/v3.11.3.tar.gz
    tar -xf v3.11.3.tar.gz
    mv json-3.11.3 json
  fi
}

function main()
{
    clean
    get_nlohmann
    build_wheel_package python3.11 "${PY311_VER}"
    build_ops Ascend310P python3.11
    build_ops Ascend910B python3.11

    package "${PY311_VER}"
}

main