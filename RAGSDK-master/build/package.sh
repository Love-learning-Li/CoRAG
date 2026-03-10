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
CUR_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=$(readlink -f "${CUR_PATH}"/..)
PKG_DIR=mindxsdk-mxrag

VERSION_FILE="${ROOT_PATH}"/ci/config/config.ini
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed -n 's/^version:[[:space:]]*//p' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="7.0.RC1"
  fi

}

get_version

{
  echo "MindX SDK mxrag:${VERSION}"
  echo "Plat: linux $(uname -m)"
} >> "$ROOT_PATH"/version.info


function package()
{
    py_version=$1

    if [ -z "$py_version" ]; then
        echo "python version invalid"
        exit 1
    fi

    cd "${ROOT_PATH}"/output/
    # package
    cp -rf "${ROOT_PATH}"/dist/mx_rag*"${py_version}"*.whl .

    mv "${ROOT_PATH}"/version.info .
    cp -rf "${ROOT_PATH}"/requirements.txt .
    cp -rf "${ROOT_PATH}"/script .

    mkdir -p ./ops/310P
    mkdir -p ./ops/910B
    mkdir -p ./ops/A3
    cp -rf "${ROOT_PATH}"/ops/Ascend910B/lib ./ops/910B
    cp -rf "${ROOT_PATH}"/ops/Ascend910B/lib ./ops/A3
    cp -rf "${ROOT_PATH}"/ops/Ascend310P/lib ./ops/310P

    mkdir -p ./ops/transformer_adapter
    cp -rf "${ROOT_PATH}"/ops/transformer_adapter/* ./ops/transformer_adapter


    cp "${ROOT_PATH}"/build/install.sh .
    cp "${ROOT_PATH}"/build/help.info .

    pkg_version=$(sed -n '1p' version.info |awk -F ':' '{print $2}')

    if [ -z "$pkg_version" ]; then
       echo "get pkg_version failed"
       exit 1
    fi
    sed -i "s/%{PACKAGE_VERSION}%/$pkg_version/g" install.sh
    pkg_arch=$(uname -m)
    sed -i "s/%{PACKAGE_ARCH}%/$pkg_arch/g" install.sh

    #将所有目录设置为750，特殊目录单独处理
    find ./ -type d -exec chmod 750 {} \;
    #将所有文件设置640，特殊文件单独处理
    find ./ -type f -exec chmod 640 {} \;

    find ./  \( -name "*.sh" -o -name "*.run"  -o -name "*.so" \)  -exec  chmod 550 {} \;

    rm -f .gitkeep
}

function patch_makeself() {
    cd "${ROOT_PATH}/opensource" || exit

    if [ ! -e "${ROOT_PATH}/opensource/makeself" ]; then
        git clone https://gitcode.com/cann-src-third-party/makeself.git
        cd makeself
        unzip makeself-release-2.5.0.zip
        cd makeself-release-2.5.0
        patch -p1 < ../*.patch
        rm -f ../*.patch
    fi
}

function build_run() {
    cd "${ROOT_PATH}/output" || exit

    bash ../opensource/makeself/makeself-release-2.5.0/makeself.sh --chown --nomd5 --sha256 --nocrc \
            --header ../opensource/makeself/makeself-release-2.5.0/makeself-header.sh \
            --help-header help.info \
            --packaging-date "" \
            --tar-extra '--owner=root --group=root' \
            ${ROOT_PATH}/output ./Ascend-${PKG_DIR}_${VERSION}_linux-${ARCH}.run "ASCEND RAG SDK RUN PACKAGE" ./install.sh

}

function main()
{
    package "$1"
    patch_makeself
    build_run
}

main "$@"