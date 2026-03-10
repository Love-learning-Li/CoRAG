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

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly RUN_PKG_PATH="${CUR_DIR}/../.."
readonly PRESMOKE_DIR="/home/ragSDK/preSmokeTestFiles"

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 安装依赖
apt-get install -y  libpq-dev 
pip3 install uvicorn

# 安装rag
cp ${RUN_PKG_PATH}/Ascend-mindxsdk-mxrag_*_linux-aarch64.run ${PRESMOKE_DIR}/pkg/
cd ${PRESMOKE_DIR}/pkg/
chmod +x *.run
./Ascend-mindxsdk-mxrag_*_linux-aarch64.run --install --install-path=/usr/local/Ascend --platform=910B
pip3 install -r  /usr/local/Ascend/mxRag/requirements.txt

# 起模型和embed服务
cd ${PRESMOKE_DIR}
python3 emb_model_service.py > /dev/null 2>&1 &
API_PID=$!
sleep 3
# 执行demo
export MX_INDEX_FINALIZE=0
python3  ragsdk-demo.py
kill $API_PID 2>/dev/null

