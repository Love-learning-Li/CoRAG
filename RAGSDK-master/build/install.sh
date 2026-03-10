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


readonly PACKAGE_VERSION=%{PACKAGE_VERSION}%
readonly INSTALL_DIRECTORY=mxRag-"${PACKAGE_VERSION}"
readonly PACKAGE_ARCH=%{PACKAGE_ARCH}%

# 自定义变量
LOG_PATH="$HOME/log/mxRag"
log_file="$LOG_PATH/deployment.log" #日志文件名

if [[ "$UID" == "0" ]]; then
    install_path="/usr/local/Ascend"
else
    install_path="${HOME}/Ascend"
fi

arch=$(uname -m)
PACKAGE_LOG_NAME=ragsdk
LOG_SIZE_THRESHOLD=$((10*1024*1024))
OWNED_CHECK_PATH_WHITELIST=("/var" "/var/log")
declare -A param_dict=()               # 参数个数统计

install_username=$(id -nu)
install_usergroup=$(id -ng)

#标识符
install_flag=n              #
install_path_flag=n
install_whitelist_flag=n
install_whitelist="whl,operator"
chip_type=310P
upgrade_flag=n
install_for_all=n
quiet_flag=n
log_init_flag=n

MAX_LEN_OF_PATH=1024
MIN_LEN_OF_PATH=0

current_uid=$(id -u)
readonly current_uid

function print() {
    # 将关键信息打印到屏幕上
    echo "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [user: $USER_NAME] [$ip_n] [$1] $2"
}

function error_exit() {
    local force_exit=$2
    if [[ "$force_exit" == "T" ]]; then
      print "ERROR" "exiting due to $1"
      log "ERROR"  "run failed on $1"
      exit 1
    fi
    log "ERROR"  "run failed on $1" "y"
    exit 1
}

function safe_path()
{
    local path=$1
    local force_exit=$2
    local allow_group_write=$3
    check_path "$path"
    while [[ ! -e "${path}" ]]; do
        path=$(dirname "${path}")
    done
    path=$(realpath -s "$path")
    local cur=${path}
    while true; do
        if [[ "${cur}" == '/' ]]; then
           break
        fi
        is_safe_owned "${cur}" "$force_exit" "$allow_group_write"
        cur=$(dirname "${cur}")
    done
}

function is_skip_owned_check() {
   local path=$1
   for whitelist_path in "${OWNED_CHECK_PATH_WHITELIST[@]}"; do
     if [[ "${path}" == "${whitelist_path}" ]]; then
        return 0
     fi
   done
   return 1
}

function check_path() {
    local path=$1
    [[ ${#path} -gt ${MAX_LEN_OF_PATH} ]] || [[ ${#path} -le ${MIN_LEN_OF_PATH} ]] && print "ERROR" "${path} length is invalid, either exceeding ${MAX_LEN_OF_PATH} or less than ${MIN_LEN_OF_PATH}, exiting" "$force_exit" && exit 1
    [[ $(echo "$path" | wc -l) -gt 1 ]]  && print "ERROR" "${path} contains newline characters, exiting"  && exit 1
    [[ -n $(echo "$path" | grep -Ev '^[/~][-_.0-9a-zA-Z/]*$') ]]  && print "ERROR" "${path} must start with '/' or '~' and characters only can contain '-_.0-9a-zA-Z/', exiting"  && exit 1
    if ! echo "$path" | grep -qv "\\.\\."; then
        print "ERROR" "${path} contains .. , exiting"  && exit 1
    fi
}

function is_safe_owned()
{
    local path=$1
    local force_exit=$2
    local allow_group_write=$3

    # 校验路径若在白名单，则跳过检验
    if is_skip_owned_check "${path}"; then
        return 0
    fi

    check_path "$path"

    if [[ -L "${path}" ]]; then
        error_exit "The $path is a soft link! exiting" "$force_exit"
    fi
    local user_id
    user_id=$(stat -c %u "${path}")
    local group_id
    group_id=$(stat -c %g "${path}")
    if [[ -z "${user_id}" ]] || [[ -z "${group_id}" ]]; then
        error_exit "user or group not exist, exiting" "$force_exit"
    fi
    if [[ "$(stat -c '%A' "${path}"|cut -c9)" == w ]]; then
        error_exit "file $path does not meet with security rules other write, other users have write permission. exiting" "$force_exit"
    fi
    if [[ "$allow_group_write" != "T" ]] && [[ "$(stat -c '%A' "${path}"|cut -c6)" == w ]]; then
        error_exit "file $path does not meet with security rules group write, group has write permission. exiting" "$force_exit"
    fi
    if [[ "${user_id}" != "0" ]] && [[ "${user_id}" != "${current_uid}" ]]; then
        error_exit "The $path is not owned by root or current user, exiting" "$force_exit"
    fi
    return 0
}

function safe_path_exit()
{
    local path=$1

    check_path "$path"

    while [[ ! -e "${path}" ]]; do
        path=$(dirname "${path}")
    done
    path=$(realpath -s "$path")
    local cur=${path}
    while true; do
        if [[ "${cur}" == '/' ]]; then
           break
        fi
        is_safe_owned_exit "${cur}"
        cur=$(dirname "${cur}")
    done
}

function is_safe_owned_exit()
{
    local path=$1

    # 校验路径若在白名单，则跳过检验
    if is_skip_owned_check "${path}"; then
        return 0
    fi

    check_path "$path"

    if [[ -L "${path}" ]]; then
        print "ERROR" "The $path is a soft link! exiting" && exit 1
    fi
    local user_id
    user_id=$(stat -c %u "${path}")
    local group_id
    group_id=$(stat -c %g "${path}")
    if [[ -z "${user_id}" ]] || [[ -z "${group_id}" ]]; then
        print "ERROR" "user or group not exist, exiting" && exit 1
    fi
    if [[ "$(stat -c '%A' "${path}"|cut -c9)" == w ]]; then
        print "ERROR" "file $path does not meet with security rules other write, other users have write permission, exiting" && exit 1
    fi
    if [[ "${user_id}" != "0" ]] && [[ "${user_id}" != "${current_uid}" ]]; then
        print "ERROR" "The $path is not owned by root or current user, exiting" && exit 1
    fi
    return 0
}

function safe_change_mode() {
    local mode=$1
    local path=$2
    local allow_group_write=$3
    safe_path "$path" F "$allow_group_write"
    chmod "${mode}" "${path}"
}

readonly USER_NAME=$(whoami)
readonly WHO_PATH=$(which who)
readonly CUT_PATH=$(which cut)
ip_n=$(${WHO_PATH} -m | ${CUT_PATH} -d '(' -f 2 | ${CUT_PATH} -d ')' -f 1)
if [[ "${ip_n}" = "" ]]; then
    ip_n="localhost"
fi
readonly ip_n

function log() {
    # 将日志打印到文件中n
    if [[ "$log_file" = "" ]] || [[ "$quiet_flag" = n ]] || [[ "$3" = "y" ]]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [user: $USER_NAME] [$ip_n] [$1] $2"
    fi
    if [[ -f "$log_file" ]]; then
        log_check "$log_file"
        safe_path_exit "$log_file"
        if ! echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [user: $USER_NAME] [$ip_n] [$1] $2" >>"$log_file"
        then
          print "ERROR" "can not write log, exiting!"
          exit 1
        fi
    fi
}

###  公用函数
function print_usage() {
    echo "Please input this command for more help: --help"
    error_exit "please check for help"
}

### 脚本入参的相关处理函数
function check_script_args() {
    # 检测脚本参数的组合关系
    ######################  check params confilct ###################
    if [[ $# -lt 3 ]]; then
        print_usage
    fi
    # 重复参数检查
    for key in "${!param_dict[@]}";do
      if [[ "${param_dict[$key]}" -gt 1 ]]; then
          log "ERROR" "parameter error! $key is repeat"
          error_exit "param repeat"
      fi
    done

    if [[ "${PACKAGE_ARCH}" != "${arch}" ]];then
        log "ERROR" "the package is ${PACKAGE_ARCH} but current is ${arch}, exit."
        error_exit "error"
    fi

    if [[ "$install_flag" != "y" ]] && [[ "$upgrade_flag" != "y" ]]; then
        log "ERROR" "parameter error ! Mode is neither install, upgrade."
        error_exit "error"
    fi

    if [[ "$install_path_flag" = "y" ]]; then
        if [[ "$install_flag" = "n" ]] && [[ "$upgrade_flag" = "n" ]]; then
            log "ERROR" "Unsupported separate 'install-path' used independently"
            error_exit "error"
        fi
    fi
}

function make_file() {
    safe_path "${1}" T T
    if touch "${1}" 2>/dev/null
    then
        log "INFO" "create $1 success" "y"
    else
        log "ERROR" "create $1 failed"  "y"
        exit 1
    fi
    safe_change_mode 640 "${1}" T
}

function log_init() {
    if [[ "${log_init_flag}" = "y" ]];then
        return
    fi
    # 日志模块初始化
    # 判断输入的安装路径路径是否存在，不存在则创建
    mkdir -p "${LOG_PATH}" 2>/dev/null
    if [[ ! -f "$log_file" ]]; then
        make_file "$log_file"
    fi
    chmod 750 "${LOG_PATH}"
    # 兼容老包升级
    chmod 640 "${log_file}"

    log "INFO" "LogFile ${log_file}"

    log_init_flag=y
}

function rotate_log() {
    safe_path_exit "$log_file"
    if [[ -f "$LOG_PATH/deployment.log.bak" ]] && [[ "$UID" == "0" ]]; then
      chown -h root:root "$LOG_PATH/deployment.log.bak"
    fi
    safe_path_exit "$LOG_PATH/deployment.log.bak"
    mv -f "$log_file" "$LOG_PATH/deployment.log.bak"
    touch "$log_file" 2>/dev/null
    safe_change_mode 440 "$LOG_PATH/deployment.log.bak" "T"
    safe_change_mode 640 "$log_file" "T"
}

function log_check() {
    local log_size=$(stat -c%s "$log_file" 2>/dev/null) || log_size=0
    if [[ "${log_size}" -ge "${LOG_SIZE_THRESHOLD}" ]];then
        rotate_log
    fi
}

log_init
safe_path_exit "$LOG_PATH"

# 解析脚本自身的参数
function parse_script_args() {
    log "INFO" "start to run"
    local all_para_len="$*"
    if [[ ${#all_para_len} -gt 1024 ]]; then
        error_exit "The total length of the parameter is too long"
    fi
    local num=0
    while true; do
        if [[ "$1" == "" ]]; then
            break
        fi
        if [[ "${1: 0: 2}" == "--" ]]; then
            num=$((num + 1))
        fi
        if [[ $num -gt 2 ]]; then
            break
        fi
        shift 1
    done
    while true; do
        case "$1" in
        --help | -h)
            print_usage
            ;;
        --version)
            echo "${PACKAGE_LOG_NAME} ${PACKAGE_VERSION}"
            exit 0
            ;;
        --install)
            install_flag=y
            ((param_dict["install"]++)) || true
            shift
            ;;
        --whitelist=*)
            install_whitelist=$(echo "$1" | cut -d"=" -f2)
            install_whitelist_flag=y
            ((param_dict["whitelist"]++)) || true
            shift
            ;;
          --platform=*)
            chip_type=$(echo "$1" | cut -d"=" -f2)
            ((param_dict["platform"]++)) || true
            shift
            ;;
        --install-path=*)
            # 去除指定安装目录后所有的 "/"
            install_path=$(echo "$1" | cut -d"=" -f2 | sed "s/\/*$//g")
            install_path_flag=y
            safe_path "$install_path"
            local home_dir="$(echo ~)"
            install_path=$(echo "$install_path" | sed -e "s#^~#${home_dir}#")
            ((param_dict["install-path"]++)) || true
            shift
            ;;
        --upgrade)
            upgrade_flag=y
            ((param_dict["upgrade"]++)) || true
            shift
            ;;
        --quiet)
            quiet_flag=y
            ((param_dict["quiet"]++)) || true
            shift
            ;;
        --check)
            print "INFO" "Check successfully, exit with 0"
            exit 0
            ;;
        -*)
            print "ERROR" "Unsupported parameters: $1"
            print_usage
            exit 1
            ;;
        *)
            if [[ "x$1" != "x" ]]; then
                print "ERROR" "Unsupported parameters: $1"
                print_usage
                exit 1
            fi
            break
            ;;
        esac
    done
}

function set_env() {
  sed -i "s!export RAG_SDK_HOME=.*!export RAG_SDK_HOME="${install_path}/mxRag"!g" "${install_path}"/mxRag/script/set_env.sh
}

check_python_version()
{
  python_version_minor=$(python3 --version | awk '{print $2}')

  if  [[ -n $(echo "$python_version_minor" | grep -Ev '^3\.11(\.[0-9]+)?$') ]]; then
      log "ERROR" "RAG SDK only support python3.11" "y"
      exit 1
  fi
}

function install_whl() {
  check_python_version

  whl_file_name=$(find ./ -maxdepth 1 -type f -name 'mx_rag*py311*.whl')
  if test x"$quiet_flag" = xn; then
      log "INFO" "Begin to install wheel package(${whl_file_name##*/})."
  fi

  if [[ -f "$whl_file_name" ]];then
      if test x"$quiet_flag" = xy; then
          python3 -m pip install --no-index --upgrade --force-reinstall --no-dependencies "${whl_file_name##*/}" --user > /dev/null 2>&1
      else
          python3 -m pip install --no-index --upgrade --force-reinstall --no-dependencies "${whl_file_name##*/}" --user
      fi
      if test $? -ne 0; then
          log "ERROR" "Install wheel package failed."
          exit 1
      else
            if test x"$quiet_flag" = xn; then
                log "INFO" "Install wheel package successfully."
            fi
      fi
  else
      log "WARNING" "There is no wheel package to install."
  fi
}

function install_ops() {
    # 校验用户输入的platform是否与当前环境的平台一致
    if [[ "$chip_type" == "310P" ]]; then
        check_ret=$(npu-smi info | awk '{print $3}' | grep 310P | sed -n 1p)
    elif [[ "$chip_type" == "910B" ]]; then
        check_ret=$(npu-smi info | awk '{print $3}' | grep 910B | sed -n 1p)
    elif [[ "$chip_type" == "A3" ]]; then
        check_ret=$(npu-smi info | awk '{print $3}' | grep Ascend910 | sed -n 1p)
    fi

    if [[ -z "$check_ret" ]]; then
        log "WARNING" "Platform mismatch for $chip_type, please check it."
    fi

    mkdir -p "${install_path}/${INSTALL_DIRECTORY}"/ops

    if [[ -d "ops/$chip_type" ]]; then
        cp -rf ops/"$chip_type"/* "${install_path}/${INSTALL_DIRECTORY}"/ops
    else
        log "ERROR" "Invalid platform: $chip_type" "y"
        exit 1;
    fi

    cp -rf ops/transformer_adapter "${install_path}/${INSTALL_DIRECTORY}"/ops
}

function modify_file_permission()
{
  local path="$1"
  find "$path/" -type d -exec chmod 750 {} +
  find "$path/" -type f -exec chmod 640 {} +
  find "$path/" -maxdepth 1 -type d  -name lib -exec chmod 550 {} +
  find "$path/" -type f -name 'version.info' -exec chmod 440 {} +
  find "$path/" -type f -name '*.so' -exec chmod 440 {} +
  find "$path/" -type f -name '*.so.*' -exec chmod 440 {} +

  find "$path/" -type f -name '*.sh' -exec chmod 500 {} +
  find "$path/" -type f -name '*.py' -exec chmod 550 {} +

  find "$path/" -perm /u+x -type f -exec chmod 500 {} +
}

function install_process() {
    log "INFO" "install start"
    log "INFO" "The install path is ${install_path} !"

    if [[ -e "$install_path"/mxRag/script/uninstall.sh ]]; then
        log "ERROR" "can not install twice, you have already installed RAG SDK." "y"
        exit 1
    fi

    if ! mkdir -p "${install_path}/${INSTALL_DIRECTORY}"; then
        log "ERROR" "create install dir failed"
        exit 1
    fi

    if [[ "$install_whitelist" =~ "whl" ]]; then
        install_whl
    fi

    if [[ "$install_whitelist" =~ "operator" ]]; then
        install_ops
    fi

    cp ./version.info "${install_path}/${INSTALL_DIRECTORY}"
    cp ./requirements.txt "${install_path}/${INSTALL_DIRECTORY}"
    cp -r ./script "${install_path}/${INSTALL_DIRECTORY}"

    if ! cd "${install_path}"; then
        log "ERROR" "cd to ${install_path} failed." "y"
        exit 1
    fi

    if [[ -e mxRag ]] && [[ ! -L mxRag ]];then
        rm "mxRag"
        ln -s "${INSTALL_DIRECTORY}" "mxRag"
    else
        ln -s "${INSTALL_DIRECTORY}" "mxRag"
    fi

    set_env "${install_path}"

    modify_file_permission "${install_path}/${INSTALL_DIRECTORY}"

    log "INFO" "Install package successfully" "y"
}

function upgrade_process() {
    log "INFO" "upgrade start"

    # check whether the old version is exist
    if [[ ! -e "$install_path"/mxRag/script/uninstall.sh ]]; then
        log "ERROR" "There is no RAG SDK installed in cur install path, please check it." "y"
        exit 1
    fi

    local doupgrade=n
    if test x"$quiet_flag" = xn; then
       read -t 60 -n1 -r -p "Do you want to upgrade to a newer version provided by this package and the old version will be removed? [Y/N]" answer
       case "${answer}" in
            Y|y)
                doupgrade=y
                ;;
            *)
                doupgrade=n
                ;;
       esac
    else
      doupgrade=y
    fi
    if [[ x"$doupgrade" == "xn" ]]; then
        log  "WARNING" "user reject to upgrade, nothing changed" "y"
        exit 1
    else
        "$install_path"/mxRag/script/uninstall.sh; res=$?;
        if test "$res" -ne 0; then
            log "ERROR" "uninstall old package failed" "y"
            exit 1
        fi
        log "INFO" "older version was removed. Installing new version..." "y"
    fi

    install_process

    log "INFO" "Upgrade package successfully" "y"
}

function process() {
    if [[ "$install_flag" = "y" ]]; then
        install_process
    elif [[ "$upgrade_flag" = "y" ]]; then
        upgrade_process
    fi
}

# 程序开始
function main() {
    parse_script_args "$@"
    check_script_args "$@"
    log_check
    process
}

main "$@"
