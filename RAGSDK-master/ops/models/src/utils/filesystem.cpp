/*
 * -------------------------------------------------------------------------
 *  This file is part of the RAGSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * RAGSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
*/
#include "atb_speed/utils/filesystem.h"
#include <fstream>
#include <dirent.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <unistd.h>
namespace atb_speed {
constexpr size_t MAX_PATH_LEN = 256;

void FileSystem::GetDirChildItems(const std::string &dirPath, std::vector<std::string> &itemPaths)
{
    GetDirChildItemsImpl(dirPath, true, true, itemPaths);
}

void FileSystem::GetDirChildFiles(const std::string &dirPath, std::vector<std::string> &filePaths)
{
    GetDirChildItemsImpl(dirPath, true, false, filePaths);
}

void FileSystem::GetDirChildDirs(const std::string &dirPath, std::vector<std::string> &dirPaths)
{
    GetDirChildItemsImpl(dirPath, false, true, dirPaths);
}

bool FileSystem::Exists(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }
    return true;
}

bool FileSystem::IsDir(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }

    return S_ISDIR(st.st_mode);
}

std::string FileSystem::Join(const std::vector<std::string> &paths)
{
    std::string retPath;
    for (const auto &path : paths) {
        if (retPath.empty()) {
            retPath.append(path);
        } else {
            retPath.append("/" + path);
        }
    }
    return retPath;
}

int64_t FileSystem::FileSize(const std::string &filePath)
{
    struct stat st;
    if (stat(filePath.c_str(), &st) < 0) {
        return -1;
    }
    return st.st_size;
}

std::string FileSystem::BaseName(const std::string &filePath)
{
    std::string fileName;
    const char *str = strrchr(filePath.c_str(), '/');
    if (str) {
        fileName = str + 1;
    } else {
        fileName = filePath;
    }
    return fileName;
}

std::string FileSystem::DirName(const std::string &path)
{
    int32_t idx = static_cast<int32_t>(path.size()) - 1;
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    std::string sub = path.substr(0, idx);
    const char *str = strrchr(sub.c_str(), '/');
    if (str == nullptr) {
        return ".";
    }
    idx = str - sub.c_str() - 1;
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    if (idx < 0) {
        return "/";
    }
    return path.substr(0, idx + 1);
}

bool FileSystem::IsOwnerSame(const std::string& path)
{
    struct stat file_info;

    // 获取文件信息
    if (stat(path.c_str(), &file_info) != 0) {
        std::cout<< "stat file:"<< path << " failed";
        return false;
    }

    // 校验文件owner
    if (file_info.st_uid != geteuid()) {
        std::cout<< "file:"<< path << " uid is not same with euid";
        return false;
    }
    return true;
}

bool FileSystem::IsPathValid(const std::string& path)
{
    struct stat buf;

    if (!IsOwnerSame(path)) {
        return false;
    }

    // 检查路径本身是否是符号链接
    if (lstat(path.c_str(), &buf) == -1) {
        // 如果路径不存在或无权限访问，返回true
        return true;
    }

    if (S_ISLNK(buf.st_mode)) {
        return false;
    }

    // 分解路径，检查父目录中的每一部分
    std::vector<std::string> parts;
    size_t pos = 0;
    size_t last = 0;
    while ((pos = path.find('/', last)) != std::string::npos) {
        if (pos != last) {
            std::string part = path.substr(last, pos - last);
            parts.push_back(part);
        }
        last = pos + 1;
    }

    std::string currentPath = "/";
    for (const std::string& part : parts) {
        currentPath += part + "/";
        if (lstat(currentPath.c_str(), &buf) == -1) {
            // 如果某部分不存在或无权限访问，跳过
            continue;
        }
        if (S_ISLNK(buf.st_mode)) {
            return false;
        }
    }

    return true;
}


bool FileSystem::ReadFile(const std::string &filePath, char *buffer, uint64_t bufferSize)
{
    if (!IsPathValid(filePath)) {
        std::cout<< "path:"<<filePath<< " is invalid";
        return false;
    }

    std::ifstream fd(filePath, std::ios::binary);
    if (!fd.is_open()) {
        return false;
    }
    fd.read(buffer, bufferSize);
    return true;
}

void FileSystem::DeleteFile(const std::string &filePath) { remove(filePath.c_str()); }

bool FileSystem::Rename(const std::string &filePath, const std::string &newFilePath)
{
    int ret = rename(filePath.c_str(), newFilePath.c_str());
    return ret == 0;
}

bool FileSystem::MakeDir(const std::string &dirPath, int mode)
{
    int ret = mkdir(dirPath.c_str(), mode);
    return ret == 0;
}

bool FileSystem::Makedirs(const std::string &dirPath, const mode_t mode)
{
    size_t offset = 0;
    size_t pathLen = dirPath.size();
    do {
        const char *str = strchr(dirPath.c_str() + offset, '/');
        offset = (str == nullptr) ? pathLen : str - dirPath.c_str() + 1;
        std::string curPath = dirPath.substr(0, offset);
        if (!Exists(curPath)) {
            if (!MakeDir(curPath, mode)) {
                return false;
            }
        }
    } while (offset != pathLen);
    return true;
}

void FileSystem::GetDirChildItemsImpl(const std::string &dirPath, bool matchFile, bool matchDir,
                                      std::vector<std::string> &itemPaths)
{
    struct stat st;
    if (stat(dirPath.c_str(), &st) < 0 || !S_ISDIR(st.st_mode)) {
        return;
    }

    DIR *dirHandle = opendir(dirPath.c_str());
    if (dirHandle == nullptr) {
        return;
    }

    struct dirent *dp = nullptr;
    while ((dp = readdir(dirHandle)) != nullptr) {
        const int parentDirNameLen = 2;
        if ((strncmp(dp->d_name, ".", 1) == 0) || (strncmp(dp->d_name, "..", parentDirNameLen) == 0)) {
            continue;
        }
        std::string itemPath = dirPath + "/" + dp->d_name;
        stat(itemPath.c_str(), &st);
        if ((matchDir && S_ISDIR(st.st_mode)) || (matchFile && S_ISREG(st.st_mode))) {
            itemPaths.push_back(itemPath.c_str());
        }
    }

    closedir(dirHandle);
}
} // namespace atb_speed