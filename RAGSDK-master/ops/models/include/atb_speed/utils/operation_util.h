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
#ifndef ATB_SPEED_UTILS_OPERATION_H
#define ATB_SPEED_UTILS_OPERATION_H
#include <atb/atb_infer.h>

#define CREATE_OPERATION(param, operation) \
    do { \
        atb::Status atbStatus = atb::CreateOperation(param, operation); \
        if (atbStatus != atb::NO_ERROR) { \
            return atbStatus; \
        } \
    } while (0)

#define CHECK_OPERATION_STATUS_RETURN(atbStatus) \
    do { \
        if ((atbStatus) != atb::NO_ERROR) { \
            return (atbStatus); \
        } \
    } while (0)

#define CHECK_PARAM_LT(param, threshold) \
    do { \
        if ((param) >= (threshold)) { \
            ATB_LOG(ERROR) << "param should be less than " << (threshold) << ", please check"; \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define CHECK_PARAM_GT(param, threshold) \
    do { \
        if ((param) <= (threshold)) { \
            ATB_LOG(ERROR) << "param should be greater than " << (threshold) << ", please check"; \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define CHECK_PARAM_GE(param, threshold) \
    do { \
        if ((param) < (threshold)) { \
            ATB_LOG(ERROR) << "param should be greater than or equal to " << (threshold) << ", please check"; \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define CHECK_PARAM_NE(param, value) \
    do { \
        if ((param) == (value)) { \
            ATB_LOG(ERROR) << "param should not be equal to " << (value) << ", please check"; \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define CHECK_TENSORDESC_DIMNUM_VALID(dimNum) \
    do { \
        if ((dimNum) > (8) || (dimNum) == (0) ) { \
            ATB_LOG(ERROR) << "dimNum should be less or equal to 8 and cannot be 0, please check"; \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#endif