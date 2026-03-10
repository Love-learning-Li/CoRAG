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

#ifndef MODEL_MODEL_TORCH_H
#define MODEL_MODEL_TORCH_H
#include <memory>
#include <string>
#include <vector>

#include <torch/custom_class.h>
#include <torch/script.h>

#include "atb_speed/base/model.h"
#include "atb_speed/utils/timer.h"

namespace atb_speed {
class ModelTorch : public torch::CustomClassHolder {
public:
    explicit ModelTorch(std::string modelName);
    ~ModelTorch() override;
    int64_t SetParam(std::string param);
    int64_t SetWeight(std::vector<torch::Tensor> atWeightTensors);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors, std::string param);
    c10::intrusive_ptr<ModelTorch> clone() const { return c10::make_intrusive<ModelTorch>(modelName_); }

private:
    int64_t AtTensor2Tensor(std::vector<torch::Tensor> &atTensors, std::vector<atb::Tensor> &opsTensors) const;
    int64_t ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                        const std::string &param);
    std::string GetSaveTensorDir() const;
    void* GetWorkSpace(const uint64_t bufferSize) const;
    atb::Tensor CreateInternalTensorFromDesc(const atb::TensorDesc &tensorDesc);
    void RunTask(std::string taskName, std::function<int()> task);
private:
    std::string modelName_;
    std::shared_ptr<atb_speed::Model> model_;
    uint64_t executeCount_ = 0;
    uint64_t modelId_ = 0;
    std::shared_ptr<atb::Context> context_;
    std::vector<torch::Tensor> atInternalTensors_;
    const size_t maxParamLength_ = 20000;
};
}
#endif