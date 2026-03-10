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
#ifndef ATB_SPEED_BASE_MODEL_H
#define ATB_SPEED_BASE_MODEL_H
#include <acl/acl.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <atb_speed/utils/timer.h>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <map>
#include <atomic>
#include <set>
#include <nlohmann/json.hpp>
#include "atb_speed/utils/operation_util.h"


namespace atb_speed {
class Model {
public:
    using ReshapeFunc = std::function<void(const atb::Dims &oldDims, atb::Dims &newDims)>;
    using GetWorkspaceFunc = std::function<void*(uint64_t bufferSize)>;
    using CreateTensorFromTensorDescFunc = std::function<atb::Tensor(const atb::TensorDesc &tensorDesc)>;
    using Task = std::function<int()>;
    using RunTaskFunc = std::function<void(const std::string &taskName, Task task)>;
    enum class TensorType {
        INTERMEDIATE_TENSOR = 0,
        NOT_INTERMEDIATE_TENSOR,
    };

    struct Node {
        std::shared_ptr<atb::Operation> operation;
        std::vector<atb::Tensor *> inTensors;
        std::vector<atb::Tensor *> outTensors;
        atb::VariantPack variantPack;
        // std::vector<torch::Tensor> torchTensors;
        atb::SVector<ReshapeFunc> inTensorReshapeFuncs;
        atb::SVector<TensorType> inTensorTypes;
        atb::SVector<TensorType> outTensorTypes;
        uint64_t workspaceSize = 0;
        void *workspace = nullptr;
    };

    struct Graph {
        std::vector<atb::Tensor> weightTensors;
        std::vector<atb::Tensor> kCacheTensors;
        std::vector<atb::Tensor> vCacheTensors;
        std::vector<atb::Tensor> inTensors;
        std::vector<atb::Tensor> outTensors;
        std::vector<atb::Tensor> internalTensors;
        std::vector<Node> nodes;
        std::map<uint64_t, std::set<atb::Tensor *>> maxNodeIdTensorMap;
        void Init();
        std::string ToString() const;

    private:
        void InitTensorType();
        bool IsInternalTensor(const atb::Tensor *tensor);
        void InitTensorMaxNodeMap();
        void FindInTensorInMaxNodeId(atb::Tensor& inTensor, uint64_t& maxNodeId, uint64_t& dependNodeCount);
    };

    Model(const std::string &modelName, const std::string &param);
    virtual ~Model();
    int64_t Init(GetWorkspaceFunc getWorkSpaceFunc, CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc,
            RunTaskFunc runTaskFunc = nullptr);

    virtual uint32_t GetInputNum() const = 0;
    virtual uint32_t GetOutputNum() const = 0;
    virtual atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                   std::vector<atb::TensorDesc> &outTensorDescs) = 0;

    int64_t SetWeight(const std::vector<atb::Tensor> &weightTensors);
    int64_t SetKVCache(const std::vector<atb::Tensor> &kCacheTensors, const std::vector<atb::Tensor> &vCacheTensors);
    atb::Status Execute(atb::Context *context, std::vector<atb::Tensor> &inTensors,
        std::vector<atb::Tensor> &outTensors, const std::string &param);

protected:
    virtual int64_t BuildGraph() = 0;
    virtual atb::Status ParseParam(const std::string &param);
    virtual atb::Status BindParamHostTensor(uint32_t nodeId);

protected:
    bool IsTensorDescEqual(const atb::TensorDesc &tensorDesc, const atb::Tensor &atbTensor) const;
    void ExecuteNodeView(int nodeId);
    void BuildNodeVariantPack(int nodeId);
    atb::Status ExecuteNode(int nodeId);
    void ThreadProcessTask();
    atb::Status ExecutePlanSync(int nodeId);
    void ExecutePlanAsync(int nodeId);
    void PushTask(int nodeId);
    int PopTask();
    void WaitAsyncPlanExecuteFinish();
    std::string GetSaveTensorDir() const;
    void ClearInternalTensors();
    atb::Tensor MallocInternalTensor(atb::Tensor* outTensor, size_t nodeId, size_t outTensorId,
        const atb::TensorDesc &tensorDesc);
    void FreeInternalTensor(const void *tensorDeviceData);
    void GetModelTensorNameList(nlohmann::json &modelJson,
        std::map<atb::Tensor *, std::string> &tensorNameMap);
    void GetNodeTopoInfo(nlohmann::json &nodeJson, const Node &opNode,
        const std::map<atb::Tensor *, std::string> tensorNameMap) const;
    std::string GetModelTopoInfo();

protected:
    GetWorkspaceFunc getWorkSpaceFunc_;
    CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc_;
    RunTaskFunc runTaskFunc_ = nullptr;
    std::string modelName_;
    std::string param_;
    Graph graph_;

    uint64_t executeCount_ = 0;
    atb::Context *context_ = nullptr;
    Timer timer_;

    bool isUsePlanExecuteAsync_ = false;
    std::queue<int> taskQueue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::thread taskProcessThread_;
    std::atomic_bool allTaskFinish_;
    int32_t currentDevId_ = 0;
    std::vector<std::pair<atb::Tensor, bool>> internalTensors_;
    std::vector<atb::Tensor*> nodeOutTensors_;

    // Max layer num
    static const int MAX_LAYER_NUM = 1024;
    // Max length of param string
    const size_t MAX_PARAM_STRING_LENGTH = 20000;
    // Max value of tokenOffset, seqLen and qLen
    const int MAX_PARAM_VALUE = 600000;
    // Max batch size
    const size_t MAX_BATCH_SIZE = 10240;
};

// Param Type Size
const size_t PACK_QUANT_TYPE_LENGTH = 2;
const size_t LINEAR_TYPE_LENGTH = 7;
int CheckPositive(const int &intParam);
void CheckLinearParamsSufficient(const std::vector<std::vector<int>> &linearParam, \
    size_t numHiddenLayers, size_t thershold);
void CheckPackQuantParamsSufficient(const std::vector<std::vector<int>> &packQuantType, size_t numHiddenLayers);
void CheckLinearPackParamsSufficient(const std::vector<std::vector<int>> &linearPackType, size_t numHiddenLayers);
} // namespace atb_speed
#endif