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
#ifndef BGE_LARGE_ZH_FLASH_ATTENTION_LAYER_H
#define BGE_LARGE_ZH_FLASH_ATTENTION_LAYER_H

#include <atb/atb_infer.h>
#include <atb/svector.h>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace bge_large {
struct FlashAttentionLayerParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string model = "bge_large";
};

void from_json(const nlohmann::json &paramJson, FlashAttentionLayerParam &param);

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation);

atb::Operation *CreateFlashAttentionLayer(const nlohmann::json &paramJson);

class FlashAttentionLayerBase : public HostTensorBinder {
public:
    FlashAttentionLayerBase();
    ~FlashAttentionLayerBase() override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace bge_large
} // namespace atb_speed
#endif