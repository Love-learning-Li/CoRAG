# RAGSDK

-   [最新消息](#最新消息)
-   [简介](#简介)
-   [目录结构](#目录结构)
-   [版本说明](#版本说明)
-   [环境部署](#环境部署)
-   [编译流程](#编译流程)
-   [快速入门](#快速入门)
-   [特性介绍](#特性介绍)
-   [API参考](#API参考)
-   [FAQ](#FAQ)
-   [安全声明](#安全声明)
-   [分支维护策略](#分支维护策略)
-   [版本维护策略](#版本维护策略)
-   [免责声明](#免责声明)
-   [License](#License)
-   [建议与交流](#建议与交流)

# 最新消息

- 2025.12.30：RAGSDK 开源发布

# 简介

RAGSDK是昇腾面向大语言模型的知识增强开发套件，为解决大模型知识更新缓慢以及垂直领域知识回答弱的问题，面向大模型知识库提供垂域调优、生成增强、知识管理等特性，帮助用户搭建专属的高性能、准确度高的大模型问答系统。
<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/RAGSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/RAGSDK)

</div>

# 目录结构

``` 
├─build
├─mx_rag
│  ├─cache
│  ├─chain
│  ├─compress
│  ├─document
│  ├─embedding
│  ├─evaluate
│  ├─graphrag
│  ├─knowledge
│  ├─llm
│  ├─reranker
│  ├─retrievers
│  ├─storage
│  ├─summary
│  ├─tools
│  ├─utils
├─ops
├─output
├─script
├─tests
│  ├─data
│  └─python
```

# 版本说明

RAGSDK版本配套详情参考：[版本配套说明](./docs/zh/release_notes.md)

# 环境部署

RAGSDK支持在容器内部署和在物理机内部署两种安装方式。

# 编译流程

在RAGSDK容器中，构建run包以及执行ut的流程。
1. 到[acendhub](https://www.hiascend.com/developer/ascendhub/detail/ragsdk)，下载ragsdk镜像，运行容器

2. 下载代码到本地
```
git clone https://gitcode.com/Ascend/RAGSDK.git
cd RAGSDK
```
3. 执行构建脚本
```
cd build
bash build.sh
```
4. 构建完之后，run包在./ouput/目录下，安装run包
```
cd ../output/
./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install --install-path=<安装路径> --platform=<chip_type>
```
>  <chip_type>表示芯片类型，可在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是--platform的取值。若是Atlas 800I A3 超节点服务器则取值为A3。

6. 执行测试用例
```
cd ../tests/
bash run_py_test.sh
```

# 快速入门

RAGSDK提供快速构建基于昇腾平台问答系统的能力，提供多模态文档解析、知识库管理等能力，降低用户大模型应用开发门槛，支持对接开源生态。

- 快速搭建：提供模块化功能接口，支持按需进行调用。通过预置的端到端工作流模板，支持用户通过极少量代码快速搭建问答服务。

- 多模态解析：支持文档、表格、PDF、图片等多种类型文件的解析，为大模型提供多样性语料。
- 高性能推理：提供昇腾亲和模型优化加速，实现更高的吞吐和更短的响应时间。

具体的操作请参考：[用户指南](./docs/zh/user_guide.md)。

# 特性介绍

RAGSDK组件提供多模态文档解析、知识库管理等能力，降低用户大模型应用开发门槛，支持对接开源生态。具体特性及使用指南参考[用户指南](./docs/zh/user_guide.md)对应章节，已发布特性如下：
- ✅文生文场景 
- ✅文本检索图片 
- ✅多轮对话 
- ✅调用 Agentic RAG 样例 
- ✅chat with ragsdk 

# API参考

API参考详见，[接口参考](./docs/zh/api/README.md)。

# FAQ

相关FAQ请参考，[FAQ](./docs/zh/faq.md)。

# 安全声明

- 当前容器方式部署本组件。
- 该容器权限具有一定风险，建议用户自行进行安全加强。
- 其他安全加固详见：[安全加固](./docs/zh/security_hardening.md)	 
 - 公网地址详见：[公网地址](./docs/zh/resource/RAG_SDK_7.3.0%20%E5%85%AC%E7%BD%91%E5%9C%B0%E5%9D%80.xlsx)

# 分支维护策略

版本分支的维护阶段如下：

| 状态                | 时间     | 说明                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| 计划                | 1-3个月  | 计划特性                                                     |
| 开发                | 3个月    | 开发新特性并修复问题，定期发布新版本                         |
| 维护                | 3-12个月 | 常规分支维护3个月，长期支持分支维护12个月。对重大BUG进行修复，不合入新特性，并视BUG的影响发布补丁版本 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                         |

# 版本维护策略

| 版本     | 维护策略 | 当前状态 | 发布日期         | 后续状态                      | EOL日期    |
| -------- | -------- | -------- | ---------------- | ----------------------------- | ---------- |
| master   | 长期支持 | 开发     | 2025-12-30 | -                     | -          |


# 免责声明

- 本仓库代码中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布前，这些分支不应被应用于任何生产环境或者依赖关键业务的项目中。请务必使用我们的正式发行版本，以确保代码的稳定性和安全性。
  使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。
- 正式版本请参考release版本 

# License

RAGSDK以Mulan PSL v2许可证许可，对应许可证文本可查阅[LICENSE](./LICENSE.md)。

介绍RAGSDK docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。
# 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[issue](https://gitcode.com/Ascend/RAGSDK/issues)，我们会尽快回复。感谢您的支持。