# 接口参考<a name="ZH-CN_TOPIC_0000001981995304"></a>

## 使用说明<a name="ZH-CN_TOPIC_0000002018595345"></a>

RAG SDK的接口包含知识管理接口、数据库接口、模型管理接口、评估模块接口、检索接口、Chain类等接口。用户可调用接口完成二次开发。

RAG SDK接口遵循异常处理机制，故用户必须在try/except语句块内进行调用以及异常处理，防止在使用的过程中出现异常抛出导致程序退出的情况。

> [!NOTICE] 须知 
>RAG SDK如果使用Cache对问题-答案缓存，生成的数据库未做加密存储，如果涉及银行卡号、身份证号、护照号、口令等个人数据，请勿存放到数据库中。

|接口类名|导航链接|
|--|--|
|通用类|[universal_api](./universal_api.md#通用)|
|知识管理|[knowledge_management](./knowledge_management.md#知识管理)|
|数据库|[databases](./databases.md#数据库)|
|对接模型客户端|[llm_client](./llm_client.md)|
|向量化|[embedding](./embedding.md)|
|排序|[reranker](./reranker.md)|
|模型推理加速|[model_inference_acceleration](./model_inference_acceleration.md)|
|embedding模型微调|[embedding_model_fine_tuning](./embedding_model_fine_tuning.md)|
|评估模块|[evaluation_module](./evaluation_module.md#评估模块)|
|缓存模块|[cache_module](./cache_module.md#缓存模块)|
|检索|[retrieval](./retrieval.md)|
|文档总结|[document_summary](./document_summary.md#文档总结)|
|Prompt压缩|[prompt_compression](./prompt_compression.md)|
|大模型Chain|[llm_chains](./llm_chains.md)|
|知识图谱|[knowledge_graph](./knowledge_graph.md)|

