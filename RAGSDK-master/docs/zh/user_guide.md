# 开发流程<a name="ZH-CN_TOPIC_0000001989074198"></a>

RAG SDK的完整开发流程如[图1](#fig1495610311102)所示。用户可参见以下步骤完成接口调用。

运行阶段请使用普通用户HwHiAiUser执行相关用例。

知识库构建和在线问答支持并发，具体参见对应Demo。

**图 1** RAG SDK开发流程<a id="fig1495610311102"></a>  

![](figures/240914150147412.png)

-   构建知识库。
    1.  上传领域文档，加载和切分。初始化文档处理器，用户可以根据上传的文件类型注册相应的文档解析器（参见[文档解析](./api/knowledge_management.md#文档解析)、[langchain文档解析API](https://python.langchain.com/v0.2/docs/integrations/document_loaders/#all-document-loaders)或基于langchain自定义）和文档切分器（参见[langchain文档切分API](https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/)或基于langchain自定义），支持的文档类型包括Docx、Excel、Pdf、PowerPoint等。用户可以根据需要加载相应的解析和切分功能，输出为文档切分后的文本chunks。
    2.  文本向量化。加载embedding模型（参见[向量化](./api/vectorization.md)），根据模型具体路径进行配置。文档切分后的文本chunks向量化后存入知识库管理中的向量数据库。
    3.  初始化知识库管理，参见[知识库文档管理](./api/knowledge_management.md#知识库文档管理)，包括初始化关系数据库和向量数据库（参见[关系型数据库](./api/databases.md#关系型数据库)和[向量数据库](./api/databases.md#向量数据库)）。

        切分后的文本chunks会存入关系数据库，chunks向量化后数据会存入向量数据库，一一对应。

-   在线问答。
    1.  初始化缓存（参见[缓存模块](./api/cache_module.md#缓存模块)，可选），RAG SDK支持配置缓存和近似搜索，当用户问答时优先从缓存中搜索答案，问题命中则直接返回缓存中的回答，未配置缓存或问题未命中缓存则继续以下推理过程。
    2.  初始化大模型Chain（参见[大模型Chain](./api/model_chains.md)），通过Chain串联大语言模型，检索和精排模块进行问答，用户可以选择文生文、文生图、图生图等Chain，支持多轮对话、检索推理并行等方式。
    3.  初始化检索方式（参见[检索](./api/retrieval.md)），用户可以定义近似检索、查询改写检索等方式。问题经过embedding模型向量化后，通过检索在知识库中找到上下文context，进行下一步处理。
    4.  对检索到的上下文context通过reranker进行精排（参见[排序](./api/ranking.md)，可选），提高检索质量。
    5.  最后将用户问题和上下文context组装成prompt，传入大语言模型（参见[大语言模型](./api/interconnecting_with_the_model_client.md#大语言模型))）进行推理并获得回答返回给用户。如果有配置缓存，问答完成后会将问答对刷新到缓存中，再次问答命中时将缩短问答耗时。


# 应用开发<a name="ZH-CN_TOPIC_0000002043287861"></a>

## 文生文场景<a name="ZH-CN_TOPIC_0000002024300245"></a>

### FlatL2检索方式<a name="ZH-CN_TOPIC_0000002043290941"></a>

#### 总体说明<a name="ZH-CN_TOPIC_0000002019748828"></a>

**样例介绍<a name="section297581914254"></a>**

本章节介绍基于Atlas 800I A2 推理服务器，使用RAG SDK  Python接口开发基于知识库的问答系统。RAG SDK运行框架如[图1](#fig17633219113617)所示，其运行步骤分为“构建知识库”和“检索问答”。

本样例是一个文生文场景，检索方法为距离检索“FLAT:L2”方法，其中框架图中每个步骤的“\[xxx\]”表示可选的方法类。推理大模型使用Llama3-8B-Chinese-Chat，embedding模型使用模型acge\_text\_embedding，reranker（可选）模型使用bge-reranker-large。

**图 1**  基于知识库的问答流程<a id="fig17633219113617"></a>  
![](figures/基于知识库的问答流程.png "基于知识库的问答流程")

**前提条件<a name="section896201815106"></a>**

-   已经在MindIE容器中下载和运行Llama3-8B-Chinese-Chat大模型，模型下载链接：<a href="https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat">链接</a>。
-   已经基于《MindIE安装指南》中的“安装MindIE \> 方式三：容器安装方式”章节完成在宿主机上的容器化部署，并参考《MindIE Motor开发指南》中的“快速入门 \> 启动服务”章节启动服务。
-   已经完成[安装RAG SDK](./installation_guide.md#安装rag-sdk)。
-   已经下载嵌入模型“acge\_text\_embedding”和reranker模型“bge-reranker-large”，并放在[2.a](./installation_guide.md#容器内部署rag-sdk)中运行容器时配置的模型存放目录下。模型下载链接：
    -   acge\_text\_embedding模型：<a href="https://huggingface.co/aspire/acge_text_embedding">链接</a>
    -   bge-reranker-large模型：<a href="https://huggingface.co/BAAI/bge-reranker-large">链接</a>

**TEI服务化说明<a name="section1734316490"></a>**

Embedding模型和Reranker模型可以支持服务化运行，如果选择TEI服务化方式，请完成Embedding服务运行和Reranker服务运行，请参见<a href="https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d">链接</a>。


#### 构建知识库<a name="ZH-CN_TOPIC_0000002018714889"></a>

**操作步骤<a name="section37221226145019"></a>**

1.  编译检索算子，以实现检索功能。

    ```
    cd $MX_INDEX_INSTALL_PATH/tools/ && python3 aicpu_generate_model.py -t <chip_type> && python3 flat_generate_model.py -d <dim> -t <chip_type>  && cp op_models/* $MX_INDEX_MODELPATH 
    ```

    > [!NOTE] 说明
    >-   MX\_INDEX\_INSTALL\_PATH、MX\_INDEX\_MODELPATH变量已在\~/.bashrc中配置，无需单独配置。具体配置值请查看\~/.bashrc。
    >-   **-d** <dim\>表示embedding模型向量化后的维度，因acge\_text\_embedding嵌入模型向量维度为1024，这里设置为-d 1024。
    >-   **-t** <i><chip\_type\></i>表示芯片类型。对于Atlas 300I Duo 推理卡，可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，将查询到的“Name”最后一位数字删掉，即是<i><chip\_type\></i>的取值。对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，取“Name”对应的字段。对于Atlas 800I A3 超节点服务器，可以通过**npu-smi info -t board -i 0 -c 0**命令进行查询，获取**NPU Name**信息，910\_<b><i><NPU Name\></i></b>即是<i><chip\_type\></i>的取值。

2.  创建领域知识文档。

    在/home/HwHiAiUser目录下创建文档gaokao.txt，编码格式为utf-8，内容如下：

    ```
    2024年高考语文作文试题
    新课标I卷
    阅读下面的材料，根据要求写作。（60分）
    随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？
    以上材料引发了你怎样的联想和思考？请写一篇文章。
    要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。
    ```

    > [!NOTE] 说明 
    >所选大模型训练截止日在2024年以前，模型本身未学习“2024年高考语文作文试题”相关知识。

3.  构建领域知识库。

    参考并运行[Demo](https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/RAGSDK/MainRepo/Samples/RagDemo)中rag\_demo\_knowledge.py样例代码，请根据实际情况修改代码中的文件路径、模型路径等默认参数，详细参数设置请参见readme.md文件。

    ```
    python3 rag_demo_knowledge.py --file_path "/path/to/gaokao.txt"
    ```

4.  运行程序获取结果。

    样例代码能打印出上传的文件名列表，则表示构建知识库成功。

    ```
    [‘gaokao.txt’]
    ```


#### 检索问答<a name="ZH-CN_TOPIC_0000001990176706"></a>

**操作步骤<a name="section37221226145019"></a>**

1.  执行在线问答样例。参考并运行[Demo](https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/RAGSDK/MainRepo/Samples/RagDemo)中rag\_demo\_query.py代码文件，请根据实际情况修改代码中的模型路径、mindie服务IP和port等默认参数，详细参数设置请参见readme.md文件。

    ```
    python3 rag_demo_query.py --query "请描述2024年高考作文题目" 
    ```

2.  运行程序获取结果。

    ```
    {
        'query': '请描述2024年高考作文题目',
        'result': '题目：新时代下的生活\n\n材料：\n\n随着科技的不断发展，人们的生活逐渐便利。各种智能设备的应用，让我们的生活更加便捷。然而，在这种便利背后，我们是否面临着一些问题？\n\n请根据以上材料，结合自己的思考，以新时代下的生活为题材，自拟标题，写一篇议论文。',
        'source_documents': [
            {
                'metadata':
                {
                    'source': '/home/HwHiAiUser/gaokao.txt'
                },
                'page_content': '2024年高考语文作文试题\n新课标I卷\n阅读下面的材料，根据要求写作。（60分）\n随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？\n以上材料引发了你怎样的联想和思考？请写一篇文章。\n要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。'
            }
        ]
    }
    ```

> [!NOTE] 说明 
>-   “构建知识库”和“检索回答”过程使用的embedding模型、关系数据库路径、向量数据库路径需对应保持一致，才能正常执行样例。
>-   执行样例代码时，当参数<b>"tei\_emb"</b>为“False”，表示本地启动embedding模型，embedding\_path传入本地模型存放目录；当参数<b>"tei\_emb"</b>为“True”，表示启动服务化模型，embedding_url传入服务化模型URI地址；reranker同理。



### MxRAGCache缓存和自动生成QA<a name="ZH-CN_TOPIC_0000001988965500"></a>

**样例介绍<a name="section28311549181418"></a>**

本样例基于[构建知识库](#构建知识库)和[检索问答](#检索问答)增加MxRAGCache缓存和生成QA的功能，自动生成QA功能支持解析markdown文档，并存入MxRAGCache缓存功能。使用memory cache和similarity cache作为缓存使用。

**图 1**  基于Cache缓存的RAG SDK问答流程<a name="fig3211467423"></a>  
![](figures/基于Cache缓存的RAG-SDK问答流程.png "基于Cache缓存的RAG-SDK问答流程")

**前提条件<a name="section1736555225910"></a>**

-   已经在MindIE容器中下载和运行Llama3-8B-Chinese-Chat大模型，模型下载链接：<a href="https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat">链接</a>。

-   RAG SDK的容器内能够访问Llama3-8B-Chinese-Chat大模型的路径下的config.json和tokenizer.json，用于计算文本token大小。
-   已经基于《MindIE安装指南》中的“安装MindIE \> 方式三：容器安装方式”章节完成在宿主机上的容器化部署，并参考《MindIE Motor开发指南》中的“快速入门 \> 启动服务”章节启动服务。
-   已经完成[安装RAG SDK](./installation_guide.md#安装rag-sdk)。
-   已经下载嵌入模型“acge\_text\_embedding”和reranker模型“bge-reranker-large”，并放在[2.a](./installation_guide.md#容器内部署rag-sdk)中运行容器时配置的模型存放目录下。模型下载链接：
    -   acge\_text\_embedding模型：<a href="https://huggingface.co/aspire/acge_text_embedding">链接</a>
    -   bge-reranker-large模型：<a href="https://huggingface.co/BAAI/bge-reranker-large">链接</a>


**操作步骤<a name="section599518311318"></a>**

1.  编译检索算子，以实现检索功能。

    ```
    cd $MX_INDEX_INSTALL_PATH/tools/ && python3 aicpu_generate_model.py -t <chip_type> && python3 flat_generate_model.py -d <dim> -t <chip_type>  && cp op_models/* $MX_INDEX_MODELPATH 
    ```

    > [!NOTE] 说明 
    >-   MX\_INDEX\_INSTALL\_PATH、MX\_INDEX\_MODELPATH变量已在\~/.bashrc中配置，无需单独配置。具体配置值请查看\~/.bashrc。
    >-   **-d** <i><dim\></i>表示embedding模型向量化后的维度，因acge\_text\_embedding嵌入模型向量维度为1024，这里设置为-d 1024。
    >-   **-t** <i><chip\_type\></i>表示芯片类型。对于Atlas 300I Duo 推理卡，可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，将查询到的“Name”最后一位数字删掉，即是<i><chip\_type\></i>的取值。对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，取“Name”对应的字段。对于Atlas 800I A3 超节点服务器，可以通过**npu-smi info -t board -i 0 -c 0**命令进行查询，获取**NPU Name**信息，910\_<b><i><NPU Name\></i></b>即是<i><chip\_type\></i>的取值。

2.  创建领域知识文档。

    在/home/HwHiAiUser目录下创建文档gaokao.md，编码格式为utf-8，内容如下：

    ```
    2024年高考语文作文试题
    新课标I卷
    阅读下面的材料，根据要求写作。（60分）
    随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？
    以上材料引发了你怎样的联想和思考？请写一篇文章。
    要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。
    ```

    > [!NOTE] 说明 
    >所选大模型训练截止日在2024年以前，模型本身未学习“2024年高考语文作文试题”相关知识。

3.  参见并运行[Demo](https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/RAGSDK/MainRepo/Samples/RagDemo)中rag\_demo\_cache\_qa.py代码文件，请根据实际情况修改代码中的文件路径、模型路径和大模型IP和port等默认参数，详细参数设置请参见readme.md文件。
4.  执行样例代码。

    ```
    python3 rag_demo_cache_qa.py  --query "请描述2024年高考作文题目"
    ```

5.  运行两次样例代码，获取结果。

    ```
    # 第一次运行结果和第二次回答一致，但第二次运行时命中缓存返回，回答时间大幅减少
    {'query': '请描述2024年高考作文题目', 'result': '根据您提供的信息，2024年高考语文作文试题的具体内容尚未公开。通常，高考作文题目会在考试当天或考试前一段时间由教育部门公布。因此，无法为您提供2024年高考作文题目具体内容。\n\n不过，根据您提供的信息，题目可能会围绕“随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？”这一主题展开。学生需要根据这个问题，选准角度，确定立意，明确文体，自拟标题，并在不少于800字的范围内进行写作。\n\n如果您需要进一步的指导或帮助，例如如何构思作文、如何组织思路、如何提高写作质量等，我可以提供一些一般性的建议。', 'source_documents': [{'metadata': {'source': '/home/HwHiAiUser/gaokao.md'}, 'page_content': '2024年高考语文作文试题\n新课标I卷\n阅读下面的材料，根据要求写作。（60分）\n随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？\n以上材料引发了你怎样的联想和思考？请写一篇文章。\n要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。\n'}]}
    耗时：0.0007343292236328125s
    ```



## 文本检索图片<a name="ZH-CN_TOPIC_0000002272375173"></a>

本章节将指导用户使用RAG SDK根据文本搜索图片样例。

**前提条件<a name="section1734316490"></a>**

已经完成[安装RAG SDK](./installation_guide.md#安装rag-sdk)。

**样例流程介绍<a name="section1281432091612"></a>**

![](figures/文本检索图片流程.png)

**操作步骤<a name="section7904194010166"></a>**

1.  在任意目录编辑创建retrieve\_img\_demo.py，内容如下：

    ```
    import argparse
    
    from mx_rag.document import LoaderMng
    from mx_rag.document.loader import ImageLoader
    
    from mx_rag.embedding.local import ImageEmbedding
    from mx_rag.knowledge import KnowledgeDB, upload_files
    from mx_rag.knowledge.knowledge import KnowledgeStore
    from mx_rag.retrievers import Retriever
    from mx_rag.storage.document_store import SQLiteDocstore
    from mx_rag.storage.vectorstore import MindFAISS
    
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--query', type=str, help="查询图片文本内容")
        parser.add_argument("--image-path", type=str, action='append', help="待入库图片路径")
    
        args = parser.parse_args().__dict__
        images: list[str] = args.pop("image_path")
        query = args.pop("query")
        loader_mng = LoaderMng()
        loader_mng.register_loader(ImageLoader, [".jpg"])
    
        dev = 0
        img_emb = ImageEmbedding("ViT-B-16", model_path="path to clip model", dev_id=dev)
    
        img_vector_store = MindFAISS(x_dim=512, devs=[dev],
                                     load_local_index="./image_faiss.index",
                                     auto_save=True)
        chunk_store = SQLiteDocstore(db_path="./sql.db")
    
        # 初始化知识管理关系数据库
        knowledge_store = KnowledgeStore(db_path="./sql.db")
    
        user_id = "fc557af8-5973-4893-9624-4a510c3e18fb"
        knowledge_store.add_knowledge("test", user_id=user_id)
    
        knowledge_db = KnowledgeDB(knowledge_store=knowledge_store, chunk_store=chunk_store, vector_store=img_vector_store,
                                   knowledge_name="test", white_paths=["/home"], user_id=user_id)
    
        upload_files(knowledge_db, images, loader_mng=loader_mng,
                     embed_func=img_emb.embed_images, force=True)
    
        img_retriever = Retriever(vector_store=img_vector_store, document_store=chunk_store,
                                  embed_func=img_emb.embed_documents, k=1, score_threshold=0.4)
        res = img_retriever.invoke(query)
        # res中包含检索到的图片路径
        print(res)
    
    ```

2.  执行如下命令运行，其他参数按实际情况配置，参考[ClientParam](./api/README.md#clientparam)。

    ```
    python3 retrieve_img_demo.py --image-path ./car1.jpg  --image-path ./car2.jpg  --query "小汽车"
    ```


## 多轮对话<a name="ZH-CN_TOPIC_0000002026661421"></a>

本章节将指导用户使用LangChain来使用多轮对话功能。

**前提条件<a name="section1736555225910"></a>**

-   已经完成[安装RAG SDK](安装RAG-SDK.md)。
-   已经基于《MindIE安装指南》中的“安装MindIE \> 方式三：容器安装方式”章节完成容器化部署。
-   已经运行Llama3-8B-Chinese-Chat大模型。

**操作步骤<a name="section599518311318"></a>**

1.  在容器内任意目录执行vim命令创建demo.py代码文件，文件内容如下：

    ```
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    from mx_rag.llm import Text2TextLLM
    from mx_rag.utils import ClientParam
    if __name__ == '__main__':
        template = """You are a chatbot having a conversation with a human. Please answer as briefly as possible.
    
        {chat_history}
        Human: {human_input}"""
        dev = 1
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=template
        )
        # k可以设置保存的历史会话轮数，还支持ConversationBufferMemory和ConversationTokenBufferMemory，参考langchain官方文档
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
        client_param = ClientParam(ca_file="/path/to/ca.crt")
        chat = Text2TextLLM(base_url="https://ip:port/v1/chat/completions", 
                            model_name="Llama3-8B-Chinese-Chat", 
                            client_param=client_param)
        llm_chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)
        questions = ["请记住小明的爸爸是小刚",
                     "七大洲前四个是啥？",
                     "后三个呢？"]
        for question in questions:
            llm_chain.predict(human_input=question)
        completion = llm_chain.predict(human_input="请问小明的爸爸是谁？")   
        print(completion)
    ```

2.  运行样例代码，请求大模型中带有历史信息，prompt拼接结果如下：

    ```
    You are a chatbot having a conversation with a human. Please answer as briefly as possible.
    
    Human: 请记住小明的爸爸是小刚
    AI: 记住了，小明的爸爸是小刚。
    Human: 七大洲前四个是啥？
    AI: 亚洲、非洲、欧洲、北美洲。
    Human: 后三个呢？
    AI: 南美洲、澳大利亚、南极洲。
    Human: 请问小明的爸爸是谁？
    小明的爸爸是小刚。
    ```


## 调用Agentic RAG样例<a name="ZH-CN_TOPIC_0000002041731821"></a>

详细介绍可参见：[RAG SDK基于LangGraph知识检索增强应用使能方案](https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/RAGSDK/MainRepo/langgraph)。


## chat with ragsdk<a name="ZH-CN_TOPIC_0000002485964970"></a>

启动WEB服务，进行参数配置、文档上传、删除、问答等操作，详细流程见：[chat\_with\_ragsdk](https://gitcode.com/printSSS/mindsdk-referenceapps_9751/blob/cache01/RAGSDK/MainRepo/Samples/RagDemo/chat_with_ascend/readme.md)

