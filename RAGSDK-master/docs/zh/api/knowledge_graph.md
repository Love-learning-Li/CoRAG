
## 知识图谱<a name="ZH-CN_TOPIC_0000002340395433"></a>

实现基于知识图谱的RAG（Retrieval-Augmented Generation）流程。它结合了大模型（LLM）、图数据库（如NetworkX或OpenGauss）、向量检索、重排序等多种技术，实现了文档的结构化知识抽取、图谱构建、概念聚类、向量化检索和多跳推理等能力。该模块适用于复杂知识问答、企业知识管理、智能检索等场景。

知识图谱利用大型语言模型（LLM）通过结构化提示实现关系抽取、节点和关系的概念化。为了方便您在默认提示的基础上进行自定义修改，以下提供中文和英文版本的默认提示供您参考：

**中文默认提示：**

-   **三元组抽取提示（TRIPLE\_INSTRUCTION\_CN）：**

    ```
    
    TRIPLE_INSTRUCTIONS_CN = {
        "entity_relation": """
    ## 目标
    请从以下文本中提取所有重要实体及其关系，并严格遵守以下规则：
    ## 要求
    1. 实体必须为名词，尽量简洁；  
    2. 关系必须为一个动词，准确描述“头实体”与“尾实体”之间的具体联系，且不得重复头、尾实体的字面信息；  
    3. 头实体与尾实体均不得为“是”，不得使用代词；
    4. 实体和关系不能为空字符串，不能为仅包含标点符号的字符串；
    4. 输出必须采用下列 JSON 格式，禁止添加、删除或修改任何字段：
    [
        {
            "头实体": "{名词}",
            "关系": "{动词}",
            "尾实体": "{名词}"
        }
    ]
    ## 示例
    [
        {
            "头实体": "中国",
            "关系": "首都",
            "尾实体": "北京",
        },
        {
            "头实体": "小狗",
            "关系": "喜欢",
            "尾实体": "骨头",
        },
        {
            "头实体": "毛泽东",
            "关系": "父亲",
            "尾实体": "毛岸英",
        },
        {
            "头实体": "中国船舶工业物资云贵有限公司",
            "关系": "成立",
            "尾实体": "1990月05月31日",
        },
        {
            "头实体": "公司",
            "关系": "地址",
            "尾实体": "云南省昆明市",
        },
        {
            "头实体": "公司",
            "关系": "经营",
            "尾实体": "电子器件",
        },
        {
            "头实体": "1999年",
            "关系": "早于",
            "尾实体": "2000年",
        },
        {
            "头实体": "2001年",
            "关系": "晚于",
            "尾实体": "2000年",
        }
    ]
    ## 待分析文本
    """,
        "event_entity": """
    ## 目标
    请对以下段落逐句进行事件抽取，并识别每个事件所涉及的全部实体。
    ## 要求
    1. 一句视为一个独立事件，保留原句，不做任何省略；
    2. 列出每个事件直接参与的所有实体，不重复、不遗漏；
    3. 输出严格使用下方 JSON 格式，不允许添加或删减字段。
    ## JSON 格式
    [
        {
            "事件": "{原句}",
            "实体": ["实体1", "实体2", "..."]
        }
    ]
    ## 待分析文本
    """,
        "event_relation": """
    ## 目标
    请对以下段落逐句抽取事件，并识别它们之间的时间或因果关系。  
    ## 要求  
    1. 一句视为一个独立事件，保留原句，不做任何省略。  
    2. 仅使用指定关系类型：在之前、在之后、同时、因为、结果。  
    3. 每个三元组中的“头事件”与“尾事件”均须为段落中完整原句，且语义对应具体、可独立理解。
    4. “头事件”和“尾事件”不能为空字符串，且不能重叠；
    4. 关系不能为空字符串；
    5. 输出严格使用下方 JSON 格式，不允许添加、删减或省略任何字段。  
    ## JSON 格式  
    [
        {
            "头事件": "{事件1完整原句}",
            "关系": "{在之前|在之后|同时|因为|结果}",
            "尾事件": "{事件2完整原句}"
        }
    ]
    ## 待分析文本
    """,
    }
    ```

-   **实体概念化提示 \(ENTITY\_PROMPT\_CN\):**

    ```
    ENTITY_PROMPT_CN = '''
    ## 目标
    给定一个实体及其背景，提供多个短语来表示该实体的抽象概念。
    ## 输出格式
    短语1, 短语2, 短语3,...
    ## 要求
    * 短语应准确代表实体，可以是其类型或相关概念。
    * 短语包含1-2个词。
    * 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
    * 严格遵循输出格式，不添加任何额外字符。
    * 尽可能提供3到10个不同抽象层次的短语。
    * 不重复使用与实体或已有短语相同的词语。
    * 如果无法生成更多短语，请立即停止。
    ## 示例
    实体：灵魂
    背景：在BFI伦敦电影节首映，成为皮克斯票房最高的影片
    概念：电影, 影片
    实体：Thinkpad X60
    背景：理查德·斯托曼宣布他在Thinkpad X60上使用Trisquel
    概念：Thinkpad, 笔记本, 机器, 设备, 硬件, 电脑, 品牌
    实体：哈里·卡拉汉
    背景：欺骗另一个抢劫犯，折磨天蝎座
    概念：人, 美国人, 角色, 警察, 侦探
    实体：黑山学院
    背景：由约翰·安德鲁·赖斯创立，吸引了教师
    概念：学院, 大学, 学校, 文理学院
    ## 待分析文本
    实体：[ENTITY]
    背景：[CONTEXT]
    概念：
    '''
    ```

-   **事件概念化提示 \(EVENT\_PROMPT\_CN\):**

    ```
    EVENT_PROMPT_CN = '''
    ## 目标
    给定一个事件，提供多个短语来表示该事件的抽象概念。 
    ## 输出格式
    短语1, 短语2, 短语3,...
    ## 要求
    * 短语应准确代表事件，可以是其类型或相关概念。
    * 短语包含1-2个词。
    * 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
    * 严格遵循输出格式，不添加任何额外字符。
    * 尽可能提供3到10个不同抽象层次的短语。
    * 不重复使用与事件或已有短语相同的词语。
    * 如果无法生成更多短语，请立即停止。
    ## 示例
    事件：一名男子隐居于山林
    概念：隐居, 放松, 逃避, 自然, 孤独
    事件：一只猫追逐猎物进入它的藏身之处
    概念：狩猎, 逃避, 捕食, 躲藏, 潜行
    事件：山姆和他的狗玩耍
    概念：放松的活动, 爱抚, 玩耍, 联结, 友谊
    事件：中国船舶工业物资云贵有限公司，成立日期：1990年3月31日成立，住所：云南省昆明市
    概念：公司成立, 公司地点, 公司住所, 省, 市, 成立时间, 年月日
    ## 待分析文本
    事件：[EVENT]
    概念：
    '''
    ```

-   **关系概念化提示 \(RELATION\_PROMPT\_CN\):**

    ```
    RELATION_PROMPT_CN = '''
    ## 目标
    给定一个关系，提供多个短语来表示该关系的抽象概念。
    ## 输出格式
    短语1, 短语2, 短语3,...
    ## 要求
    * 短语应准确代表关系，可以是其类型或相关概念。
    * 短语包含1-2个词。
    * 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
    * 严格遵循输出格式，不添加任何额外字符。
    * 尽可能提供3到10个不同抽象层次的短语。
    * 不重复使用与关系或已有短语相同的词语。
    * 如果无法生成更多短语，请立即停止。
    ## 示例
    关系：参与
    概念：成为一部分, 参加, 投入, 涉及
    关系：被包括在内
    概念：加入, 成为一部分, 成为成员, 成为组成部分
    ## 待分析文本
    关系：[RELATION]
    概念：
    '''
    ```

**英文默认提示：**

-   **三元组抽取提示（TRIPLE\_INSTRUCTIONS\_EN）：**

    ```
    TRIPLE_INSTRUCTIONS_EN = {
        "entity_relation": """Given a passage, summarize all the important entities and the relations between them in 
        a concise manner. Relations should briefly capture the connections between entities, without repeating information 
        from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from 
        being considered as entities. The output should strictly adhere to the following JSON format:
        [
            {
                "Head": "{a noun}",
                "Relation": "{a verb}",
                "Tail": "{a noun}",
            },
            {
                "Head": "China",
                "Relation": "Capital",
                "Tail": "Beijing",
            },
            {
                "Head": "Dog",
                "Relation": "like",
                "Tail": "bone",
            },
            {
                "Head": "Mao Zedong",
                "Relation": "Father",
                "Tail": "Mao Anying",
            },
            {
                "Head": "China Shipbuilding Materials Yungui Co., Ltd.",
                "Relation": "Established",
                "Tail": "May 31, 1990",
            },
            {
                "Head": "Company",
                "Relation": "Address",
                "Tail": "Kunming City, Yunnan Province",
            },
            {
                "Head": "Company",
                "Relation": "Operation",
                "Tail": "Electronics",
            },
            {
                "Head": "Year 1999",
                "Relation": "Before",
                "Tail": "Year 2000",
            },
            {
                "Head": "Year 2001",
                "Relation": "After",
                "Tail": "Year 2000",
            },
        ]""",
        "event_entity": """Please analyze and summarize the participation relations between the events and entities 
        in the given paragraph. Each event is a single independent sentence. Additionally, identify all the entities 
        that participated in the events. Do not use ellipses. Please strictly output in the following JSON format:
        [
            {
                "Event": "{a simple sentence describing an event}",
                "Entity": ["entity 1", "entity 2", "..."]
            }...
        ] """,
        "event_relation": """Please analyze and summarize the relationships between the events in the paragraph. 
        Each event is a single independent sentence. Identify temporal and causal relationships between the events using the following types: before, after, at the same time, because, and as a result. Each extracted triple should be specific, meaningful, and able to stand alone.  Do not use ellipses.  The output should strictly adhere to the following JSON format:
        [
            {
                "Head": "{a simple sentence describing the event 1}",
                "Relation": "{temporal or causality relation between the events}",
                "Tail": "{a simple sentence describing the event 2}"
            }...
        ]"""
    }
    ```

-   **实体概念化提示 \(ENTITY\_PROMPT\_EN\):**

    ```
    ENTITY_PROMPT_EN = '''I will give you an ENTITY. You need to give several phrases containing 1-2 words for the 
                ABSTRACT ENTITY of this ENTITY.
                You must return your answer in the following format: phrases1, phrases2, phrases3,...
                You can't return anything other than answers.
                These abstract intention words should fulfill the following requirements.
                1. The ABSTRACT ENTITY phrases can well represent the ENTITY, and it could be the type of the ENTITY or 
                the related concepts of the ENTITY.
                2. Strictly follow the provided format, do not add extra characters or words.
                3. Write at least 3 or more phrases at different abstract level if possible.
                4. Do not repeat the same word and the input in the answer.
                5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
                ENTITY: Soul
                CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release
                Your answer: movie, film
                ENTITY: Thinkpad X60
                CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60
                Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand
                ENTITY: Harry Callahan
                CONTEXT: bluffs another robber, tortures Scorpio
                Your answer: person, Amarican, character, police officer, detective
                ENTITY: Black Mountain College
                CONTEXT: was started by John Andrew Rice, attracted faculty
                Your answer: college, university, school, liberal arts college
                EVENT: 1st April
                CONTEXT: Utkal Dibas celebrates
                Your answer: date, day, time, festival
                ENTITY: [ENTITY]
                CONTEXT: [CONTEXT]
                Your answer:
                '''
    ```

-   **事件概念化提示 \(EVENT\_PROMPT\_EN\):**

    ```
    EVENT_PROMPT_EN = '''I will give you an EVENT. You need to give several phrases containing 1-2 words for the 
                ABSTRACT EVENT of this EVENT.
                You must return your answer in the following format: phrases1, phrases2, phrases3,...
                You can't return anything other than answers.
                These abstract event words should fulfill the following requirements.
                1. The ABSTRACT EVENT phrases can well represent the EVENT, and it could be the type of the EVENT or the related concepts of the EVENT.    
                2. Strictly follow the provided format, do not add extra characters or words.
                3. Write at least 3 or more phrases at different abstract level if possible.
                4. Do not repeat the same word and the input in the answer.
                5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
                EVENT: A man retreats to mountains and forests
                Your answer: retreat, relaxation, escape, nature, solitude
                
                EVENT: A cat chased a prey into its shelter
                Your answer: hunting, escape, predation, hidding, stalking
                EVENT: Sam playing with his dog
                Your answer: relaxing event, petting, playing, bonding, friendship
                EVENT: [EVENT]
                Your answer:
                '''
    ```

-   **关系概念化提示 \(RELATION\_PROMPT\_EN\):**

    ```
    RELATION_PROMPT_EN = '''I will give you an RELATION. You need to give several phrases containing 1-2 words for 
                the ABSTRACT RELATION of this RELATION.
                You must return your answer in the following format: phrases1, phrases2, phrases3,...
                You can't return anything other than answers.
                These abstract intention words should fulfill the following requirements.
                1. The ABSTRACT RELATION phrases can well represent the RELATION, and it could be the type of the RELATION 
                or the simplest concepts of the RELATION.
                2. Strictly follow the provided format, do not add extra characters or words.
                3. Write at least 3 or more phrases at different abstract level if possible.
                4. Do not repeat the same word and the input in the answer.
                5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
                
                RELATION: participated in
                Your answer: become part of, attend, take part in, engage in, involve in
                RELATION: be included in
                Your answer: join, be a part of, be a member of, be a component of
                RELATION: [RELATION]
                Your answer:
                '''
    ```

### GraphRAGPipeline<a name="ZH-CN_TOPIC_0000002306556168"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002340515621"></a>

**功能描述<a name="section957011509130"></a>**

提供知识图谱创建、检索的统一入口。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.graphrag import GraphRAGPipeline

GraphRAGPipeline(work_dir, llm, embedding_model, dim, rerank_model, graph_type,graph_name, encrypt_fn,decrypt_fn,kwargs)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|work_dir|str|必选|知识图谱工作目录，其剩余空间至少为5GB，保存生成的图json中间文件，如果使用的MindFAISS，对应的向量数据也在该路径下。<br>不能为相对路径，路径长度不能超过1024，不能为软链接且不允许存在".."。<br>路径不能在路径列表中：["/etc", "/usr/bin", "/usr/lib", "/usr/lib64", "/sys/", "/dev/", "/sbin", "/tmp"]。|
|llm|Text2TextLLM|必选|大模型接口实例对象。|
|embedding_model|Embeddings|必选|langchain_core.embeddings.Embeddings的子类，取值包含：<li>mx_rag.embedding.local.TextEmbedding<li>mx_rag.embedding.service.TEIEmbedding|
|dim|int|必选|嵌入模型生成的向量维度，其取值范围为[1, 1024 * 1024]。|
|rerank_model|Reranker|可选|mx_rag_reranker.Reranker的子类，默认为None，取值包含：<li>mx_rag.reranker.local.LocalReranker<li>mx_rag.reranker.service.TEIReranker|
|graph_type|str|可选|图数据库类型，默认为“networkx”，其取值仅支持["networkx", "opengauss"]。|
|graph_name|str|可选|知识图谱名称，默认为“graph”，其取值范围为[1, 255]，只能由标识符组成。|
|encrypt_fn|Callable|可选|回调方法，对调用[build_graph](#build_graph)产生的json文件内容加密。请注意提供正确加密方法并保证安全性，返回值是加密后的字符串。<br>> [!NOTICE] 须知 如果上传的文档涉及银行卡号、身份证号、护照号、口令等个人数据，请配置该参数保证个人数据安全。|
|decrypt_fn|Callable|可选|回调方法，在graph_type为"networkx"时，在检索时会对"{graph_name}.json"解密读取。请注意提供正确解密方法并保证安全性，返回值是解密后的字符串。|
|kwargs|Dict|可选|扩展参数列表：<li>age_graph：当图数据库类型为openGauss时，需要指定该参数，类型为openGaussAGEGraph，为openGuass图数据库连接实例。<li>devs：指定NPU设备，为一个只包含一个元素的list，类型list[int]。<li>node_vector_store: 用于存储向量化节点以实现相似节点搜索的向量数据库。默认为None，此时将使用MindFAISS作为向量数据库。<li>concept_vector_store: 在对概念进行聚类时，用于存储向量化概念以实现相似概念搜索的向量数据库。默认为None，此时将使用MindFAISS作为向量数据库。<br>> [!NOTE] 说明 age_graph由用户控制传入，请使用安全的连接方式。|


**返回值说明<a name="section53998444524"></a>**

GraphRAGPipeline对象。

**调用示例<a name="section8509453104117"></a>**

```
import getpass
from paddle.base import libpaddle  # fix std::bad_alloc
from langchain_opengauss import OpenGaussSettings, openGaussAGEGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from mx_rag.chain.single_text_to_text import GraphRagText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.embedding.local import TextEmbedding
from mx_rag.graphrag import GraphRAGPipeline
from mx_rag.llm import LLMParameterConfig, Text2TextLLM
from mx_rag.reranker.local import LocalReranker
from mx_rag.utils import ClientParam
work_dir = "test_pipeline"
llm = Text2TextLLM(
    base_url="https://x.x.x.x:port/v1/chat/completions",
    model_name="model_name",
    llm_config=LLMParameterConfig(max_tokens=64 * 1024, temperature=0.6, top_p=0.9),
    client_param=ClientParam(timeout=180, ca_file="/path/to/ca.crt"),
)
rerank_model = LocalReranker("/data/models/bge-reranker-v2-m3/", 0, 20, False)
embedding_model = TextEmbedding.create(model_path="/data/models/bge-large-en-v1.5")
data_load_mng = LoaderMng()
data_load_mng.register_loader(TextLoader, [".txt"])
data_load_mng.register_splitter(
    RecursiveCharacterTextSplitter,
    [".txt"],
    dict(chunk_size=512, chunk_overlap=20)
)
graph_name = "hotpotqa"
graph_type = "opengauss"

conf = OpenGaussSettings(user="gaussdb",
                         password=getpass.getpass(),
                         host="x.x.x.x",
                         port="x",
                         database="postgres")
age_graph = openGaussAGEGraph(graph_name, conf,
                              sslmode="verify-ca",
                              sslcert="client.crt",
                              sslkey="client.key",
                              sslrootcert="cacert.pem")
pipeline = GraphRAGPipeline(work_dir, llm, embedding_model, 1024, rerank_model, graph_name=graph_name,
                            age_graph=age_graph)
pipeline.upload_files(["./test_graph/hotpotqa.500.txt"], data_load_mng)
pipeline.build_graph()
question = "Which case was brought to court first Miller v. California or Gates v. Collier ?"
contexts = pipeline.retrieve_graph(question)
text2text_chain = GraphRagText2TextChain(
    llm=llm,
    retriever=pipeline.as_retriever(),
    reranker=rerank_model)
result = text2text_chain.query(question)
print(f"#contexts: {len(contexts)}")
print(contexts)
print(result)
```


#### upload\_files<a name="ZH-CN_TOPIC_0000002306396444"></a>

**功能描述<a name="section53998444524"></a>**

调用此函数上传创建知识图谱所需要的文档列表。

**函数原型<a name="section18789201331417"></a>**

```
def upload_files(file_list, loader_mng)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|file_list|list|是|文件列表，注意同一批次文档列表只支持同一种语言类型，一次上传过多文档构建知识图谱时会比较慢，文档个数限制[1, 100]。<br>单个文档路径长度取值范围为[1, 1024]，文档路径不能为软链接且不允许存在".."，且每个文档大小不超过10G。|
|loader_mng|LoaderMng|是|提供文档解析函数的管理类对象，数据类型参见[LoaderMng](./knowledge_management.md#loadermng)。|


**返回值说明<a name="section11818153884917"></a>**

无


#### build\_graph<a name="ZH-CN_TOPIC_0000002340395437"></a>

**功能<a name="section53998444524"></a>**

调用此函数创建文本节点索引以及生成对应文本的知识图谱。

**函数原型<a name="section18789201331417"></a>**

```
def build_graph(lang, pad_token, conceptualize, **kwargs)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|lang|Lang|否|语料所用的语言，默认取值为Lang.EN，即英文语料。|
|pad_token|str|否|大语言模型使用填充字符，默认为空字符，其取值范围为[0, 255]。|
|conceptualize|bool|否|是否进行对节点概念化，默认为False。|
|kwargs|dict|否|扩展参数列表：<li>max_workers：构建知识图谱的线程数。默认值为5。<li>batch_size，节点向量化，检索等操作批大小，默认为32。<li>top_k：在对图节点概念进行聚类时，向量检索返回的最相似概念数量。默认值为5，取值范围为[1, 100]。<li>threshold：向量相似性阈值。低于此值的相似性结果将被过滤。默认值为0.3，取值范围为[0.0,1.0]。triple_instructions: 用于指导大型语言模型（LLM）从文档中抽取关系的指令，字典类型。默认值为None，此时将根据语言使用默认值（中文为TRIPLE_INSTRUCTIONS_CN，英文为TRIPLE_INSTRUCTIONS_EN）。用户可以通过提供一个字典来覆盖默认的抽取指令。该字典必须包含以下键：<ul><li>entity_relation： 对应的值定义实体关系抽取的指令, 字符串类型，长度范围为[1, 1048576]。<li>event_entity：对应的值定义事件实体抽取的指令, 字符串类型，长度范围为[1, 1048576]。<li>event_relation：对应的值定义事件关系抽取的指令, 字符串类型，长度范围为[1, 1048576]。<br>每个键对应的值定义了特定提取任务的指令。</ul><li>conceptualizer_prompts: 用于指导LLM进行概念化的提示，字典类型。默认值为None。用户可以通过提供一个字典来覆盖默认的概念化提示。该字典必须包含以下键：<ul><li>entity: 对应的值定义对图中实体进行概念化的提示， 字符串类型，长度范围为[1, 1048576]。当conceptualizer_prompts为None时将根据语言使用默认值（中文为ENTITY_PROMPT_CN， 英文为ENTITY_PROMPT_EN）。<li>event: 定义对图中事件进行概念化的提示, 字符串类型，长度范围为[1, 1048576]。当conceptualizer_prompts为None时将根据语言使用默认值（中文为EVENT_PROMPT_CN， 英文为EVENT_PROMPT_EN）。<li>relation: 定义对图中关系进行概念化的提示, 字符串类型，长度范围为[1, 1048576]。当conceptualizer_prompts为None时将根据语言使用默认值（中文为RELATION_PROMPT_CN， 英文为RELATION_PROMPT_EN）。|


**返回值说明<a name="section14945144616426"></a>**

无

方法执行后会在work\_dir下生成过程文件：

**表 1** 

|文件名|说明|
|--|--|
|"{graph_name}.json"|用于保存图，graph_type为"networkx"时，检索会通过该文件加载图。|
|"{graph_name}_relations.json"|保存实体关系信息。|
|"{graph_name}_concepts.json"|保存概念信息。|
|"{graph_name}_synset.json"|保存概念聚类之后的类别信息。|
|"{graph_name}_node_vectors.index"|实体的向量索引文件。|
|"{graph_name}_concept_vectors.index"|概念的向量索引文件。|



#### retrieve\_graph<a name="ZH-CN_TOPIC_0000002340515629"></a>

**功能<a name="section53998444524"></a>**

调用该接口检索返回相关文档片段。

**函数原型<a name="section18789201331417"></a>**

```
def retrieve_graph(question, **kwargs)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|question|str|是|用户问题，字符串长度范围[1, 1000*1000]|
|kwargs|dict|否|扩展参数列表：<li>use_text：布尔类型，默认为True，表示在检索子图时仅使用文本类型的节点包含的文本构建上下文。<li>batch_size：整数类型，默认为4，表示在对节点向量化时的批次大小，其范围为[1, 1024]。<li>similarity_tail_threshold：向量相似阈值，默认为0.0，低于该值将被过滤，其范围为[0.0, 1.0]。<li>retrieval_top_k：整数类型，默认为40，根据实体从节点向量数据库检索相似节点时的topk，其范围为[1, 1000]。<li>reranker_top_k：reranker需要的topk，默认为20，其范围为[1， 1000]。<li>subgraph_depth：整数类型，默认为2，图检索最大探索的深度，其取值范围为[1, 5]。|


**返回值说明<a name="section14945144616426"></a>**

|数据类型|说明|
|--|--|
|List[str]|检索到的上下文片段。|



#### as\_retriever<a name="ZH-CN_TOPIC_0000002306396448"></a>

**功能<a name="section53998444524"></a>**

返回检索器。

**函数原型<a name="section18789201331417"></a>**

```
def as_retriever(**kwargs)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|kwargs|dict|否|扩展参数列表：<li>use_text：布尔类型，默认为True，表示在检索子图时仅使用文本类型的节点包含的文本构建上下文。<li>batch_size：整数类型，默认为4，表示在对节点向量化时的批次大小，其范围为[1, 1024]。<li>similarity_tail_threshold：向量相似阈值，默认为0.0，低于该值将被过滤，其范围为[0.0, 1.0]。<li>retrieval_top_k：整数类型，默认为40，根据实体从节点向量数据库检索相似节点时的topk，其范围为[1, 1000]。<li>reranker_top_k：reranker需要的topk，默认为20，其范围为[1， 1000]。<li>subgraph_depth：整数类型，默认为2，图检索最大探索的深度，其取值范围为[1, 5]。|


**返回值说明<a name="section14945144616426"></a>**

|数据类型|说明|
|--|--|
|GraphRetriever|该检索器继承自langchain_core.retrievers.BaseRetriever。|




### GraphEvaluator<a name="ZH-CN_TOPIC_0000002336118394"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002370116525"></a>

**功能描述<a name="section957011509130"></a>**

用于对知识图谱的质量进行评估。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.graphrag import GraphEvaluator

GraphEvaluator(llm, llm_config)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|llm|Text2TextLLM|必选|大模型接口实例对象。|
|llm_config|LLMParameterConfig|必选|描述参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|


**返回值说明<a name="section53998444524"></a>**

GraphEvaluator对象。

**调用示例<a name="section8509453104117"></a>**

```
import json
from paddle.base import libpaddle
from mx_rag.graphrag.graph_evaluator import GraphEvaluator
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
llm_config = LLMParameterConfig(temperature=0.5, top_p=0.8, max_tokens=8192)
llm = Text2TextLLM(
    base_url="https://ip:port/v1/chat/completions",
    model_name="Llama3-8B-Chinese-Chat",
    llm_config=llm_config,
    client_param=ClientParam(ca_file="/path/to/ca.crt", timeout=120),
)
graph_evaluator = GraphEvaluator(llm, llm_config)
relations_path = "/path/to/graph_relations.json"
with open(relations_path, "r", encoding="utf-8") as f:
    relations = json.load(f)
    graph_evaluator.evaluate(relations)
```


#### evaluate<a name="ZH-CN_TOPIC_0000002370036373"></a>

**功能描述<a name="section53998444524"></a>**

评估大语言模型抽取的三元组关系。

**函数原型<a name="section18789201331417"></a>**

```
def evaluate(relations)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|relations|list[dict]|是|知识图谱关系列表，每个元素为一个字典，必须包括如下键：raw_text，键值entity_relations、event_entity_relations、event_relations可选。列表长度范围[1, 50000]，嵌套深度最高为5，文本长度限制4096。|


**返回值说明<a name="section11818153884917"></a>**

无。函数会打印输出三组精度、召回以及F1得分，分别对应实体，事件中的实体以及事件的抽取情况。输出的得分越高，抽取的质量越好。




