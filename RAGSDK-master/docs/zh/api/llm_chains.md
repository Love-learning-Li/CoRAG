
## 大模型Chain<a name="ZH-CN_TOPIC_0000001981995492"></a>

Chain定义实现了对接大模型客户端。

### Chain抽象类<a name="ZH-CN_TOPIC_0000001982155020"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000001983469186"></a>

**功能描述<a name="section957011509130"></a>**

大模型Chain抽象基类，定义抽象接口。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.chain.base import Chain
Chain()
```


#### query<a name="ZH-CN_TOPIC_0000002020148565"></a>

**功能描述<a name="section177691234412"></a>**

通过query方法调用大模型问答。

**函数原型<a name="section3491114194215"></a>**

```
def query(text, llm_config, *args, **kwargs)
```

**参数说明<a name="section3101137124315"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|text|str|必选|待查询文本信息。|
|llm_config|LLMParameterConfig|可选|调用大模型的参数，描述参考[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|
|args|-|可选|传入的有效参数根据具体的Chain而定。|
|kwargs|-|可选|传入的有效参数根据具体的Chain而定。|




### Text2ImgChain<a name="ZH-CN_TOPIC_0000001981995532"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002018595501"></a>

**功能描述<a name="section439205710473"></a>**

构建文生图对接大模型对象，继承实现抽象Chain类。

**函数原型<a name="section18642312124813"></a>**

```
from mx_rag.chain import Text2ImgChain
Text2ImgChain(multi_model)
```

**参数说明<a name="section1647273884918"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|multi_model|Text2ImgMultiModel|必选|对接大模型对象，传入[Text2ImgMultiModel](./llm_client.md#text2imgmultimodel)实例。|



#### query<a name="ZH-CN_TOPIC_0000002018595241"></a>

**功能描述<a name="section1933110414379"></a>**

从给定的文本提示生成图片。

**函数原型<a name="section1011494243817"></a>**

```
def query(text, llm_config, *args, **kwargs)
```

**参数说明<a name="section4350184332814"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|text|str|必选|图片生成提示词，非空，最大长度为1024*1024。|
|llm_config|LLMParameterConfig|可选|调用大模型的参数，具体说明可参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|
|args|list|可选|继承自基类，未使用。|
|kwargs["output_format"]|str|可选|输出的图片格式，通过kwargs["output_format"]获取，支持["png", "jpeg", "jpg", "webp"]，默认取值为png。|
|kwargs["size"]|str|可选|图片生成尺寸，表示为"height*width"，由入参kwargs传递，具体支持的尺寸由对应的大模型决定，正则匹配格式为"^\d{1,5}\*\d{1,5}$"，默认为512*512。|


**返回值说明<a name="section5555330124016"></a>**

|数据类型|说明|
|--|--|
|Dict,{<br>"prompt": prompt, "result": data}|其中data为图片base64编码后的数据。|


**调用示例<a name="section17535145902914"></a>**

```
from mx_rag.chain import Text2ImgChain
from mx_rag.llm import Text2ImgMultiModel
from mx_rag.utils import ClientParam
client_param = ClientParam(ca_file="/path/to/ca.crt")
multi_model=Text2ImgMultiModel(model_name="sd", url="text to img url", client_param=client_param)
text2img_chain = Text2ImgChain(multi_model=multi_model)
llm_data = text2img_chain.query("dog wearing black glasses", output_format="jpg")
print(llm_data)
```



### Img2ImgChain<a name="ZH-CN_TOPIC_0000002018714905"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002018595397"></a>

**功能描述<a name="section99771719978"></a>**

构建图片生成图对接大模型对象，继承Chain。

**函数原型<a name="section33741681180"></a>**

```
from mx_rag.chain import Img2ImgChain
Img2ImgChain(multi_model, retriever)
```

**参数说明<a name="section1873013175815"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|multi_model|Img2ImgMultiModel|必选|对接大模型对象，[Img2ImgMultiModel](./llm_client.md#img2imgmultimodel)实例。|
|retriever|BaseRetriever|必选|相似检索器，[Retriever](./retrieval.md#retriever)实例。|



#### query<a name="ZH-CN_TOPIC_0000001982155060"></a>

**功能描述<a name="section1945141620153"></a>**

根据文本检索出相关图片，结合提示词发送给大模型生成图片。

**函数原型<a name="section17985124181613"></a>**

```
def query(text, llm_config, *args, **kwargs)
```

**参数说明<a name="section1358545161816"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|text|str|必选|检索图片的描述文字，长度取值：(0, 1* 1000 * 1000]。|
|llm_config|LLMParameterConfig|可选|继承父类方法，此处未使用。|
|args|list|可选|当前chain未使用。|
|kwargs["prompt"]|str|必选|图片生成提示词，由入参kwargs传递，长度取值：(0, 1* 1024 * 1024]。|
|kwargs["size"]|str|可选|图片生成尺寸，表示为"height*width"，由入参kwargs传递，具体支持的尺寸由对应的大模型决定，正则匹配格式为: "^\d{1,5}\\\*\d{1,5}$"，默认512\*512。|


**返回值说明<a name="section57446519215"></a>**

|数据类型|说明|
|--|--|
|Dict,{"prompt": prompt, "result": data}|其中data为图片base64编码后的数据。|


**调用示例<a name="section055181421218"></a>**

```
# 该用例基于知识库中已上传的图片检索出相关图片，结合提示词发送给大模型生成图片
from paddle.base import libpaddle
from mx_rag.chain import Img2ImgChain
from mx_rag.llm import Img2ImgMultiModel
from mx_rag.retrievers import Retriever
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.embedding.local import ImageEmbedding
from mx_rag.utils import ClientParam
dev = 0
img_emb = ImageEmbedding(model_name="ViT-B-16", model_path="/path/to/chinese-clip-vit-base-patch16", dev_id=dev)
img_vector_store = MindFAISS(x_dim=512,
                             devs=[dev],
                             load_local_index="/path/to/image_faiss.index",
                             auto_save=True)
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
img_retriever = Retriever(vector_store=img_vector_store, document_store=chunk_store,
                          embed_func=img_emb.embed_documents, k=1, score_threshold=0.5)
multi_model = Img2ImgMultiModel(model_name="sd",
                                url="img to image url",
                                client_param=ClientParam(ca_file="/path/to/ca.crt"))
img2img_chain = Img2ImgChain(multi_model=multi_model, retriever=img_retriever)
llm_data = img2img_chain.query("查找小男孩图片",
                               prompt="he is a knight, wearing armor, big sword in right hand. Blur the background, focus on the knight")
print(llm_data)
```



### SingleText2TextChain<a name="ZH-CN_TOPIC_0000002018595485"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000001981995500"></a>

**功能描述<a name="section957011509130"></a>**

单轮对话Chain，实现基本的问答对话功能，继承Chain，参考[基本对话功能](#section175571825169)。也可实现图文并茂对话功能，参考[图文并茂对话功能](#section175571825169)

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.chain import SingleText2TextChain
SingleText2TextChain(llm, retriever, reranker, prompt, sys_messages, source, user_content_builder)
```

**参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|llm|Text2TextLLM|必选|大模型对象，具体可参见[Text2TextLLM](./llm_client.md#text2textllm)。|
|retriever|Retriever|必选|Retriever对象，具体可参见[Retriever](./retrieval.md#retriever)。|
|reranker|Reranker|可选|Reranker对象，实现对检索的文档进行精排，默认为None，具体可参见Reranker。|
|prompt|str|可选|添加知识检索内容同时可以添加系统prompt，对大模型进行更精确的控制，默认值是："根据上述已知信息，简洁和专业地回答用户的问题。如果无法从已知信息中得到答案，请根据自身经验做出回答"。如果用户需要自定义prompt，请参考大模型的提示词工程增加。长度取值范围：[1, 1024*1024]|
|sys_messages|List[dict]|可选|系统消息，默认值为None，列表最大长度为16，列表每个字典长度最大为16，字典key字符串长度最大为16，value字符串最大长度为4 * 1024* 1024，参考格式：[{"role": "system", "content":"你是一个友好助手" }]|
|source|bool|可选|在对话过程中，是否返回检索到的相关文档，Chain返回字典中key值为source_documents，默认为True。|
|user_content_builder|Callable|可选|回调函数，返回值必须为字符串且长度最大为4*1024*1024，默认函数为_user_content_builder，功能是整合「原始问题、检索到的文档列表、用户提示词」这三类信息，生成可直接作为大模型对话中 user 角色消息 content 字段的文本（即 {"role": "user", "content": 生成结果}）。|


-   <a name="li19440111017591"></a>参数user\_content\_builder的默认函数：

```
def _user_content_builder(query: str, docs: List[Document], prompt: str) -> str:
    """
       默认的用户输入拼接逻辑。
       参数说明：
       ----------
       query : str
           用户原始提问内容。
           例如：“请根据以下材料总结关键要点。”
       docs : List[Document]
           从检索器（retriever）返回的文档对象列表。
           每个 Document 通常包含：
           - page_content：文档内容文本；
           - metadata：元信息（如来源、标题、分数等）。
       prompt : str
           系统提示词。默认为"根据上述已知信息，简洁和专业地回答用户的问题。
           如果无法从已知信息中得到答案，请根据自身经验做出回答"
       返回：
       -----
       str : 拼接后的完整 prompt 文本，作为大模型输入内容。
       """
    final_prompt = ""
    document_separator: str = "\n\n"
    if len(docs) != 0:
        if prompt != "":
            last_doc = docs[-1]
            last_doc.page_content = (last_doc.page_content
                                     + f"{document_separator}{prompt}")
            docs[-1] = last_doc
        final_prompt = document_separator.join(x.page_content for x in docs)
    if final_prompt != "":
        final_prompt += document_separator
    final_prompt += query
    return final_prompt
```


#### query<a name="ZH-CN_TOPIC_0000002018714737"></a>

**功能描述<a name="section5434255810"></a>**

RAG SDK对话功能。

**函数原型<a name="section18789201331417"></a>**

```
def query(text, llm_config, *args, **kwargs) 
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|text|str|必选|原始问题，取值范围为(0, 1000*1000]|
|llm_config|LLMParameterConfig|可选|调用大模型参数，此处默认值temperature为0.5，top_p为0.95，其余参数说明请参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|
|args|列表|可选|继承父类方法签名，此处未使用。|
|kwargs|字典|可选|继承父类方法签名，此处未使用。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|Union[Dict, Iterator[Dict]]|大模型返回结果，其中Dict内容为：带知识来源：{"query": query, "result": data, "source_documents": [{'metadata': xxx, 'page_content': xxx}]}不带知识来源：{"query": query, "result": data}|


**调用示例<a id="section175571825169"></a>**

-   **基本对话功能**

```
from paddle.base import libpaddle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.chain import SingleText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader, PowerPointLoader
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeDB, KnowledgeStore
from mx_rag.knowledge.handler import upload_files
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.llm import Text2TextLLM, Img2TextLLM, LLMParameterConfig
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.utils import ClientParam
from mx_rag.llm.llm_parameter import LLMParameterConfig

loader_mng = LoaderMng()
# 加载文档加载器，可以使用mxrag自有的，也可以使用langchain的
loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"])
loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])
loader_mng.register_loader(loader_class=PowerPointLoader, file_types=[".pptx"])
# 加载文档切分器，使用langchain的
loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                             file_types=[".pdf", ".docx", ".txt", ".md", ".xlsx", ".pptx"],
                             splitter_params={"chunk_size": 750,
                                              "chunk_overlap": 150,
                                              "keep_separator": False })

dev = 0
# 加载embedding模型
emb = TextEmbedding("/path/to/acge_text_embedding/", dev_id=dev)
# 初始化向量数据库
vector_store = MindFAISS(x_dim=1024,  devs=[dev],
                                 load_local_index="/path/to/faiss.index",
                                 auto_save=True)
# 初始化文档chunk关系数据库
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
# 初始化知识管理关系数据库
knowledge_store = KnowledgeStore(db_path="/path/to/sql.db")
# 添加知识库
knowledge_store.add_knowledge("test", "Default", "admin")
# 初始化知识库管理
knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                           chunk_store=chunk_store,
                           vector_store=vector_store,
                           knowledge_name="test",
                           white_paths=["/path/"],
                           user_id="Default"
                           )
# 上传文档到知识库
upload_files(knowledge_db, ["/path/to/file1", "/path/to/file2"], loader_mng, emb.embed_documents, True)
client_param = ClientParam(ca_file="/path/to/ca.crt")
llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct", 
                   base_url="https://x.x.x.x:port/v1/chat/completions", 
                   client_param=client_param)
r = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents, k=1, score_threshold=0.6)
rag = SingleText2TextChain(retriever=r, llm=llm)
response = rag.query("mxVision软件架构包含哪些模块？", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
print(response)

```

-   **图文并茂对话功能**

```
from paddle.base import libpaddle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.chain import SingleText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader, PowerPointLoader
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeDB, KnowledgeStore
from mx_rag.knowledge.handler import upload_files
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.llm import Text2TextLLM, Img2TextLLM, LLMParameterConfig
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.utils import ClientParam
from mx_rag.llm.llm_parameter import LLMParameterConfig
from typing import List
from langchain_core.documents import Document

# 加载用于解析文档中图片的视觉大模型
vlm = Img2TextLLM(base_url="https://x.x.x.x:port/openai/v1/chat/completions",
                   model_name="Qwen2.5-VL-7B-Instruct",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
loader_mng = LoaderMng()
# 文档加载器，可以使用mxrag自有的，也可以使用langchain的
loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"], loader_params={"vlm": vlm})
loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"], loader_params={"vlm": vlm})
loader_mng.register_loader(loader_class=PowerPointLoader, file_types=[".pptx"], loader_params={"vlm": vlm})
# 加载文档切分器，使用langchain的
loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                             file_types=[".pdf", ".docx", ".txt", ".md", ".xlsx", ".pptx"],
                             splitter_params={"chunk_size": 750,
                                              "chunk_overlap": 150,
                                              "keep_separator": False })

dev = 0
# 加载embedding模型
emb = TextEmbedding("/path/to/acge_text_embedding/", dev_id=dev)
client_param = ClientParam(ca_file="/path/to/ca.crt")
# 初始化向量数据库
vector_store = MindFAISS(x_dim=1024,  devs=[dev],
                                 load_local_index="/path/to/faiss.index",
                                 auto_save=True)
# 初始化文档chunk关系数据库
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
# 初始化知识管理关系数据库
knowledge_store = KnowledgeStore(db_path="/path/to/sql.db")
# 添加知识库
knowledge_store.add_knowledge("test", "Default", "admin")
# 初始化知识库管理
knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                           chunk_store=chunk_store,
                           vector_store=vector_store,
                           knowledge_name="test",
                           white_paths=["/path/"],
                           user_id="Default"
                           )
# 上传文档到知识库
upload_files(knowledge_db, ["/path/to/file1", "/path/to/file2"], loader_mng, emb.embed_documents, True)
# 定义回调函数，整合问题和检索到的文本与图片描述，生成大模型对话中角色user的content的内容
def user_content_builder(query: str, docs: List[Document], *args, **kwargs):
       """
       参数说明：
       query : str, 用户原始提问内容。例如：“请根据以下材料总结关键要点。”
       docs : List[Document],从检索器（retriever）返回的文档对象列表。
              每个 Document 通常包含：page_content：文档内容文本；metadata：元信息（如来源、标题、分数等）。
       返回：
       str : 拼接后的完整 prompt 文本，作为大模型输入内容。
       """
    text_docs = [doc for doc in docs if doc.metadata.get("type", "") == "text"]
    img_docs = [doc for doc in docs if doc.metadata.get("type", "") == "image"]
    user_message = []
    if len(text_docs) > 0:
        # 2. Add text quotes
        user_message.append(f"Text Quotes are:")
        for i, doc in enumerate(text_docs):
            user_message.append(f"\n[{i + 1}] {doc.page_content}")
    if len(img_docs) > 0:
        # 3. Add image quotes vlm-text or ocr-text
        user_message.append("\nImage Quotes are:")
        for i, doc in enumerate(img_docs):
            user_message.append(f"\nimage{i + 1} is described as: {doc.page_content}")
    user_message.append("\n\n")
    # 4. add user question
    user_message.append(f"The user question is: {query}")
    return ''.join(user_message)

# 系统prompt
TEXT_INFER_PROMPT = '''
You are a helpful question-answering assistant. Your task is to generate a interleaved text and image response based on provided questions and quotes. Here‘s how to refine your process:

1. **Evidence Selection**:
   - From both text and image quotes, pinpoint those really relevant for answering the question. Focus on significance and direct relevance.
   - Each image quote is the description of the image.

2. **Answer Construction**:
   - Use Markdown to embed text and images in your response, avoid using obvious headings or divisions; ensure the response flows naturally and cohesively.
   - Conclude with a direct and concise answer to the question in a simple and clear sentence.

3. **Quote Citation**:
   - Cite images using the format `![{conclusion}](image index)`; for the first image, use `![{conclusion}](image1)`;The {conclusion} should be a concise one-sentence summary of the image’s content.
   - Ensure the cite of the image must strict follow `![{conclusion}](image index)`, do not simply stating "See image1", "image1 shows" ,"[image1]" or "image1".
   - Each image or text can only be quoted once.

- Do not cite irrelevant quotes.
- Compose a detailed and articulate interleaved answer to the question.
- Ensure that your answer is logical, informative, and directly ties back to the evidence provided by the quotes.
- If Quote contain text and image, answer must contain both text and image response.
- If Quote only contain text, answer must contain text response, do not contain image.
- Answer in chinese.
'''

client_param = ClientParam(ca_file="/path/to/ca.crt")
# 用于对话的语言大模型
llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct", 
                   base_url="https://x.x.x.x:port/v1/chat/completions", 
                   client_param=client_param)
sys_messages=[{"role": "system", "content": TEXT_INFER_PROMPT}]
r = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents, k=1, score_threshold=0.6)
rag = SingleText2TextChain(retriever=r, llm=llm, sys_messages=sys_messages, user_content_builder=user_content_builder)
response = rag.query("mxVision软件架构包含哪些模块？", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
# 回答的source_documents中可能会有图片，可在字典metadata中获取图片base64编码
print(response)
```



### ParallelText2TextChain<a name="ZH-CN_TOPIC_0000001987173374"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002023612945"></a>

**功能描述<a name="section957011509130"></a>**

支持检索并行推理文生文对话Chain。继承了SingleText2TextChain基类，节省检索时延。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.chain import ParallelText2TextChain
class ParallelText2TextChain(SingleText2TextChain)
```

**参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|llm|Text2TextLLM|必选|大模型对象，具体可参见[Text2TextLLM](./llm_client.md#text2textllm)。|
|retriever|Retriever|必选|Retriever对象，具体可参见[Retriever](./retrieval.md#retriever)。|
|reranker|Reranker|可选|Reranker对象，实现对检索的文档进行精排，默认为None，具体可参见[Reranker](./reranker.md#reranker)。|
|prompt|str|可选|添加知识检索内容同时可以添加系统prompt，对大模型进行更精确的控制，默认值是："根据上述已知信息，简洁和专业地回答用户的问题。如果无法从已知信息中得到答案，请根据自身经验做出回答"。如果用户需要自定义prompt，请参考大模型的提示词工程增加。长度取值范围：[1, 1024*1024]|
|sys_messages|List[dict]|可选|系统消息，默认值为None，列表最大长度为16，列表每个字典长度最大为16，字典key字符串长度最大为16，value字符串最大长度为4 * 1024* 1024，参考格式：[{"role": "system", "content":"你是一个友好助手" }]|
|source|bool|可选|在对话过程中，是否返回检索到的相关文档，Chain返回字典中key值为source_documents，默认为True。|
|user_content_builder|Callable|可选|回调函数，返回值必须为字符串且长度最大为4*1024*1024，默认函数为_user_content_builder，功能是整合「原始问题、检索到的文档列表、用户提示词」这三类信息，生成可直接作为大模型对话中 user 角色消息 content 字段的文本（即 {"role": "user", "content": 生成结果}）。|


-   <a name="zh-cn_topic_0000001981995500_li19440111017591"></a>参数user\_content\_builder的默认函数：

```
def _user_content_builder(query: str, docs: List[Document], prompt: str) -> str:
    """
       默认的用户输入拼接逻辑。
       参数说明：
       ----------
       query : str
           用户原始提问内容。
           例如：“请根据以下材料总结关键要点。”
       docs : List[Document]
           从检索器（retriever）返回的文档对象列表。
           每个 Document 通常包含：
           - page_content：文档内容文本；
           - metadata：元信息（如来源、标题、分数等）。
       prompt : str
           系统提示词。默认为"根据上述已知信息，简洁和专业地回答用户的问题。
           如果无法从已知信息中得到答案，请根据自身经验做出回答"
       返回：
       -----
       str : 拼接后的完整 prompt 文本，作为大模型输入内容。
       """
    final_prompt = ""
    document_separator: str = "\n\n"
    if len(docs) != 0:
        if prompt != "":
            last_doc = docs[-1]
            last_doc.page_content = (last_doc.page_content
                                     + f"{document_separator}{prompt}")
            docs[-1] = last_doc
        final_prompt = document_separator.join(x.page_content for x in docs)
    if final_prompt != "":
        final_prompt += document_separator
    final_prompt += query
    return final_prompt
```


#### query<a name="ZH-CN_TOPIC_0000002023613429"></a>

**功能描述<a name="section5434255810"></a>**

RAG SDK对话功能。

**函数原型<a name="section18789201331417"></a>**

```
def query(text: str, llm_config, *args, **kwargs)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|text|str|必选|原始问题，非空，最大的问题长度为1000*1000。|
|llm_config|LLMParameterConfig|可选|调用大模型参数，此处默认值temperature为0.5，top_p为0.95，其余参数说明请参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|
|args|列表|可选|继承父类方法签名，此处未使用。|
|kwargs|字典|可选|继承父类方法签名，此处未使用。|


**返回值说明<a name="section15414449287"></a>**

|数据类型|说明|
|--|--|
|Union[Dict, Iterator[Dict]]|返回字典或者迭代器，stream设置成True表示返回迭代器，否则返回字典。其中Dict内容为：<li>带知识来源：{"prompt": prompt, "result": data, "source_documents": [{'metadata': xxx, 'page_content': xxx}]}<li>不带知识来源：{"prompt": prompt, "result": data}|


**调用示例<a name="section352434893611"></a>**

```
from mx_rag.chain import ParallelText2TextChain
from mx_rag.llm import Text2TextLLM
from mx_rag.embedding.local import TextEmbedding
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.retrievers import Retriever
from mx_rag.utils import ClientParam
dev = 0
emb = TextEmbedding("/path/to/acge_text_embedding/", dev_id=dev)
client_param = ClientParam(ca_file="/path/to/ca.crt")
llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct",
                   base_url="https://x.x.x.x:port/v1/chat/completions",
                   client_param=client_param)
vector_store = MindFAISS(x_dim=1024,  devs=[dev],
                                 load_local_index="/path/to/faiss.index",
                                 auto_save=True)
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
retriever = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents, k=1, score_threshold=0.6)
parallel_chain = ParallelText2TextChain(llm=llm, retriever=retriever)
answer = parallel_chain.query(text="123456")
print(answer)
```



### GraphRagText2TextChain<a name="ZH-CN_TOPIC_0000002195845708"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002195686116"></a>

**功能描述<a name="section957011509130"></a>**

知识图谱Chain，继承自[SingleText2TextChain](#singletext2textchain)，调用示例可参考：[调用示例](./knowledge_graph.md#类功能))。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.chain.single_text_to_text import GraphRagText2TextChain
GraphRagText2TextChain(llm, retriever, reranker)
```

**输入参数说明<a name="section207808578417"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|llm|Text2TextLLM|必选|大模型对象，具体可参见[Text2TextLLM](./llm_client.md#text2textllm)。|
|retriever|GraphRetriever|必选|GraphRetriever对象，由[GraphRAGPipeline](./knowledge_graph.md#graphragpipeline)的[as_retriever](./knowledge_graph.md#as_retriever)方法返回。|
|reranker|Reranker|可选|Reranker对象，实现对检索的文档进行精排，默认为None，具体可参见[Reranker](./reranker.md#reranker)。|


其余参数见父类。


#### 函数query<a name="ZH-CN_TOPIC_0000002231091445"></a>

**功能描述<a name="section53998444524"></a>**

调用此接口进行知识图谱知识问答。

**函数原型<a name="section18789201331417"></a>**

```
def query(text, llm_config, *args, **kwargs)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|text|str|必选|原始问题，取值范围为(0, 1000*1000]|
|llm_config|LLMParameterConfig|可选|调用大模型参数，此处默认值temperature为0.5，top_p为0.95，其余参数说明请参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|
|args|列表|可选|继承父类方法签名，此处未使用。|
|kwargs|字典|可选|继承父类方法签名，此处未使用。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|dict|返回大模型回答，格式{'query': "who is Teutberga's parents?", 'result': "Teutberga's parents are Bosonid Boso the Elder and an unknown mother."}|




