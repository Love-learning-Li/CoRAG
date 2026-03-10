
## 文档总结<a name="ZH-CN_TOPIC_0000002026578193"></a>

### Summary<a name="ZH-CN_TOPIC_0000001989939282"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002026657721"></a>

**功能描述<a name="section957011509130"></a>**

该类实现了对文档提取总结内容。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.summary import Summary
Summary(llm, llm_config)
```

**参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|llm|Text2TextLLM|必选|大模型对象实例，具体类型请参考[Text2TextLLM](./llm_client.md#text2textllm)。|
|llm_config|LLMParameterConfig|可选|调用大模型参数，此处默认值temperature为0.5，top_p为0.95，其余参数说明请参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|


**调用示例<a name="section129100236713"></a>**

```
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mx_rag.document.loader import DocxLoader
from mx_rag.llm import Text2TextLLM
from mx_rag.summary import Summary
from mx_rag.utils import ClientParam
client_param = ClientParam(ca_file="/path/to/ca.crt")
llm = Text2TextLLM(base_url="https://ip:port/v1/chat/completions", model_name="qianwen-7b", client_param=client_param)
loader=DocxLoader("/home/HwHiAiUser/MindIE.docx")
docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150))
summary = Summary(llm=llm)
# 调用summarize方法
sub_summaries = summary.summarize([doc.page_content for doc in docs])
# 调用merge_text_summarize方法
res = summary.merge_text_summarize(sub_summaries)
print(res)
```


#### summarize<a name="ZH-CN_TOPIC_0000001990103142"></a>

**功能描述<a name="section1933110414379"></a>**

对文档通过大模型提取总结内容。

**函数原型<a name="section1011494243817"></a>**

```
def summarize(texts, not_summarize_threshold, prompt)
```

**参数说明<a name="section4350184332814"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|texts|List[str]|必选|输入的文本列表，列表中的所有文本长度和的取值范围：（0, 1024*1024]，列表长度取值范围：（0,1024]。|
|not_summarize_threshold|int|可选|单次总结时由于给定的文本太短，大模型无法进行总结或总结错误，此值设定需要大模型进行总结的文本长度阈值，如果给定的文本内容小于等于not_summarize_threshold，不调用大模型进行总结操作，总结内容为文本原始内容，默认值为30，取值范围（0, 1024*1024]。|
|prompt|langchain_core.prompts.PromptTemplate|可选|默认值如下，prompt中input_variables必须等于["text"]，表示输入的文本，template长度取值范围（0，1024 * 1024]。实际请求大模型的query为prompt拼接text，其有效取值依赖MindIE的配置，请参见《MindIE LLM开发指南》中的“核心概念与配置 > 配置参数说明（服务化）”章节中关于maxSeqLen的说明。注意：prompt和text的语言类型最好保持一致或者指明大模型返回语言类型，否则会影响大模型回答效果。<br>_SUMMARY_TEMPLATE = PromptTemplate(input_variables=["text"],<br>template="""使用简洁的语言提取以下内容的摘要，包含尽可能多的关键信息，输出只包含内容信息，请用中文回答\n\n{text}""")。|


**返回值说明<a name="section5555330124016"></a>**

|数据类型|说明|
|--|--|
|List[str]|对应总结后的文本列表|



#### merge\_text\_summarize<a name="ZH-CN_TOPIC_0000002026662581"></a>

**功能描述<a name="section1933110414379"></a>**

由于大模型输入token限制，长文本总结需要进行拆分成多个短文本，然后对短文本总结得到子总结，再对子总结进行合并再次通过大模型总结，多次迭代（最大10次）得到最终总结。

**函数原型<a name="section1011494243817"></a>**

```
def merge_text_summarize(texts, merge_threshold, not_summarize_threshold, prompt)
```

**参数说明<a name="section4350184332814"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|texts|List[str]|必选|文本子总结列表，列表中所有文本长度总和的取值范围（0, 1024*1024]，列表长度取值范围：（0,1024]。|
|merge_threshold|int|可选|合并总结时由于大模型token限制，需对子总结列表进行拆分发送给大模型合并总结，此值用于设置拆分门限值，保证每个拆分后的总长度不大于门限值，默认值为4 * 1024，取值范围：[1024, 1024*1024]，merge_threshold参数值大于not_summarize_threshold值。|
|not_summarize_threshold|int|可选|单次总结时由于给定的文本太短大模型无法进行总结或总结错误，此值设定需要大模型进行总结的文本长度阈值，如果给定的文本内容小于等于not_summarize_threshold，不调用大模型进行总结操作，总结内容为文本原始内容，默认值为30，取值范围：（0, 1024*1024]。|
|prompt|langchain_core.prompts.PromptTemplate|可选|默认值如下，prompt中input_variables必须等于["text"]，template长度取值范围：（0，1024 * 1024]，实际请求大模型的query为prompt拼接text，其有效取值依赖MindIE的配置，请参见《MindIE LLM开发指南》中的“核心概念与配置 > 配置参数说明（服务化）”章节中关于maxSeqLen的说明。<br>注意：prompt和text的语言类型最好保持一致，或者指明大模型回答语言类型，否则会影响大模型回答效果。<br>PromptTemplate(<br>input_variables=["text"],<br>template="""使用简洁的语言把下面的多个摘要提炼合并成一个摘要，包含尽可能多的关键信息，输出只包含内容信息，请用中文回答\n\n{text}""")|


**返回值说明<a name="section5555330124016"></a>**

|数据类型|说明|
|--|--|
|str|返回合并总结后的最终总结内容。|




