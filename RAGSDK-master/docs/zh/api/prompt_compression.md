## Prompt压缩<a name="ZH-CN_TOPIC_0000002319268421"></a>

### PromptCompressor<a name="ZH-CN_TOPIC_0000002284793848"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002319451761"></a>

**功能描述<a name="section29524313490"></a>**

prompt压缩抽象类

**函数原型<a name="section546771414342"></a>**

```
from mx_rag.compress.base_compressor import PromptCompressor
class PromptCompressor(ABC)
```


#### compress\_texts<a name="ZH-CN_TOPIC_0000002284802432"></a>

**功能描述<a name="section1031631414920"></a>**

压缩prompt文本

**函数原型<a name="section1247913102108"></a>**

```
@abstractmethod
def compress_texts(self, context, question)
```

**参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|context|str|必选|待总结的长文本。|
|question|str|必选|总结长文本的指令。|




### RerankCompressor<a name="ZH-CN_TOPIC_0000002319290097"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002284699168"></a>

**功能描述<a name="section5434255810"></a>**

通过排序模型计算question（总结长文本的指令）和context（总结长文本的指令）切片之间的相关性得分，根据设定的压缩率阈值，优先保留相关性高的切片，从而实现对长文本的有效压缩。

**函数原型<a name="section18789201331417"></a>**

```
from mx_rag.compress.rerank_compressor import RerankCompressor
class RerankCompressor(reranker, splitter)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|reranker|Reranker|必选|排序模型实例，实现对文本切片进行精排，只能为mx_rag.reranker的Reranker对象，具体可参见[Reranker](./reranker.md#reranker)。|
|splitter|TextSplitter|可选|文档切分函数，只能为继承自langchain的TextSplitter的子类。默认为langchain.text_splitter的RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0, separators=["\n", ""], keep_separator=True)|


**调用示例<a name="section11818153884917"></a>**

```
from mx_rag.compress.rerank_compressor import RerankCompressor
from mx_rag.reranker.local import LocalReranker
from mx_rag.reranker.service import TEIReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.utils import ClientParam

context="""需要压缩的prompt文本"""
question="请给上述内容起一个标题"
tei_reranker=False
if tei_reranker:
    reranker = TEIReranker.create(url="https://ip:port/rerank",
                            client_param=ClientParam(ca_file="/path/to/ca.crt"))
else:
    reranker = LocalReranker(model_path="reranker_path", dev_id=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0, separators=["\n", ""], keep_separator=True)
compressor=RerankCompressor(reranker=reranker, splitter=text_splitter)
res=compressor.compress_texts(context, question, 0.3)
print(res)
```


#### compress\_texts<a name="ZH-CN_TOPIC_0000002284690566"></a>

**功能描述<a name="section5434255810"></a>**

根据指令（question）、长文本（context）以及压缩率（compress\_rate）压缩文本

**函数原型<a name="section18789201331417"></a>**

```
def compress_texts(context, question, compress_rate, context_reorder)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|context|str|必选|待总结的长文本。长度范围：[1, 16MB]|
|question|str|必选|总结长文本的指令，用于计算与context文本切片的相关性。长度范围：[1, 1000*1000]|
|compress_rate|float|可选|压缩率，默认为0.6，取值范围：(0, 1)|
|context_reorder|bool|可选|是否根据得分重排，默认为False，若为True，计算完相关性之后，将根据压缩率优先保留相关性低的文本切片。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|str|压缩后的文本。|




### ClusterCompressor<a name="ZH-CN_TOPIC_0000002319443213"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002319298729"></a>

**功能描述<a name="section5434255810"></a>**

通过聚类模型将嵌入后的文本进行聚类，将其划分为多个语义簇。随后，计算context的切片与question（总结长文本的指令）的余弦相似度。根据设定的压缩率，在每个簇内删除相似度较低的切片，从而保留与指令最相关的信息，实现长文本的压缩式总结。

**函数原型<a name="section18789201331417"></a>**

```
from mx_rag.compress.cluster_compressor import ClusterCompressor
class ClusterCompressor(cluster_func, embed, splitter, dev_id):
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|cluster_func|Callable[[List[List[float]]], Union[List[int], np.ndarray]]|必选|聚类函数，将嵌入后的文本切片进行聚类，将其划分为多个语义簇，返回的结果必须为List[int]或ndarray，长度不能超过1000*1000，且长度要和文本切片数量一致。|
|embed|Embeddings|必选|嵌入对象，把文本切片转换为向量，只能为继承自langchain_core.embeddings的Embeddings的子类。|
|splitter|TextSplitter|可选|文档切分对象，只能为继承自langchain_text_splitters.base的TextSplitter的子类。默认为langchain.text_splitter的RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators=["。", "！", "？", "\n", "，", "；", " ", ""])|
|dev_id|int|可选|NPU id，通过**npu-smi info**查询可用ID，取值范围[0, 63]，默认为卡0。|


**调用示例<a name="section11818153884917"></a>**

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import HDBSCAN
from mx_rag.compress.cluster_compressor import ClusterCompressor
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.utils import ClientParam

context="""需要压缩的prompt文本"""
question="请给上述内容起一个标题"
tei_emb=False
if tei_emb:
    emb = TEIEmbedding.create(url="https://ip:port/embed", client_param=ClientParam(ca_file="/path/to/ca.crt"))
else:
    emb = TextEmbedding(model_path="embedding_path", dev_id=0)
def _get_community(sentences_embedding):
    # 社区划分
    node_num=len(sentences_embedding)
    min_cluster_size=2
    hdbscan = HDBSCAN(min_cluster_size=min(min_cluster_size, node_num))
    labels = hdbscan.fit_predict(sentences_embedding)
    return labels
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators=["。", "！", "？", "\n", "，", "；", " ", ""], )
compressor=ClusterCompressor(cluster_func=_get_community, embed=emb, splitter=splitter, dev_id=0)
res=compressor.compress_texts(context, question, 0.6)
print(res)
```


#### compress\_texts<a name="ZH-CN_TOPIC_0000002284793852"></a>

**功能描述<a name="section5434255810"></a>**

根据提供的指令（question）、长文本（context）和压缩率（compress\_rate）压缩文本。

**函数原型<a name="section18789201331417"></a>**

```
def compress_texts(context, question, compress_rate)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|context|str|必选|待总结的长文本。长度范围：[1, 16MB]|
|question|str|必选|总结长文本的指令，用于计算与context文本切片的相关性。长度范围：[1, 1000*1000]|
|compress_rate|float|可选|压缩率，默认值为0.6，取值范围：(0, 1)|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|str|压缩后的文本。|




