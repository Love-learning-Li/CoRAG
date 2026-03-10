## 缓存模块<a name="ZH-CN_TOPIC_0000002018595221"></a>

### 总体说明<a name="ZH-CN_TOPIC_0000001984862788"></a>

MxRAGCache的功能主要是基于开源组件GPTCache进行二次开发，支持以下cache基本功能：

-   Cache初始化
-   Cache更新
-   Cache老化
-   Cache查询
-   Cache级联

相比GPTCache，MxRAGCache扩展了以下功能：

-   语义近似Cache矢量检索过程支持FAISS\_NPU检索（Index SDK）。
-   语义近似Cache embedding支持RAG优化的TEI Embedding。
-   语义近似Cache相似度计算过程支持RAG优化的TEI Reranker。
-   支持RAG SDK  chain（图生图，文生文，文生图）的Cache功能。

在原始RAG SDK流程中，在知识文档检索前，增加问答Cache，如果查询命中了Cache，就不会经过大模型推理过程，节省了知识文档检索和大模型推理时延，提升了端到端性能，经过性能测试，缓存命中相比缓存未命中可以提升50倍性能。


### 模块介绍<a name="ZH-CN_TOPIC_0000002018595301"></a>

#### 缓存类型<a name="ZH-CN_TOPIC_0000002018714985"></a>

##### 章节介绍<a name="ZH-CN_TOPIC_0000002021663593"></a>

缓存类型分为memory cache（hash map + 精确匹配）和similarity cache（向量数据库 + embedding + 相似度计算），其中memory cache用于用户问题完全匹配场景，similarity cache用于用户问题不完全一样，但是比较相似的场景。在实际使用的过程中，也可以将Cache进行串联。


##### memory cache<a name="ZH-CN_TOPIC_0000001982155332"></a>

Memory cache完全匹配的cache，在内存采用的是hash map结构，hash key是用户的query hash值，hash value是用户的问题，用户问题需要满足完全匹配。

**图 1**  memory cache结构<a name="fig855724183211"></a>  
![](../figures/memory-cache结构.png "memory-cache结构")


##### similarity cache<a name="ZH-CN_TOPIC_0000001982155272"></a>

Similarity cache是语义相似匹配的cache，存储结构是sqlite + 向量数据库（faiss, npu\_faiss,milvus）。

查询时首先对用户的问题做embedding，从向量数据库查询相似TOPK的结果，然后从sqlite获取缓存答案和问题，再将缓存的问题和用户的问题进行reranker精排，得到最相似的结果返回给用户。该cache不需要满足完全匹配，只需要语义相似即可命中。

**图 1**  similarity cache结构<a name="fig7942321123315"></a>  
![](../figures/similarity-cache结构.png "similarity-cache结构")




### 缓存配置<a name="ZH-CN_TOPIC_0000001981995432"></a>

#### 章节介绍<a name="ZH-CN_TOPIC_0000002021546637"></a>

本章主要用于阐述提供给用户进行配置数据的内容。


#### CacheConfig<a name="ZH-CN_TOPIC_0000002018595477"></a>

##### 类功能<a id="ZH-CN_TOPIC_0000002020105781"></a>

**功能描述<a name="section714219220293"></a>**

用于配置memory cache的配置数据结构。

**函数原型<a name="section339074703020"></a>**

```
from gptcache.config import Config
from mx_rag.cache import CacheConfig
# 继承自Config
CacheConfig(cache_size, eviction_policy, auto_flush, data_save_folder, min_free_space, similarity_threshold, disable_report, lock)
```

**输入参数说明<a name="section680452314333"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|cache_size|int|必选|缓存大小，配置缓存条目数。<br>cache_size不能小于等于0。<br>取值范围(0, 100000]|
|eviction_policy|EvictPolicy|可选|缓存老化策略。<br>默认值为LRU，具体可参考[EvictPolicy](#evictpolicy)。|
|auto_flush|int|可选|数据落盘频度，即缓存多少条条目之后，进行一次落盘操作。<br>默认值：20<br>取值范围(0, cache_size]|
|data_save_folder|str|可选|缓存落盘路径，路径长度不能超过1024，不能为软链接和相对路径。<li>目录下的各文件大小不能超过100GB、深度不超过64，且文件总个数不超过512。<li>运行用户的属组，以及非运行用户不能有该目录下文件的写权限。<li>目录下的文件以及文件的上一级目录的属组必须是运行用户。<br>默认值为当前用户家目+"/Ascend/mxRag/cache_save_folder"，若不存在需要用户创建。存放路径不能在路径列表中：["/etc", "/usr/bin", "/usr/lib", "/usr/lib64", "/sys/", "/dev/", "/sbin", "/tmp"]。|
|min_free_space|int|可选|用于检查落盘路径的可用空间，单位字节，默认值为1GB。<br>取值范围[20MB, 100GB]|
|similarity_threshold|float|可选|相似度计算阈值<br>默认值：0.8<br>取值范围[0.0, 1.0]|
|disable_report|bool|可选|是否需要支持维测数据功能<br>默认值：False<br>取值范围：True表示不支持；False表示支持。|
|lock|multiprocessing.synchronize.Lock, _thread.LockType|可选|CacheConfig不支持多线程或者多进程进行处理，如果用户需要多进程或者多线程调用此接口需要申请锁。默认值为None。<br>可选值：<br>None：表示不使用锁，此时该接口不支持并发。<br>multiprocessing.Lock()：表示进程锁，此时该接口支持多进程调用。<br>threading.Lock()：表示线程锁。此时该接口支持多线程调用。|


> [!NOTE] 说明 
>-   本接口内部使用了pickle模块，有被恶意构造的数据在unpickle期间攻击的风险。需要保证在被加载的落盘数据data\_save\_folder是安全存储，仅可加载可信的落盘数据。
>-   对于memory cache来说，它的落盘文件不能超过100MB大小。

**调用示例<a name="section1211769151311"></a>**

```
from paddle.base import libpaddle
from mx_rag.cache import CacheConfig
from mx_rag.cache import EvictPolicy
from mx_rag.cache import MxRAGCache
cache_config = CacheConfig(
    cache_size=100,
    eviction_policy=EvictPolicy.LRU,
    data_save_folder="path_to_cache_save_folder"
)
mxrag_l1_cache = MxRAGCache("memory_cache", cache_config)
```



#### SimilarityCacheConfig<a name="ZH-CN_TOPIC_0000002018595333"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000001983545962"></a>

**功能描述<a name="section14130756172810"></a>**

用于配置similarity cache的配置数据结构。

**函数原型<a name="section339074703020"></a>**

```
from mx_rag.cache import SimilarityCacheConfig
SimilarityCacheConfig(vector_config, cache_config, emb_config, similarity_config, retrieval_top_k, clean_size, **kwargs)
```

**输入参数说明<a name="section198665819815"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|vector_config|Dict[str, Any]|必选|配置矢量数据库。具体配置参见[表1](#vector_config)，注意这里如果是faiss数据库，MindFAISS的load_local_index参数会被data_save_folder落盘路径覆盖，auto_save参数置为False，配置中的字符串长度不能超过1024，字典中包含的可迭代序列长度不能超过1024，字典长度不能超过1024，字典嵌套深度不能超过2层。|
|cache_config|str|必选|配置标量数据库。当前只支持配置为“sqlite”。|
|emb_config|Dict[str, Any]|必选|配置embedding模型，请参见[表2](#emb_config)。字典长度不能超过1024，字典中字符串长度不能超过1024，字典嵌套深度不能超过1层。|
|similarity_config|Dict[str, Any]|必选|配置相似度计算模型，字典长度不能超过1024，字典中字符串长度不能超过1024，字典嵌套深度不能超过1层。请参考[表3](#similarity_config)。|
|retrieval_top_k|int|可选|相似检索时的topk值。默认值为1。取值范围：(0, 1000]。|
|clean_size|int|可选|每次缓存数据添加超过cache_size时，老化的个数，默认值为1。取值范围：(0, cache_size]|
|**kwargs|Any|必选|参数介绍可参见[CacheConfig](#cacheconfig)。|


> [!NOTE] 说明 
>-   本接口内部使用了pickle模块，有被恶意构造的数据在unpickle期间攻击的风险。需要保证在被加载的落盘数据data\_save\_folder是安全存储，仅可加载可信的落盘数据。
>-   vector\_config和cache\_config必须同时为None或同时不为None。如果vector\_config和cache\_config同时为None，则等同于memory cache。
>-   对于sqlite数据库来说，它的落盘文件不能超过30GB，对于矢量数据库来说，它的落盘文件不能超过20GB。

**表 1**  vector\_config<a id="vector_config"></a>

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|**kwargs|Dict[str, Any]|必选|具体介绍可参见[create_storage](./databases.md#create_storage)。|
|top_k|int|可选|相似度检索时的topk个数。默认值：5。|
|vector_save_file|str|必选|落盘路径。vector_type为"npu_faiss_db"时，该参数会覆盖MindFAISS中的load_local_index参数作为落盘路径；对于milvus_db，该参数不生效。|


**表 2**  emb\_config参数说明<a id="emb_config"></a>

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|x_dim|int|可选|embedding模型的维度，默认值为0。|
|skip_emb|bool|可选|是否跳过embedding，默认值为False。|
|**kwargs|Dict[str, Any]|必选|具体介绍可参见create_embedding。|


**表 3**  similarity\_config参数说明<a id="similarity_config"></a>

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|score_min|float|可选|相似度计算可能存在范围的最小值，默认值为0.0。<br>取值范围：[0.0, 100.0]|
|score_max|float|可选|相似度计算可能存在范围的最大值，默认值为1。<br>取值范围：[1.0, 100.0]，"score_max"必须大于等于"score_min"。|
|reverse|bool|可选|相似度分数和相似度的关系，默认值为"False"。<li>False：相似度分数越高，相似度越高。<li>True：相似度分数越高，相似度越低。|
|**kwargs|Dict[str, Any]|必选|具体介绍可参见[create_reranker](./reranker.md#create_reranker)。|


**调用示例<a name="section209387181454"></a>**

示例1：faiss\_npu + local\_embedding + local\_reranker

```
from mx_rag.cache import SimilarityCacheConfig
from mx_rag.cache import MxRAGCache
dim = 1024
dev = 1

similarity_config = SimilarityCacheConfig(
        vector_config={
            "vector_type": "npu_faiss_db",
            "x_dim": dim,
            "devs": [dev],
           
        },
        cache_config="sqlite",
        emb_config={
            "embedding_type": "local_text_embedding",
            "x_dim": dim,
            "model_path": "path_to_embedding_model", # emb 模型路径
            "dev_id": dev
        },
        similarity_config={
            "similarity_type": "local_reranker",
            "model_path": "path_to_reranker_model",  # reranker 模型路径
            "dev_id": dev
        },
        retrieval_top_k=1,
        cache_size=1000,
        clean_size=20,
        similarity_threshold=0.86,
        data_save_folder="path_to_cache_save_folder", # 落盘路径
        disable_report=True
    )
similarity_cache = MxRAGCache("similarity_cache", similarity_config)
```

示例2：milvus\_db + tei\_embedding + tei\_reranker

```
import getpass
from paddle.base import libpaddle
from mx_rag.cache import SimilarityCacheConfig
from mx_rag.cache import EvictPolicy
from mx_rag.cache import MxRAGCache
from mx_rag.utils import ClientParam
from pymilvus import MilvusClient
dim = 1024

client = MilvusClient("https://x.x.x.x:port", user="xxx", password=getpass.getpass(), secure=True,   client_pem_path="path_to/client.pem",   client_key_path="path_to/client.key",   ca_pem_path="path_to/ca.pem",   server_name="localhost")
similarity_config = SimilarityCacheConfig(
    vector_config={
        "client": client,
        "vector_type": "milvus_db",
        "x_dim": dim,
        "collection_name": "mxrag_cache_123",  # milvus db的标签
        "param": None
    },
    cache_config="sqlite",
    emb_config={
        "embedding_type": "tei_embedding",
        "url": "https://<ip>:<port>/embed",  # tei_embedding 服务的IP地址和侦听端口
        "client_param": ClientParam(ca_file="/path/to/ca.crt")
    },
    similarity_config={
        "similarity_type": "tei_reranker",
        "url": "https://<ip>:<port>/rerank",  # tei_reranker 服务的IP地址和侦听端口
        "client_param": ClientParam(ca_file="/path/to/ca.crt")
    },
    retrieval_top_k=1,
    cache_size=100,
    auto_flush=100,
    similarity_threshold=0.70,
    data_save_folder="path_to_cache_save_folder",
    disable_report=True,
    eviction_policy=EvictPolicy.FIFO
)
similarity_cache = MxRAGCache("similarity_cache", similarity_config)
```



#### EvictPolicy<a name="ZH-CN_TOPIC_0000002018715013"></a>

##### 类功能<a id="ZH-CN_TOPIC_0000002020225309"></a>

**功能描述<a name="section14130756172810"></a>**

用于配置缓存的替换策略。

**函数原型<a name="section339074703020"></a>**

```
from mx_rag.cache import EvictPolicy
class EvictPolicy(Enum)
```

**输入参数说明<a name="section18394927124010"></a>**

|属性名|数据类型|说明|
|--|--|--|
|LRU|str|替换最久没有访问的缓存。|
|LFU|str|替换使用频率最低的缓存。|
|FIFO|str|按照先进先出的规则进行替换。|
|RR|str|随机替换缓存。|





### 缓存数据<a name="ZH-CN_TOPIC_0000001981995616"></a>

#### 章节介绍<a name="ZH-CN_TOPIC_0000002021666745"></a>

本章节主要用于阐述给用户提供缓存数据更新，缓存数据查找，缓存数据落盘功能的MxRAGCache。


#### MxRAGCache<a name="ZH-CN_TOPIC_0000002018714849"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000001983528348"></a>

**功能描述<a name="section531511142318"></a>**

主要提供用户问题和答案的缓存存储、缓存更新、缓存落盘功能。

**函数原型<a name="section1885124712320"></a>**

```
from mx_rag.cache import MxRAGCache
MxRAGCache(cache_name, config)
```

**输入参数说明<a name="section94511016123411"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|cache_name|str|必选|缓存名字，该命名会体现在落盘文件名中，字符串长度范围：(0, 64)。取值范围：[0-9a-zA-Z_]<br>只能是字母、数字或下划线。|
|config|CacheConfig/SimilarityCacheConfig|必选|缓存配置，参考[缓存配置](#缓存配置)。|


**调用示例<a name="section1963211562113"></a>**

```
import json
import getpass
from paddle.base import libpaddle
from pymilvus import MilvusClient
from mx_rag.cache import CacheConfig, SimilarityCacheConfig
from mx_rag.cache import EvictPolicy
from mx_rag.cache import MxRAGCache
from mx_rag.utils import ClientParam

dim = 1024

cache_config = CacheConfig(
    cache_size=100,
    eviction_policy=EvictPolicy.LRU,
    data_save_folder="path_to_cache_save_folder"
)
cache = MxRAGCache("memory_cache", cache_config)
# 检查cache实例是否初始化成功
cache_obj = cache.get_obj()
if cache_obj is None:
    print(f"cache init failed")
similarity_config = SimilarityCacheConfig(
    vector_config={
        "vector_type": "milvus_db",
        "x_dim": dim,
        "client": MilvusClient("https://x.x.x.x:port", user="xxx", password=getpass.getpass(), secure=True,   client_pem_path="path_to/client.pem", client_key_path="path_to/client.key", ca_pem_path="path_to/ca.pem", server_name="localhost")
        "collection_name": "mxrag_cache_123",  # milvus db的标签
        "use_http": False,
        "param": None
    },
    cache_config="sqlite",
    emb_config={
        "embedding_type": "tei_embedding",
        "url": "https://<ip>:<port>/embed",  # tei_embedding 服务的IP地址和侦听端口
        "client_param": ClientParam(ca_file="/path/to/ca.crt")
    },
    similarity_config={
        "similarity_type": "tei_reranker",
        "url": "https://<ip>:<port>/rerank",  # tei_reranker 服务的IP地址和侦听端口
        "client_param": ClientParam(ca_file="/path/to/ca.crt")
    },
    retrieval_top_k=1,
    cache_size=100,
    auto_flush=100,
    similarity_threshold=0.70,
    data_save_folder="path_to_cache_save_folder",
    disable_report=True,
    eviction_policy=EvictPolicy.FIFO
)
similarity_cache = MxRAGCache("similarity_cache", similarity_config)
# 设置缓存级联
cache.join(similarity_cache)
# 设置缓存每条的字符限制为4000个字符
cache.set_cache_limit(4000)
# 设置是否详细显示缓存过程
cache.set_verbose(False)
# 手动更新缓存
cache.update("小明的爸爸是谁?", json.dumps({"小明的爸爸是谁?": "小明的爸爸名字是大明"}))
# 精确匹配结果
res = cache.search("小明的爸爸是谁?")
print(f"memory match res: {res}")
# 语义近似匹配结果
res = cache.search("小明的爸爸叫什么名字")
print(f"similarity match res: {res}")
# 手动调用flush 将缓存落盘，也会按照auto_flush配置进行自动落盘
cache.flush()
# 删除已落盘的文件和数据
cache.clear()
```


##### clear<a name="ZH-CN_TOPIC_0000002109544962"></a>

**功能描述<a name="section292317486370"></a>**

删除已落盘的数据和文件。

**函数原型<a name="section725633832220"></a>**

```
def clear()
```


##### flush<a name="ZH-CN_TOPIC_0000002020088177"></a>

**功能描述<a name="section292317486370"></a>**

该函数主要是将用户的缓存数据强制从内存中刷新至磁盘空间，刷新地址为[类功能](#ZH-CN_TOPIC_0000002020105781)中的配置参数“data\_save\_folder”。

必须在初始化之后进行调用。

**函数原型<a name="section725633832220"></a>**

```
def flush()
```


##### get\_obj<a name="ZH-CN_TOPIC_0000001983528352"></a>

**功能描述<a name="section9311431449"></a>**

获得gptcache对象，用于适配LangChain等开源RAG框架，必须在初始化之后进行调用。

**函数原型<a name="section9196890185"></a>**

```
def get_obj()
```

**返回值说明<a name="section1756123482917"></a>**

|数据类型|说明|
|--|--|
|gptcache对象|gptcache对象|



##### join<a name="ZH-CN_TOPIC_0000001983688032"></a>

**功能描述<a name="section778813916511"></a>**

将两个cache进行串联，达到多级缓存目的。

**函数原型<a name="section195173421755"></a>**

```
def join(next_cache)
```

**输入参数说明<a name="section663711453511"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|next_cache|MxRAGCache|必选|下一级缓存，缓存必须先初始化，不能串联自己。下级缓存必须是MxRAGCache类型。缓存串联时不能成环，并且最大串联深度为6|



##### search<a name="ZH-CN_TOPIC_0000002020207693"></a>

**功能描述<a name="section292317486370"></a>**

该函数主要负责根据用户的问题找到对应的答案，必须在初始化之后进行调用。

**函数原型<a name="section8743103910378"></a>**

```
def search(query)
```

**输入参数说明<a name="section685215278388"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|query|str|必选|用户的问题，字符串长度范围(0, 128\*1024\*1024]。|


**返回值说明<a name="section1756123482917"></a>**

|数据类型|说明|
|--|--|
|Dict|返回QA对的 Unicode 编码字符串，如：{"\u66aa....?": "\u6ae6..."，...}|



##### set\_cache\_limit<a name="ZH-CN_TOPIC_0000002020207697"></a>

**功能描述<a name="section1252711771617"></a>**

设置cache缓存时LLM返回答案的字符数限制。如果LLM返回的字符串超过这个限制，则不会被缓存。

**函数原型<a name="section41822194419"></a>**

```
@classmethod
def set_cache_limit(cache_limit: int)
```

**输入参数说明<a name="section18524162011160"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|cache_limit|int|必选|每一条被缓存答案的字符数限制，默认值是一百万个字符，取值范围(0, 1000000]|



##### set\_verbose<a name="ZH-CN_TOPIC_0000002020088181"></a>

**功能描述<a name="section890719421814"></a>**

设置是否开启日志记录。

**函数原型<a name="section18524162011160"></a>**

```
@classmethod
def set_verbose(verbose: bool)
```

**输入参数说明<a name="section3116128191811"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|verbose|bool|必选|是否开启详细日志记录，默认值为False。<li>True：记录cache过程的命中或者未命中。<li>False：不会记录cache过程的命中或者未命中。|



##### update<a name="ZH-CN_TOPIC_0000001983688028"></a>

**功能描述<a name="section292317486370"></a>**

该函数主要负责将用户问题和答案进行存储，必须在初始化之后调用，否则会抛出异常。

**函数原型<a name="section8743103910378"></a>**

```
def update(query, answer)
```

**输入参数说明<a name="section685215278388"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|query|str|必选|用户的问题，字符串长度(0, 128\*1024\*1024]范围|
|answer|str|必选|用户的问题所对应的答案，其字符数范围(0, min(1000000 , cache_limit)]，否则不会被缓存，具体可参考[set_cache_limit](#set_cache_limit)。|





### 缓存适配<a name="ZH-CN_TOPIC_0000002018714697"></a>

#### 章节介绍<a name="ZH-CN_TOPIC_0000002021547249"></a>

本章节主要介绍MxRAGCache的chain适配。


#### CacheChainChat<a name="ZH-CN_TOPIC_0000002018715021"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000002020211437"></a>

**功能描述<a name="section3010217492"></a>**

用于适配RAG SDK的文生文、图生图、文生图的各种chain，同时也提供访问MxRAGCache的能力，当缓存未命中时，将进行大模型推理，然后将结果再刷新至缓存。

**函数原型<a name="section072452410494"></a>**

```
from mx_rag.cache import CacheChainChat
CacheChainChat(cache,chain,convert_data_to_cache,convert_data_to_user)
```

**输入参数说明<a name="section11487481490"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|cache|MxRAGCache|必选|RAG SDK缓存。|
|chain|Chain|必选|RAG SDK chain，用于访问大模型。|
|convert_data_to_cache|Callable[[Any], Dict]|可选|该回调函数主要用于当用户数据无法转换为字符串格式时，由用户提供转换函数。<br>默认为不做转换。|
|convert_data_to_user|Callable[[Dict], Any]|可选|该回调函数主要是配合convert_data_to_cache使用，当用户问题命中时，将cache存储的格式转换为用户格式。默认为不做转换。|


**调用示例<a name="section159212456107"></a>**

```
import time
from paddle.base import libpaddle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.chain import SingleText2TextChain
from mx_rag.document.loader import DocxLoader
from mx_rag.embedding.local import TextEmbedding
from mx_rag.knowledge import KnowledgeDB
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.llm import Text2TextLLM
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.knowledge.handler import upload_files
from mx_rag.document import LoaderMng
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.utils import ClientParam
from mx_rag.cache import CacheChainChat, MxRAGCache, SimilarityCacheConfig

#向量维度
dim = 1024
# NPU卡id
dev = 0

similarity_config = SimilarityCacheConfig(
    vector_config={
        "vector_type": "npu_faiss_db",
        "x_dim": dim,
        "devs": [dev],

    },
    cache_config="sqlite",
    emb_config={
        "embedding_type": "local_text_embedding",
        "x_dim": dim,
        "model_path": "/path to emb",  # emb 模型路径
        "dev_id": dev
    },
    similarity_config={
        "similarity_type": "local_reranker",
        "model_path": "/path to reranker",  # reranker 模型路径
        "dev_id": dev
    },

    retrieval_top_k=1,
    cache_size=1000,
    clean_size=20,
    similarity_threshold=0.86,
    data_save_folder="/save path",  # 落盘路径
    disable_report=True
)
similarity_cache = MxRAGCache("similarity_cache", similarity_config)

# cache 初始化
cache = MxRAGCache("similarity_cache", similarity_config)
# Step1离线构建知识库,首先注册文档处理器
loader_mng = LoaderMng()
# 加载文档加载器，可以使用RAG SDK自有的，也可以使用langchain的
loader_mng.register_loader(DocxLoader, [".docx"])
# 加载文档切分器，使用langchain的
loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".pdf"],
                             {"chunk_size": 200, "chunk_overlap": 50, "keep_separator": False})

emb = TextEmbedding(model_path="/path to emb", dev_id=dev)

# 初始化文档chunk关系数据库
chunk_store = SQLiteDocstore(db_path="./sql.db")
# 初始化知识管理关系数据库
knowledge_store = KnowledgeStore(db_path="./sql.db")
# 初始化矢量检索

vector_store = MindFAISS(x_dim=dim,
                         devs=[dev],
                         load_local_index="./faiss.index"
                         )

#添加知识库及管理员
knowledge_store.add_knowledge(knowledge_name="test", user_id='Default', role='admin')
# 初始化知识库管理
knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                           chunk_store=chunk_store,
                           vector_store=vector_store,
                           knowledge_name="test",
                           user_id='Default',
                           white_paths=["/home"])
# 完成离线知识库构建,上传领域知识test.docx文档。
upload_files(knowledge_db, ["/path to files"],
             loader_mng=loader_mng,
             embed_func=emb.embed_documents,
             force=True)
# Step2在线问题答复,初始化检索器
retriever = vector_store.as_retriever(document_store=chunk_store,
                                      embed_func=emb.embed_documents, k=3, score_threshold=0.3)
# 配置reranker

# 配置text生成text大模型chain，具体ip端口请根据实际情况适配修改
llm = Text2TextLLM(base_url="https://<ip>:<port>",
                   model_name="Llama3-8B-Chinese-Chat",
                   client_param=ClientParam(ca_file="/path/to/ca.crt"))
text2text_chain = SingleText2TextChain(llm=llm, retriever=retriever)
cache_chain = CacheChainChat(chain=text2text_chain, cache=cache)
start_time = time.time()
res = cache_chain.query("请描述2024年高考作文题目")
end_time = time.time()
print(f"no cache query time cost:{(end_time - start_time) * 1000}ms")
print(f"no cache answer {res}")
start_time = time.time()
res = cache_chain.query("2024年的高考题目是什么", )
end_time = time.time()
print(f"cache query time cost:{(end_time - start_time) * 1000}ms")
print(f"cache answer {res}")

```


##### query<a name="ZH-CN_TOPIC_0000001983691808"></a>

**功能描述<a name="section3010217492"></a>**

提供给用户查询缓存的接口，当缓存无法查询时，会访问大模型。

**函数原型<a name="section072452410494"></a>**

```
def query(text, *args, **kwargs) -> Union[Dict, Iterator[Dict]]
```

**输入参数说明<a name="section11487481490"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|text|str|必选|用户的原始问题，字符数范围(0, 128M]|
|llm_config|LLMParameterConfig|可选|大模型参数，具体介绍可参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|
|*args/**kwargs|Any|可选|继承父类方法签名，RAG SDK不涉及使用。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|Union[Dict, Iterator[Dict]]|返回问答结果，其中Dict内容为：<li>带知识来源：{"query": query, "result": data, "source_documents": [{'metadata': xxx, 'page_content': xxx}]}<li>不带知识来源：{"query": query, "result": data}|





### 自动生成QA作为缓存<a name="ZH-CN_TOPIC_0000001981995540"></a>

#### 章节介绍<a name="ZH-CN_TOPIC_0000001985027148"></a>

本章节主要介绍生成QA，并更新到MxRAGCache缓存中的接口。


#### QAGenerate<a name="ZH-CN_TOPIC_0000001981995640"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000001982155268"></a>

**功能描述<a name="section957011509130"></a>**

问答生成类，输入标题和正文，调用大模型根据该正文生成问答对。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.cache import QAGenerate
QAGenerate(config: QAGenerationConfig)
```

**参数说明<a name="section73451452114318"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|config|QAGenerationConfig|必选|QAGenerationConfig对象，生成QA的相关参数。<br>关于QAGenerationConfig原型说明请参见[QAGenerationConfig](#qagenerationconfig)。|


**调用示例<a name="section143597458412"></a>**

```
from paddle.base import libpaddle
from transformers import AutoTokenizer
from mx_rag.cache import QAGenerationConfig, QAGenerate
from mx_rag.llm import Text2TextLLM
from mx_rag.utils import ClientParam
llm = Text2TextLLM(base_url="https://ip:port/v1/chat/completions", model_name="llama3-chinese-8b-chat",
                   client_param=ClientParam(ca_file="/path/to/ca.crt"))
# 使用模型的tokenizer, 传入模型存放路径
tokenizer = AutoTokenizer.from_pretrained("/home/model/Llama3-8B-Chinese-Chat/", local_files_only=True)
# 可以调用MarkDownParser生成titles和contents
titles = ["2024年高考语文作文题目"]
contents = ['2024年高考语文作文试题\n新课标I卷\n阅读下面的材料，根据要求写作。（60分）\n'
            '随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？\n'
            '以上材料引发了你怎样的联想和思考？请写一篇文章。'
            '要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。']
config = QAGenerationConfig(titles, contents, tokenizer, llm, qas_num=1)
qa_generate = QAGenerate(config)
qas = qa_generate.generate_qa()
print(qas)
```


##### generate\_qa<a name="ZH-CN_TOPIC_0000002018714949"></a>

**功能描述<a name="section192361437127"></a>**

通过QAGenerationConfig传入标题和正文，会按照QAGenerationConfig中的max\_tokens值截断正文。

大模型返回的QA如果不符合格式和个数要求，则会跳过。比如生成三个符合要求的QA：

```
Q1：如何查询成都火车站的停运列车？
参考段落：'查询方式：铁路12306网页首页。查询流程：第一步：进入铁路12306app首页，点击【车站大屏】；第二步：左上角车站名下拉选择成都东站；第三步：搜索框输入车次即可查询车次情况。'
Q2：四川省将洪水灾害防御响应提升至哪个级别？
参考段落：四川将洪水灾害防御四级响应提升至三级。
Q3：在7月14日，四川省气象台发布了哪种天气预警？
参考段落：7月14日15时30分，四川省气象台继续发布暴雨蓝色预警。
```

**函数原型<a name="section168811421126"></a>**

```
def generate_qa(llm_config)
```

**参数说明<a name="section1927740111518"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|llm_config|LLMParameterConfig|可选|调用大模型的参数，此处修改默认值temperature为0.5，top_p为0.95，其余参数说明请参见[LLMParameterConfig](./llm_client.md#llmparameterconfig)。|


**返回值说明<a name="section1756123482917"></a>**

|数据类型|说明|
|--|--|
|Dict|返回生成的QA对列表，格式为<br>{"从成都到重庆要多久？ : 乘坐高铁1个小时"，...}|




#### QAGenerationConfig<a name="ZH-CN_TOPIC_0000001981995556"></a>

**功能描述<a name="section169811132143713"></a>**

生成QA的参数。

**函数原型<a name="section1486053512535"></a>**

```
from mx_rag.cache import QAGenerationConfig
QAGenerationConfig(titles, contents, tokenizer, llm, max_tokens, qas_num)
```

**参数说明<a name="section42019717567"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|titles|List[str]|必选|标题列表，标题和正文列表一一对应。列表长度范围[1, 10000]，字符串长度范围[1, 100]。|
|contents|List[str]|必选|正文列表，标题和正文列表一一对应。列表长度范围[1, 10000]，字符串长度范围[1, 1048576]。|
|tokenizer|transformers.PreTrainedTokenizerBase|必选|tokenizer实例，通过AutoTokenizer.from_pretrained加载。加载外部模型有安全风险，local_files_only设置为True。|
|llm|Text2TextLLM|必选|大模型对象实例，具体类型请参见[Text2TextLLM](./llm_client.md#text2textllm)。|
|max_tokens|int|可选|用于截断正文的最大token大小，超出部分丢弃，取值范围[500, 100000]，默认值为1000。该参数实际有效取值依赖MindIE的配置，请参考《MindIE LLM开发指南》中的“核心概念与配置 > 配置参数说明（服务化）”章节中关于maxSeqLen的说明。|
|qas_num|int|可选|生成QA对数量，取值范围[1, 10]，默认值为5。|


**调用示例<a name="section7976512121919"></a>**

```
from paddle.base import libpaddle
from transformers import AutoTokenizer
from mx_rag.cache import QAGenerationConfig, QAGenerate
from mx_rag.llm import Text2TextLLM
from mx_rag.utils import ClientParam
llm = Text2TextLLM(base_url="https://ip:port/v1/chat/completions", model_name="llama3-chinese-8b-chat",
                   client_param=ClientParam(ca_file="/path/to/ca.crt"))
# 使用模型的tokenizer, 传入模型存放路径
tokenizer = AutoTokenizer.from_pretrained("/home/model/Llama3-8B-Chinese-Chat/", local_files_only=True)
# 可以调用MarkDownParser生成titles和contents
titles = ["2024年高考语文作文题目"]
contents = ['2024年高考语文作文试题\n新课标I卷\n阅读下面的材料，根据要求写作。（60分）\n'
            '随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？\n'
            '以上材料引发了你怎样的联想和思考？请写一篇文章。'
            '要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。']
config = QAGenerationConfig(titles, contents, tokenizer, llm, qas_num=1)
qa_generate = QAGenerate(config)
qas = qa_generate.generate_qa()
print(qas)
```


#### MarkDownParser<a name="ZH-CN_TOPIC_0000001982155252"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000002018714941"></a>

**功能描述<a name="section1169530131517"></a>**

解析Markdown，返回的标题和正文的类。

**函数原型<a name="section4202122620152"></a>**

```
from mx_rag.cache import MarkDownParser
MarkDownParser(file_path, max_file_num)
```

**参数说明<a name="section9272549111610"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|file_path|str|必选|markdown文件所在的文件夹路径，路径长度不能超过1024。调用parse时会校验：不能为软链接和相对路径，文件夹路径下的.md文件大小不能超过10MB，.md文件数量不能超过max_file_num。路径不能在路径列表中：["/etc", "/usr/bin", "/usr/lib", "/usr/lib64", "/sys/", "/dev/", "/sbin", "/tmp"]。|
|max_file_num|int|可选|最大解析的markdown文件个数，默认值为1000，取值范围[1, 10000]。|


**返回值说明<a name="section104931419101818"></a>**

|数据类型|说明|
|--|--|
|Tuple[List[str], List[str]]|返回的是Markdown解析后的titles列表和contents列表。|


**调用示例<a name="section81390573417"></a>**

```
from paddle.base import libpaddle
from mx_rag.cache import MarkDownParser
dir_path = "path of .md document "
parser = MarkDownParser(dir_path)
titles, contents = parser.parse()
print(titles)
print(contents)
```


##### parse<a name="ZH-CN_TOPIC_0000002018714945"></a>

**功能描述<a name="section425174761911"></a>**

返回markdown的标题和正文，文件夹内markdown文件数量不能超过max\_file\_num。

**函数原型<a name="section5138652112017"></a>**

```
def parse()
```

**返回值说明<a name="section1236418436222"></a>**

|数据类型|说明|
|--|--|
|Tuple[List[str], List[str]]|返回的是markdown对应的titles列表和contents列表。|





