## 向量化<a name="ZH-CN_TOPIC_0000002419262684"></a>

### TextEmbedding<a name="ZH-CN_TOPIC_0000002419262688"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002452701717"></a>

**功能描述<a name="section957011509130"></a>**

本地使用transformers启动模型，提供文本至向量的embedding功能。需要使用transformers支持的BertModel类模型权重。类继承实现了langchain\_core.embeddings.Embeddings接口。当前支持的模型：[BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)，[aspire/acge\_text\_embedding](https://huggingface.co/aspire/acge_text_embedding)。

> [!NOTE] 说明 
>配置的模型如果不是safetensors权重格式，请先将模型权重转换为safetensors格式后再使用，防止使用ckpt、bin等不安全的模型权重格式引入安全问题。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.embedding.local import TextEmbedding
TextEmbedding(model_path, dev_id, use_fp16, pooling_method, lock)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|model_path|str|必选|模型权重文件目录，路径长度不能超过1024，不能为软链接和相对路径。目录下的各文件大小不能超过10GB、深度不超过64，且文件总个数不超过512。运行用户的属组，以及非运行用户不能有该目录下文件的写权限。目录下的文件以及文件的上一级目录的属组必须是运行用户。存放路径不能在路径列表中：["/etc", "/usr/bin", "/usr/lib", "/usr/lib64", "/sys/", "/dev/", "/sbin", "/tmp"]。|
|dev_id|int|可选|模型运行NPU ID，通过**npu-smi info**查询可用ID，取值范围[0, 63]，默认为卡0。|
|use_fp16|bool|可选|是否将模型转换为半精度，默认为True。|
|pooling_method|str|可选|选择处理last_hidden_state的方式，取值范围['cls', 'mean', 'max', 'lasttoken']，默认'cls'。|
|lock|multiprocessing.synchronize.Lock或_thread.LockType|可选|local model不支持多线程或者多进程进行处理，如果用户需要多进程或者多线程调用此接口需要申请锁。默认值为None。可选值：<li>None：表示不使用锁，此时该接口不支持并发。<li>multiprocessing.Lock()：表示进程锁，此时该接口支持多进程调用。<li>threading.Lock()：表示线程锁。此时该接口支持多线程调用。|


**不启用推理加速调用示例<a name="section551785315254"></a>**

```
from paddle.base import libpaddle
from mx_rag.embedding.local import TextEmbedding
# 同embed = TextEmbedding("/path/to/model", 1)
embed = TextEmbedding.create(model_path="/path/to/model", dev_id=1)
print(embed.embed_documents(['abc', 'bcd']))
print(embed.embed_query('abc'))
```

**开启推理加速调用示例<a name="section18289184984712"></a>**

```
import os
from paddle.base import libpaddle
import torch_npu
# 适配向量化推理加速
from modeling_bert_adapter import enable_bert_speed
# 使能向量化推理加速（设置为"True"时表示使能,"False"表示不使能）
os.environ["ENABLE_BOOST"] = "True"
from mx_rag.embedding.local import TextEmbedding
device_id = 1
torch_npu.npu.set_device(f"npu:{device_id}")
# 同embed = TextEmbedding("/path/to/model", 1)
embed = TextEmbedding.create(model_path="/path/to/model", dev_id=device_id )
print(embed.embed_documents(['abc', 'bcd']))
print(embed.embed_query('abc'))
```

**多线程调用示例（其余嵌入模型均可参考该示例）<a name="section1434214302272"></a>**

```
from paddle.base import libpaddle
import threading
from mx_rag.embedding.local import TextEmbedding
def infer(k, embed):
    print(f"thread_{k}")
    print(embed.embed_query('abc'))
    print(embed.embed_documents(['abc', 'bcd']))

if __name__ == '__main__':
    worker_nums=2
    threads = []
    embed = TextEmbedding.create(model_path='/path/to/model', dev_id=1, pooling_method='cls', lock=threading.Lock())
    for i in range(worker_nums):
        thread = threading.Thread(target=infer, args=(i,embed,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
```


#### create<a name="ZH-CN_TOPIC_0000002419102840"></a>

**功能描述<a name="section118111227123016"></a>**

创建并返回一个TextEmbedding对象。

**函数原型<a name="section544124513018"></a>**

```
@staticmethod
def create(**kwargs)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|kwargs|dict|必选|关键字参数，参考[类功能](#类功能)的入参，必选参数必须传入，否则将抛出KeyError。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|TextEmbedding|TextEmbedding对象。|



#### embed\_documents<a name="ZH-CN_TOPIC_0000002452821605"></a>

**功能描述<a name="section53998444524"></a>**

使用模型将用户提供的文本转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_documents(texts, batch_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|texts|List[str]|必选|文本列表，列表长度取值[1，1000*1000]，字符串长度范围[1, 128 * 1024 * 1024]。|
|batch_size|int|可选|组batch的大小，每次会组合batch_size的texts进行embed操作，取值范围：[1, 1024]，默认值为32。可配置的值由设备显存决定。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[List[float]]|texts转换后的向量数组。如果texts为长度是4的数组，embedding模型的输出是1024维向量，最终的输出结果为（4，1024）大小的数组。|



#### embed\_query<a name="ZH-CN_TOPIC_0000002419262692"></a>

**功能描述<a name="section53998444524"></a>**

使用模型将用户提供的文本转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_query(text)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|text|str|必选|待向量化的文本，文本长度范围：[1, 128 * 1024 * 1024]。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[float]|text转换后的向量。如果embedding模型的输出是1024维向量，最终的输出结果为（1，1024）大小的数组。|




### SparseEmbedding<a name="ZH-CN_TOPIC_0000002452701721"></a>

#### 类功能<a id="ZH-CN_TOPIC_0000002419102844"></a>

**功能描述<a name="section957011509130"></a>**

本地使用transformers启动模型，提供文本至向量的sparse embedding功能。需要使用transformers支持的BertModel类模型权重。类继承实现了langchain\_core.embeddings.Embeddings接口。当前支持的模型：BAAI/bge-m3。

> [!NOTE] 说明 
>配置的模型如果不是safetensors权重格式，请先将模型权重转换为safetensors格式后再使用，防止使用ckpt、bin等不安全的模型权重格式引入安全问题。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.embedding.local import SparseEmbedding
SparseEmbedding(model_path, dev_id, use_fp16)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|model_path|str|必选|模型权重文件目录，路径长度不能超过1024，不能为软链接和相对路径。<li>目录下的各文件大小不能超过10GB、深度不超过64，且文件总个数不超过512。<li>运行用户的属组，以及非运行用户不能有该目录下文件的写权限。<li>目录下的文件以及文件的上一级目录的属组必须是运行用户。存放路径不能在路径列表中：["/etc", "/usr/bin", "/usr/lib", "/usr/lib64", "/sys/", "/dev/", "/sbin", "/tmp"]。|
|dev_id|int|可选|模型运行NPU ID，通过**npu-smi info**查询可用ID，取值范围[0, 63]，默认为卡0。|
|use_fp16|bool|可选|是否将模型转换为半精度，默认为True。|


**调用示例<a name="section74949341082"></a>**

```
from paddle.base import libpaddle
from mx_rag.embedding.local import SparseEmbedding
# 同embed = SparseEmbedding("/path/to/model", 1)
embed = SparseEmbedding.create(model_path="/path/to/model", dev_id=1)
print(embed.embed_documents(['abc', 'bcd']))
print(embed.embed_query('abc'))
```


#### create<a name="ZH-CN_TOPIC_0000002452821609"></a>

**功能描述<a name="section118111227123016"></a>**

创建并返回一个TextEmbedding对象。

**函数原型<a name="section544124513018"></a>**

```
@staticmethod
def create(**kwargs)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|kwargs|dict|必选|关键字参数，参考[类功能](#ZH-CN_TOPIC_0000002419102844)的入参，必选参数必须传入，否则将抛出KeyError。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|SparseEmbedding|SparseEmbedding对象。|



#### embed\_documents<a name="ZH-CN_TOPIC_0000002419262696"></a>

**功能描述<a name="section53998444524"></a>**

使用模型将用户提供的文本转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_documents(texts, batch_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|texts|List[str]|必选|文本列表，列表长度取值[1，1000*1000]，字符串长度范围[1, 128 * 1024 * 1024]。|
|batch_size|int|可选|组batch的大小，每次会组合batch_size的texts进行embed操作，取值范围：[1, 1024]，默认值为32。可配置的值由设备显存决定。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[Dict[int, float]]|texts转换后的向量数组。如果texts为长度是4的数组，embedding模型的输出是key为token_id，value为token_weights的字典，最终的输出结果为4维的数组，每个元素为一组字典。|



#### embed\_query<a name="ZH-CN_TOPIC_0000002452701725"></a>

**功能描述<a name="section53998444524"></a>**

使用模型将用户提供的文本转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_query(text)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|text|str|必选|待向量化的文本，文本长度范围：[1, 128 * 1024 * 1024]。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|Dict[int, float]|text转换后的稀疏向量。|




### TEIEmbedding<a name="ZH-CN_TOPIC_0000002419102848"></a>

#### 类功能<a id="ZH-CN_TOPIC_0000002452821613"></a>

**功能描述<a name="section957011509130"></a>**

连接TEI服务，提供文本至向量的embedding功能。类继承实现了langchain\_core.embeddings.Embeddings接口。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.embedding.service import TEIEmbedding
TEIEmbedding(url, client_param, embed_mode)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|url|str|必选|TEI embed服务地址，字符串长度范围[1, 128]。支持"/v1/embed"、"/v1/embeddings"、"/embed_sparse"接口。<br>> [!NOTE] 说明 当前基于TEI框架创建的embed服务不支持https协议，为安全起见可通过搭建一个nginx服务，让该服务与embed服务处于一个可信网络。使用时客户端以https方式访问nginx，nginx转发请求到embed服务。|
|client_param|ClientParam|可选|https客户端配置参数，默认值为ClientParam()，具体描述请参见[ClientParam](./univers_api.md#clientparam)。|
|embed_mode|str|可选|与TEI服务提供的向量化类型对应，默认为"dense"，值只能为"sparse"或"dense"，"sparse"表示稀疏向量化，"dense"表示稠密向量化，该参数当前已弃用。|


**返回值说明<a name="section53998444524"></a>**

TEIEmbedding对象。

**调用示例<a name="section7248521123815"></a>**

```
from paddle.base import libpaddle
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.utils import ClientParam
# 同tei_embed = TEIEmbedding("https://ip:port/embed", client_param=ClientParam(xxx))
tei_embed = TEIEmbedding.create(url="https://ip:port/embed",
                                client_param=ClientParam(ca_file="/path/to/ca.crt"))
print(tei_embed.embed_documents(['abc', 'bcd']))
print(tei_embed.embed_query('abc'))
```


#### create<a name="ZH-CN_TOPIC_0000002419262700"></a>

**功能描述<a name="section118111227123016"></a>**

创建并返回一个TEIEmbedding对象。

**函数原型<a name="section544124513018"></a>**

```
@staticmethod
def create(**kwargs)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|kwargs|dict|必选|关键字参数，参考[类功能](#ZH-CN_TOPIC_0000002452821613)的入参，必选参数必须传入，否则将抛出KeyError。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|TEIEmbedding|TEIEmbedding对象。|



#### embed\_documents<a name="ZH-CN_TOPIC_0000002452701729"></a>

**功能描述<a name="section53998444524"></a>**

调用TEI服务，将用户提供的文本列表转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_documents(texts, batch_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|texts|List[str]|必选|文本列表，列表长度取值(0，1000*1000]，字符串长度范围：[1, 128 * 1024 * 1024]。|
|batch_size|int|可选|组batch的大小，每次会组合batch_size的texts进行embed操作，取值范围：[1, 1024]，默认值为32。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[List[float]]|texts转换后的向量数组。如果texts为长度是4的数组，embedding模型的输出是1024维向量，最终的输出结果为（4，1024）大小的数组。|



#### embed\_query<a name="ZH-CN_TOPIC_0000002419102852"></a>

**功能描述<a name="section53998444524"></a>**

调用TEI服务，将用户提供的文本转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_query(text)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|text|str|必选|待向量化文本，长度范围：[1, 128 * 1024 * 1024]。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[float]|text转换后的向量数组。如果embedding模型的输出是1024维向量，最终的输出结果为（1，1024）大小的数组。|




### CLIPEmbedding<a name="ZH-CN_TOPIC_0000002452821617"></a>

#### 类功能<a id="ZH-CN_TOPIC_0000002419262704"></a>

**功能描述<a name="section957011509130"></a>**

连接CLIP服务，提供文本或图片至向量的embedding功能。类继承实现了langchain\_core.embeddings.Embeddings接口。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.embedding.service import CLIPEmbedding
CLIPEmbedding(url, client_param)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|url|str|必选|CLIP embedding服务地址，url字符串长度不能超过128。|
|client_param|ClientParam|可选|https客户端配置参数，默认值为ClientParam()，具体描述请参见[ClientParam](./univers_api.md#clientparam)。|


**返回值说明<a name="section53998444524"></a>**

CLIPEmbedding对象。

**调用示例<a name="section7248521123815"></a>**

```
from paddle.base import libpaddle
from mx_rag.embedding.service import CLIPEmbedding
from mx_rag.utils import ClientParam
clip_embed = CLIPEmbedding.create(url="https://ip:port/encode",
                                  client_param=ClientParam(ca_file="/path/to/ca.crt"))
print(clip_embed.embed_documents(['abc', 'bcd']))
print(clip_embed.embed_query('abc'))
```


#### create<a name="ZH-CN_TOPIC_0000002452701733"></a>

**功能描述<a name="section118111227123016"></a>**

创建并返回一个CLIPEmbedding对象。

**函数原型<a name="section544124513018"></a>**

```
@staticmethod
def create(**kwargs)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|kwargs|dict|必选|关键字参数，参考[类功能](#ZH-CN_TOPIC_0000002419262704)的入参，必选参数必须传入，否则将抛出KeyError。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|CLIPEmbedding|CLIPEmbedding对象。|



#### embed\_documents<a name="ZH-CN_TOPIC_0000002419102856"></a>

**功能描述<a name="section53998444524"></a>**

调用CLIP服务，将用户提供的文本列表转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_documents(texts, batch_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|texts|List[str]|必选|文本列表，列表长度取值(0，1000*1000]，字符串长度范围：[1, 128 * 1024 * 1024]。|
|batch_size|int|可选|组batch的大小，每次会组合batch_size的texts进行embed操作，取值范围：[1, 1024]，默认值为32。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[List[float]]|texts转换后的向量数组。如果texts为长度是4的数组，embedding模型的输出是512维向量，最终的输出结果为（4，512）大小的数组。|



#### embed\_images<a name="ZH-CN_TOPIC_0000002452821621"></a>

**功能描述<a name="section53998444524"></a>**

调用CLIP服务，将用户提供的图像列表转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_images(images, batch_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|images|List[str]|必选|图片列表，列表长度取值[1，1000]，字符串长度范围：[1, 10 * 1024 * 1024]。每个图片为一个使用base64编码后的字符串。|
|batch_size|int|可选|组batch的大小，每次会组合batch_size的texts进行embed操作，取值范围：[1, 1024]，默认值为32。如果batch_size太大，可能导致服务器返回500错误，此时需要调小batch_size。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[List[float]]|images转换后的向量数组。如果images为长度是4的数组，embedding模型的输出是512维向量，最终的输出结果为（4，512）大小的数组。|



#### embed\_query<a name="ZH-CN_TOPIC_0000002419262708"></a>

**功能描述<a name="section53998444524"></a>**

调用CLIP服务，将用户提供的文本转换至向量。

**函数原型<a name="section18789201331417"></a>**

```
def embed_query(text)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|text|str|必选|待向量化文本，长度范围：[1, 128 * 1024 * 1024]。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[float]|text转换后的向量数组。如果embedding模型的输出是512维向量，最终的输出结果为大小为512的浮点数列表。|




### ImageEmbedding<a name="ZH-CN_TOPIC_0000002452701737"></a>

#### 类功能<a id="ZH-CN_TOPIC_0000002419102860"></a>

**功能描述<a name="section957011509130"></a>**

本地使用cn\_clip启动模型，提供将图片和文本的embedding功能。ImageEmbedding类继承实现了langchain\_core.embeddings.Embeddings接口。

> [!NOTE] 说明 
>cn\_clip采用torch.load加载权重文件，请确保权重文件安全可信，防止加载权重时引入命令注入等安全问题。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.embedding.local import ImageEmbedding
ImageEmbedding(model_name, model_path, dev_id)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|model_name|str|必选|模型名称，必须为['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']其中之一。对应模型下载链接参见网页说明。|
|model_path|str|必选|模型权重文件目录，路径长度不能超过1024，不能为软链接和相对路径。<li>目录下的各文件大小不能超过10GB、深度不超过64，且文件总个数不超过512。<li>运行用户的属组，以及非运行用户不能有该目录下文件的写权限。<li>目录下的文件以及文件的上一级目录的属组必须是运行用户。<br>存放路径不能在路径列表中：["/etc", "/usr/bin", "/usr/lib", "/usr/lib64", "/sys/", "/dev/", "/sbin", "/tmp"]。|
|dev_id|int|可选|模型运行的NPU卡ID，取值范围[0, 63]，默认为0。|


**返回值说明<a name="section171041117571"></a>**

ImageEmbedding对象。

**调用示例<a name="section1930245418425"></a>**

```
import sys
from paddle.base import libpaddle
from mx_rag.document.loader import ImageLoader
from mx_rag.embedding.local import ImageEmbedding
embed = ImageEmbedding.create(model_name="ViT-B-16", model_path="/data/chinese-clip-vit-base-patch16")
print(embed.embed_documents(['abc', 'bcd']))
print(embed.embed_query('abc'))
                                
loader = ImageLoader("image path")
docs = loader.load()
if len(docs) < 1:
    print("load image failed")
    sys.exit(1)
    
print(embed.embed_images([docs[0].page_content]))
```


#### create<a name="ZH-CN_TOPIC_0000002452821625"></a>

**功能描述<a name="section118111227123016"></a>**

创建并返回一个ImageEmbedding对象。

**函数原型<a name="section544124513018"></a>**

```
@staticmethod
def create(**kwargs)
```

**输入参数说明<a name="section19434210583"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|kwargs|dict|必选|关键字参数，参考[类功能](#ZH-CN_TOPIC_0000002419102860)的入参，必选参数必须传入，否则将抛出KeyError。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|ImageEmbedding|ImageEmbedding对象。|



#### embed\_documents<a name="ZH-CN_TOPIC_0000002419262712"></a>

**功能描述<a name="section53998444524"></a>**

将文本列表进行向量化。

**函数原型<a name="section18789201331417"></a>**

```
def embed_documents(texts, batch_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|texts|List[str]|必选|文本列表，列表长度取值[1, 1000*1000]，列表中每个文本长度取值[1, 256]。|
|batch_size|int|可选|组batch大小，每次会组合batch_size的texts进行embed操作，取值范围：[1, 1024]，默认值为32。可配置的值由设备显存决定。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[List[float]]|texts转换后的向量数组。如果texts为长度是4的数组，embedding模型的输出是512维向量，最终的输出结果为（4，512）大小的数组|



#### embed\_query<a name="ZH-CN_TOPIC_0000002452701741"></a>

**功能描述<a name="section53998444524"></a>**

将单个文本进行向量化。

**函数原型<a name="section18789201331417"></a>**

```
def embed_query(text)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|是否必选|说明|
|--|--|--|--|
|text|str|必选|文本长度取值范围为：[1, 256]。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[float]|text转换后的向量。如果embedding模型的输出是512维向量，最终的输出结果为大小为512的浮点数列表。|



#### embed\_images<a name="ZH-CN_TOPIC_0000002419102864"></a>

**功能描述<a name="section53998444524"></a>**

将给定的图片进行向量化。

**函数原型<a name="section18789201331417"></a>**

```
def embed_images(images, batch_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|images|Union[List[str], List[Image.Image]]|必选|入参类型为List[str]时，列表中每个元素为图片base64编码后对应的字符串，列表总长度取值范围为[1, 1000],每个元素长度取值范围为[1, 10 * 1024 * 1024]；入参类型为List[Image.Image]时，表示输入的数据类型为PIL.Image.Image。|
|batch_size|int|可选|组batch大小，每次会组合batch_size的images进行embed操作，取值范围：[1, 1024]，默认值为32。可配置的值由设备显存决定。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|List[List[float]]|images转换后的向量数组。如果images为长度是4的数组，embedding模型的输出是512维向量，最终的输出结果为（4，512）大小的数组。|




### EmbeddingFactory<a name="ZH-CN_TOPIC_0000002452821629"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002419262716"></a>

**功能描述<a name="section957011509130"></a>**

embedding的工厂方法类，用于生产RAG SDK的embedding。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.embedding import EmbeddingFactory
class EmbeddingFactory(ABC)
```

**调用示例<a name="section17546115484515"></a>**

```
from paddle.base import libpaddle
from mx_rag.embedding import EmbeddingFactory
from mx_rag.utils import ClientParam
txt_embed = EmbeddingFactory.create_embedding(embedding_type="local_text_embedding",
                                              model_path="path to model", dev_id=0)
print(txt_embed.embed_query("abc"))
# 根据实际情况修改参数
tei_embed = EmbeddingFactory.create_embedding(embedding_type="tei_embedding",
                                              url="https://ip:port/embed",
                                              client_param=ClientParam(ca_file="/path/to/ca.crt"))
print(tei_embed.embed_query("abc"))
img_embed = EmbeddingFactory.create_embedding(embedding_type="local_images_embedding", model_name="model_name", 
                                              model_path="path to model", dev_id=0)
print(img_embed.embed_query("abc"))
```


#### create\_embedding<a name="ZH-CN_TOPIC_0000002452701745"></a>

**功能描述<a name="section957011509130"></a>**

构造embedding实例。会调用TextEmbedding、ImageEmbedding和TEIEmbedding类的静态方法create创建类实例。

**函数原型<a name="section12411139493"></a>**

```
@classmethod
def create_embedding(**kwargs):
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|embedding_type|str|必选|该参数在kwargs中，用于指定生成的embedding类型。通过关键字参数传递。<br>可取值：<li>local_text_embedding<li>local_images_embedding<li>tei_embedding|
|**kwargs|Any|可选|除去embedding_type，其余参数为embedding的构造参数。调用对应类的静态方法create返回实例。<li>如果是local_text_embedding，请参见[类功能](#类功能)。<li>如果是local_images_embedding，请参见[类功能](#ZH-CN_TOPIC_0000002419102860)。<li>如果是tei_embedding，请参见[类功能](#ZH-CN_TOPIC_0000002452821613)。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|langchain_core.embeddings.Embeddings|返回Embeddings实例对象。|




