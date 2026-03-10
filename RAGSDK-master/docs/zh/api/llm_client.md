
## 对接模型客户端<a name="ZH-CN_TOPIC_0000001990172976"></a>

### 大语言模型<a name="ZH-CN_TOPIC_0000002018595309"></a>

#### Text2TextLLM<a name="ZH-CN_TOPIC_0000001982155120"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000002018595341"></a>

**功能描述<a name="section957011509130"></a>**

建立客户端对接语言大模型服务，提供大模型交互功能，当前只支持兼容OpenAI接口/v1/chat/completions。该类继承实现了langchain.llms.base.LLM。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.llm import Text2TextLLM
# 所有参数需通过关键字参数传递
Text2TextLLM(base_url, model_name, llm_config, client_param)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|base_url|str|必选|大模型服务地址。长度取值范围[1, 128]。|
|model_name|str|必选|LLM模型名称。长度取值范围[1, 128]。|
|llm_config|LLMParameterConfig|可选|通过langchain调用时生效，描述参见[LLMParameterConfig](#llmparameterconfig)；非langchain方式调用通过chat和chat_streamly方法传入参数，参见[chat](#chat)和[chat_streamly](#chat_streamly)。|
|client_param|ClientParam|可选|https客户端配置参数，默认值为ClientParam()，具体描述请参见[ClientParam](./univers_api.md#clientparam)。|


**调用示例<a name="section1743812130014"></a>**

```
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
res = llm.chat("请介绍下北京")
print(res)
for res in llm.chat_streamly("请介绍下北京"):    
    print(res)
```


##### chat<a name="ZH-CN_TOPIC_0000002018595377"></a>

**功能描述<a name="section53998444524"></a>**

与LLM服务进行对话，获取LLM模型的推理结果。

**函数原型<a name="section18789201331417"></a>**

```
def chat(query, sys_messages, role, llm_config)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|query|str|必选|推理请求文本，字符串长度范围[1, 4 * 1024 * 1024]。|
|sys_messages|List[dict]|可选|系统消息，列表最大长度为16，列表每个字典长度最大为16，字典key字符串长度最大为16，value字符串最大长度为4 * 1024* 1024，默认值为None。|
|role|str|可选|推理请求消息角色，长度取值[1, 16]，默认值为user。|
|llm_config|LLMParameterConfig|可选|调用大模型的参数，描述参见[LLMParameterConfig](#llmparameterconfig)，默认为None。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|str|LLM文本推理的结果。|



##### chat\_streamly<a name="ZH-CN_TOPIC_0000002018595177"></a>

**功能描述<a name="section53998444524"></a>**

与LLM服务进行对话，获取LLM模型推理的流式结果。

**函数原型<a name="section18789201331417"></a>**

```
def chat_streamly(query, sys_messages, role, llm_config)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|query|str|必选|推理请求文本，字符串长度范围[1, 4 * 1024 * 1024]。|
|sys_messages|List[dict]|可选|系统消息，列表最大长度为16，列表每个字典长度最大为16，字典key字符串长度最大为16，value字符串最大长度为4 * 1024* 1024，默认值为None。|
|role|str|可选|推理请求消息角色，长度取值[1, 16]，默认值为user。|
|llm_config|LLMParameterConfig|可选|调用大模型的参数，描述参见[LLMParameterConfig](#llmparameterconfig)。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|Iterator[str]|LLM文本推理的流式结果。|





### 图像生成模型<a name="ZH-CN_TOPIC_0000001981995332"></a>

#### Text2ImgMultiModel<a name="ZH-CN_TOPIC_0000001982155112"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000001981995300"></a>

**功能描述<a name="section957011509130"></a>**

对接文生图大模型服务，提供大模型交互功能。

当前只支持模型：stable-diffusion-v1-5和stable-diffusion-2-1-base。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.llm import Text2ImgMultiModel
Text2ImgMultiModel(url, model_name, client_param)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|url|str|必选|大模型访问url。长度取值[1, 128]。|
|model_name|str|可选|SD模型名称。默认值为None。长度取值范围(0,128]。|
|client_param|ClientParam|可选|https客户端配置参数，默认值为ClientParam()，具体描述请参见[ClientParam](./univers_api.md#clientparam)|


**返回值说明<a name="section53998444524"></a>**

Text2ImgMultiModel对象。

**调用示例<a name="section145571842142214"></a>**

```
from mx_rag.llm import Text2ImgMultiModel
from mx_rag.utils import ClientParam
multi_model = Text2ImgMultiModel(model_name="sd", url="txt to image url",
                                 client_param=ClientParam(ca_file="/path/to/ca.crt"))
res = multi_model.text2img(prompt="dog wearing black glasses", output_format="jpg", size="512*512")
print(res)
```


##### text2img<a name="ZH-CN_TOPIC_0000002018714797"></a>

**功能描述<a name="section53998444524"></a>**

与SD服务交互进行文生图，获取SD模型推理的结果。

请求body数据格式如下：

```
{
"prompt": 文本生成提示词,
"output_format": 生成的图片格式,
"size": 生成图片尺寸,
"model_name": 模型名字
}
```

**函数原型<a name="section18789201331417"></a>**

```
def text2img(prompt, output_format, size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|prompt|str|必选|生成图片的提示词。长度取值范围[1, 1024 * 1024]。|
|output_format|str|可选|生成图片的格式。取值类型png、jpeg、jpg和webp，默认值为png。|
|size|str|可选|图片生成尺寸，表示为"height*width"，具体支持的尺寸由对应的大模型决定，正则匹配格式为: "^\d{1,5}\*\d{1,5}$"，默认"512*512"，当前支持的模型生成的图片支持"512 * 512"。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|dict|返回格式为{"prompt": prompt, "result": data}，其中prompt为图片生成的提示词，result为大模型推理结果的图片base64编码后的数据。|




#### Img2ImgMultiModel<a name="ZH-CN_TOPIC_0000001981995588"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000002018714685"></a>

**功能描述<a name="section957011509130"></a>**

对接图生图大模型服务，提供大模型交互功能。

当前只支持IP-Adapter搭建的模型：stable-diffusion-v1-5

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.llm import Img2ImgMultiModel
Img2ImgMultiModel(url, model_name, client_param)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|url|str|必选|大模型访问url。长度取值范围[1, 128]。|
|model_name|str|可选|SD模型名称。默认值为None。长度取值范围[1, 128]。|
|client_param|ClientParam|可选|https客户端配置参数，默认值为ClientParam()，具体描述请参见[ClientParam](./univers_api.md#clientparam)。|


**返回值说明<a name="section53998444524"></a>**

Img2ImgMultiModel对象。

**调用示例<a name="section175571825169"></a>**

```

import sys
from mx_rag.document.loader import ImageLoader
from mx_rag.llm import Img2ImgMultiModel
from mx_rag.utils import ClientParam
multi_model = Img2ImgMultiModel(url="image to image url", model_name="sd",
                                client_param=ClientParam(ca_file="/path/to/ca.crt")
                                )
loader = ImageLoader("image path")
docs = loader.load()
if len(docs) < 1:
    print("load image failed")
    sys.exit(1)
res = multi_model.img2img(
    prompt="he is a knight, wearing armor, big sword in right hand. Blur the background, focus on the knight",
    image_content=docs[0].page_content,
    size="512*512")
print(res)

```


##### img2img<a name="ZH-CN_TOPIC_0000001982155128"></a>

**功能描述<a name="section53998444524"></a>**

与大模型服务交互进行文生图，获取模型推理的结果。发送给大模型数据格式如下：

```
请求body数据格式如下：
{
"prompt": 生成图片提示词,
"image": 图片数据base64编码后的数据,
"size": 图片生成尺寸，
"model_name": 模型名字
}
```

**函数原型<a name="section18789201331417"></a>**

```
def img2img(prompt, image_content, size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|prompt|str|必选|生成图片的提示词。长度取值范围[1, 1024 * 1024]。|
|image_content|str|必选|图片数据base64编码后对应的字符串，长度取值范围：(0, 10 * 1024 * 1024]。|
|size|str|可选|图片生成尺寸，表示为"height*width"，具体支持的尺寸由对应的大模型决定，正则匹配格式为: "^\d{1,5}\*\d{1,5}$"，默认"512*512"。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|dict|返回格式为{"prompt": prompt, "result": data}，其中prompt为图片生成的提示词，result为大模型推理结果的图片base64编码后的数据。|





### 参数类说明<a name="ZH-CN_TOPIC_0000002005999484"></a>

#### LLMParameterConfig<a name="ZH-CN_TOPIC_0000002042197585"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000002006157752"></a>

**功能描述<a name="section957011509130"></a>**

对接语言大模型参数类，每个参数具体有效值会根据不同模型配置有差别。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.llm import LLMParameterConfig
LLMParameterConfig(max_tokens, presence_penalty, frequency_penalty, temperature, top_p, seed, stream)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|max_tokens|int|可选|允许推理生成的最大token个数，取值范围[1，100000]，默认值为512。由kwargs传递获取。实际有效取值依赖MindIE的配置，请参见《MindIE LLM开发指南》中的“核心概念与配置 > 配置参数说明（服务化）”章节中关于maxSeqLen的说明。|
|presence_penalty|float, int|可选|影响模型如何根据到目前为止是否出现在文本中来惩罚新token。正值将通过惩罚已经使用的词，增加模型谈论新主题的可能性。取值范围：[-2.0, 2.0]，默认值为0.0。|
|frequency_penalty|float. int|可选|影响模型如何根据文本中词汇（token）的现有频率惩罚新词汇（token）。正值将通过惩罚已经频繁使用的词来降低模型一行中重复用词的可能性。取值范围：[-2.0, 2.0]，默认值为0.0。|
|seed|int|可选|用于指定推理过程的随机种子，相同的seed值可以确保推理结果的可重现性，不同的seed值会提升推理结果的随机性。取值范围[0, 2 ** 31 - 1]，不传递该参数，系统会产生一个随机seed值，默认值为None。|
|temperature|float, int|可选|控制生成的随机性，较高的值会产生更多样化的输出，取值范围[0.0, 2.0]，默认值为1.0。|
|top_p|float, int|可选|控制模型生成过程中考虑的词汇范围，使用累积概率选择候选词，直到累积概率超过给定的阈值。该参数也可以控制生成结果的多样性，它基于累积概率选择候选词，直到累积概率超过给定的阈值为止。取值范围(0.0, 1.0]，默认值为1.0。|
|stream|bool|可选|是否流式回答，默认值为False，该参数在如下场景生效<br>["ParallelText2TextChain", "SingleText2TextChain"，"GraphRagText2TextChain"]。|


**调用示例<a name="section96001515205720"></a>**

```

from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
res = llm.chat("请介绍下北京")
print(res)
for res in llm.chat_streamly("请介绍下北京"):    
    print(res)
```




### 视觉大模型<a name="ZH-CN_TOPIC_0000002445441861"></a>

#### Img2TextLLM<a name="ZH-CN_TOPIC_0000002411922676"></a>

##### 类功能<a name="ZH-CN_TOPIC_0000002411762796"></a>

**功能描述<a name="section957011509130"></a>**

建立客户端对接视觉大模型服务，提供大模型交互功能，当前只支持兼容OpenAI接口/openai/v1/chat/completions。该类继承实现了langchain.llms.base.LLM。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.llm import Img2TextLLM
# 所有参数需通过关键字参数传递
Img2TextLLM(base_url, prompt, model_name, llm_config, client_param)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|base_url|str|必选|大模型服务地址。长度取值范围[1, 128]。|
|prompt|str|可选|提示词，用于指导视觉大模型生成结构化、详细且符合要求的图像描述，默认值为[IMG_TO_TEXT_PROMPT](#li183183578215)，用户也可根据需求配置。长度范围[1, 1024 * 1024]|
|model_name|str|必选|LLM模型名称。长度取值范围[1, 128]。|
|llm_config|LLMParameterConfig|可选|通过langchain调用时生效，描述参见[LLMParameterConfig](#llmparameterconfig)；非langchain方式调用通过chat方法传入参数，参见[chat](#chat)。|
|client_param|ClientParam|可选|https客户端配置参数，默认值为ClientParam()，具体描述请参见[ClientParam](./univers_api.md#clientparam)。|


-   <a id="li183183578215"></a>**图像结构化描述提示（IMG\_TO\_TEXT\_PROMPT）**

```
IMG_TO_TEXT_PROMPT = '''Given an image containing a table or figure, please provide a structured and detailed
description in chinese with two levels of granularity:

  Coarse-grained Description:
  - Summarize the overall content and purpose of the image.
  - Briefly state what type of data or information is presented (e.g., comparison, trend, distribution).
  - Mention the main topic or message conveyed by the table or figure.

  Fine-grained Description:
  - Describe the specific details present in the image.
  - For tables: List the column and row headers, units, and any notable values, patterns, or anomalies.
  - For figures (e.g., plots, charts): Explain the axes, data series, legends, and any significant trends, outliers,
  or data points.
  - Note any labels, captions, or annotations included in the image.
  - Highlight specific examples or noteworthy details.

  Deliver the description in a clear, organized, and reader-friendly manner, using bullet points or paragraphs
  as appropriate, answer in chinese'''
```

**调用示例<a name="section1743812130014"></a>**

```
from mx_rag.llm import Img2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
from PIL import Image
import io
import base64

vlm = Img2TextLLM(base_url="https://{ip}:{port}/openai/v1/chat/completions",
                   model_name="Qwen2.5-VL-7B-Instruct",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
# 生成图片base64编码
with Image.open("/path/to/image.jpeg") as img:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

image_url = {"url": f"data:image/jpeg;base64,{img_base64}"}
res = vlm.chat(image_url=image_url)
print(res)
```


##### chat<a name="ZH-CN_TOPIC_0000002445521929"></a>

**功能描述<a name="section53998444524"></a>**

与VLM服务进行交互，获取VLM模型的推理结果。

**函数原型<a name="section18789201331417"></a>**

```
def chat(image_url, prompt, sys_messages, role, llm_config)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|image_url|dict|必选|包含图片base64编码的字典，键为"url"，值为以"img_base64"为变量的字符串，示例：{"url": f"data:image/jpeg;base64,{image_base64}"}，其中image_base64为图片base64编码。长度范围[1, 4 * 1024 * 1024]。|
|sys_messages|List[dict]|可选|系统消息，列表最大长度为16，列表每个字典长度最大为16，字典key字符串长度最大为16，value字符串最大长度为4 * 1024* 1024，默认值为None。|
|role|str|可选|推理请求消息角色，长度取值[1, 16]，默认值为user。|
|llm_config|LLMParameterConfig|可选|调用大模型的参数，描述参见[LLMParameterConfig](#llmparameterconfig)。|


**返回值说明<a name="section11818153884917"></a>**

|数据类型|说明|
|--|--|
|str|VLM对图片内容的描述总结。|





