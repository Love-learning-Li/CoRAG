## 通用<a name="ZH-CN_TOPIC_0000002503733171"></a>

### ClientParam<a name="ZH-CN_TOPIC_0000002022246688"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002058325153"></a>

**功能描述<a name="section957011509130"></a>**

对接服务端配置参数。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.utils import ClientParam
ClientParam(use_http, ca_file, crl_file, timeout, response_limit_size)
```

**输入参数说明<a name="section1054013414143"></a>**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|use_http|bool|可选|指定客户端是否可以使用HTTP协议，默认值为False，即使用HTTPS协议。<br>HTTP协议存在安全风险，建议使用HTTPS安全协议。|
|ca_file|str|可选|对接服务端根证书，默认值为""。路径长度不能超过1024，不能为软链接且不允许存在".."，文件大小不超过1M。|
|crl_file|str|可选|对接服务端吊销列表，默认值为""。路径长度不能超过1024，不能为软链接且不允许存在".."，文件大小不超过1M。|
|timeout|int|可选|对接服务端响应超时时间，取值(0, 600]，默认值为60，单位为秒。|
|response_limit_size|int|可选|客户端接受服务端响应的最大字节，取值(0, 10MB]，默认1M。|


> [!NOTE]说明 
>创建客户端时，判断use\_http，如果启用HTTPS时，ca\_file是必传参数，如果只传入了ca\_file参数，创建单向认证的TLS/SSL context；如果未启用HTTPS，客户端配置默认的TLS/SSL context。

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



### Lang<a name="ZH-CN_TOPIC_0000002470814566"></a>

#### 类功能<a name="ZH-CN_TOPIC_0000002503734751"></a>

语言枚举类。

当前支持两种语言，包括英文（EN）和中文（CH）。

-   EN：设定工作语言为英文。
-   CH：设定工作语言为中文。

**函数原型<a name="section12411139493"></a>**

```
from mx_rag.utils import Lang
class Lang(Enum):
    EN: str = 'en'
    CH: str = 'ch'
```