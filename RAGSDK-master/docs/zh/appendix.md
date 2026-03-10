# 附录<a name="ZH-CN_TOPIC_0000002031193448"></a>

## 软件包含的公网地址<a name="ZH-CN_TOPIC_0000002018714937"></a>

RAG SDK编译的开源软件中会存在公开网址和邮箱地址，SDK本身不会访问，不会造成风险。

更多公网地址和邮箱地址请参考[RAG SDK 7.3.0 公网地址.xlsx](./resource/RAG%20SDK%207.3.0%20公网地址.xlsx)。


## 环境变量说明<a name="ZH-CN_TOPIC_0000002353214097"></a>

**表 1**  环境变量

|环境变量名|说明|
|--|--|
|PATH|可执行程序的文件路径。|
|LD_LIBRARY_PATH|动态链接库路径。|
|PYTHONPATH|python模块文件的默认搜索路径。|
|HOME|当前用户的家目录。|
|PWD|当前系统路径。|
|TMPDIR|临时文件路径。|
|LANG|语言环境。|
|RAG_SDK_HOME|RAG SDK软件工作路径|
|ATB_LOG_TO_STDOUT|设置为1时，算子加速日志记录到标准输出|
|ATB_LOG_TO_FILE|设置为1时，算子加速日志记录到文件|
|ATB_LOG_LEVEL|设置算子加速日志等级，可配置为TRACE，DEBUG, INFO，WARN， ERROR， FATAL|
|ENABLE_BOOST|是否激活向量模型推理加速，设置值为"True"或"False"|
|DISABLE_RAGS_LOGGING|设置为0时，启用ragas的日志输出，默认不开启。|
|RAGAS_DO_NOT_TRACK|是否对RAGAS报告上传到远端网站，rag sdk已固定设置为"true"，即表示不上传报告到远端网站。|
|HF_HUB_OFFLINE|加载权重时是否只支持加载离线文件，rag sdk已固定设置为"1"，即只支持加载离线权重文件。|
|HF_DATASETS_OFFLINE|是否支持加载hugging face离线数据，rag sdk已固定设置为"1"，即只支持加载离线数据集。|
|AUTO_DOWNLOAD_NLTK|Markdown文档首次解析时是否自动下载NLTK分词模型，rag sdk已固定设置为"false"，即表示不自动下载NLTK分词模型。|


> [!NOTE] 说明 
>考虑业务数据安全，请勿配置RAGAS\_DO\_NOT\_TRACK、HF\_HUB\_OFFLINE、HF\_DATASETS\_OFFLINE、AUTO\_DOWNLOAD\_NLTK。


## 用户信息列表<a name="ZH-CN_TOPIC_0000002009306806"></a>

请周期性地更新用户的密码，避免长期使用同一个密码带来的风险。

**系统用户<a name="section1570875794310"></a>**

**表 1**  用户信息列表

|用户名|描述|初始密码|密码修改方法|
|--|--|--|--|
|root|操作系统用户，安装驱动。|用户自定义|使用**passwd**命令修改。|
|HwHiAiUser|操作系统用户，RAG SDK部署后普通运行用户。|用户自定义|使用**passwd**命令修改。|
|openguass用户名|连接openguass数据库用户名。|用户自定义|请参考openguass官方网站。|
|milvus用户名|连接milvus数据库用户名。|用户自定义|请参考milvus官方网站。|


**Ubuntu基础镜像中的用户<a name="zh-cn_topic_0000001951504565_zh-cn_topic_0000001515257736_zh-cn_topic_0000001446965016_section158195363315"></a>**

|用户|初始密码|密码修改方法|
|--|--|--|
|root|无|-|
|daemon|无|-|
|bin|无|-|
|sys|无|-|
|sync|无|-|
|games|无|-|
|man|无|-|
|lp|无|-|
|mail|无|-|
|news|无|-|
|uucp|无|-|
|proxy|无|-|
|www-data|无|-|
|backup|无|-|
|list|无|-|
|irc|无|-|
|gnats|无|-|
|nobody|无|-|
|_apt|无|-|



## 修订记录<a name="ZH-CN_TOPIC_0000002067272857"></a>

|发布日期|修订记录|
|--|--|
|2025-06-30|第一次正式发布|



