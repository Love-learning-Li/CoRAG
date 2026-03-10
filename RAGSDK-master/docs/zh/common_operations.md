# 常用操作<a name="ZH-CN_TOPIC_0000001981995336"></a>

## 日志说明<a name="ZH-CN_TOPIC_0000002018595381"></a>

RAG SDK安装包为run格式，RAG SDK的安装、卸载日志记录\~/log/mxRag/deployment.log。

RAG SDK运行日志模块使用的loguru，默认输出在控制台，如果需要请配置重定向文件。

为防止日志注入安全问题（如特殊字符 \`\\n\`、\`\\b\` 被转义），需要配置环境变量 \`LOGURU\_FORMAT\`，配置示例如下，主要是在message中使用\{message!r\}，确保特殊字符被安全处理，其他参数根据用户偏好设置。

```
export LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message!r}</level>'
```


