# RAG本地文档智能问答机器人
基于LangChain + 通义千问大模型实现的本地PDF文档问答系统，可以读取本地PDF文件，严格基于文档内容回答问题，不会编造信息。

## ✨ 功能特点
- ✅ 本地PDF文档自动加载与解析
- ✅ 智能文本分段与向量化存储
- ✅ 向量库持久化，无需每次重新生成
- ✅ 基于相似度的精准文档检索
- ✅ 通义千问大模型生成准确回答
- ✅ 流式输出打字机效果，体验流畅
- ✅ 严格基于文档内容，不编造信息

## 🚀 快速开始
### 1. 安装依赖
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb pypdf dashscope
