from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import dashscope
from dashscope import TextEmbedding
import os

# ===================== 配置区（只改这里！）=====================
API_KEY = "sk-da6ab4fb9490463a8c1e32617b0e2a5c"
PDF_PATH = "test.pdf"
# ==============================================================

# 配置通义千问API
dashscope.api_key = API_KEY


# 自定义通义千问嵌入模型（解决格式不兼容问题）
class QwenEmbeddings:
    def embed_documents(self, texts):
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v1,
            input=texts
        )
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def main():
    print("正在加载并处理PDF文档...")
    # 1. 加载PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 2. 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"文档分割完成，共 {len(splits)} 个文本块")

    # 3. 创建向量库
    print("正在生成向量并构建知识库...")
    embeddings = QwenEmbeddings()
    db = Chroma.from_documents(splits, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 4. 初始化大模型
    llm = ChatOpenAI(
        model="qwen-turbo",
        temperature=0.1,
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    print("\n===== 本地文档智能问答机器人已启动 =====")
    print("输入问题即可提问，输入 exit 退出程序\n")

    while True:
        question = input("请输入问题：")
        if question.lower() == "exit":
            print("程序已退出")
            break

        # 检索相关文档
        relevant_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # 生成回答
        prompt = f"""
        请严格根据以下提供的上下文内容回答问题，不要编造任何上下文以外的信息。
        如果上下文中没有相关信息，请直接回答"抱歉，我在文档中没有找到相关内容"。

        上下文：
        {context}

        问题：{question}
        """

        response = llm.invoke(prompt)
        print("\n回答：", response.content)
        print("-" * 60)


if __name__ == "__main__":
    main()