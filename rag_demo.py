from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import dashscope
from dashscope import TextEmbedding
import os
import sys
import warnings

# 忽略所有警告，运行时干干净净
warnings.filterwarnings("ignore")

# ===================== 配置区（只改这里！）=====================
API_KEY = "sk-da6ab4fb9490463a8c1e32617b0e2a5c"
DOCS_FOLDER = "./docs"  # 所有PDF文档都放在这个文件夹里
VECTOR_DB_PATH = "./chroma_db"  # 向量库本地保存路径
# ==============================================================

# 配置通义千问API，强制关闭代理解决网络问题
dashscope.api_key = API_KEY
os.environ["NO_PROXY"] = "dashscope.aliyuncs.com"
os.environ["no_proxy"] = "dashscope.aliyuncs.com"


# 自定义通义千问嵌入模型
class QwenEmbeddings:
    def embed_documents(self, texts):
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v1,
            input=texts
        )
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# 流式输出打字机效果
def stream_print(response):
    print("\n回答：", end="", flush=True)
    for chunk in response:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n" + "-" * 60)


def main():
    embeddings = QwenEmbeddings()

    # 向量库持久化：存在就加载，不存在就创建
    if os.path.exists(VECTOR_DB_PATH):
        print("检测到已存在的向量库，正在加载...")
        db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
    else:
        print("未检测到向量库，正在加载docs文件夹中的所有PDF文档...")

        # 自动创建docs文件夹（如果不存在）
        if not os.path.exists(DOCS_FOLDER):
            os.makedirs(DOCS_FOLDER)
            print(f"✅ 已自动创建 {DOCS_FOLDER} 文件夹")
            print("请将需要问答的PDF文件放入该文件夹后重新运行程序")
            return

        # 加载文件夹中所有PDF文件（包括子文件夹）
        loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.pdf",  # 匹配所有.pdf后缀的文件
            loader_cls=PyPDFLoader,
            show_progress=True  # 显示加载进度条
        )
        docs = loader.load()

        # 文本智能分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print(f"\n✅ 文档加载完成，共 {len(docs)} 个PDF，分割为 {len(splits)} 个文本块")

        # 生成向量并构建知识库
        print("正在生成向量并构建知识库...")
        db = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print("✅ 向量库已自动保存到本地")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 初始化大模型
    llm = ChatOpenAI(
        model="qwen-turbo",
        temperature=0.1,
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        streaming=True
    )

    print("\n" + "=" * 50)
    print("===== 本地多文档智能问答机器人已启动 =====")
    print("输入问题即可提问，输入 exit 退出程序")
    print("=" * 50 + "\n")

    while True:
        question = input("请输入问题：")
        if question.lower() == "exit":
            print("程序已退出")
            break

        # 检索相关文档
        relevant_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # 生成回答并流式输出
        prompt = f"""
        请严格根据以下提供的上下文内容回答问题，不要编造任何上下文以外的信息。
        如果上下文中没有相关信息，请直接回答"抱歉，我在文档中没有找到相关内容"。

        上下文：
        {context}

        问题：{question}
        """

        response = llm.stream(prompt)
        stream_print(response)


if __name__ == "__main__":
    main()