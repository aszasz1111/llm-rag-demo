from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import dashscope
from dashscope import TextEmbedding
import os
import warnings
import gradio as gr

# 忽略所有警告
warnings.filterwarnings("ignore")

# ===================== 配置区（只改这里！）=====================
API_KEY = "你自己的sk-开头的API Key"  # 改成占位符，避免泄露密钥
DOCS_FOLDER = "./docs"
VECTOR_DB_PATH = "./chroma_db"
# ==============================================================

# 配置通义千问不走代理
dashscope.api_key = API_KEY
os.environ["NO_PROXY"] = "dashscope.aliyuncs.com"
os.environ["no_proxy"] = "dashscope.aliyuncs.com"


# 自定义嵌入模型
class QwenEmbeddings:
    def embed_documents(self, texts):
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v1,
            input=texts
        )
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# 提取引用来源
def get_source_files(docs):
    sources = []
    for doc in docs:
        file_name = os.path.basename(doc.metadata.get("source", "未知文档"))
        if file_name not in sources:
            sources.append(file_name)
    return sources


# 全局初始化
embeddings = QwenEmbeddings()
retriever = None
llm = None


# 初始化知识库
def init_knowledge_base():
    global retriever, llm
    if os.path.exists(VECTOR_DB_PATH):
        print("检测到已存在向量库，正在加载...")
        db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
    else:
        print("未检测到向量库，正在加载docs文件夹所有PDF...")
        if not os.path.exists(DOCS_FOLDER):
            os.makedirs(DOCS_FOLDER)
            print(f"已自动创建 {DOCS_FOLDER}，请放入PDF后重启程序")
            return False

        loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print(f"\n文档加载完成：共{len(docs)}个PDF，分割为{len(splits)}个文本块")

        print("正在生成向量知识库...")
        db = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print("向量库保存完成！")

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model="qwen-turbo",
        temperature=0.1,
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return True


# ✅ 适配新版Gradio的对话逻辑（核心修复）
def chat_response(message, chat_history):
    if not message.strip():
        return "", chat_history

    # 1. 检索文档并获取来源
    relevant_docs = retriever.invoke(message)
    context = "\n".join([d.page_content for d in relevant_docs])
    source_list = get_source_files(relevant_docs)
    source_text = "📚 引用来源：" + "、".join(source_list)

    # 2. 拼接历史对话（适配新版格式）
    history_text = ""
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        history_text += f"{role}：{content}\n"

    # 3. 构造提示词
    prompt = f"""
你是本地PDF智能问答助手，严格只根据上下文回答，不许编造。
若无相关内容，直接回复：抱歉，我在文档中没有找到相关内容。

【历史对话】
{history_text}
【参考上下文】
{context}
【用户问题】
{message}
"""
    # 4. 获取回答
    res = llm.invoke(prompt)
    reply = res.content + "\n\n" + source_text

    # 5. 按新版格式更新对话历史
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})

    return "", chat_history


# 清空对话
def clear_chat():
    return []


# 创建网页界面
def create_web_ui():
    with gr.Blocks(title="本地PDF智能问答系统") as demo:
        gr.Markdown("# 📖 本地多PDF文档智能问答")
        gr.Markdown("自动加载docs文件夹PDF | 多轮对话 | 显示引用来源")

        # ✅ 新版Chatbot，默认支持字典格式
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="输入你的问题...")
        clear = gr.Button("清空对话")

        # 绑定交互事件
        msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat, [], chatbot)

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    if init_knowledge_base():
        create_web_ui()