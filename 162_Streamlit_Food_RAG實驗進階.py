import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb

# Embedding Model（向量模型）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Vector Database（向量資料庫）
vectorstore = Chroma(
    collection_name = "my_docs",
    embedding_function = embeddings,
    persist_directory = "./food_chroma_db",
    client_settings = chromadb.Settings(anonymized_telemetry = False)
)

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Streamlit UI（前端介面）
# 主標題（應用名稱）
st.title("台灣美食助手")

# Sidebar（側邊欄 UI）
# 用於資料新增與資料庫狀態顯示
with st.sidebar:

    # 目前資料庫內資料數量
    count = vectorstore._collection.count()
    st.write(f"目前資料總數：{count}")

    # 新增資料輸入區
    new_content = st.text_area("輸入美食資料：", height = 150)

    # 寫入向量資料庫按鈕
    if st.button("存入向量資料庫"):

        # 防呆：避免空資料寫入
        if new_content.strip():

            # 封裝成 Document（RAG 格式）
            new_doc = Document(
                page_content = new_content,
                metadata = {"source": "food"}
            )

            # 寫入 Chroma 向量資料庫
            vectorstore.add_documents([new_doc])
            st.success("資料已經存入！")

            # 更新資料數量
            count = vectorstore._collection.count()
            st.write(f"寫入後資料總數：{count}")

        # 防呆提示
        else:
            st.warning("請輸入內容後再儲存！")

# Query（查詢區）
question1 = st.text_input(
    "請輸入詢問的美食資訊",
    placeholder = "南部粽與北部粽有什麼差別？"
)

# RAG 流程（檢索 + 生成）
if question1:

    # 向量檢索（找最相關資料）
    results = vectorstore.similarity_search(question1, k = 8, filter = {"source": "food"})

    # 組合上下文（提供給 LLM）
    context = "\n".join([doc.page_content for doc in results])

    # Prompt（提示詞）
    prompt = f"""
    只能依據提供資料回答。

    資料：
    {context}

    問題：
    {question1}

    請根據資料回答問題。
    """

    # LLM 生成回答
    response = llm.invoke(prompt)

    # 顯示結果
    st.write("回答結果：")
    st.info(response)

# 如果沒有找到相關資料，給予提示
else:
    st.warning("資料庫中找不到相關的美食資訊，請先在側邊欄補充資料喔！")