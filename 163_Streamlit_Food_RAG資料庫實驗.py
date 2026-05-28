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
    client_settings = chromadb.Settings(
        anonymized_telemetry = False
    )
)

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Streamlit UI（前端介面）
# 主標題（應用名稱）
st.set_page_config(
    page_title = "台灣美食助手",
    layout = "wide"
)

st.title("台灣美食助手")

# Sidebar（側邊欄 UI）
# 用於資料新增、修改與刪除
with st.sidebar:

    # 資料庫管理標題
    st.header("向量資料庫管理")

    # 目前資料庫內資料數量
    count = vectorstore._collection.count()
    st.write(f"目前資料總數：{count}")

    st.markdown("---")

    # 新增資料輸入區
    st.subheader("新增美食資料")

    new_content = st.text_area(
        "輸入美食資料",
        key = "new_content",
        height = 120
    )

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
            st.success("資料已成功存入！")

        # 防呆提示
        else:
            st.warning("請輸入內容後再儲存！")

    st.markdown("---")

    # 資料修改與刪除
    st.subheader("修改 / 刪除資料")

    if count > 0:

        # 取得全部資料
        all_docs = vectorstore.get()
        doc_ids = all_docs["ids"]
        doc_contents = all_docs["documents"]

        # 建立選單資料
        doc_options = [
            (doc_id, content)
            for doc_id, content in zip(doc_ids, doc_contents)
        ]

        # Selectbox（資料選單）
        selected_option = st.selectbox(
            "選取資料：",
            doc_options,
            format_func = lambda x: x[1][:20] + "..."
        )

        # 取得目前選取資料
        selected_id = selected_option[0]
        current_content = selected_option[1]

        # 編輯內容輸入區
        updated_content = st.text_area(
            "編輯內容：",
            value = current_content,
            height = 150
        )

        # 建立兩欄佈局
        col1, col2 = st.columns(2)

        # 更新資料
        with col1:
            if st.button("確認更新", type = "secondary"):

                # 防呆：避免空內容
                if updated_content.strip():

                    # 內容未變更
                    if updated_content == current_content:
                        st.info("內容沒有變更！")

                    else:
                        # 重新建立向量
                        new_embedding = embeddings.embed_query(updated_content)

                        # 更新資料
                        vectorstore._collection.update(
                            ids = [selected_id],
                            documents = [updated_content],
                            metadatas = [{"source": "food"}],
                            embeddings = [new_embedding]
                        )

                        st.success("資料更新成功！")

                # 防呆提示
                else:
                    st.warning("更新內容不可為空！")

        # 刪除資料
        with col2:
            if st.button("刪除此資料", type = "primary"):
                vectorstore.delete(ids = [selected_id])
                st.success("資料已成功刪除！")

    else:
        st.info("目前沒有任何資料，請先新增資料！")

# Query（查詢區）
st.markdown("---")

question = st.text_input(
    "請輸入詢問的美食資訊：",
    placeholder = "南部粽與北部粽有什麼差別？"
)

# RAG 流程（檢索 + 生成）
if question:
    results = vectorstore.similarity_search(question, k = 5, filter = {"source": "food"})

    if results:
        # 組合上下文（提供給 LLM）
        context = "\n".join([doc.page_content for doc in results])

        # Prompt（提示詞）
        prompt = f"""
        你是一個台灣美食資料助理。

        規則：
        1. 只能依據提供資料回答
        2. 不可以自行補充外部知識
        3. 如果資料不足，請回答「資料不足，無法判斷」
        4. 請使用繁體中文回答

        資料：
        {context}

        問題：
        {question}

        請用條列方式回答：
        """

        # LLM 生成回答
        with st.spinner("AI 正在思考中..."):
            response = llm.invoke(prompt)

        st.subheader("AI 回答：")
        st.info(response)
        print("=" * 50)

        # 顯示檢索資料
        with st.expander("查看檢索到的資料："):
            for i, doc in enumerate(results, start = 1):
                st.write(f"### 資料 {i}")
                st.write(doc.page_content)
                st.markdown("---")

    else:
        st.warning("資料庫找不到相關資訊，請先新增資料！")