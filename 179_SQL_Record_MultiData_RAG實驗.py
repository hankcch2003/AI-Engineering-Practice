import os
import glob
import shutil
import time
import gc

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager
from langchain.indexes import index
from langchain_community.vectorstores.utils import filter_complex_metadata

print("===== 179_SQL_Record_MultiData_RAG實驗 =====\n")

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader
)

# Base Directory（基礎路徑）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Target Directory（資料來源）
TARGET_DIR = os.path.join(BASE_DIR, "multi_data")

# Vector DB Directory（向量資料庫）
DB_DIR = os.path.join(BASE_DIR, "newsdb_multi")

# SQL Record Path（索引記錄）
SQL_FILE_PATH = os.path.join(BASE_DIR, "record_manager_multi.sql")
SQL_DB_PATH = f"sqlite:///{SQL_FILE_PATH}"

print("目前程式所在目錄：", BASE_DIR)
print("=" * 50)
print("資料來源資料夾：", TARGET_DIR)
print("=" * 50)

# Create Folder（建立資料夾）
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print("=" * 50)
    print("已建立資料夾，請放入多格式檔案")

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Text Splitter（文字切割）
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

# Load Single File（單檔載入函式）
def load_single_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".txt":
            return TextLoader(file_path, encoding = "utf-8").load()

        elif ext == ".pdf":
            return PyPDFLoader(file_path).load()

        elif ext == ".docx":
            return Docx2txtLoader(file_path).load()

        elif ext in [".ppt", ".pptx"]:
            return UnstructuredPowerPointLoader(file_path).load()

        elif ext in [".xls", ".xlsx"]:
            return UnstructuredExcelLoader(file_path, mode = "elements").load()

        elif ext == ".csv":
            return CSVLoader(file_path, encoding = "utf-8").load()

    except Exception as e:
        print("讀取失敗：", file_path, e)

    return []

# Collect Documents（收集文件）
all_docs = []

for ext in ["*.txt", "*.pdf", "*.docx", "*.pptx", "*.xlsx", "*.csv"]:
    for file_path in glob.glob(os.path.join(TARGET_DIR, ext)):
        print("讀取檔案：", file_path)
        all_docs.extend(load_single_file(file_path))

# Split + Clean（切分 + 清理）
split_docs = splitter.split_documents(all_docs)
cleaned_docs = filter_complex_metadata(split_docs)

print("=" * 50)

# Initialize DB（初始化向量資料庫）
def initialize_db():
    print("[系統] 初始化 Chroma + SQLRecordManager...")

    vector_store = Chroma(
        persist_directory = DB_DIR,
        embedding_function = embedding
    )

    record_manager = SQLRecordManager(
        namespace = "chroma/multi_docs",
        db_url = SQL_DB_PATH
    )

    record_manager.create_schema()

    if cleaned_docs:
        result = index(
            cleaned_docs,
            record_manager,
            vector_store,
            cleanup = "full",
            source_id_key = "source"
        )

        print("=" * 50)
        print("索引完成：", result)
        print("=" * 50)
    
    return vector_store

# Auto Recovery（自動修復機制）
vector_store = None

try:
    vector_store = initialize_db()
    _ = vector_store.similarity_search("test", k = 1)

except Exception as e:
    msg = str(e)

    if "header" in msg or "HNSW" in msg:
        print("=" * 50)
        print("偵測資料庫損壞，啟動修復...")

        vector_store = None
        gc.collect()
        time.sleep(1)

        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR, ignore_errors = True)

        if os.path.exists(SQL_FILE_PATH):
            try:
                os.remove(SQL_FILE_PATH)
            except:
                pass

        vector_store = initialize_db()

    else:
        raise e

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Query（查詢問題）
user_input = "總統就職演說的重點是甚麼？"

# Optional File Filter（檔案精準搜尋）
search_filter = None

for ext in ["*.txt", "*.pdf", "*.docx", "*.pptx", "*.xlsx", "*.csv"]:
    for file_path in glob.glob(os.path.join(TARGET_DIR, ext)):
        filename = os.path.basename(file_path)

        if filename in user_input:
            search_filter = {"source": file_path}
            print("鎖定檔案：", filename)
            break

# RAG Retrieval（RAG 檢索）
if search_filter:
    print("===== 檔案過濾模式（Similarity Search） =====")
    results = vector_store.similarity_search(
        user_input,
        k = 5,
        filter = search_filter
    )
else:
    print("===== 全域搜尋模式（Similarity Search） =====")
    results = vector_store.similarity_search(
        user_input,
        k = 5
    )

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])

print()
print("相似文件內容：")
print()
print(context)
print("=" * 50)

print(f"找到 {len(results)} 筆相關資料：")

for i, doc in enumerate(results, start = 1):
    print()
    print(f"[{i}]")
    print(doc.page_content)

print("=" * 50)

# Prompt（提示詞）
prompt = f"""
你是公司制度助理，只能依據提供資料回答。

資料：
{context}

問題：
{user_input}

若資料沒有答案，請回答：查無資料。
"""

# Response（模型輸出）
response = llm.invoke(prompt)
print("回答結果：")
print(response)