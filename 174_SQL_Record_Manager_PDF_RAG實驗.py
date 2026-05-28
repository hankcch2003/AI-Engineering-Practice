import os
import glob

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 174_SQL_Record_Manager_PDF_RAG實驗 =====\n")

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager
from langchain.indexes import index
from langchain_community.document_loaders import PyPDFLoader

# Base Directory（基礎路徑）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Target Directory（目標資料夾）
TARGET_DIR = os.path.join(BASE_DIR, "multi_pdf")

# Database Directory（資料庫資料夾）
DB_DIR = os.path.join(BASE_DIR, "newsdb_pdf")

# SQL Database Path（SQL 資料庫路徑）
SQL_DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'record_manager_pdf.sql')}"

print("目前程式所在目錄：", BASE_DIR)
print("=" * 50)
print("預期讀取的資料夾：", TARGET_DIR)

# Create Folder（建立資料夾）
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

    print("=" * 50)
    print(f"已建立資料夾：{TARGET_DIR}")
    print("請將 pdf 檔案放入資料夾")

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Text Splitter（文字切割器）
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

# Vector Database（向量資料庫）
db = Chroma(
    persist_directory = DB_DIR,
    embedding_function = embedding
)

# SQL Record Manager（SQL 記錄管理器）
record_manager = SQLRecordManager(
    namespace = "chroma/pdf_docs",
    db_url = SQL_DB_PATH
)

record_manager.create_schema()

# Search Path（搜尋路徑）
search_path = os.path.join(TARGET_DIR, "*.pdf")

# File Search（搜尋檔案）
files = glob.glob(search_path)

# Documents（文件資料）
all_docs = []

print("=" * 50)
print("開始掃描檔案：", search_path)
print("=" * 50)

# Load Documents（載入文件）
for file in files:
    print("成功讀取 PDF 檔：", file)

    loader = PyPDFLoader(file)
    docs = loader.load_and_split(text_splitter = splitter)
    all_docs.extend(docs)

print("=" * 50)
print("文本片段總數量：", len(all_docs))

# Incremental Index（增量更新）
if all_docs:
    indexing_result = index(
        all_docs,
        record_manager,
        db,
        cleanup = "incremental",
        source_id_key = "source"
    )

    print("=" * 50)
    print("資料庫更新狀態：")
    print(indexing_result)

    print("=" * 50)
    print("資料庫資料夾：")
    print(DB_DIR)

    print("=" * 50)
    print("SQL 記錄檔：")
    print(os.path.join(BASE_DIR, "record_manager_pdf.sql"))

else:
    print("=" * 50)
    print("資料夾內沒有任何 pdf 檔案")
    print("=" * 50)
    print("無法建立資料庫")

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Query（查詢問題）
query = "RNN 是什麼？"

# Similarity Search（相似度搜尋）
results = db.similarity_search(query, k = 10)

# RAG Retrieval（RAG 檢索）
if results:
    context = "\n".join([doc.page_content for doc in results])

    print("=" * 50)
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
    {query}

    若資料沒有答案，請回答：查無資料。
    """

    # Response（模型輸出）
    response = llm.invoke(prompt)
    print("回答結果：")
    print(response)