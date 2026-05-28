import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 164_MultiPDF_RAG實驗 =====")

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Text Splitter（文件切塊工具）
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

# PDF Loader（讀取資料夾內 PDF）
loader = DirectoryLoader(
    "./aipdf",
    glob = "**/*.pdf",
    loader_cls = PyPDFLoader,
    show_progress = True,
    use_multithreading = True,
    silent_errors = True
)

# 讀取 PDF 文件
documents = loader.load()
print("=" * 50)
print("PDF頁數：", len(documents))

# 文件切塊（Chunking）
docs = splitter.split_documents(documents)
print("=" * 50)
print("切塊後數量：", len(docs))

# 列出成功讀取的檔案清單
loaded_files = set([
    doc.metadata.get("source")
    for doc in documents
])

print("=" * 50)
print("成功讀取到的檔案清單：")
print()

for f in loaded_files:
    print("- ", f)

# Vector Database（向量資料庫）
db = Chroma.from_documents(
    documents = docs,
    embedding = embedding,
    persist_directory = "./multipdfdb"
)

# Query（查詢問題）
query = "LLM 是甚麼"

# Similarity Search（向量檢索）
results = db.similarity_search(query, k = 5)

# 列出相似文件內容
print("=" * 50)
print("相似文件內容：")
print()

for i, doc in enumerate(results):
    print(f"--- 第 {i + 1} 筆文件 ---")
    print(doc.page_content)
    print("=" * 50)

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

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