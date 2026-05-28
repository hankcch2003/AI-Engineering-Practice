import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 172_Conversational_Retrieval_Chain實驗 =====\n")

from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Vector Database（向量資料庫）
vectordb = Chroma(
    persist_directory = "./multipdfdb",
    embedding_function = embedding
)

# Retriever（檢索器）
retriever = vectordb.as_retriever(search_kwargs = {"k": 3})

# Memory（摘要記憶）
memory = ConversationSummaryMemory(
    llm = llm,
    memory_key = "chat_history",
    return_messages = True
)

# Conversational Retrieval Chain（對話檢索鏈）
qa = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    memory = memory
)

# Chat Loop（對話迴圈）
while True:
    question = input("你：")

    if question == "exit":
        break

    result = qa.invoke({
        "question": question
    })

    print("=" * 50)
    print("回答結果：")
    print(result["answer"])
    print("=" * 50)