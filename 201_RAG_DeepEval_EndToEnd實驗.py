import os
import warnings
import logging

# 忽略警告訊息（避免輸出過多 LangChain / Deprecation Warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

print("===== 201_RAG_DeepEval_EndToEnd實驗 =====\n")

# Base Directory（專案目錄）
base_dir = r"E:\Python\Python4-人工智慧整合開發實務"

# Load Document（讀取文件）
file_path = os.path.join(base_dir, "請假制度.txt")

with open(file_path, "r", encoding = "utf-8") as f:
    text = f.read()

# Split Document（文件切分）
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

chunks = splitter.split_text(text)

# Create Documents（建立文件物件）
documents = [
    Document(page_content = chunk)
    for chunk in chunks]

# Create Embedding（建立向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Create Vector DB（建立向量資料庫）
persist_directory = os.path.join(base_dir, "db")

db = Chroma.from_documents(
    documents = documents,
    embedding = embedding,
    persist_directory = persist_directory
)

# Retriever（建立檢索器）
retriever = db.as_retriever(search_kwargs = {"k": 2})

# LLM（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt Template（提示模板）
prompt = ChatPromptTemplate.from_template("""
你是一個公司 HR 助理。

請根據以下內容回答問題：

{context}

問題：
{question}
""")

question = "滿一年有幾天年假？"

# Retrieve Context（檢索內容）
docs = retriever.invoke(question)

retrieval_context_list = [doc.page_content for doc in docs]
context = "\n".join(retrieval_context_list)

# Generate Answer（產生回答）
chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": question
})

print("回答結果：")
print(response)
print("=" * 50)

# DeepEval Model（評估模型）
class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOllama(
            model = "llama3.2:latest",
            temperature = 0.0,
            format = "json"
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()

        return content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        content = response.content.strip()

        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        return content

    def get_model_name(self):
        return "llama3.2:latest"

judge_model = OllamaJudge()

# Metric（評估指標）
faithfulness_metric = FaithfulnessMetric(
    threshold = 0.7,
    model = judge_model
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold = 0.7,
    model = judge_model
)

precision_metric = ContextualPrecisionMetric(
    threshold = 0.7,
    model = judge_model
)

recall_metric = ContextualRecallMetric(
    threshold = 0.7,
    model = judge_model
)

relevancy_metric = ContextualRelevancyMetric(
    threshold = 0.7,
    model = judge_model
)

# Test Case（測試案例）
test_case = LLMTestCase(
    input = question,
    actual_output = response,
    retrieval_context = retrieval_context_list,
    expected_output = "滿一年未滿三年者，可獲得7天年假。"
)

print("DeepEval 開始評估...")
print("=" * 50)

# Contextual Precision Evaluation（上下文精確率評估）
precision_metric.measure(test_case)

print("===== Contextual Precision Evaluation =====\n")

print("Contextual Precision Score:", precision_metric.score)
print("Reason:", precision_metric.reason)
print("=" * 50)

# Contextual Recall Evaluation（上下文召回率評估）
recall_metric.measure(test_case)

print("===== Contextual Recall Evaluation =====\n")

print("Contextual Recall Score:", recall_metric.score)
print("Reason:", recall_metric.reason)
print("=" * 50)

# Contextual Relevancy Evaluation（上下文相關性評估）
relevancy_metric.measure(test_case)

print("===== Contextual Relevancy Evaluation =====\n")

print("Contextual Relevancy Score:", relevancy_metric.score)
print("Reason:", relevancy_metric.reason)
print("=" * 50)

# Faithfulness Evaluation（忠實度評估）
faithfulness_metric.measure(test_case)

print("===== Faithfulness Evaluation =====\n")

print("Faithfulness Score:", faithfulness_metric.score)
print("Reason:", faithfulness_metric.reason)
print("=" * 50)

# Answer Relevancy Evaluation（答案相關性評估）
answer_relevancy_metric.measure(test_case)

print("===== Answer Relevancy Evaluation =====\n")

print("Answer Relevancy Score:", answer_relevancy_metric.score)
print("Reason:", answer_relevancy_metric.reason)