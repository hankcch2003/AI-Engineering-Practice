from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import ContextualRelevancyMetric

print("===== 193_DeepEval_ContextualRelevancy_Robustness實驗 =====\n")

# Test Case（測試案例）
# 使用「噪音 retrieval_context」進行 robustness 測試
# 用來觀察 irrelevant context 對評分的影響
test_case = LLMTestCase(
    input = "Python是甚麼",
    actual_output = "Python是一種程式語言。",
    expected_output = "Python是一種程式語言。",
    retrieval_context = [
        "香蕉是黃色水果",
        "貓咪喜歡睡覺"
    ]
)

# Ollama Model（評估模型）
ollama_model = OllamaModel(model = "llama3.2:latest")

# Metric（Contextual Relevancy 評估）
metric = ContextualRelevancyMetric(
    threshold = 0.7,
    model = ollama_model
)

# Run Evaluation（執行評估）
score = metric.measure(test_case)

# Output result（輸出結果）
print("Score:", score)
print("Reason:", metric.reason)