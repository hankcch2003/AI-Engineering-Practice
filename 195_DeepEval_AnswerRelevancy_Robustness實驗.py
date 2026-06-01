from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric

print("===== 195_DeepEval_AnswerRelevancy_Robustness實驗 =====\n")

# Test Case（測試案例）
# 使用不相關答案測試模型對答案相關性的判斷能力
test_case = LLMTestCase(
    input = "Python是甚麼",
    actual_output = "台北101很高。"
)

# Ollama Model（評估模型）
ollama_model = OllamaModel(model = "llama3.2:latest")

# Metric（答案相關性評估）
metric = AnswerRelevancyMetric(
    threshold = 0.7,
    model = ollama_model
)

# Run Evaluation（執行評估）
score = metric.measure(test_case)

# Output Result（輸出結果）
print("Score:", score)
print("Reason:", metric.reason)