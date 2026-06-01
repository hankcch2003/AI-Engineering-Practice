from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric

print("===== 199_DeepEval_Faithfulness_MixedNoise實驗 =====\n")

# Test Case（測試案例）
# 混合錯誤與干擾資訊，測試模型穩定性
test_case = LLMTestCase(
    input = "Python是甚麼",
    actual_output = "Python是OpenAI在2025年開發的作業系統。",
    expected_output = "Python是一種作業系統。",
    retrieval_context = [
        "Python是一種2025年高階程式語言。",
        "Python常用於OpenAI與資料分析。"
    ]
)

# Ollama Model（評估模型）
ollama_model = OllamaModel(model = "llama3.2:latest")

# Metric（忠實度評估）
metric = FaithfulnessMetric(
    threshold = 0.7,
    model = ollama_model
)

# Run Evaluation（執行評估）
score = metric.measure(test_case)

# Output Result（輸出結果）
print("Score:", score)
print("Reason:", metric.reason)