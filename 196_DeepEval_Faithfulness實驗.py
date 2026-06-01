from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric

print("===== 196_DeepEval_Faithfulness實驗 =====\n")

# Test Case（測試案例）
test_case = LLMTestCase(
    input = "Python是甚麼",
    actual_output = "公司年假為7天。",
    expected_output = "員工年假為7天。",
    retrieval_context = [
        "員工年假為7天。",
        "病假一年可請30天。"
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