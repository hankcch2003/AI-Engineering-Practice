from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import OllamaModel

print("===== 190_DeepEval_ContextualPrecision_FailureCase實驗 =====\n")

# Test Case（測試案例：刻意錯誤輸出，用來測試模型評估能力）
test_case = LLMTestCase(
    input = "Python是甚麼",
    actual_output = "Python是蛇類動物。",
    expected_output = "Python是一種程式語言。",
    retrieval_context = [
        "香蕉是黃色水果。",
        "台北101很高。",
        "貓咪喜歡睡覺。"
    ]
)

# Ollama Eval Model（評估模型）
ollama_model = OllamaModel(model = "llama3.2:latest")

# Metric（上下文精準度評估）
metric = ContextualPrecisionMetric(
    threshold = 0.7,
    model = ollama_model
)

# Run Evaluation（執行評估）
score = metric.measure(test_case)

# Output result（輸出結果）
print("Score:", score)
print("Reason:", metric.reason)