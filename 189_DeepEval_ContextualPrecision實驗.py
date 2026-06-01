import warnings

# 忽略警告訊息（避免 DeepEval / LangChain 輸出干擾）
warnings.filterwarnings("ignore")

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import OllamaModel

print("===== 189_DeepEval_ContextualPrecision實驗 =====\n")

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

# Ollama Eval Model（評測用模型）
ollama_model = OllamaModel(model = "llama3.2:latest")

# Metric（評估指標：上下文精準度）
metric = ContextualPrecisionMetric(
    threshold = 0.7,
    model = ollama_model
)

# Run Evaluation（執行評估）
score = metric.measure(test_case)

# Output result（輸出結果）
print("Score:", score)
print("Reason:", metric.reason)