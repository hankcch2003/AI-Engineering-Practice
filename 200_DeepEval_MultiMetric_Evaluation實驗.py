from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

print("===== 200_DeepEval_MultiMetric_Evaluation實驗 =====\n")

# Test Case（測試案例）
# 同時測試 Faithfulness + Answer Relevancy
test_case = LLMTestCase(
    input = "Python是甚麼",
    actual_output = "台北101很高。",
    retrieval_context = ["台北101很高。"]
)

# Ollama Model（評估模型）
ollama_model = OllamaModel(model = "llama3.2:latest")

# Metric 1（忠實度）
faithfulness_metric = FaithfulnessMetric(
    threshold = 0.7,
    model = ollama_model
)

# Metric 2（答案相關性）
relevancy_metric = AnswerRelevancyMetric(
    threshold = 0.7,
    model = ollama_model
)

# Run Evaluation（Faithfulness）
print("===== Faithfulness Evaluation =====\n")

faith_score = faithfulness_metric.measure(test_case)
print("Faithfulness Score:", faith_score)
print("Reason:", faithfulness_metric.reason)

print("=" * 50)

# Run Evaluation（Answer Relevancy）
print("===== Answer Relevancy Evaluation =====\n")

rel_score = relevancy_metric.measure(test_case)
print("Relevancy Score:", rel_score)
print("Reason:", relevancy_metric.reason)