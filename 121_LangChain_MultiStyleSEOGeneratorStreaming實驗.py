from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("===== Multi-Style SEO Generator Streaming 實驗 =====")

# LLM initialization (LLM 初始化)
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt template (提示詞模板)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一位頂尖 SEO 文案專家與內容行銷顧問。
        你的任務是根據使用者提供的文章內容，產生 10 個高點擊率 SEO 標題。

        要求：
        1. 一次產生 10 個標題
        2. 每個標題需不同
        3. 需符合指定風格
        4. 需吸引人、自然、具搜尋價值
        5. 使用繁體中文輸出

        可用風格：
        - 幽默風
        - 專業風
        - 驚喜風
        - 溫馨風
        - 恐怖風

        輸出格式：
        SEO 標題（{style}）：
        1.
        2.
        3.
        4.
        5.
        6.
        7.
        8.
        9.
        10.
        """
    ),
    (
        "user",
        """
        文章內容：
        {article}

        請產生 10 個 {style} SEO 標題
        """
    )
])

# Input (使用者輸入)
article = input("請輸入文章內容：")

print("\n請選擇風格：")
print("1. 幽默風")
print("2. 專業風")
print("3. 驚喜風")
print("4. 溫馨風")
print("5. 恐怖風")

choice = input("\n請輸入數字：")

# Style mapping (風格對應)
style_map = {
    "1": "幽默風",
    "2": "專業風",
    "3": "驚喜風",
    "4": "溫馨風",
    "5": "恐怖風"
}

style = style_map.get(choice, "專業風")

# Chain (建立鏈式結構)
chain = prompt | llm

print()
print("正在生成 SEO 標題...\n")

# Streaming output (串流輸出)
for chunk in chain.stream({
    "article": article,
    "style": style
}):
    print(chunk, end = "", flush = True)