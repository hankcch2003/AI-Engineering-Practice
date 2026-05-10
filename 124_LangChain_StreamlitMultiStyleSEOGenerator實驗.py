from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# LLM initialization (LLM 初始化)
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt template (提示詞模板)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位頂尖 SEO 文案專家與內容行銷顧問。\n"
        "你的任務是根據使用者提供的文章內容與風格，生成 10 個 SEO 標題。\n"
        "要求：\n"
        "1. 一次產生 10 個標題\n"
        "2. 每個標題需不同\n"
        "3. 需符合指定風格\n"
        "4. 需吸引人、自然、具搜尋價值\n"
        "5. 使用繁體中文\n"
        "6. 只輸出標題，不要任何解釋\n"
        "7. 嚴格依照編號輸出，不可少於或多於\n\n"
        "輸出格式：\n"
        "SEO 標題：\n"
        "1. xxx\n"
        "2. xxx\n"
        "3. xxx\n"
        "4. xxx\n"
        "5. xxx\n"
        "6. xxx\n"
        "7. xxx\n"
        "8. xxx\n"
        "9. xxx\n"
        "10. xxx"
    ),
    (
        "user",
        "文章內容：{article}\n風格：{style}\n請生成 10 個 SEO 標題"
    )
])

# Chain (建立鏈式結構)
chain = prompt | llm

# Streamlit UI (使用者介面)
st.title("SEO標題生成器（多風格）")

# 使用者輸入區（文章內容）
article = st.text_area("請輸入文章內容：", height = 200)

# 風格選擇器
style = st.selectbox("選擇風格", ["幽默風", "專業風", "驚喜風", "溫馨風", "恐怖風"])

# 生成按鈕（事件觸發）
if st.button("生成 SEO 標題"):

    # 防呆：避免空輸入
    if not article.strip():
        st.warning("請先輸入內容！")

    else:
        st.subheader("產生結果：")

        # 動態輸出容器（streaming 更新用）
        output_box = st.empty()
        result = ""

        # Streaming output (串流輸出)
        for chunk in chain.stream({
            "article": article,
            "style": style
        }):
            result += chunk
            output_box.markdown(result)