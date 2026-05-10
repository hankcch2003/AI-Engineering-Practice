from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# LLM initialization（LLM 初始化）
# 初始化多個模型，用於比較不同模型的翻譯效果
llama = OllamaLLM(model = "llama3.2:latest")
gemma = OllamaLLM(model = "gemma2:2b")
phi = OllamaLLM(model = "phi3:mini")

# Prompt template（提示詞模板）
# 統一翻譯格式，確保不同模型輸出一致
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位多國語言翻譯助手，請只輸出翻譯結果，不要多餘解釋。"
    ),
    (
        "user",
        "請將以下文字翻譯成 {language}：{original}"
    )
])

# Create chains（建立多模型 Chain）
llama_chain = prompt | llama
gemma_chain = prompt | gemma
phi_chain = prompt | phi

# Streamlit title（網頁標題）
st.title("多模型翻譯比較工具")

# User input area（使用者輸入區）
article = st.text_area("請輸入需要翻譯的文字", height = 200)

# Target language selector（目標語言選擇）
target_language = st.selectbox(
    "目標語言", ["英文", "日文", "法文", "韓文", "阿拉伯文", "西班牙文", "俄文", "德文"])

# Execute button（執行按鈕）
if st.button("開始比較翻譯"):

    # Prevent empty input（避免空白輸入）
    if not article.strip():
        st.warning("請先輸入內容！")

    else:
        # Result title（結果標題）
        st.subheader("翻譯結果比較")

        # llama3.2 translation（llama3.2 翻譯）
        with st.spinner("llama3.2 翻譯中..."):
            llama_result = llama_chain.invoke({
                "original": article,
                "language": target_language
            })

        # gemma2 translation（gemma2 翻譯）
        with st.spinner("gemma2 翻譯中..."):
            gemma_result = gemma_chain.invoke({
                "original": article,
                "language": target_language
            })

        # phi3 translation（phi3 翻譯）
        with st.spinner("phi3 翻譯中..."):
            phi_result = phi_chain.invoke({
                "original": article,
                "language": target_language
            })

        # Display results（顯示結果）
        st.markdown("----- llama3.2 翻譯結果 -----")
        st.write(llama_result)
        print()

        st.markdown("----- gemma2 翻譯結果 -----")
        st.write(gemma_result)
        print()

        st.markdown("----- phi3 翻譯結果 -----")
        st.write(phi_result)

        # Success message（完成提示）
        st.success("三模型翻譯完成！")