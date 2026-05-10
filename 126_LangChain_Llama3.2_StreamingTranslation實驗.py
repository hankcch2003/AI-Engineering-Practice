from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st

# LLM initialization（LLM 初始化）
# streaming 模式會即時輸出翻譯結果，讓使用者能夠看到翻譯過程中的變化，提升互動體驗。
llm = OllamaLLM(model = "llama3.2:latest", streaming = True)

# Prompt template（提示詞模板）
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位多國語言翻譯助手，能夠將用戶提供的文字翻譯成多種語言。"
        "請根據用戶的要求，翻譯成指定的目標語言。"
        "輸出格式為：\n原文：\n目標語言：\n翻譯："
    ),
    (
        "user",
        "請將以下文字翻譯成 {language}：{original}"
    )
])

# Create chain（建立 Chain）
chain = prompt | llm

# Streaming callback handler（Streaming 回呼處理器）
class StreamHandler(BaseCallbackHandler):

    # Initialize container（初始化容器）
    def __init__(self, container):
        self.container = container
        self.text = ""

    # Update streaming text（即時更新文字）
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Streamlit title（網頁標題）
st.title("中文翻譯工具")

# User input area（使用者輸入區）
article = st.text_area("請輸入需要翻譯的文字", height = 200)

# Target language selector（目標語言選擇）
target_language = st.selectbox(
    "目標語言", ["英文", "日文", "法文", "韓文", "阿拉伯文", "西班牙文", "俄文", "德文"]
)

# Translation button（翻譯按鈕）
if st.button("進行翻譯"):

    # Prevent empty input（避免空白輸入）
    if not article.strip():
        st.warning("請先輸入內容！")

    else:
        # Result title（結果標題）
        st.subheader("翻譯結果：")

        # Output container（輸出容器）
        output_box = st.empty()

        # Initialize handler（初始化回呼處理器）
        handler = StreamHandler(output_box)

        # Show spinner while translating（翻譯過程中顯示轉圈圖示）
        with st.spinner("翻譯中，請稍候..."):

            # Execute chain（執行 Chain）
            chain.invoke(
                {
                    "original": article,
                    "language": target_language
                },
                {
                    "callbacks": [handler]
                }
            )

        # Show success message（顯示成功訊息）
        st.success("翻譯完成！")