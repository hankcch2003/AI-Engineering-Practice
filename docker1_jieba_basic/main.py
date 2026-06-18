import jieba

# Input text (輸入文字)
text = "我正在學習Docker與人工智慧整合應用"

# Word segmentation (斷詞處理)
words = jieba.lcut(text)

# Output result (輸出結果)
print("輸出結果：")
print("/ ".join(words))