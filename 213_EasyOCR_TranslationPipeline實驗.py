import easyocr
import warnings
import ssl
from deep_translator import GoogleTranslator

# ignore all warnings (忽略所有警告輸出)
warnings.filterwarnings("ignore")

# SSL Fix (SSL 修復) - bypass certificate verification (跳過 SSL 驗證)
ssl._create_default_https_context = ssl._create_unverified_context

print("===== 213_OCR_TranslationPipeline實驗 =====\n")

# OCR Reader (OCR 辨識器)
reader = easyocr.Reader(['ja', 'en'])

# Image path (圖片路徑)
image_path = "./images/trump.jpg"

# OCR process (執行 OCR)
results = reader.readtext(image_path)

ocrlist = []

# Output result (輸出結果)
for bbox, text, prob in results:
    print("=" * 50)
    print(f"文字：{text}")
    print(f"信心指數：{prob:.4f}")
    ocrlist.append(text)

# Combine text (合併文字)
full_text = " ".join(ocrlist)

print("=" * 50)
print("===== OCR 完整文字 =====\n")
print(full_text)

print("=" * 50)
print("開始進行翻譯.....")
print("=" * 50)

# Translation (翻譯)
translated = GoogleTranslator(
    source = 'en',
    target = 'zh-TW'
).translate(full_text)

print("翻譯結果：")
print(translated)
print("=" * 50)