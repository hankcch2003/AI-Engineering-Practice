import easyocr
import warnings
import ssl

# ignore all warnings (忽略所有警告輸出)
warnings.filterwarnings("ignore")

# SSL Fix (SSL 修復) - bypass certificate verification (跳過 SSL 驗證)
ssl._create_default_https_context = ssl._create_unverified_context

print("===== 208_EasyOCR_BasicRecognition實驗 =====\n")

# OCR Reader (OCR 辨識器)
reader = easyocr.Reader(['ch_tra', 'en'])

# Image path (圖片路徑)
image_path = "./images/invoice.png"

# OCR process (執行 OCR)
results = reader.readtext(image_path)

# Output result (輸出結果)
for bbox, text, prob in results:
    print("=" * 50)
    print(f"文字：{text}")
    print(f"信心指數：{prob:.4f}")