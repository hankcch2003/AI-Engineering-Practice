import easyocr
import warnings
import ssl
import re
import json

# ignore all warnings (忽略所有警告輸出)
warnings.filterwarnings("ignore")

# SSL Fix (SSL 修復) - bypass certificate verification (跳過 SSL 驗證)
ssl._create_default_https_context = ssl._create_unverified_context

print("===== 214_EasyOCR_InvoiceDataParsing實驗 =====\n")

# OCR Reader (OCR 辨識器)
reader = easyocr.Reader(['ch_tra', 'en'])

# Invoice extraction function (發票資料擷取)
def extract_invoice_data(image_path):

    # OCR process (執行 OCR)
    results = reader.readtext(image_path, detail = 0)

    # Combine text (合併文字)
    full_text = " ".join(results)

    print("=" * 50)
    print("===== OCR 完整文字 =====")
    print()
    print(full_text)

    print("=" * 50)
    print("開始進行資料解析.....")
    print("=" * 50)

    # Tax ID extraction (統編)
    tax_ids = re.findall(r'\b\d{8}\b', full_text)

    # Amount extraction (金額)
    amount_match = re.search(
        r'(?:總計|金額|Total|SUM)[:\s]*\$?([\d,]+)',
        full_text,
        re.IGNORECASE
    )
    amount = amount_match.group(1).replace(',', '') if amount_match else "未偵測到"

    # JSON output (輸出結果)
    data = {
        "status": "success",
        "tax_id": tax_ids[0] if tax_ids else "未偵測到",
        "total_amount": amount,
        "raw_text_snippets": results[:10]
    }
    return json.dumps(data, ensure_ascii = False, indent = 4)

# Run pipeline (執行流程)
result_json = extract_invoice_data("./images/invoice.png")

print("資料剖析結果：")
print(result_json)
print("=" * 50)