import requests

# API endpoint (API位置)
url = "http://ollama:11434/api/generate"

# Request payload (請求內容)
data = {
    "model": "llama3.2:latest",
    "prompt": "Explain Docker in simple terms"
}

# Send request (發送請求)
response = requests.post(url, json = data)

# Print result (輸出結果)
print(response.text)