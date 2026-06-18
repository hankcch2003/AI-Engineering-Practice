# Base image (基礎映像)
FROM python:3.10-slim

# Working directory (工作目錄)
WORKDIR /app

# Copy files (複製專案)
COPY . /app

# Install dependencies (安裝套件)
RUN pip install --no-cache-dir -r requirements.txt

# Default command (預設執行 chatbot)
CMD ["python", "chatbot.py"]