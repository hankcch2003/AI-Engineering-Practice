# Base image (基礎映像)
FROM python:3.10-slim

# Working directory (工作目錄)
WORKDIR /app

# Copy files (複製檔案)
COPY . /app

# Install dependency (安裝套件)
RUN pip install --no-cache-dir requests

# Run script (執行程式)
CMD ["python", "main.py"]