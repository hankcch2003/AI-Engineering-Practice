# Base image (基礎映像)
FROM python:3.10-slim

# Working directory (工作目錄)
WORKDIR /app

# Copy files (複製檔案)
COPY . /app

# Install dependencies (安裝套件)
RUN pip install --no-cache-dir streamlit

# Expose port (開放埠號)
EXPOSE 8501

# Run Streamlit (啟動服務)
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]