FROM python:3.12-slim

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-serialization-dev \
    libboost-python-dev \
    libboost-regex-dev \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libxrender1 \
    libsm6 \
    libxt6 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの作成
WORKDIR /app

# 必要なPythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコピー
COPY src/ /app/src/

# ポートの公開
EXPOSE 8501

# Streamlitを実行
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]