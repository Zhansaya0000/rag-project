FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY scipy-1.13.1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl .

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install numpy==1.26.4
RUN pip install --no-cache-dir scipy-1.13.1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

RUN pip install faiss-cpu
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=1200 --retries=20 -i https://pypi.org/simple

RUN pip cache purge
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
