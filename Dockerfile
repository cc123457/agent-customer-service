# 后端服务镜像：FastAPI + uvicorn
# 用官方 slim 基础镜像（体积小）；chromadb 依赖 onnxruntime，运行需要 libgomp1
FROM python:3.13-slim

# 容器内工作目录
WORKDIR /app

# 先只复制依赖文件：依赖不变时 Docker 会复用这一层缓存，改代码就不用重装依赖
COPY requirements.txt .

# 先装系统库（onnxruntime 运行需要 libgomp1），再装 Python 依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# 再复制项目代码（.dockerignore 已排除 .venv / .env / 数据等）
COPY . .

# 暴露 FastAPI 端口
EXPOSE 8000

# 启动后端
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
