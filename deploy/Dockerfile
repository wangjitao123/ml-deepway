# ──────────────────────────────────────────────────────
# 多阶段构建：车辆故障预测API服务镜像
# ──────────────────────────────────────────────────────

# ===== 第一阶段：构建阶段 =====
FROM python:3.10-slim AS builder

# 设置工作目录
WORKDIR /build

# 安装系统依赖（编译某些包需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖到独立目录（避免污染系统包）
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ===== 第二阶段：运行阶段 =====
FROM python:3.10-slim AS final

# 设置元数据标签
LABEL maintainer="ML DeepWay Team"
LABEL description="工业级智能车辆故障预测API服务"
LABEL version="1.0.0"

# 设置工作目录
WORKDIR /app

# 安装运行时系统依赖（精简版）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已安装的包
COPY --from=builder /install /usr/local

# 复制项目源代码
COPY . .

# 创建必要目录
RUN mkdir -p checkpoints evaluation_results runs

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露API端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "serving.api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
