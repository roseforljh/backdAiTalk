# ---- Build Stage ----
FROM python:3.9-slim as builder

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Final Stage ----
FROM python:3.9-slim

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY . .

# 确保Python可以找到已安装的库
ENV PATH=/root/.local/bin:$PATH

# 暴露端口
EXPOSE 7860

# 运行应用
CMD ["uvicorn", "eztalk_proxy.main:app", "--host", "0.0.0.0", "--port", "7860"]