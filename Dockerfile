# 使用一个标准的、轻量级的 Python 镜像
FROM python:3.9-slim

# 设置工作目录，后续所有操作都在这个目录里
WORKDIR /app

# 将当前文件夹（包含run.py, requirements.txt等）的所有内容复制到容器的/app目录
COPY . .

# 安装在 requirements.txt 中定义的所有依赖项
# 使用 --no-cache-dir 来减小最终镜像的体积
RUN pip install --no-cache-dir -r requirements.txt

# 暴露我们在README.md中为应用指定的端口
EXPOSE 7860

# 容器启动时执行的最终命令
# 它会用uvicorn来运行run.py文件中的名为"app"的FastAPI实例
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "7860"]