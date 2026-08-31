FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Word 文档转 PDF；纯 PDF/Markdown 部署可在自定义镜像中移除该系统依赖。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/knowledge/requirements.txt
RUN pip install --no-cache-dir -r /app/knowledge/requirements.txt

COPY . /app/knowledge

EXPOSE 18000 18001

# 默认启动入库服务；查询容器通过 docker run 的 command 覆盖该命令。
CMD ["python", "-m", "uvicorn", "knowledge.api.import_api:app", "--host", "0.0.0.0", "--port", "18000", "--workers", "1"]
