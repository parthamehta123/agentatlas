FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV AGENTATLAS_INFRA=/data/infra
EXPOSE 8080
CMD ["uvicorn", "agentatlas.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
