FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
