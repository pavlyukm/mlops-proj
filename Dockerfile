FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3

COPY . .

# Create necessary directories
RUN mkdir -p mlruns encoders vectorizer model scripts

# Copy scripts
COPY scripts/ scripts/

# Initialize MLflow
RUN python scripts/init_mlflow.py || echo "MLflow init skipped"

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]