# Customer Support Ticket Classifier App

MLOps platform for customer support ticket classification with Kubeflow integration, MLflow experiment tracking, and XGBoost-based models.

Dataset: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

## Quick Start

### Prerequisites
- Docker Desktop (8GB+ memory allocation recommended when using Kubeflow deployment)
- Kind (Kubernetes in Docker)
- AWS credentials configured
- Dataset uploaded to S3 as `dataset-tickets-multi-lang-4-20k.csv`

### Environment Setup

Create a `.env` file:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=pavliukmmlops
```

## Docker Compose Deployment (No Kubeflow required)

### Run the Application

```bash
docker-compose up -d
docker-compose ps
docker-compose logs -f app
```

## Kubeflow Deployment

### Prerequisites for Kubeflow Setup
- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/): `brew install kind`
- [kubectl](https://kubernetes.io/docs/tasks/tools/): `brew install kubectl`
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/v1/installation/localcluster-deployment/) installed

### 1. Setup Kind Cluster with Kubeflow

```bash
kind create cluster --name kubeflow
export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

### 2. Deploy ML Services to Kubeflow

```bash
kubectl create namespace ml-service

export $(cat .env | xargs)
kubectl create secret generic aws-credentials \
    --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --namespace=ml-service

docker build -t ticket-classifier:latest .
kind load docker-image ticket-classifier:latest --name kubeflow

kubectl apply -f k8s-deployment.yaml
```

### 3. Setup Port Forwarding

```
pkill -f "kubectl port-forward" 2>/dev/null
kubectl port-forward svc/ticket-classifier-service -n ml-service 8001:8000 &
kubectl port-forward svc/mlflow-service -n ml-service 5001:5000 &
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8082:80 &
kubectl port-forward svc/ml-pipeline -n kubeflow 8888:8888 &
```

### 4. Access Services

- **ML API**: http://localhost:8001
- **MLflow UI**: http://localhost:5001
- **Kubeflow Pipelines**: http://localhost:8082
- **Kubeflow API**: http://localhost:8888

## API Endpoints

### Base URL: `http://localhost:8001` (Kubeflow) or `http://localhost:8000` (Docker Compose)

#### GET `/`
- **Description**: API information and available endpoints
- **Response**: JSON with API details

#### GET `/health`
- **Description**: Health check
- **Response**: Service status and model loading status

#### POST `/train`
- **Description**: Train a new XGBoost model with hyperparameter tuning
- **Request Body**:
```json
{
  "max_depth": 8,
  "learning_rate": 0.05,
  "n_estimators": 200,
  "subsample": 0.7,
  "colsample_bytree": 0.7,
  "reg_alpha": 0,
  "reg_lambda": 1,
  "min_child_weight": 1
}
```
- **Response**: Training results with accuracy and champion/challenger promotion info

#### POST `/predict`
- **Description**: Classify a customer support ticket
- **Request Body** (all fields except subject and body are optional):
```json
{
  "subject": "Problem with Integration",
  "body": "The integration stopped working unexpectedly, causing synchronization errors",
  "answer": "I will look into the problem and call you to discuss it further",
  "type": "Problem",
  "priority": "high", 
  "language": "en",
  "tag_1": "Technical",
  "tag_2": "Integration",
  "tag_3": "Bug"
}
```
- **Response**:
```json
{
  "predicted_queue": "IT Support",
  "confidence": 0.92,
  "model_type": "xgboost",
  "model_version": "2",
  "probabilities": {
    "IT Support": 0.92,
    "Customer Service": 0.05,
    "Technical Support": 0.03
  }
}
```

#### GET `/model-info`
- **Description**: Get information about the current champion model
- **Response**: Model details, hyperparameters, and registry info

#### GET `/model-registry`
- **Description**: Get champion/challenger model information from MLflow registry

## Kubeflow Pipeline Integration

### Single-Component Pipeline

1. **Compile pipeline:**
```bash
python3 simple_kf_pipeline.py
```

2. **Upload to Kubeflow:**
   - Go to http://localhost:8082
   - Click "Pipelines" → "Upload Pipeline"
   - Upload `ticket_classifier_pipeline.yaml`

3. **Create and run experiment:**
   - Click "Experiments" → "Create Experiment"
   - Name: "customer-support-ml"
   - Create run with desired hyperparameters

### Multi-Component Pipeline (Advanced)

```bash
python3 multi_component_pipeline_fixed.py
```

**Pipeline Components:**
1. **Data Loading**: Load dataset from S3
2. **Preprocessing**: TF-IDF vectorization + feature engineering  
3. **Training**: XGBoost model training with hyperparameters
4. **Evaluation**: Model validation and deployment decision

**Resource Requirements:**
- Data Loading: 2Gi memory
- Preprocessing: 4Gi memory  
- Training: 6Gi memory
- Evaluation: 1Gi memory

## Architecture

### Core Components
- **`main.py`**: FastAPI application with Kubeflow integration
- **`train.py`**: Standalone training script
- **`predict.py`**: Prediction service with champion model loading
- **`data_utils.py`**: Data processing and S3 integration
- **`model_utils.py`**: XGBoost model creation and MLflow tracking

### Infrastructure
- **Kind Cluster**: Local Kubernetes with Kubeflow
- **MLflow**: Experiment tracking and model registry
- **S3**: Dataset and artifact storage
- **MySQL**: Kubeflow pipeline metadata
- **Minio**: Kubeflow artifact storage

## Usage Examples

### 1. Train Model (API)
```bash
curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.7
  }'
```

### 2. Train Model (Kubeflow Pipeline)
- Open http://localhost:8082
- Go to your experiment
- Create run with hyperparameters
- Monitor execution in visual pipeline graph

### 3. Make Predictions
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Security breach in medical data",
    "body": "We detected unauthorized access to patient records",
    "type": "Incident",
    "priority": "critical",
    "language": "en"
  }'
```

### 4. Monitor Experiments
- **MLflow**: http://localhost:5001 - Model metrics, parameters, artifacts
- **Kubeflow**: http://localhost:8082 - Pipeline runs, visual graphs

## Model Management

### Champion/Challenger System
- **Champion**: Current production model (highest accuracy)
- **Challenger**: New models competing for promotion
- **Automatic Promotion**: New models become champion if accuracy improves
- **Model Registry**: Track all model versions in MLflow

### Experiment Tracking
- All training runs logged to MLflow
- Hyperparameters, metrics, and artifacts stored
- S3 integration for large model artifacts
- Champion/challenger tags for easy identification

## Data Processing Pipeline

### Automated Feature Engineering
1. **Text Processing**: TF-IDF vectorization (5000 features)
2. **Categorical Encoding**: Label encoding for metadata fields
3. **Feature Combination**: Text + categorical features
4. **Data Splitting**: 80/20 train/test with stratification

### Supported Data Formats
- **Primary**: CSV files with customer support tickets
- **Required Columns**: `subject`, `body`, `queue` (target)
- **Optional Columns**: `answer`, `type`, `priority`, `language`, `tag_*`
- **International Support**: UTF8MB4 encoding for global datasets

## Troubleshooting

### Memory Issues
```bash
# Check cluster resources
kubectl describe node kubeflow-control-plane | grep -A 5 "Allocated resources"

# Increase Docker Desktop memory (Recommended: 8GB+)
# Docker Desktop → Settings → Resources → Memory
```

### Pipeline Stuck/Failed
```bash
# Check pipeline pod logs
kubectl get pods -n kubeflow | grep multi-component
kubectl logs <pod-name> -n kubeflow

# Clean up failed runs
kubectl delete pods -n kubeflow -l workflows.argoproj.io/completed=true
```

## Shutdown

```bash
# Stop port forwards
pkill -f "kubectl port-forward"

# Scale down deployments (preserves data)
kubectl scale deployment --all --replicas=0 -n ml-service
kubectl scale deployment --all --replicas=0 -n kubeflow

# Delete cluster (complete cleanup)
kind delete cluster --name kubeflow
```
---
