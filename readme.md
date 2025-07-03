# Customer Support Ticket Classifier v2.0

## Quick Start

### Prerequisites
- Docker and Docker Compose
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

### Run the Application

```bash
# Start the services
docker-compose up -d

# Check if services are healthy
docker-compose ps

# View logs
docker-compose logs -f app
```

## API Endpoints

### Base URL: `http://localhost:8000`

#### GET `/`
- **Description**: API information and available endpoints
- **Response**: JSON with API details

#### GET `/health`
- **Description**: Health check
- **Response**: Service status and model loading status

#### POST `/train`
- **Description**: Train a new model using the dataset from S3
- **Response**: Training results with accuracy and run information
- **Note**: This will download data from S3, train the model, and reload the prediction service

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
  "probabilities": {
    "IT Support": 0.92,
    "Customer Service": 0.05,
    "Technical Support": 0.03
  }
}
```

#### GET `/model-info`
- **Description**: Get information about the current model
- **Response**: Model details, classes, and input shape

## Architecture

### Simplified Components
- **`main.py`**: FastAPI application with clean endpoints
- **`train.py`**: Standalone training script
- **`predict.py`**: Prediction service with model loading
- **`data_utils.py`**: Data processing and S3 integration
- **`model_utils.py`**: Model creation and MLflow integration


## Usage Examples

### 1. Train a Model
```bash
curl -X POST http://localhost:8000/train
```

### 2. Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Security breach in medical data",
    "body": "We detected unauthorized access to patient records",
    "answer": "Thank you for reporting this. We will investigate immediately",
    "type": "Incident",
    "priority": "high",
    "language": "en",
    "tag_1": "Security",
    "tag_2": "Breach"
  }'
```

**Minimal prediction** (only required fields):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Need help with billing",
    "body": "I have questions about my invoice"
  }'
```

### 3. Check Model Status
```bash
curl http://localhost:8000/model-info
```

## MLflow Integration

- **Tracking Server**: http://localhost:5000
- **Experiment**: `customer_support_classification_v2`
- **Model Registry**: `customer_support_classifier`
- **Artifacts**: Stored in S3 at `s3://your-bucket/mlflow-v2/`

## Data Processing

The system automatically detects dataset structure:
- **Text columns**: Looks for columns containing 'text', 'message', 'description', 'content', 'ticket'
- **Label columns**: Looks for columns containing 'label', 'category', 'class', 'type', 'priority'
- **Fallback**: Uses first string column as text, last column as labels

### Preprocessing Pipeline
1. Load CSV from S3
2. Auto-detect text and label columns
3. Clean missing data
4. TF-IDF vectorization (5000 features)
5. Label encoding
6. Train/test split (80/20)
