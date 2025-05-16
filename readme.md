# MLOps Project

This project contains a FastAPI application for a machine learning model that predicts customer support ticket types.

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install the dependencies using `pip install -r requirements.txt`.
4. Run the application using `uvicorn main:app --reload`.

NOTE: Train endpoint will not run without updating envrionmental variables with AWS Access Key and Secret Key.

## Docker

To build and run the Docker container:

```sh
docker build -t mlops-app .
docker run -p 80:80 mlops-app
```

## Endpoints

### GET/health
Healthcheck endpoint to show if the web server is healthy

### POST/train
Request that loads training data from S3 bucket into the model. 
When new training data is uplaoded, code should be changed to reflect new parquet file name

### GET/labels
Get a list of possible priority names and purhcaased products that are included in the encoder files that help us build the payload for /predict request.

### POST/predict
Predicts the ticket type based on the payload data.
Example request:
```
{
  "customer_email": "email@gmail.com",
  "product_purchased": "xbox",
  "ticket_subject": "Xbox is not turing on",
  "ticket_priority": "critical",
  "combined_text": "Please help my console is not turning on and the red light is blinking"
}
```