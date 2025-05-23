# MLOps Project

This project contains a FastAPI application for a machine learning model that predicts customer support ticket types.

## To run locally

1. Create a virtual environment and activate it.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Run the application using `uvicorn main:app --reload`.

NOTE: Train endpoint will not run without updating envrionmental variables with AWS Access Key and Secret Key.

## Docker set up

To build and run the Docker container:

```sh
docker build -t mlops-app .
docker run -p 80:80 mlops-app
```

## Endpoints

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
  "ticket_subject": "Xbox not working",
  "ticket_priority": "critical",
  "combined_text": "Console is not turning on and red light is blinking, I want to return it"
}
```