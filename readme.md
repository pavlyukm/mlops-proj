# MLOps Project

This project contains a FastAPI application for a machine learning model that predicts customer support ticket types.

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install the dependencies using `pip install -r requirements.txt`.
4. Run the application using `uvicorn main:app --reload`.

## Docker

To build and run the Docker container:

```sh
docker build -t mlops-app .
docker run -p 80:80 mlops-app
