# Save as simple_kf_pipeline.py
import kfp
from kfp import dsl
from kfp.dsl import component

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "requests==2.31.0",
        "pandas==2.1.4"
    ]
)
def trigger_training_component(
    api_endpoint: str = "http://ticket-classifier-service.ml-service.svc.cluster.local:8000",
    max_depth: int = 6,
    n_estimators: int = 100,
    learning_rate: float = 0.1
) -> str:
    """Trigger training via API call"""
    import requests
    import json
    import time
    
    training_params = {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate
    }
    
    print(f"Triggering training with params: {training_params}")
    
    try:
        response = requests.post(
            f"{api_endpoint}/train",
            json=training_params,
            timeout=1800  # 30 minutes
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Training completed successfully!")
            print(f"Accuracy: {result.get('accuracy', 'Unknown')}")
            print(f"Run ID: {result.get('run_id', 'Unknown')}")
            return f"Training successful - Accuracy: {result.get('accuracy', 'N/A')}"
        else:
            print(f"Training failed: {response.status_code}")
            print(f"Response: {response.text}")
            return f"Training failed: {response.status_code}"
            
    except Exception as e:
        print(f"Error calling training API: {e}")
        return f"Error: {str(e)}"

@dsl.pipeline(
    name="ticket-classifier-training",
    description="Train customer support ticket classifier"
)
def training_pipeline(
    max_depth: int = 6,
    n_estimators: int = 50,
    learning_rate: float = 0.1
):
    training_task = trigger_training_component(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path='ticket_classifier_pipeline.yaml'
    )
    print("Pipeline compiled to ticket_classifier_pipeline.yaml")