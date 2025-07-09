# xgboost_tuning_examples.py
"""
Examples of different XGBoost hyperparameter configurations to test
Run these via API calls to find the best performing model
"""

import requests
import json

# API endpoint
API_BASE = "http://localhost:8000"

# Different hyperparameter configurations to test
HYPERPARAMETER_CONFIGS = [
    {
        "name": "baseline",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "min_child_weight": 1
        }
    },
    {
        "name": "faster_training",
        "params": {
            "max_depth": 4,
            "learning_rate": 0.2,
            "n_estimators": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "min_child_weight": 1
        }
    },
    {
        "name": "deeper_trees",
        "params": {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "min_child_weight": 1
        }
    },
    {
        "name": "regularized",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2,
            "min_child_weight": 3
        }
    },
    {
        "name": "conservative",
        "params": {
            "max_depth": 3,
            "learning_rate": 0.01,
            "n_estimators": 500,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "min_child_weight": 5
        }
    }
]

def train_model_with_params(params, config_name):
    """Train model with specific hyperparameters"""
    try:
        response = requests.post(
            f"{API_BASE}/train",
            json=params,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n{config_name} - Training successful!")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   Promoted: {result['promotion_result'].get('promoted', False)}")
            print(f"   Reason: {result['promotion_result'].get('reason', 'Unknown')}")
            return result
        else:
            print(f"\n{config_name} - Training failed!")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n{config_name} - Exception: {e}")
        return None

def get_model_registry_info():
    """Get current model registry information"""
    try:
        response = requests.get(f"{API_BASE}/model-registry")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get registry info: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception getting registry info: {e}")
        return None

def test_prediction():
    """Test prediction with current champion"""
    test_data = {
        "subject": "Login Problem",
        "body": "I cannot access my account, getting authentication error",
        "type": "Problem",
        "priority": "high",
        "language": "en"
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction Test:")
            print(f"   Predicted Queue: {result['predicted_queue']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Model Version: {result.get('model_version', 'Unknown')}")
            return result
        else:
            print(f"\nPrediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"\nPrediction exception: {e}")
        return None

def main():
    """Run hyperparameter tuning experiment"""
    print("🚀 Starting XGBoost Hyperparameter Tuning Experiment")
    print("=" * 60)
    
    # Get initial registry state
    print("\nInitial Model Registry State:")
    registry_info = get_model_registry_info()
    if registry_info:
        print(f"   Total versions: {registry_info.get('total_versions', 0)}")
        champion = registry_info.get('champion')
        if champion:
            print(f"   Current champion: Version {champion['version']} (Accuracy: {champion.get('accuracy', 'Unknown')})")
        else:
            print("   No current champion")
    
    # Test each configuration
    results = []
    for config in HYPERPARAMETER_CONFIGS:
        print(f"\nTesting configuration: {config['name']}")
        print(f"   Parameters: {config['params']}")
        
        result = train_model_with_params(config['params'], config['name'])
        if result:
            results.append({
                'config_name': config['name'],
                'params': config['params'],
                'accuracy': result['accuracy'],
                'promoted': result['promotion_result'].get('promoted', False),
                'reason': result['promotion_result'].get('reason', 'Unknown')
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    if results:
        # Sort by accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n🏆 Best performing configurations:")
        for i, result in enumerate(results[:3]):  # Top 3
            print(f"   {i+1}. {result['config_name']}: {result['accuracy']:.4f} {'(PROMOTED)' if result['promoted'] else ''}")
        
        print(f"\n📊 All results:")
        for result in results:
            status = "CHAMPION" if result['promoted'] else "CHALLENGER"
            print(f"   {result['config_name']}: {result['accuracy']:.4f} {status}")
    
    # Final registry state
    print("\nFinal Model Registry State:")
    registry_info = get_model_registry_info()
    if registry_info:
        print(f"   Total versions: {registry_info.get('total_versions', 0)}")
        champion = registry_info.get('champion')
        if champion:
            print(f"   Current champion: Version {champion['version']} (Accuracy: {champion.get('accuracy', 'Unknown')})")
        
        challenger = registry_info.get('challenger')
        if challenger:
            print(f"   Current challenger: Version {challenger['version']} (Accuracy: {challenger.get('accuracy', 'Unknown')})")
    
    # Test prediction with final champion
    test_prediction()
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()


# Example usage commands:

"""
# 1. Train with default parameters
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{}'

# 2. Train with custom parameters (faster model)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "max_depth": 4,
    "learning_rate": 0.2,
    "n_estimators": 50
  }'

# 3. Train with deeper trees
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.7
  }'

# 4. Check model registry
curl http://localhost:8000/model-registry

# 5. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Security Issue",
    "body": "Found a potential security vulnerability",
    "type": "Incident",
    "priority": "critical",
    "language": "en"
  }'
"""