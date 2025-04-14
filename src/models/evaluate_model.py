import pandas as pd
import os
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(input_dir: str, model_file: str, output_dir: str, metrics_dir: str):
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    pd.DataFrame({'Actual': y_test.values.ravel(), 'Predicted': y_pred}).to_csv(predictions_path, index=False)
    
    os.makedirs(metrics_dir, exist_ok=True)
    scores_path = os.path.join(metrics_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump({'MSE': mse, 'R2': r2}, f, indent=4)
    
    print("Predictions saved in", predictions_path)
    print("Evaluation metrics saved in", scores_path)

if __name__ == "__main__":
    evaluate_model("./data/processed", "./models/trained_model.pkl", "./data/", "./metrics/")