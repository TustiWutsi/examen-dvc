import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor

def train_model(input_dir: str, param_file: str, output_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    
    with open(param_file, 'rb') as f:
        best_params = pickle.load(f)
    
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train.values.ravel())
    
    model_path = os.path.join(output_dir, 'trained_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Trained model saved in", model_path)

if __name__ == "__main__":
    train_model("data/processed", "models/best_params.pkl", "./models/")