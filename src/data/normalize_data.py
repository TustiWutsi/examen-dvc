import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import requests
from import_raw_data import main

def normalize_data(input_dir: str, output_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    
    print("Data normalization completed. Processed files saved in", output_dir)

if __name__ == "__main__":    
    if not (os.path.exists("data/processed/X_train.csv") and os.path.exists("data/processed/X_test.csv")):
        main()
    normalize_data("data/processed", "data/processed")